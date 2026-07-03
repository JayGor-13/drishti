from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MoEDiagnostics:
    """Rich routing diagnostics for MoE monitoring (DeepSeek-style)."""

    balance_loss: Tensor
    # Per-expert fraction of tokens routed (shape: [num_experts])
    expert_utilization: Tensor
    # Mean routing probability per expert (shape: [num_experts])
    routing_probabilities: Tensor
    # Shannon entropy of the router distribution, averaged over tokens (scalar)
    router_entropy: Tensor
    # Fraction of tokens that received zero expert assignment (scalar)
    token_drop_rate: Tensor
    # Average number of times each expert is reused per token (scalar)
    expert_reuse_frequency: Tensor
    # Coefficient of variation of expert load — lower is better balanced (scalar)
    load_balance_cv: Tensor


class Expert(nn.Module):
    """Single feed-forward expert."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SparseMoE(nn.Module):
    """Sparse top-k Mixture-of-Experts with load-balancing loss."""

    def __init__(
        self,
        d_model: int = 256,
        num_experts: int = 8,
        top_k: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        dense: bool = False,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.dense = dense
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, ffn_dim, dropout) for _ in range(num_experts)])

    def forward(self, x: Tensor) -> tuple[Tensor, MoEDiagnostics]:
        *leading, dim = x.shape
        x_flat = x.reshape(-1, dim)
        num_tokens = x_flat.shape[0]
        probs = torch.softmax(self.router(x_flat), dim=-1)

        # --- Router entropy: H = -sum(p * log(p)) averaged over tokens ---
        log_probs = torch.log(probs.clamp_min(1e-8))
        per_token_entropy = -(probs * log_probs).sum(dim=-1)
        router_entropy = per_token_entropy.mean()

        if self.dense:
            expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
            out = (expert_outputs * probs.unsqueeze(-1)).sum(dim=1)
            # In dense mode all experts process all tokens — perfect balance
            diag = MoEDiagnostics(
                balance_loss=probs.new_tensor(0.0),
                expert_utilization=probs.new_ones(self.num_experts) / self.num_experts,
                routing_probabilities=probs.mean(dim=0).detach(),
                router_entropy=router_entropy.detach(),
                token_drop_rate=probs.new_tensor(0.0),
                expert_reuse_frequency=probs.new_tensor(float(self.num_experts)),
                load_balance_cv=probs.new_tensor(0.0),
            )
            return out.reshape(*leading, dim), diag

        top_probs, top_indices = probs.topk(self.top_k, dim=-1)
        top_weights = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        out = torch.zeros_like(x_flat)

        for rank in range(self.top_k):
            expert_ids = top_indices[:, rank]
            weights = top_weights[:, rank]
            for expert_idx, expert in enumerate(self.experts):
                mask = expert_ids == expert_idx
                if mask.any():
                    out[mask] += expert(x_flat[mask]) * weights[mask].unsqueeze(-1)

        # --- Dispatch matrix for load-balance loss ---
        dispatch = torch.zeros_like(probs)
        dispatch.scatter_add_(1, top_indices, torch.ones_like(top_probs))
        fraction = dispatch.mean(dim=0) / float(self.top_k)
        probability = probs.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(fraction * probability)

        # --- Expert utilization: fraction of tokens routed to each expert ---
        expert_load = dispatch.sum(dim=0)  # total assignments per expert
        expert_utilization = expert_load / (num_tokens * self.top_k)

        # --- Token drop rate: tokens with zero assignment (shouldn't happen with top-k but logged for safety) ---
        tokens_assigned = (dispatch.sum(dim=-1) > 0).float()
        token_drop_rate = 1.0 - tokens_assigned.mean()

        # --- Expert reuse frequency: avg experts used per token ---
        experts_per_token = (dispatch > 0).float().sum(dim=-1)
        expert_reuse_frequency = experts_per_token.mean()

        # --- Coefficient of variation of expert load ---
        load_mean = expert_load.mean()
        load_std = expert_load.std()
        load_balance_cv = load_std / load_mean.clamp_min(1e-8)

        diag = MoEDiagnostics(
            balance_loss=balance_loss,
            expert_utilization=expert_utilization.detach(),
            routing_probabilities=probability.detach(),
            router_entropy=router_entropy.detach(),
            token_drop_rate=token_drop_rate.detach(),
            expert_reuse_frequency=expert_reuse_frequency.detach(),
            load_balance_cv=load_balance_cv.detach(),
        )
        return out.reshape(*leading, dim), diag
