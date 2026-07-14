from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MoEDiagnostics:
    balance_loss: Tensor
    expert_utilization: Tensor
    routing_probabilities: Tensor
    router_entropy: Tensor
    token_drop_rate: Tensor
    expert_reuse_frequency: Tensor
    load_balance_cv: Tensor
    expert_overlap: Tensor
    router_z_loss: Tensor


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

    def _diagnostics(self, probs: Tensor, dispatch: Tensor, balance_loss: Tensor, z_loss: Tensor) -> MoEDiagnostics:
        utilization = dispatch.mean(dim=0)
        mean_prob = probs.mean(dim=0)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        load_cv = utilization.std(unbiased=False) / utilization.mean().clamp_min(1e-8)
        token_drop_rate = probs.new_tensor(0.0)
        assigned = dispatch > 0
        overlaps = []
        for left in range(self.num_experts):
            for right in range(left + 1, self.num_experts):
                intersection = (assigned[:, left] & assigned[:, right]).to(probs.dtype).sum()
                union = (assigned[:, left] | assigned[:, right]).to(probs.dtype).sum().clamp_min(1.0)
                overlaps.append(intersection / union)
        expert_overlap = torch.stack(overlaps).mean() if overlaps else probs.new_tensor(0.0)
        return MoEDiagnostics(
            balance_loss=balance_loss,
            expert_utilization=utilization.detach(),
            routing_probabilities=probs.detach(),
            router_entropy=entropy.detach(),
            token_drop_rate=token_drop_rate,
            expert_reuse_frequency=mean_prob.detach(),
            load_balance_cv=load_cv.detach(),
            expert_overlap=expert_overlap.detach(),
            router_z_loss=z_loss,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, MoEDiagnostics]:
        *leading, dim = x.shape
        x_flat = x.reshape(-1, dim)
        router_logits = self.router(x_flat)
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        probs = torch.softmax(router_logits, dim=-1)

        if self.dense:
            expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
            out = (expert_outputs * probs.unsqueeze(-1)).sum(dim=1)
            balance_loss = probs.new_tensor(0.0)
            diagnostics = self._diagnostics(probs, probs, balance_loss, z_loss)
            return out.reshape(*leading, dim), diagnostics

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

        dispatch = torch.zeros_like(probs)
        dispatch.scatter_add_(1, top_indices, torch.ones_like(top_probs))
        fraction = dispatch.mean(dim=0) / float(self.top_k)
        probability = probs.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(fraction * probability)
        diagnostics = self._diagnostics(probs, dispatch / float(self.top_k), balance_loss, z_loss)
        return out.reshape(*leading, dim), diagnostics
