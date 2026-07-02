from __future__ import annotations

import torch
from torch import Tensor, nn


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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        *leading, dim = x.shape
        x_flat = x.reshape(-1, dim)
        probs = torch.softmax(self.router(x_flat), dim=-1)

        if self.dense:
            expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
            out = (expert_outputs * probs.unsqueeze(-1)).sum(dim=1)
            balance_loss = probs.new_tensor(0.0)
            return out.reshape(*leading, dim), balance_loss

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
        return out.reshape(*leading, dim), balance_loss
