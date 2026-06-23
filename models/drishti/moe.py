"""Stage 5: sparse top-2 mixture of experts."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import DRISHTIConfig
from .types import DRISHTIMoEOutput


class DRISHTIMoE(nn.Module):
    """Eight-expert sparse MoE with learned top-k router."""

    def __init__(self, config: DRISHTIConfig) -> None:
        super().__init__()
        self.feature_dim = config.feature_dim
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.load_balance_weight = config.load_balance_weight
        self.router = nn.Linear(config.feature_dim, config.moe_num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.feature_dim, config.moe_ffn_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(config.moe_ffn_dim, config.feature_dim),
                )
                for _ in range(config.moe_num_experts)
            ]
        )

    def forward(self, features: Tensor) -> DRISHTIMoEOutput:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, crops, feature_dim]")
        batch, crops, hidden = features.shape
        if hidden != self.feature_dim:
            raise ValueError(f"expected feature dim {self.feature_dim}, got {hidden}")

        flat = features.reshape(batch * crops, hidden)
        logits = self.router(flat)
        probs = torch.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
        topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        output = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            selected = topk_indices == expert_idx
            rows = selected.any(dim=-1)
            if not rows.any():
                continue
            expert_output = expert(flat[rows])
            weights = (selected[rows].float() * topk_weights[rows]).sum(dim=-1)
            output[rows] = output[rows] + expert_output * weights.unsqueeze(-1)

        counts = torch.bincount(topk_indices.reshape(-1), minlength=self.num_experts).float()
        usage = counts / counts.sum().clamp_min(1.0)
        load_balance = usage.var(unbiased=False) * self.load_balance_weight
        return DRISHTIMoEOutput(
            hidden_states=output.reshape(batch, crops, hidden),
            router_logits=logits.reshape(batch, crops, self.num_experts),
            router_probs=probs.reshape(batch, crops, self.num_experts),
            topk_indices=topk_indices.reshape(batch, crops, self.top_k),
            topk_scores=topk_scores.reshape(batch, crops, self.top_k),
            load_balance_loss=load_balance,
        )
