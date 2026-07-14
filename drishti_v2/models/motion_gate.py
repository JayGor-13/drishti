from __future__ import annotations

import torch
from torch import Tensor, nn


class MotionGate(nn.Module):
    """Predict whether the heatmap is trustworthy enough for selective crops."""

    def __init__(self, hidden_dim: int = 16, active_threshold: float = 0.5) -> None:
        super().__init__()
        self.active_threshold = active_threshold
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _stats(self, heatmap: Tensor) -> Tensor:
        batch = heatmap.shape[0]
        flat = heatmap.flatten(1).clamp(0.0, 1.0)
        top = flat.topk(k=min(2, flat.shape[1]), dim=1).values
        top1 = top[:, 0]
        top2 = top[:, 1] if top.shape[1] > 1 else flat.new_zeros(batch)

        probs = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=1)
        entropy = entropy / torch.log(flat.new_tensor(float(flat.shape[1]))).clamp_min(1e-6)

        return torch.stack(
            [
                flat.max(dim=1).values,
                flat.mean(dim=1),
                flat.std(dim=1, unbiased=False),
                entropy,
                top1 - top2,
                (flat > self.active_threshold).to(flat.dtype).mean(dim=1),
            ],
            dim=1,
        )

    def forward(self, heatmap: Tensor) -> Tensor:
        if heatmap.ndim != 4 or heatmap.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, H, W], got {tuple(heatmap.shape)}")
        return self.net(self._stats(heatmap)).squeeze(-1)
