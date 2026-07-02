from __future__ import annotations

from torch import Tensor, nn


class DetectionHead(nn.Module):
    """Per-crop objectness and crop-relative box regression head."""

    def __init__(self, feature_dim: int = 256, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.objectness_head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 1))
        self.box_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        return self.objectness_head(features), self.box_head(features)
