"""Stage 6: DRISHTI detection head."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DRISHTIDetectionHead(nn.Module):
    """Predict normalized box coordinates plus objectness/confidence."""

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.objectness = nn.Linear(feature_dim, 1)
        self.box_regressor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 4),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        normed = self.norm(features)
        object_logits = self.objectness(normed)
        boxes = torch.sigmoid(self.box_regressor(normed))
        confidence = torch.sigmoid(object_logits)
        predictions = torch.cat([boxes, confidence, confidence], dim=-1)
        return object_logits, boxes, predictions
