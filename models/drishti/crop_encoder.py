"""Stage 3: LocateAnything-compatible frozen crop encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FrozenCropEncoder(nn.Module):
    """Lightweight local stand-in for a frozen LocateAnything crop encoder."""

    def __init__(
        self,
        feature_dim: int = 256,
        image_channels: int = 3,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.frozen = frozen
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(feature_dim, feature_dim)
        if frozen:
            self.requires_grad_(False)

    def forward(self, crops: Tensor) -> Tensor:
        if crops.ndim != 4:
            raise ValueError("crops must have shape [batch*num_crops, channels, size, size]")
        with torch.set_grad_enabled(not self.frozen and torch.is_grad_enabled()):
            features = self.encoder(crops).flatten(1)
            return self.proj(features)
