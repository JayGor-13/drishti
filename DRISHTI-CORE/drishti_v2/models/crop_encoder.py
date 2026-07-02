from __future__ import annotations

from torch import Tensor, nn


class CropEncoder(nn.Module):
    """Lightweight CNN patch encoder."""

    def __init__(self, out_dim: int = 256, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, crops: Tensor) -> Tensor:
        x = self.features(crops).flatten(1)
        return self.proj(x)

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        self.train()
        for parameter in self.parameters():
            parameter.requires_grad = True
