from __future__ import annotations

import torch
from torch import Tensor, nn


class MotionCNN(nn.Module):
    """Small convolutional anomaly localizer producing an H/4 by W/4 heatmap."""

    def __init__(
        self,
        image_channels: int = 3,
        hidden_channels: tuple[int, ...] = (32, 64, 64),
        in_channels: int | None = None,
    ) -> None:
        super().__init__()
        in_channels = in_channels or image_channels * 5
        layers: list[nn.Module] = []
        for idx, out_channels in enumerate(hidden_channels):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2 if idx < 2 else 1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        layers.extend([nn.Conv2d(in_channels, 1, kernel_size=1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, filtered_triplet: Tensor) -> Tensor:
        return self.net(filtered_triplet)

    @staticmethod
    def make_gt_heatmap(boxes: Tensor, heatmap_size: tuple[int, int], sigma: float = 2.0) -> Tensor:
        height, width = heatmap_size
        device = boxes.device
        dtype = boxes.dtype if boxes.is_floating_point() else torch.float32
        y = torch.arange(height, device=device, dtype=dtype).view(height, 1)
        x = torch.arange(width, device=device, dtype=dtype).view(1, width)
        heatmap = torch.zeros(1, height, width, device=device, dtype=dtype)
        if boxes.numel() == 0:
            return heatmap
        centers_x = (boxes[:, 0].clamp(0, 1) * (width - 1)).view(-1, 1, 1)
        centers_y = (boxes[:, 1].clamp(0, 1) * (height - 1)).view(-1, 1, 1)
        gaussian = torch.exp(-((x - centers_x) ** 2 + (y - centers_y) ** 2) / (2.0 * sigma**2))
        heatmap[0] = gaussian.amax(dim=0)
        return heatmap.clamp(0.0, 1.0)
