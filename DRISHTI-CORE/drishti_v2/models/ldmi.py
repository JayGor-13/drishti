from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LocalDifferentialMotion(nn.Module):
    """Parameter-free Local Differential Motion Invariant preprocessing."""

    def __init__(self, image_channels: int = 3, scales: tuple[int, ...] = (15, 31)) -> None:
        super().__init__()
        if not scales:
            raise ValueError("At least one LDMI scale is required")
        for scale in scales:
            if scale < 1 or scale % 2 == 0:
                raise ValueError("LDMI scales must be positive odd integers")
        self.image_channels = image_channels
        self.scales = tuple(scales)

    def _compute_residual(self, diff: Tensor) -> Tensor:
        residuals = []
        for kernel in self.scales:
            local_mean = F.avg_pool2d(
                diff,
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
                count_include_pad=False,
            )
            residuals.append(torch.abs(diff - local_mean))
        return torch.stack(residuals, dim=0).amax(dim=0)

    def forward(self, triplet: Tensor) -> Tensor:
        channels = self.image_channels
        expected = channels * 3
        if triplet.ndim != 4 or triplet.shape[1] != expected:
            raise ValueError(f"Expected [B, {expected}, H, W], got {tuple(triplet.shape)}")

        f_old = triplet[:, 0:channels]
        f_prev = triplet[:, channels : 2 * channels]
        f_curr = triplet[:, 2 * channels : 3 * channels]
        r_old = self._compute_residual(f_prev - f_old)
        r_new = self._compute_residual(f_curr - f_prev)
        return torch.cat([r_old, f_curr, r_new], dim=1)
