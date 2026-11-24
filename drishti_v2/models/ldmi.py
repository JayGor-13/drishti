from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
).view(1, 1, 3, 3)


class LocalDifferentialMotion(nn.Module):
    """Parameter-free LDMI v2 preprocessing.

    A triplet of infrared or RGB frames is converted into motion residuals,
    motion magnitudes, scale hints, the current image, appearance/disappearance
    cues, and an optional Sobel edge map of the newest frame difference. The
    output has ``3C + 7`` channels with Sobel enabled and ``3C + 6`` without it.
    """

    def __init__(
        self,
        image_channels: int = 1,
        scales: tuple[int, ...] = (15, 31),
        use_sobel_edge: bool = True,
    ) -> None:
        super().__init__()
        if not scales:
            raise ValueError("At least one LDMI scale is required")
        for scale in scales:
            if scale < 1 or scale % 2 == 0:
                raise ValueError("LDMI scales must be positive odd integers")
        self.image_channels = image_channels
        self.scales = tuple(scales)
        self.use_sobel_edge = use_sobel_edge
        self.register_buffer("sobel_x", SOBEL_X, persistent=False)
        self.register_buffer("sobel_y", SOBEL_Y, persistent=False)

    def _signed_residual_and_scale(self, diff: Tensor) -> tuple[Tensor, Tensor]:
        residuals = []
        for kernel in self.scales:
            local_mean = F.avg_pool2d(
                diff,
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
                count_include_pad=False,
            )
            residuals.append(diff - local_mean)

        stacked = torch.stack(residuals, dim=0)
        indices = stacked.abs().argmax(dim=0, keepdim=True)
        residual = stacked.gather(0, indices).squeeze(0)

        if len(self.scales) == 1:
            scale = diff.new_zeros(diff.shape[0], 1, diff.shape[-2], diff.shape[-1])
        else:
            scale = indices.squeeze(0).to(diff.dtype).mean(dim=1, keepdim=True)
            scale = scale / float(len(self.scales) - 1)
        return residual, scale

    def _compute_sobel_edge(self, diff: Tensor) -> Tensor:
        gray_diff = diff.mean(dim=1, keepdim=True)
        kernel_x = self.sobel_x.to(device=diff.device, dtype=diff.dtype)
        kernel_y = self.sobel_y.to(device=diff.device, dtype=diff.dtype)
        gx = F.conv2d(gray_diff, kernel_x, padding=1)
        gy = F.conv2d(gray_diff, kernel_y, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-8)

    def forward(self, triplet: Tensor) -> Tensor:
        channels = self.image_channels
        expected = channels * 3
        if triplet.ndim != 4 or triplet.shape[1] != expected:
            raise ValueError(f"Expected [B, {expected}, H, W], got {tuple(triplet.shape)}")

        f_old = triplet[:, 0:channels]
        f_prev = triplet[:, channels : 2 * channels]
        f_curr = triplet[:, 2 * channels : 3 * channels]
        d_old = f_prev - f_old
        d_new = f_curr - f_prev
        r_old, s_old = self._signed_residual_and_scale(d_old)
        r_new, s_new = self._signed_residual_and_scale(d_new)

        m_old = d_old.norm(p=2, dim=1, keepdim=True)
        m_new = d_new.norm(p=2, dim=1, keepdim=True)
        old_strength = r_old.abs().mean(dim=1, keepdim=True)
        new_strength = r_new.abs().mean(dim=1, keepdim=True)
        disappearance = torch.relu(old_strength - new_strength)
        appearance = torch.relu(new_strength - old_strength)

        components = [
            r_old,
            m_old,
            s_old,
            f_curr,
            s_new,
            m_new,
            r_new,
            disappearance,
            appearance,
        ]
        if self.use_sobel_edge:
            components.append(self._compute_sobel_edge(d_new))
        return torch.cat(components, dim=1)
