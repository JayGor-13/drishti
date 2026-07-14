from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LocalDifferentialMotion(nn.Module):
    """Parameter-free LDMI v2 preprocessing.

    A triplet of RGB frames is converted into motion residuals, motion
    magnitudes, scale hints, the current image, and appearance/disappearance
    cues. For RGB input this produces 15 channels.
    """

    def __init__(self, image_channels: int = 3, scales: tuple[int, ...] = (15, 31)) -> None:
        super().__init__()
        if not scales:
            raise ValueError("At least one LDMI scale is required")
        for scale in scales:
            if scale < 1 or scale % 2 == 0:
                raise ValueError("LDMI scales must be positive odd integers")
        self.image_channels = image_channels
        self.scales = tuple(scales)

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

        return torch.cat(
            [
                r_old,
                m_old,
                s_old,
                f_curr,
                s_new,
                m_new,
                r_new,
                disappearance,
                appearance,
            ],
            dim=1,
        )
