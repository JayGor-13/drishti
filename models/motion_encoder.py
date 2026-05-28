"""Kinematic motion features and motion confidence for video tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class MotionEncoderOutput:
    """Projected motion tokens and scalar motion confidence."""

    embeddings: Tensor
    confidence: Tensor
    raw_features: Tensor


class KinematicMotionEncoder(nn.Module):
    """A lightweight X3D-style proxy for patch-wise motion features.

    This module intentionally avoids a heavyweight video backbone dependency.
    It consumes consecutive-frame differences, applies a small 3D CNN, pools
    to a patch grid, and projects motion features to the LLM hidden dimension.
    The public tensor contract mirrors the future X3D-Tiny replacement.
    """

    def __init__(
        self,
        hidden_dim: int,
        motion_dim: int = 256,
        patch_grid_size: int = 4,
        in_channels: int = 3,
        confidence_bias: float = -5.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.motion_dim = motion_dim
        self.patch_grid_size = patch_grid_size

        mid_dim = max(16, motion_dim // 2)
        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_channels,
                mid_dim,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.SiLU(),
            nn.Conv3d(
                mid_dim,
                motion_dim,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.SiLU(),
        )
        self.proj = nn.Linear(motion_dim, hidden_dim, bias=False)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.confidence_head.bias, confidence_bias)

    def forward(self, frames: Tensor) -> MotionEncoderOutput:
        """Encode motion from frames.

        Args:
            frames: Float tensor shaped ``[batch, time, channels, height, width]``.

        Returns:
            MotionEncoderOutput with embeddings ``[batch, time, patches, hidden]``
            and confidence ``[batch, time, patches]``.
        """

        if frames.ndim != 5:
            raise ValueError(
                "frames must have shape [batch, time, channels, height, width]"
            )

        diffs = torch.zeros_like(frames)
        diffs[:, 1:] = frames[:, 1:] - frames[:, :-1]

        encoded = self.encoder(diffs.transpose(1, 2)).transpose(1, 2)
        batch, time, channels, _, _ = encoded.shape
        pooled = F.adaptive_avg_pool3d(
            encoded.transpose(1, 2),
            output_size=(time, self.patch_grid_size, self.patch_grid_size),
        ).transpose(1, 2)
        raw = pooled.flatten(3).transpose(2, 3).contiguous()
        raw = raw.view(batch, time, self.patch_grid_size**2, channels)

        embeddings = self.proj(raw)
        confidence = torch.sigmoid(self.confidence_head(embeddings)).squeeze(-1)
        return MotionEncoderOutput(
            embeddings=embeddings,
            confidence=confidence,
            raw_features=raw,
        )
