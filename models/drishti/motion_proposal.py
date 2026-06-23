"""Stage 1+2: MotionCNN heatmap and top-k crop proposals."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import CropProposalOutput


class MotionCropProposal(nn.Module):
    """MotionCNN followed by top-k fixed-size crop extraction."""

    def __init__(
        self,
        image_channels: int = 3,
        crop_size: int = 64,
        num_crops: int = 8,
    ) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.motion_cnn = nn.Sequential(
            nn.Conv2d(image_channels * 3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def _current_frame(self, triplet: Tensor) -> Tensor:
        start = self.image_channels
        end = start + self.image_channels
        return triplet[:, start:end]

    def _extract_crops(self, frame: Tensor, centers: Tensor) -> Tensor:
        batch, channels, height, width = frame.shape
        half = self.crop_size // 2
        padded = F.pad(frame, (half, half, half, half), mode="replicate")
        crops = []
        for batch_idx in range(batch):
            for crop_idx in range(centers.shape[1]):
                cx = int(round(float(centers[batch_idx, crop_idx, 0]) * width))
                cy = int(round(float(centers[batch_idx, crop_idx, 1]) * height))
                x0 = max(0, min(cx, width + half * 2 - self.crop_size))
                y0 = max(0, min(cy, height + half * 2 - self.crop_size))
                crops.append(
                    padded[
                        batch_idx,
                        :,
                        y0 : y0 + self.crop_size,
                        x0 : x0 + self.crop_size,
                    ]
                )
        return torch.stack(crops, dim=0).reshape(
            batch * centers.shape[1],
            channels,
            self.crop_size,
            self.crop_size,
        )

    def forward(self, triplet: Tensor) -> CropProposalOutput:
        if triplet.ndim != 4:
            raise ValueError("triplet must have shape [batch, channels*3, height, width]")
        expected_channels = self.image_channels * 3
        if triplet.shape[1] != expected_channels:
            raise ValueError(f"expected {expected_channels} channels, got {triplet.shape[1]}")

        batch, _, height, width = triplet.shape
        heatmap = self.motion_cnn(triplet)
        _, _, heat_h, heat_w = heatmap.shape
        flat = heatmap.flatten(2)
        actual_k = min(self.num_crops, flat.shape[-1])
        scores, indices = torch.topk(flat, k=actual_k, dim=-1)
        scores = scores.squeeze(1)
        indices = indices.squeeze(1)
        if actual_k < self.num_crops:
            pad = self.num_crops - actual_k
            scores = torch.cat([scores, scores[:, :1].expand(-1, pad)], dim=1)
            indices = torch.cat([indices, indices[:, :1].expand(-1, pad)], dim=1)

        ys = torch.div(indices, heat_w, rounding_mode="floor").to(triplet.dtype)
        xs = (indices % heat_w).to(triplet.dtype)
        centers = torch.stack(
            [
                ((xs + 0.5) / heat_w).clamp(0.0, 1.0),
                ((ys + 0.5) / heat_h).clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        crop_w = min(float(self.crop_size) / max(width, 1), 1.0)
        crop_h = min(float(self.crop_size) / max(height, 1), 1.0)
        boxes = torch.cat(
            [
                centers,
                centers.new_full((batch, self.num_crops, 1), crop_w),
                centers.new_full((batch, self.num_crops, 1), crop_h),
            ],
            dim=-1,
        )
        crops = self._extract_crops(self._current_frame(triplet), centers)
        return CropProposalOutput(
            crops=crops,
            motion_scores=scores.reshape(batch * self.num_crops, 1),
            boxes=boxes,
            centers=centers,
            heatmap=heatmap,
        )
