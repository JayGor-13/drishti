from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from drishti_v2.models.config import DRISHTIConfig


@dataclass
class ProposalOutput:
    crops: Tensor
    centers: Tensor
    scores: Tensor
    source_labels: Tensor
    heatmap: Tensor


class CropProposalEngine(nn.Module):
    """Multi-source scheduler and crop extractor."""

    MOTION = 0
    EDGE = 1
    GRID = 2
    GUIDED = 3
    PAD = 4

    def __init__(self, config: DRISHTIConfig) -> None:
        super().__init__()
        self.config = config
        grid = torch.tensor(
            [(0.30, 0.30), (0.30, 0.70), (0.70, 0.30), (0.70, 0.70)],
            dtype=torch.float32,
        )
        bw = config.border_width_frac
        edge_horizontal = torch.tensor([(bw / 2.0, 0.5), (1.0 - bw / 2.0, 0.5)], dtype=torch.float32)
        edge_vertical = torch.tensor([(0.5, bw / 2.0), (0.5, 1.0 - bw / 2.0)], dtype=torch.float32)
        self.register_buffer("interior_grid", grid, persistent=False)
        self.register_buffer("edge_horizontal", edge_horizontal, persistent=False)
        self.register_buffer("edge_vertical", edge_vertical, persistent=False)

    def _get_motion_centers(self, heatmap: Tensor, n: int) -> tuple[Tensor, Tensor]:
        batch, _, height, width = heatmap.shape
        if n <= 0 or not self.config.use_motion_crops:
            empty_centers = heatmap.new_zeros(batch, 0, 2)
            empty_scores = heatmap.new_zeros(batch, 0)
            return empty_centers, empty_scores
        n = min(n, height * width)
        pooled = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
        peaks = torch.where(heatmap == pooled, heatmap, torch.zeros_like(heatmap))
        scores, indices = torch.topk(peaks.flatten(1), k=n, dim=-1)
        rows = torch.div(indices, width, rounding_mode="floor").to(heatmap.dtype)
        cols = (indices % width).to(heatmap.dtype)
        centers = torch.stack([cols / max(width - 1, 1), rows / max(height - 1, 1)], dim=-1)
        return centers.clamp(0.0, 1.0), scores

    def _get_edge_centers(self, frame_index: int, batch_size: int, device: torch.device) -> Tensor:
        pattern = self.edge_horizontal if frame_index % 2 == 1 else self.edge_vertical
        return pattern.to(device).unsqueeze(0).expand(batch_size, -1, -1)

    def _get_grid_centers(self, batch_size: int, device: torch.device) -> Tensor:
        return self.interior_grid.to(device).unsqueeze(0).expand(batch_size, -1, -1)

    def _scores_at_centers(self, heatmap: Tensor, centers: Tensor) -> Tensor:
        if centers.shape[1] == 0:
            return heatmap.new_zeros(heatmap.shape[0], 0)
        grid = centers.mul(2.0).sub(1.0).view(heatmap.shape[0], centers.shape[1], 1, 2)
        return F.grid_sample(heatmap, grid, mode="bilinear", align_corners=True).squeeze(1).squeeze(-1)

    def _extract_crops(self, frame: Tensor, centers: Tensor) -> Tensor:
        batch, channels, height, width = frame.shape
        num_crops = centers.shape[1]
        crop_size = self.config.crop_size
        half = crop_size / 2.0
        crops: list[Tensor] = []
        padded = F.pad(frame, (crop_size, crop_size, crop_size, crop_size), mode="replicate")

        px = centers[..., 0].clamp(0, 1) * (width - 1) + crop_size
        py = centers[..., 1].clamp(0, 1) * (height - 1) + crop_size
        for b_idx in range(batch):
            for k_idx in range(num_crops):
                x0 = int(round(float(px[b_idx, k_idx] - half)))
                y0 = int(round(float(py[b_idx, k_idx] - half)))
                crops.append(padded[b_idx, :, y0 : y0 + crop_size, x0 : x0 + crop_size])
        return torch.stack(crops, dim=0).reshape(batch * num_crops, channels, crop_size, crop_size)

    def _append(
        self,
        centers_list: list[Tensor],
        sources: list[int],
        centers: Tensor,
        source_label: int,
        max_slots: int,
    ) -> int:
        slots = min(max_slots, centers.shape[1])
        if slots > 0:
            centers_list.append(centers[:, :slots])
            sources.extend([source_label] * slots)
        return slots

    def forward(
        self,
        frame: Tensor,
        heatmap: Tensor,
        frame_index: int,
        guided_centers: Tensor | None = None,
    ) -> ProposalOutput:
        batch = frame.shape[0]
        total = self.config.num_crops
        device = frame.device
        centers_list: list[Tensor] = []
        sources: list[int] = []

        remaining = total
        if guided_centers is not None and self.config.use_guided_crops:
            guided = guided_centers.to(device).clamp(0.0, 1.0)
            if guided.shape[0] == 1 and batch > 1:
                guided = guided.expand(batch, -1, -1)
            reserve = 2 if total >= 3 else 0
            used = self._append(centers_list, sources, guided, self.GUIDED, max(0, remaining - reserve))
            remaining -= used

        if self.config.use_grid_crops and frame_index % self.config.scan_period == 0 and remaining > 0:
            reserve = 2 if remaining >= 3 else 0
            used = self._append(
                centers_list,
                sources,
                self._get_grid_centers(batch, device),
                self.GRID,
                max(0, remaining - reserve),
            )
            remaining -= used

        if self.config.use_edge_crops and remaining > 0:
            reserve = 1 if remaining >= 2 else 0
            used = self._append(
                centers_list,
                sources,
                self._get_edge_centers(frame_index, batch, device),
                self.EDGE,
                max(0, remaining - reserve),
            )
            remaining -= used

        if remaining > 0:
            motion_centers, _ = self._get_motion_centers(heatmap, remaining)
            used = self._append(centers_list, sources, motion_centers, self.MOTION, remaining)
            remaining -= used

        if remaining > 0:
            pad = frame.new_tensor((0.5, 0.5)).view(1, 1, 2).expand(batch, remaining, 2)
            centers_list.append(pad)
            sources.extend([self.PAD] * remaining)

        centers = torch.cat(centers_list, dim=1)[:, :total].contiguous()
        sources_tensor = torch.tensor(sources[:total], device=device, dtype=torch.long).view(1, total).expand(batch, -1)
        scores = self._scores_at_centers(heatmap, centers)
        crops = self._extract_crops(frame, centers)
        return ProposalOutput(crops=crops, centers=centers, scores=scores, source_labels=sources_tensor, heatmap=heatmap)
