from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def soft_argmax2d(heatmap: Tensor, temperature: float = 0.1) -> Tensor:
    """Return differentiable normalized (x, y) peak locations for [B, 1, H, W]."""

    batch, _, height, width = heatmap.shape
    weights = F.softmax(heatmap.flatten(1) / max(temperature, 1e-6), dim=-1)
    ys = torch.linspace(0.0, 1.0, height, device=heatmap.device, dtype=heatmap.dtype)
    xs = torch.linspace(0.0, 1.0, width, device=heatmap.device, dtype=heatmap.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    x = (weights * grid_x.flatten().view(1, -1)).sum(dim=1)
    y = (weights * grid_y.flatten().view(1, -1)).sum(dim=1)
    return torch.stack([x, y], dim=1).view(batch, 2)


def motion_displacement_loss(
    heatmaps: list[Tensor],
    targets: list[list[dict]],
    temperature: float = 0.1,
) -> Tensor:
    """Match heatmap peak displacement to Anti-UAV box-center displacement."""

    if len(heatmaps) < 2:
        return heatmaps[0].sum() * 0.0

    batch = heatmaps[0].shape[0]
    device = heatmaps[0].device
    dtype = heatmaps[0].dtype
    gt_centers = []
    for time_idx in range(len(heatmaps)):
        centers = []
        for batch_idx in range(batch):
            boxes = targets[batch_idx][time_idx].get("boxes", torch.empty(0, 4))
            if boxes.numel() == 0:
                centers.append(torch.zeros(2, device=device, dtype=dtype))
            else:
                centers.append(boxes[0, :2].to(device=device, dtype=dtype))
        gt_centers.append(torch.stack(centers, dim=0))

    pred_centers = [soft_argmax2d(heatmap, temperature) for heatmap in heatmaps]
    total = heatmaps[0].sum() * 0.0
    for time_idx in range(1, len(heatmaps)):
        pred_delta = pred_centers[time_idx] - pred_centers[time_idx - 1]
        gt_delta = gt_centers[time_idx] - gt_centers[time_idx - 1]
        total = total + (pred_delta - gt_delta).pow(2).sum(dim=1).mean()
    return total / float(len(heatmaps) - 1)
