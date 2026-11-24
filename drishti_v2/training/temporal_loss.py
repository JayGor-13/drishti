from __future__ import annotations

import torch
from torch import Tensor


def temporal_consistency_loss(
    logits_seq: list[Tensor],
    centers_seq: list[Tensor],
    sigma_spatial: float = 0.1,
) -> Tensor:
    """Penalize objectness flicker for crop slots that stayed spatially close."""

    if len(logits_seq) < 2:
        return logits_seq[0].sum() * 0.0

    total = logits_seq[0].sum() * 0.0
    for time_idx in range(1, len(logits_seq)):
        prev_scores = torch.sigmoid(logits_seq[time_idx - 1].squeeze(-1))
        curr_scores = torch.sigmoid(logits_seq[time_idx].squeeze(-1))
        distance = (centers_seq[time_idx] - centers_seq[time_idx - 1]).pow(2).sum(dim=-1)
        weight = torch.exp(-distance / (2.0 * sigma_spatial**2))
        total = total + (weight * (curr_scores - prev_scores).pow(2)).mean()
    return total / float(len(logits_seq) - 1)


def trajectory_smoothness_loss(boxes_seq: list[Tensor], labels_seq: list[Tensor]) -> Tensor:
    """Penalize acceleration in positive crop box trajectories."""

    if len(boxes_seq) < 3:
        return boxes_seq[0].sum() * 0.0

    positive = torch.stack([labels.bool() for labels in labels_seq], dim=0).any(dim=0)
    total = boxes_seq[0].sum() * 0.0
    count = positive.to(boxes_seq[0].dtype).sum().clamp_min(1.0)
    for time_idx in range(2, len(boxes_seq)):
        velocity = boxes_seq[time_idx] - boxes_seq[time_idx - 1]
        previous_velocity = boxes_seq[time_idx - 1] - boxes_seq[time_idx - 2]
        acceleration = (velocity - previous_velocity).pow(2).sum(dim=-1)
        total = total + (acceleration * positive.to(acceleration.dtype)).sum() / count
    return total / float(len(boxes_seq) - 2)
