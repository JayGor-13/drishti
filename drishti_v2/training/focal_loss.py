from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sigmoid_focal_loss(logits: Tensor, targets: Tensor, gamma: float = 2.0, alpha: float = 0.25) -> Tensor:
    """Binary focal loss for imbalanced crop objectness labels."""

    probabilities = torch.sigmoid(logits)
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def heatmap_focal_loss(pred: Tensor, target: Tensor, alpha: float = 2.0, beta: float = 4.0) -> Tensor:
    """CenterNet-style focal loss for Gaussian target heatmaps."""

    pred = pred.clamp(1e-6, 1.0 - 1e-6)
    pos = (target >= 1.0).to(pred.dtype)
    neg = 1.0 - pos
    num_pos = pos.sum().clamp_min(1.0)

    pos_loss = (1.0 - pred).pow(alpha) * pred.log() * pos
    neg_loss = (1.0 - target).pow(beta) * pred.pow(alpha) * torch.log1p(-pred) * neg
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos
