from __future__ import annotations

import math

import torch
from torch import Tensor


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    half = boxes[..., 2:] / 2.0
    return torch.cat([boxes[..., :2] - half, boxes[..., :2] + half], dim=-1)


def ciou_loss(pred_boxes: Tensor, gt_boxes: Tensor, eps: float = 1e-7) -> Tensor:
    """Complete IoU loss for boxes in normalized [cx, cy, w, h] format."""

    if pred_boxes.numel() == 0:
        return pred_boxes.sum() * 0.0

    pred_boxes = pred_boxes.clamp(0.0, 1.0)
    gt_boxes = gt_boxes.clamp(0.0, 1.0)
    pred_xyxy = _cxcywh_to_xyxy(pred_boxes)
    gt_xyxy = _cxcywh_to_xyxy(gt_boxes)

    inter_min = torch.maximum(pred_xyxy[..., :2], gt_xyxy[..., :2])
    inter_max = torch.minimum(pred_xyxy[..., 2:], gt_xyxy[..., 2:])
    inter_wh = (inter_max - inter_min).clamp_min(0.0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    pred_area = pred_boxes[..., 2].clamp_min(eps) * pred_boxes[..., 3].clamp_min(eps)
    gt_area = gt_boxes[..., 2].clamp_min(eps) * gt_boxes[..., 3].clamp_min(eps)
    union = pred_area + gt_area - inter_area + eps
    iou = inter_area / union

    enclosing_min = torch.minimum(pred_xyxy[..., :2], gt_xyxy[..., :2])
    enclosing_max = torch.maximum(pred_xyxy[..., 2:], gt_xyxy[..., 2:])
    diagonal = (enclosing_max - enclosing_min).pow(2).sum(dim=-1).clamp_min(eps)
    center_distance = (pred_boxes[..., :2] - gt_boxes[..., :2]).pow(2).sum(dim=-1)

    pred_ratio = pred_boxes[..., 2] / pred_boxes[..., 3].clamp_min(eps)
    gt_ratio = gt_boxes[..., 2] / gt_boxes[..., 3].clamp_min(eps)
    aspect = (4.0 / math.pi**2) * (torch.atan(gt_ratio) - torch.atan(pred_ratio)).pow(2)
    with torch.no_grad():
        aspect_weight = aspect / (1.0 - iou + aspect + eps)

    return (1.0 - iou + center_distance / diagonal + aspect_weight * aspect).mean()
