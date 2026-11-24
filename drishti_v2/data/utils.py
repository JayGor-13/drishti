from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor


def xywh_to_cxcywh(boxes: Tensor, image_size: tuple[int, int] | None = None) -> Tensor:
    boxes = boxes.to(torch.float32)
    out = boxes.clone()
    out[..., 0] = boxes[..., 0] + boxes[..., 2] / 2.0
    out[..., 1] = boxes[..., 1] + boxes[..., 3] / 2.0
    if image_size is not None:
        height, width = image_size
        scale = boxes.new_tensor([width, height, width, height])
        out = out / scale
    return out.clamp(0.0, 1.0)


def xyxy_to_cxcywh(boxes: Tensor, image_size: tuple[int, int] | None = None) -> Tensor:
    boxes = boxes.to(torch.float32)
    out = boxes.clone()
    out[..., 2] = (boxes[..., 2] - boxes[..., 0]).clamp_min(0.0)
    out[..., 3] = (boxes[..., 3] - boxes[..., 1]).clamp_min(0.0)
    out[..., 0] = boxes[..., 0] + out[..., 2] / 2.0
    out[..., 1] = boxes[..., 1] + out[..., 3] / 2.0
    if image_size is not None:
        height, width = image_size
        scale = boxes.new_tensor([width, height, width, height])
        out = out / scale
    return out.clamp(0.0, 1.0)


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    out = torch.empty_like(boxes)
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2.0
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2.0
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2.0
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2.0
    return out


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    b1 = cxcywh_to_xyxy(boxes1).clamp(0, 1)
    b2 = cxcywh_to_xyxy(boxes2).clamp(0, 1)
    left_top = torch.maximum(b1[:, None, :2], b2[None, :, :2])
    right_bottom = torch.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = (right_bottom - left_top).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (b1[:, 2] - b1[:, 0]).clamp_min(0) * (b1[:, 3] - b1[:, 1]).clamp_min(0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp_min(0) * (b2[:, 3] - b2[:, 1]).clamp_min(0)
    return inter / (area1[:, None] + area2[None, :] - inter).clamp_min(1e-8)


def list_image_files(directory: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in extensions)


def flatten_targets(targets: Iterable[dict]) -> Tensor:
    boxes = [target.get("boxes", torch.empty(0, 4)) for target in targets]
    if not boxes:
        return torch.empty(0, 4)
    return torch.cat([box for box in boxes if box.numel() > 0], dim=0) if any(box.numel() > 0 for box in boxes) else torch.empty(0, 4)
