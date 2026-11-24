"""Turn the model's per-crop boxes into the final per-frame output.

The detector emits one box per crop (8 per frame). Anti-UAV has a single drone per frame, and nearby
crops often find the same drone, which shows up as duplicate boxes (false positives). Two standard,
training-free steps fix that:

* ``nms``: merge boxes that overlap more than ``iou`` (keep the highest score), and
* ``top1``: keep only the best box per frame (what every tracker outputs, and what the 2025
  Anti-UAV challenge winner does); a frame whose best score is below the threshold reports "no drone".
"""
from __future__ import annotations

import torch
from torch import Tensor

from drishti_v2.data.utils import box_iou


def nms(boxes: Tensor, scores: Tensor, iou: float) -> tuple[Tensor, Tensor]:
    """Greedy non-maximum suppression for a handful of cx,cy,w,h boxes (no torchvision needed)."""

    if boxes.numel() == 0 or iou >= 1.0:
        return boxes, scores
    order = torch.argsort(scores, descending=True)
    boxes, scores = boxes[order], scores[order]
    overlaps = box_iou(boxes, boxes)
    keep: list[int] = []
    suppressed = torch.zeros(len(boxes), dtype=torch.bool)
    for idx in range(len(boxes)):
        if suppressed[idx]:
            continue
        keep.append(idx)
        suppressed |= overlaps[idx] > iou
    keep_t = torch.tensor(keep, dtype=torch.long)
    return boxes[keep_t], scores[keep_t]


def postprocess(boxes: Tensor, scores: Tensor, nms_iou: float | None = None, top1: bool = False) -> tuple[Tensor, Tensor]:
    """Apply optional NMS, then optional top-1. Thresholding is left to the metrics."""

    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    if nms_iou is not None:
        boxes, scores = nms(boxes, scores, nms_iou)
    if top1 and scores.numel() > 1:
        best = int(scores.argmax())
        boxes, scores = boxes[best : best + 1], scores[best : best + 1]
    return boxes, scores


def smooth_sequence(
    predictions: list[dict[str, Tensor]],
    threshold: float,
    hold: int = 8,
    alpha: float = 0.5,
    min_iou: float = 0.2,
) -> list[dict[str, Tensor]]:
    """Temporal smoothing of a ONE-box-per-frame output (idea C); causal, frame by frame.

    * Box smoothing: when this frame's box overlaps the previous output box (IoU >= ``min_iou``), output
      ``alpha * new + (1 - alpha) * previous`` (less jitter, higher frame-to-frame IoU).
    * Presence hold: if the drone was detected recently and this frame falls below ``threshold``, keep
      reporting the last box (moved by its recent velocity) for up to ``hold`` frames, scored exactly at
      the threshold. This removes one- or two-frame dropouts (detection flicker).
    """

    out: list[dict[str, Tensor]] = []
    last: Tensor | None = None
    velocity = torch.zeros(4)
    missed = 0
    for pred in predictions:
        boxes, scores = pred["boxes"].reshape(-1, 4), pred["scores"].reshape(-1)
        if scores.numel() and float(scores.max()) >= threshold:
            idx = int(scores.argmax())
            box, score = boxes[idx].clone(), scores[idx].clone()
            if last is not None and float(box_iou(box[None], last[None])[0, 0]) >= min_iou:
                box = alpha * box + (1 - alpha) * last
            if last is not None:
                velocity = 0.5 * velocity + 0.5 * (box - last)
                velocity[2:] = 0.0
            last, missed = box, 0
            out.append({"boxes": box[None], "scores": score[None]})
        elif last is not None and missed < hold:
            missed += 1
            last = last + velocity
            out.append({"boxes": last[None].clamp(0, 1), "scores": torch.tensor([threshold])})
        else:
            last, velocity = None, torch.zeros(4)
            out.append({"boxes": boxes[:0], "scores": scores[:0]})
    return out
