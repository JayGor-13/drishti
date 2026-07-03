from __future__ import annotations

import torch
from torch import Tensor

from drishti_v2.data.utils import box_iou


def match_detections(pred_boxes: Tensor, pred_scores: Tensor, gt_boxes: Tensor, iou_threshold: float) -> tuple[int, int, int]:
    if pred_boxes.numel() == 0:
        return 0, 0, int(gt_boxes.shape[0])
    if gt_boxes.numel() == 0:
        return 0, int(pred_boxes.shape[0]), 0
    order = torch.argsort(pred_scores, descending=True)
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    ious = box_iou(pred_boxes[order], gt_boxes)
    for row_idx in range(ious.shape[0]):
        best_iou, gt_idx = ious[row_idx].max(dim=0)
        if float(best_iou) >= iou_threshold and int(gt_idx) not in matched_gt:
            tp += 1
            matched_gt.add(int(gt_idx))
        else:
            fp += 1
    fn = gt_boxes.shape[0] - len(matched_gt)
    return tp, fp, fn


def detection_metrics(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    score_threshold: float = 0.3,
) -> dict[str, float]:
    totals = {0.5: [0, 0, 0], 0.75: [0, 0, 0]}
    fp_images = 0
    total_predictions = 0
    total_kept_predictions = 0
    total_gt_boxes = 0
    images_with_gt = 0
    images_with_predictions = 0
    score_sum = 0.0
    max_score_sum = 0.0
    for pred, target in zip(predictions, targets):
        scores = pred["scores"]
        total_predictions += int(scores.numel())
        if scores.numel() > 0:
            score_sum += float(scores.sum().item())
            max_score_sum += float(scores.max().item())
        keep = scores >= score_threshold
        pred_boxes = pred["boxes"][keep]
        pred_scores = scores[keep]
        gt_boxes = target.get("boxes", torch.empty(0, 4, device=pred_boxes.device)).to(pred_boxes.device)
        total_kept_predictions += int(pred_boxes.shape[0])
        total_gt_boxes += int(gt_boxes.shape[0])
        images_with_gt += int(gt_boxes.shape[0] > 0)
        images_with_predictions += int(pred_boxes.shape[0] > 0)
        for threshold in totals:
            tp, fp, fn = match_detections(pred_boxes, pred_scores, gt_boxes, threshold)
            totals[threshold][0] += tp
            totals[threshold][1] += fp
            totals[threshold][2] += fn
        fp_images += int(pred_boxes.shape[0])

    tp50, fp50, fn50 = totals[0.5]
    precision = tp50 / max(1, tp50 + fp50)
    recall = tp50 / max(1, tp50 + fn50)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    map50 = precision * recall
    tp75, fp75, fn75 = totals[0.75]
    precision75 = tp75 / max(1, tp75 + fp75)
    recall75 = tp75 / max(1, tp75 + fn75)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map75": precision75 * recall75,
        "false_positives_per_image": fp50 / max(1, len(predictions)),
        "score_threshold": score_threshold,
        "gt_boxes": float(total_gt_boxes),
        "images": float(len(predictions)),
        "images_with_gt": float(images_with_gt),
        "images_with_predictions": float(images_with_predictions),
        "raw_predictions_per_image": total_predictions / max(1, len(predictions)),
        "kept_predictions_per_image": total_kept_predictions / max(1, len(predictions)),
        "mean_score": score_sum / max(1, total_predictions),
        "mean_top_score": max_score_sum / max(1, len(predictions)),
    }
