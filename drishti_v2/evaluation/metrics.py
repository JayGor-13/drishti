from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

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
    for pred, target in zip(predictions, targets):
        scores = pred["scores"]
        keep = scores >= score_threshold
        pred_boxes = pred["boxes"][keep]
        pred_scores = scores[keep]
        gt_boxes = target.get("boxes", torch.empty(0, 4, device=pred_boxes.device)).to(pred_boxes.device)
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
    }


def _top_box(prediction: dict[str, Tensor]) -> tuple[Tensor | None, Tensor | None]:
    boxes = prediction.get("boxes", torch.empty(0, 4))
    scores = prediction.get("scores", torch.empty(0))
    if boxes.numel() == 0 or scores.numel() == 0:
        return None, None
    index = scores.argmax()
    return boxes[index], scores[index]


def temporal_iou(predictions: list[dict[str, Tensor]]) -> dict[str, float]:
    """Average IoU of the top prediction across consecutive frames/clips."""

    values: list[float] = []
    for previous, current in zip(predictions, predictions[1:]):
        prev_box, _ = _top_box(previous)
        curr_box, _ = _top_box(current)
        if prev_box is None or curr_box is None:
            continue
        values.append(float(box_iou(prev_box.view(1, 4), curr_box.view(1, 4))[0, 0].item()))
    return {"temporal_iou": float(sum(values) / max(len(values), 1))}


def trajectory_smoothness(predictions: list[dict[str, Tensor]]) -> dict[str, float]:
    """Inverse mean acceleration magnitude for top predicted boxes."""

    centers = []
    for prediction in predictions:
        box, _ = _top_box(prediction)
        if box is not None:
            centers.append(box[:2].float())
    if len(centers) < 3:
        return {"trajectory_smoothness": 0.0}

    accelerations = []
    for idx in range(2, len(centers)):
        velocity = centers[idx] - centers[idx - 1]
        previous_velocity = centers[idx - 1] - centers[idx - 2]
        accelerations.append(float((velocity - previous_velocity).norm(p=2).item()))
    mean_acceleration = sum(accelerations) / max(len(accelerations), 1)
    return {"trajectory_smoothness": float(1.0 / (1.0 + mean_acceleration))}


def detection_flicker_rate(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    score_threshold: float = 0.3,
) -> dict[str, float]:
    """Fraction of visible adjacent frames where detection state changes."""

    states = []
    for prediction, target in zip(predictions, targets):
        visible = target.get("boxes", torch.empty(0, 4)).numel() > 0
        _, score = _top_box(prediction)
        detected = score is not None and float(score) >= score_threshold
        if visible:
            states.append(bool(detected))
    if len(states) < 2:
        return {"detection_flicker_rate": 0.0}
    changes = sum(int(curr != prev) for prev, curr in zip(states, states[1:]))
    return {"detection_flicker_rate": float(changes / max(len(states) - 1, 1))}


def temporal_detection_metrics(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    score_threshold: float = 0.3,
) -> dict[str, float]:
    """Convenience bundle for Stage-2 style temporal diagnostics."""

    metrics = {}
    metrics.update(temporal_iou(predictions))
    metrics.update(trajectory_smoothness(predictions))
    metrics.update(detection_flicker_rate(predictions, targets, score_threshold))
    return metrics


def heatmap_peak_metrics(
    heatmaps: list[Tensor],
    targets: list[dict[str, Tensor]],
    pixel_threshold_frac: float = 5.0 / 448.0,
) -> dict[str, float]:
    """Measure how close each heatmap peak is to the first GT box center."""

    distances: list[float] = []
    within: list[float] = []
    for heatmap, target in zip(heatmaps, targets):
        boxes = target.get("boxes", torch.empty(0, 4))
        if boxes.numel() == 0:
            continue
        height, width = heatmap.shape[-2:]
        flat_idx = heatmap.reshape(-1).argmax()
        y = torch.div(flat_idx, width, rounding_mode="floor").to(torch.float32) / max(height - 1, 1)
        x = (flat_idx % width).to(torch.float32) / max(width - 1, 1)
        pred_center = torch.stack([x, y])
        distance = (pred_center - boxes[0, :2].cpu()).pow(2).sum().sqrt().item()
        distances.append(distance)
        within.append(float(distance <= pixel_threshold_frac))

    return {
        "heatmap_peak_distance": float(sum(distances) / max(len(distances), 1)),
        "heatmap_peak_within_5px": float(sum(within) / max(len(within), 1)),
    }


def motion_direction_accuracy(pred_displacements: list[Tensor], gt_displacements: list[Tensor]) -> dict[str, float]:
    """Average cosine similarity between predicted and GT motion vectors."""

    similarities = []
    for pred, gt in zip(pred_displacements, gt_displacements):
        similarities.append(float(F.cosine_similarity(pred.view(1, -1), gt.view(1, -1)).item()))
    return {"motion_direction_accuracy": float(sum(similarities) / max(len(similarities), 1))}


def hard_heatmap_peak(heatmap: Tensor) -> Tensor:
    """Return normalized hard-argmax heatmap peaks as [B, 2]."""

    batch, _, height, width = heatmap.shape
    flat = heatmap.flatten(1)
    indices = flat.argmax(dim=1)
    y = torch.div(indices, width, rounding_mode="floor").to(torch.float32) / max(height - 1, 1)
    x = (indices % width).to(torch.float32) / max(width - 1, 1)
    return torch.stack([x, y], dim=1).view(batch, 2)


def heatmap_motion_direction_metrics(heatmaps: list[Tensor], targets: list[list[dict]]) -> dict[str, float]:
    """Cosine similarity between heatmap peak displacement and GT displacement."""

    if len(heatmaps) < 2 or not targets:
        return {"motion_direction_accuracy": 0.0}
    peaks = [hard_heatmap_peak(heatmap.detach().cpu()) for heatmap in heatmaps]
    pred_deltas: list[Tensor] = []
    gt_deltas: list[Tensor] = []
    batch = peaks[0].shape[0]
    for time_idx in range(1, len(peaks)):
        for batch_idx in range(batch):
            prev_boxes = targets[batch_idx][time_idx - 1].get("boxes", torch.empty(0, 4))
            curr_boxes = targets[batch_idx][time_idx].get("boxes", torch.empty(0, 4))
            if prev_boxes.numel() == 0 or curr_boxes.numel() == 0:
                continue
            pred_deltas.append(peaks[time_idx][batch_idx] - peaks[time_idx - 1][batch_idx])
            gt_deltas.append(curr_boxes[0, :2].cpu() - prev_boxes[0, :2].cpu())
    return motion_direction_accuracy(pred_deltas, gt_deltas)
