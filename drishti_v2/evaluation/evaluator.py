from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from drishti_v2.evaluation.metrics import (
    detection_metrics,
    heatmap_motion_direction_metrics,
    heatmap_peak_metrics,
    temporal_detection_metrics,
)
from drishti_v2.evaluation.visualize import save_metrics_bar, save_moe_diagnostics


class DRISHTIEvaluator:
    """Runs detector evaluation and prints result metrics."""

    def __init__(self, model: nn.Module, loader: DataLoader, device: str | torch.device = "cpu", threshold: float = 0.3) -> None:
        self.model = model
        self.loader = loader
        self.device = torch.device(device)
        self.threshold = threshold

    @torch.no_grad()
    def evaluate(
        self,
        print_results: bool = True,
        output_path: str | Path | None = None,
        save_visualizations: bool = True,
    ) -> dict[str, Any]:
        self.model.eval()
        predictions = []
        targets = []
        heatmaps = []
        motion_scores = []
        moe_scalars: dict[str, list[float]] = {
            "load_balance_cv": [],
            "router_entropy": [],
            "token_drop_rate": [],
            "expert_overlap": [],
        }
        expert_utilization = []
        expert_reuse = []
        source_counts = {name: 0 for name in SOURCE_NAMES.values()}
        tp_source_counts = {name: 0 for name in SOURCE_NAMES.values()}
        gt_visible_frames = 0
        gt_covered_frames = 0

        iterator = tqdm(self.loader, desc="eval", leave=False) if print_results else self.loader
        for batch in iterator:
            frames = batch["frames"].to(self.device)
            output = self.model(frames)
            scores = torch.sigmoid(output.objectness_logits.squeeze(-1))
            for b_idx in range(frames.shape[0]):
                predictions.append({"boxes": output.boxes[b_idx].detach().cpu(), "scores": scores[b_idx].detach().cpu()})
                target = batch["targets"][b_idx][-1]
                targets.append(target)
                heatmaps.append(output.heatmap[b_idx].detach().cpu())
                attribution = attribute_detection(
                    output.boxes[b_idx].detach().cpu(),
                    scores[b_idx].detach().cpu(),
                    output.proposal_sources[b_idx].detach().cpu(),
                    output.proposal_centers[b_idx].detach().cpu(),
                    target,
                    self.threshold,
                    crop_half_extent(self.model),
                )
                if attribution["top_source"] is not None:
                    source_counts[attribution["top_source"]] += 1
                    if attribution["top_is_tp"]:
                        tp_source_counts[attribution["top_source"]] += 1
                if attribution["gt_visible"]:
                    gt_visible_frames += 1
                    gt_covered_frames += int(attribution["gt_covered"])

            if output.all_heatmaps is not None and batch["targets"] and isinstance(batch["targets"][0], list):
                motion_scores.append(heatmap_motion_direction_metrics([h.detach().cpu() for h in output.all_heatmaps], batch["targets"]))

            diagnostics = output.moe_diagnostics
            expert_utilization.append(diagnostics.expert_utilization.detach().cpu())
            expert_reuse.append(diagnostics.expert_reuse_frequency.detach().cpu())
            for key in moe_scalars:
                moe_scalars[key].append(float(getattr(diagnostics, key).detach().cpu()))

        metrics: dict[str, Any] = detection_metrics(predictions, targets, score_threshold=self.threshold)
        metrics.update(heatmap_peak_metrics(heatmaps, targets))
        metrics.update(_average_metric_dicts(motion_scores))
        metrics.update(temporal_detection_metrics(predictions, targets, self.threshold))
        metrics.update(source_attribution_metrics(source_counts, tp_source_counts, gt_visible_frames, gt_covered_frames))
        if expert_utilization:
            metrics["expert_utilization"] = torch.stack(expert_utilization).mean(dim=0).tolist()
            metrics["expert_reuse_frequency"] = torch.stack(expert_reuse).mean(dim=0).tolist()
            for key, values in moe_scalars.items():
                metrics[key] = float(sum(values) / max(len(values), 1))

        if print_results:
            print(json.dumps(_jsonable(metrics), indent=2, sort_keys=True))
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_jsonable(metrics), indent=2, sort_keys=True), encoding="utf-8")
            if save_visualizations:
                save_metrics_bar({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, path.with_name("metrics_bar.png"))
                save_moe_diagnostics(metrics, path.with_name("moe_diagnostics.png"))
        return metrics


SOURCE_NAMES = {0: "motion", 1: "edge", 2: "grid", 3: "guided", 4: "pad"}


def crop_half_extent(model: nn.Module) -> float:
    """Half-width of the base crop window in normalized frame units."""

    cfg = getattr(model, "config", None)
    if cfg is None:
        return 0.0
    return 0.5 * max(cfg.crop_size - 1, 1) / float(max(cfg.image_width - 1, 1))


def attribute_detection(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sources: torch.Tensor,
    centers: torch.Tensor,
    target: dict[str, Any],
    threshold: float,
    half_extent: float,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Which crop source produced the top-scoring box, and was the GT inside any crop window.

    This answers "where do the detections actually come from" (motion peaks, edge, grid,
    guided) so that the motion-guidance claim can be checked rather than assumed.
    """

    from drishti_v2.data.utils import box_iou

    gt_boxes = target.get("boxes", torch.empty(0, 4))
    gt_visible = gt_boxes.numel() > 0
    result: dict[str, Any] = {"top_source": None, "top_is_tp": False, "gt_visible": gt_visible, "gt_covered": False}
    if scores.numel() > 0:
        top = int(scores.argmax())
        if float(scores[top]) >= threshold:
            result["top_source"] = SOURCE_NAMES.get(int(sources[top]), "pad")
            if gt_visible:
                iou = box_iou(boxes[top].view(1, 4), gt_boxes[:1].to(boxes.dtype))[0, 0]
                result["top_is_tp"] = bool(float(iou) >= iou_threshold)
    if gt_visible and centers.numel() > 0:
        offset = (centers - gt_boxes[0, :2].to(centers.dtype)).abs().amax(dim=-1)
        result["gt_covered"] = bool((offset <= half_extent).any())
    return result


def source_attribution_metrics(
    source_counts: dict[str, int],
    tp_source_counts: dict[str, int],
    gt_visible_frames: int,
    gt_covered_frames: int,
) -> dict[str, float]:
    total = max(sum(source_counts.values()), 1)
    total_tp = max(sum(tp_source_counts.values()), 1)
    metrics: dict[str, float] = {}
    for name in SOURCE_NAMES.values():
        metrics[f"top_box_source_{name}"] = source_counts[name] / total
        metrics[f"true_positive_source_{name}"] = tp_source_counts[name] / total_tp
    metrics["gt_inside_some_crop"] = gt_covered_frames / max(gt_visible_frames, 1)
    return metrics


def _average_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {"motion_direction_accuracy": 0.0}
    keys = set().union(*(item.keys() for item in items))
    return {key: float(sum(item.get(key, 0.0) for item in items) / max(len(items), 1)) for key in keys}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return float(value)
