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
