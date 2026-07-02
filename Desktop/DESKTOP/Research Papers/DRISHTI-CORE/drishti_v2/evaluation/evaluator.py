from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from drishti_v2.evaluation.metrics import detection_metrics


class DRISHTIEvaluator:
    """Runs detector evaluation and prints result metrics."""

    def __init__(self, model: nn.Module, loader: DataLoader, device: str | torch.device = "cpu", threshold: float = 0.3) -> None:
        self.model = model
        self.loader = loader
        self.device = torch.device(device)
        self.threshold = threshold

    @torch.no_grad()
    def evaluate(self, print_results: bool = True, output_path: str | Path | None = None) -> dict[str, float]:
        self.model.eval()
        predictions = []
        targets = []
        for batch in self.loader:
            frames = batch["frames"].to(self.device)
            output = self.model(frames)
            scores = torch.sigmoid(output.objectness_logits.squeeze(-1))
            for b_idx in range(frames.shape[0]):
                predictions.append({"boxes": output.boxes[b_idx].detach().cpu(), "scores": scores[b_idx].detach().cpu()})
                targets.append(batch["targets"][b_idx][-1])
        metrics = detection_metrics(predictions, targets, score_threshold=self.threshold)
        if print_results:
            print(json.dumps(metrics, indent=2, sort_keys=True))
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return metrics
