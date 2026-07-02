from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from drishti_v2.evaluation.evaluator import DRISHTIEvaluator
from drishti_v2.training.losses import DRISHTILoss
from drishti_v2.training.scheduler import make_scheduler
from drishti_v2.training.stage_control import apply_training_stage


class DRISHTITrainer:
    """Production training loop with printed and persisted results."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        loss_fn: DRISHTILoss,
        output_dir: str | Path = "results/train",
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

    def fit(
        self,
        stage: str,
        epochs: int,
        lr: float,
        weight_decay: float = 1e-4,
        checkpoint_name: str | None = None,
    ) -> list[dict[str, float]]:
        apply_training_stage(self.model, stage)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(f"No trainable parameters for stage {stage}")
        optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
        scheduler = make_scheduler(optimizer, epochs)
        history: list[dict[str, float]] = []
        best_score = -1.0
        checkpoint_name = checkpoint_name or f"{stage}_best.pt"

        for epoch in range(1, epochs + 1):
            self.model.train()
            apply_training_stage(self.model, stage)
            accum = {"loss": 0.0, "heatmap": 0.0, "cls": 0.0, "bbox": 0.0, "balance": 0.0}
            steps = 0
            for batch in tqdm(self.train_loader, desc=f"{stage} epoch {epoch}/{epochs}", leave=False):
                frames = batch["frames"].to(self.device)
                output = self.model(frames)
                losses = self.loss_fn(output, batch["targets"])
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
                optimizer.step()
                for key in accum:
                    accum[key] += float(losses[key].detach().cpu())
                steps += 1
            scheduler.step()
            record = {f"train_{key}": value / max(1, steps) for key, value in accum.items()}
            record["epoch"] = float(epoch)
            record["learning_rate"] = float(scheduler.get_last_lr()[0])

            if self.val_loader is not None:
                evaluator = DRISHTIEvaluator(self.model, self.val_loader, device=self.device)
                metrics = evaluator.evaluate(print_results=False)
                record.update({f"val_{key}": float(value) for key, value in metrics.items()})
                score = float(metrics.get("map50", 0.0))
            else:
                score = -record["train_loss"]

            history.append(record)
            self._append_csv(record)
            print(json.dumps(record, indent=2, sort_keys=True))
            if score > best_score:
                best_score = score
                self.save_checkpoint(self.output_dir / checkpoint_name, epoch, record)
            self.save_checkpoint(self.output_dir / f"{stage}_last.pt", epoch, record)
        return history

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "model": self.model.state_dict(), "metrics": metrics}, path)

    def _append_csv(self, record: dict[str, float]) -> None:
        path = self.output_dir / "history.csv"
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(record))
            if not exists:
                writer.writeheader()
            writer.writerow(record)
