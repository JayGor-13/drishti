from __future__ import annotations

import csv
import inspect
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from drishti_v2.evaluation.evaluator import DRISHTIEvaluator
from drishti_v2.evaluation.visualize import save_training_curves
from drishti_v2.training.scheduler import make_scheduler
from drishti_v2.training.stage_control import apply_training_stage


class DRISHTITrainer:
    """Production training loop with printed and persisted results."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        loss_fn: nn.Module,
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
        resume_checkpoint: str | Path | None = None,
    ) -> list[dict[str, float]]:
        apply_training_stage(self.model, stage)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(f"No trainable parameters for stage {stage}")
        optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
        scheduler = make_scheduler(optimizer, epochs)
        history: list[dict[str, float]] = []
        best_score = -float("inf")

        start_epoch = 1
        if resume_checkpoint is not None:
            resume_path = Path(resume_checkpoint)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
            payload = torch.load(resume_path, map_location=self.device)
            if not isinstance(payload, dict):
                raise ValueError("A training resume checkpoint must be a state dictionary payload")
            required = {"model", "optimizer", "scheduler", "epoch"}
            missing = sorted(required - payload.keys())
            if missing:
                raise ValueError(f"Incomplete resume checkpoint; missing keys: {missing}")
            checkpoint_stage = payload.get("stage")
            if checkpoint_stage is not None and str(checkpoint_stage).lower() != stage.lower():
                raise ValueError(f"Checkpoint stage {checkpoint_stage!r} does not match requested stage {stage!r}")
            self.model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            start_epoch = int(payload["epoch"]) + 1
            best_score = float(payload.get("best_score", best_score))
            print(f"Resumed {stage} from {resume_path} at epoch {start_epoch}")

        checkpoints_dir = self.output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, epochs + 1):
            self.model.train()
            apply_training_stage(self.model, stage)
            loss_keys = [
                "loss",
                "heatmap",
                "cls",
                "bbox",
                "offset",
                "balance",
                "motion_disp",
                "temporal_consist",
                "traj_smooth",
                "z_loss",
                "gate",
                "focus",
                "focus_halt",
            ]
            accum = {key: 0.0 for key in loss_keys}
            steps = 0
            progress = tqdm(self.train_loader, desc=f"{stage} epoch {epoch}/{epochs}", leave=False)
            data_timer = time.perf_counter()
            for batch in progress:
                data_time = time.perf_counter() - data_timer
                gpu_timer = time.perf_counter()

                started = time.perf_counter()
                frames = batch["frames"].to(self.device)
                output = self.model(frames)
                loss_kwargs = {"targets": batch["targets"]}
                params = inspect.signature(self.loss_fn.forward).parameters
                accepts_extra = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
                extras = {
                    "all_heatmaps": output.all_heatmaps,
                    "logits_seq": output.objectness_logits_seq,
                    "centers_seq": output.proposal_centers_seq,
                    "boxes_seq": output.boxes_seq,
                }
                for key, value in extras.items():
                    if accepts_extra or key in params:
                        loss_kwargs[key] = value
                losses = self.loss_fn(output, **loss_kwargs)
                focus = getattr(self.model, "focus", None)
                if focus is not None and any(p.requires_grad for p in focus.parameters()):
                    focus_terms = focus.training_loss(output.frame_last, output.boxes, output.objectness_logits, batch["targets"])
                    cfg = self.model.config
                    losses = dict(losses)
                    losses["loss"] = losses["loss"] + cfg.w_focus * focus_terms["focus"] + cfg.w_focus_halt * focus_terms["focus_halt"]
                    losses.update(focus_terms)
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
                optimizer.step()
                
                gpu_time = time.perf_counter() - gpu_timer
                elapsed = max(time.perf_counter() - started, 1e-8)
                throughput = float(frames.shape[0] / elapsed)

                for key in accum:
                    value = losses.get(key)
                    if value is not None:
                        accum[key] += float(value.detach().cpu())
                steps += 1

                step_record = self._make_step_record(
                    stage=stage,
                    epoch=epoch,
                    step=steps,
                    losses=losses,
                    grad_norm=float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                    throughput=throughput,
                    sparse_mfu=self._estimate_sparse_mfu(frames, elapsed),
                    output=output,
                )
                self._append_jsonl(self.output_dir / f"{stage}_steps.jsonl", step_record)

                progress.set_postfix(
                    loss=f"{step_record['loss/total']:.4f}",
                    grad=f"{step_record['grad/global_norm']:.2f}",
                    sps=f"{throughput:.2f}",
                    mem=f"{step_record['gpu/memory_allocated_mb']:.0f}MB",
                    data=f"{data_time:.2f}s",
                    gpu=f"{gpu_time:.2f}s"
                )
                data_timer = time.perf_counter()
            scheduler.step()
            record = {f"train_{key}": value / max(1, steps) for key, value in accum.items()}
            record["epoch"] = float(epoch)
            record["learning_rate"] = float(scheduler.get_last_lr()[0])

            if self.val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    evaluator = DRISHTIEvaluator(self.model, self.val_loader, device=self.device)
                    metrics = evaluator.evaluate(
                        print_results=False,
                        output_path=self.output_dir / f"{stage}_eval_metrics.json",
                        save_visualizations=True,
                    )
                record.update({f"val_{key}": float(value) for key, value in metrics.items() if isinstance(value, (int, float))})
                score = float(metrics.get("map50", 0.0))
            else:
                score = -record["train_loss"]

            history.append(record)
            self._append_csv(record)
            save_training_curves(history, self.output_dir / "training_curves.png")
            print(json.dumps(record, indent=2, sort_keys=True))
            if score > best_score:
                best_score = score
                best_filename = checkpoint_name or f"{stage}_best.pt"
                self.save_checkpoint(
                    checkpoints_dir / best_filename,
                    epoch,
                    record,
                    optimizer,
                    scheduler,
                    stage=stage,
                    best_score=best_score,
                )
            
            self.save_checkpoint(
                checkpoints_dir / f"{stage}_latest.pt",
                epoch,
                record,
                optimizer,
                scheduler,
                stage=stage,
                best_score=best_score,
            )
            
            if epoch == 1 or epoch % 10 == 0:
                self.save_checkpoint(
                    checkpoints_dir / f"{stage}_epoch_{epoch}.pt",
                    epoch,
                    record,
                    optimizer,
                    scheduler,
                    stage=stage,
                    best_score=best_score,
                )
        return history

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: dict[str, Any],
        optimizer=None,
        scheduler=None,
        stage: str | None = None,
        best_score: float | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"epoch": epoch, "model": self.model.state_dict(), "metrics": metrics}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        if stage is not None:
            payload["stage"] = stage
        if best_score is not None:
            payload["best_score"] = best_score
        torch.save(payload, path)

    def _append_csv(self, record: dict[str, float]) -> None:
        path = self.output_dir / "history.csv"
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(record))
            if not exists:
                writer.writeheader()
            writer.writerow(record)

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _make_step_record(
        self,
        stage: str,
        epoch: int,
        step: int,
        losses: dict[str, torch.Tensor],
        grad_norm: float,
        throughput: float,
        sparse_mfu: float,
        output: Any,
    ) -> dict[str, Any]:
        zero = losses["loss"] * 0.0
        record = {
            "stage": stage,
            "epoch": epoch,
            "step": step,
            "loss/total": self._loss_value(losses, "loss", zero),
            "loss/heatmap": self._loss_value(losses, "heatmap", zero),
            "loss/cls": self._loss_value(losses, "cls", zero),
            "loss/bbox": self._loss_value(losses, "bbox", zero),
            "loss/offset": self._loss_value(losses, "offset", zero),
            "loss/balance": self._loss_value(losses, "balance", zero),
            "loss/motion_disp": self._loss_value(losses, "motion_disp", zero),
            "loss/temporal_consist": self._loss_value(losses, "temporal_consist", zero),
            "loss/traj_smooth": self._loss_value(losses, "traj_smooth", zero),
            "loss/z_loss": self._loss_value(losses, "z_loss", zero),
            "loss/focus": self._loss_value(losses, "focus", zero),
            "loss/focus_halt": self._loss_value(losses, "focus_halt", zero),
            "grad/global_norm": grad_norm,
            "perf/throughput_samples_sec": throughput,
            "perf/sparse_mfu": sparse_mfu,
            "gpu/memory_allocated_mb": self._gpu_memory_allocated(),
            "gpu/max_memory_allocated_mb": self._gpu_max_memory_allocated(),
        }
        diagnostics = getattr(output, "moe_diagnostics", None)
        if diagnostics is not None:
            record.update(
                {
                    "moe/load_balance_cv": float(diagnostics.load_balance_cv.detach().cpu()),
                    "moe/router_entropy": float(diagnostics.router_entropy.detach().cpu()),
                    "moe/token_drop_rate": float(diagnostics.token_drop_rate.detach().cpu()),
                    "moe/expert_overlap": float(diagnostics.expert_overlap.detach().cpu()),
                }
            )
        return record

    @staticmethod
    def _loss_value(losses: dict[str, torch.Tensor], key: str, default: torch.Tensor) -> float:
        value = losses.get(key, default)
        return float(value.detach().cpu())

    def _gpu_memory_allocated(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return float(torch.cuda.memory_allocated(self.device) / (1024**2))

    def _gpu_max_memory_allocated(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return float(torch.cuda.max_memory_allocated(self.device) / (1024**2))

    def _estimate_sparse_mfu(self, frames: torch.Tensor, elapsed: float) -> float:
        if self.device.type != "cuda":
            return 0.0
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return 0.0
        tokens = frames.shape[0] * frames.shape[1] * max(cfg.num_crops, cfg.dense_num_crops)
        active_experts = min(cfg.top_k, cfg.num_experts)
        rough_flops = tokens * cfg.encoder_feature_dim * cfg.expert_ffn_dim * active_experts * 4
        assumed_peak_flops = 100e12
        return float(min(rough_flops / (elapsed * assumed_peak_flops), 1.0))
