from __future__ import annotations

import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from drishti_v2.evaluation.evaluator import DRISHTIEvaluator
from drishti_v2.models.moe import MoEDiagnostics
from drishti_v2.training.losses import DRISHTILoss
from drishti_v2.training.scheduler import make_scheduler
from drishti_v2.training.stage_control import apply_training_stage


def _setup_step_logger(output_dir: Path) -> logging.Logger:
    """Create a dedicated file logger for per-step JSONL logs."""
    logger = logging.getLogger("drishti.step_log")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # Remove existing handlers to avoid duplicates on resume
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    log_path = output_dir / "step_log.jsonl"
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def _count_params(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _estimate_flops_per_step(model: nn.Module, batch_size: int) -> float:
    """Rough FLOPs estimate: 2 * params * batch_size (fwd) + 4 * params * batch_size (bwd) = 6 * trainable * B."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 6.0 * trainable * batch_size


def _score_from_metrics(metrics: dict[str, Any]) -> float:
    if "val_map50" in metrics:
        return float(metrics["val_map50"])
    if "map50" in metrics:
        return float(metrics["map50"])
    if "train_loss" in metrics:
        return -float(metrics["train_loss"])
    return -math.inf


class DRISHTITrainer:
    """Production training loop with comprehensive per-step metrics logging.

    Metrics logged per step (JSONL):
        - All loss components (total, heatmap, cls, bbox, balance)
        - MoE diagnostics: expert_utilization, routing_probabilities, router_entropy,
          token_drop_rate, expert_reuse_frequency, load_balance_cv
        - Gradient norm (global L2)
        - Learning rate
        - Step wall-clock time
        - Throughput (samples/sec)
        - GPU memory allocated / reserved (if CUDA)
        - Sparse MFU estimate

    Metrics logged per epoch (CSV + JSON):
        - Averaged training metrics
        - Full validation metrics
        - Epoch-level throughput, memory peaks, total time
    """

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
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.metrics_dir = self.output_dir / "epoch_metrics"
        self.device = torch.device(device)
        self.step_logger = _setup_step_logger(self.output_dir)

    # ------------------------------------------------------------------
    # Step-level metric collection
    # ------------------------------------------------------------------
    def _collect_moe_diagnostics(self, diag: MoEDiagnostics | None) -> dict[str, Any]:
        """Extract all MoE routing metrics into a flat dict."""
        if diag is None:
            return {}
        return {
            "moe/balance_loss": float(diag.balance_loss.detach().cpu()),
            "moe/router_entropy": float(diag.router_entropy.cpu()),
            "moe/token_drop_rate": float(diag.token_drop_rate.cpu()),
            "moe/expert_reuse_frequency": float(diag.expert_reuse_frequency.cpu()),
            "moe/load_balance_cv": float(diag.load_balance_cv.cpu()),
            "moe/expert_utilization": [round(float(v), 6) for v in diag.expert_utilization.cpu()],
            "moe/routing_probabilities": [round(float(v), 6) for v in diag.routing_probabilities.cpu()],
        }

    def _collect_memory_metrics(self) -> dict[str, float]:
        """Collect GPU memory stats (bytes -> MB)."""
        if self.device.type != "cuda":
            return {}
        return {
            "gpu/memory_allocated_mb": round(torch.cuda.memory_allocated(self.device) / 1e6, 2),
            "gpu/memory_reserved_mb": round(torch.cuda.memory_reserved(self.device) / 1e6, 2),
            "gpu/max_memory_allocated_mb": round(torch.cuda.max_memory_allocated(self.device) / 1e6, 2),
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def fit(
        self,
        stage: str,
        epochs: int,
        lr: float,
        weight_decay: float = 1e-4,
        checkpoint_name: str | None = None,
        resume_from: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        apply_training_stage(self.model, stage)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(f"No trainable parameters for stage {stage}")
        optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
        scheduler = make_scheduler(optimizer, epochs)

        start_epoch = 1
        global_step = 0
        if resume_from:
            payload = torch.load(resume_from, map_location=self.device)
            if "model" in payload:
                self.model.load_state_dict(payload["model"])
            else:
                self.model.load_state_dict(payload)
            if "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])
            if "scheduler" in payload:
                scheduler.load_state_dict(payload["scheduler"])
            start_epoch = payload.get("epoch", 0) + 1
            global_step = int(payload.get("global_step", 0))
            print(f"Resuming {stage} training from {Path(resume_from)} at epoch {start_epoch}")

        history = self._load_existing_history(before_epoch=start_epoch)
        best_checkpoint_name = checkpoint_name or "best_model.pt"
        best_path = self.checkpoint_dir / best_checkpoint_name
        latest_path = self.checkpoint_dir / "latest.pt"
        best_score = self._load_best_score(best_path, history)
        total_params, trainable_params = _count_params(self.model)

        # Log run metadata once
        run_meta = {
            "event": "run_start",
            "stage": stage,
            "epochs": epochs,
            "lr": lr,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
        }
        self.step_logger.info(json.dumps(run_meta, sort_keys=True))

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        if start_epoch > epochs:
            summary = self._write_final_summary(
                stage=stage,
                epochs=epochs,
                history=history,
                best_score=best_score,
                best_checkpoint=best_path,
                latest_checkpoint=latest_path,
            )
            self._print_final_summary(summary, history[-1] if history else None)
            return history

        print(f"Training {stage}: epochs {start_epoch}-{epochs} | checkpoints: {self.checkpoint_dir}")

        for epoch in range(start_epoch, epochs + 1):
            self.model.train()
            apply_training_stage(self.model, stage)

            # Epoch-level accumulators
            accum = {"loss": 0.0, "heatmap": 0.0, "cls": 0.0, "bbox": 0.0, "balance": 0.0}
            moe_accum = {
                "router_entropy": 0.0, "token_drop_rate": 0.0,
                "expert_reuse_frequency": 0.0, "load_balance_cv": 0.0,
            }
            grad_norm_accum = 0.0
            epoch_samples = 0
            steps = 0
            epoch_start_time = time.perf_counter()

            progress = tqdm(
                self.train_loader,
                desc=f"{stage} epoch {epoch}/{epochs}",
                leave=True,
                dynamic_ncols=True,
                unit="batch",
            )
            for batch in progress:
                step_start = time.perf_counter()
                frames = batch["frames"].to(self.device)
                batch_size = frames.shape[0]
                output = self.model(frames)
                losses = self.loss_fn(output, batch["targets"])
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()

                # --- Gradient norm (before clipping) ---
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0).detach().cpu())

                optimizer.step()
                step_elapsed = time.perf_counter() - step_start

                # --- Accumulate losses ---
                for key in accum:
                    accum[key] += float(losses[key].detach().cpu())

                # --- MoE diagnostics ---
                moe_metrics = self._collect_moe_diagnostics(output.moe_diagnostics)
                for k in moe_accum:
                    moe_accum[k] += moe_metrics.get(f"moe/{k}", 0.0)

                # --- Throughput & MFU ---
                throughput = batch_size / max(step_elapsed, 1e-8)
                flops_per_step = _estimate_flops_per_step(self.model, batch_size)
                # MFU = achieved_flops / peak_device_flops (RTX 4090 approx. 82.6 TFLOPS FP32)
                peak_flops = 82.6e12  # conservative FP32 estimate
                sparse_mfu = (flops_per_step / max(step_elapsed, 1e-8)) / peak_flops

                # --- Memory ---
                mem_metrics = self._collect_memory_metrics()

                # --- Per-step JSONL log ---
                step_record = {
                    "event": "step",
                    "stage": stage,
                    "epoch": epoch,
                    "global_step": global_step,
                    "step_in_epoch": steps,
                    "batch_size": batch_size,
                    # Losses
                    "loss/total": round(float(losses["loss"].detach().cpu()), 6),
                    "loss/heatmap": round(float(losses["heatmap"].detach().cpu()), 6),
                    "loss/cls": round(float(losses["cls"].detach().cpu()), 6),
                    "loss/bbox": round(float(losses["bbox"].detach().cpu()), 6),
                    "loss/balance": round(float(losses["balance"].detach().cpu()), 6),
                    # Gradient
                    "grad/global_norm": round(grad_norm, 6),
                    # Learning rate
                    "lr": float(scheduler.get_last_lr()[0]),
                    # Throughput & efficiency
                    "perf/step_time_sec": round(step_elapsed, 4),
                    "perf/throughput_samples_sec": round(throughput, 2),
                    "perf/sparse_mfu": round(sparse_mfu, 6),
                }
                step_record.update(moe_metrics)
                step_record.update(mem_metrics)
                self.step_logger.info(json.dumps(step_record, sort_keys=True))

                grad_norm_accum += grad_norm
                epoch_samples += batch_size
                steps += 1
                global_step += 1
                progress.set_postfix(loss=f"{step_record['loss/total']:.4f}", lr=f"{step_record['lr']:.2e}")

            scheduler.step()
            epoch_elapsed = time.perf_counter() - epoch_start_time

            # --- Epoch-level averages ---
            record = {f"train_{key}": value / max(1, steps) for key, value in accum.items()}
            record["epoch"] = float(epoch)
            record["learning_rate"] = float(scheduler.get_last_lr()[0])
            record["train_grad_norm_avg"] = round(grad_norm_accum / max(1, steps), 6)
            record["train_epoch_time_sec"] = round(epoch_elapsed, 2)
            record["train_throughput_samples_sec"] = round(epoch_samples / max(epoch_elapsed, 1e-8), 2)

            # MoE epoch averages
            for k, v in moe_accum.items():
                record[f"train_moe_{k}"] = round(v / max(1, steps), 6)

            # Memory peaks
            if self.device.type == "cuda":
                record["gpu_peak_memory_mb"] = round(torch.cuda.max_memory_allocated(self.device) / 1e6, 2)

            if self.val_loader is not None:
                evaluator = DRISHTIEvaluator(self.model, self.val_loader, device=self.device)
                metrics = evaluator.evaluate(print_results=False)
                record.update({f"val_{key}": float(value) for key, value in metrics.items()})
                score = float(metrics.get("map50", 0.0))
            else:
                score = -record["train_loss"]

            history.append(record)
            self._write_history(history)
            self._write_epoch_metrics(epoch, record)

            # Log epoch summary to step log as well
            epoch_record = dict(record)
            epoch_record["event"] = "epoch_end"
            self.step_logger.info(json.dumps(epoch_record, sort_keys=True))

            if score > best_score or not best_path.exists():
                best_score = score
                self.save_checkpoint(best_path, epoch, record, optimizer, scheduler, stage=stage, global_step=global_step)
            self.save_checkpoint(latest_path, epoch, record, optimizer, scheduler, stage=stage, global_step=global_step)
            checkpointed_epoch_path = None
            if epoch % 10 == 0:
                checkpointed_epoch_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
                self.save_checkpoint(
                    checkpointed_epoch_path,
                    epoch,
                    record,
                    optimizer,
                    scheduler,
                    stage=stage,
                    global_step=global_step,
                )
            self._write_checkpoint_manifest(
                stage=stage,
                latest_path=latest_path,
                best_path=best_path,
                epoch=epoch,
                metrics=record,
                best_score=best_score,
                checkpointed_epoch_path=checkpointed_epoch_path,
            )
        summary = self._write_final_summary(
            stage=stage,
            epochs=epochs,
            history=history,
            best_score=best_score,
            best_checkpoint=best_path,
            latest_checkpoint=latest_path,
        )
        self._print_final_summary(summary, history[-1] if history else None)
        return history

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: dict[str, Any],
        optimizer=None,
        scheduler=None,
        stage: str | None = None,
        global_step: int = 0,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "stage": stage,
            "model": self.model.state_dict(),
            "metrics": metrics,
        }
        if optimizer:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler:
            payload["scheduler"] = scheduler.state_dict()
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

    def _load_existing_history(self, before_epoch: int) -> list[dict[str, Any]]:
        path = self.output_dir / "history.json"
        if not path.exists():
            return []
        records = json.loads(path.read_text(encoding="utf-8"))
        return [record for record in records if int(record.get("epoch", 0)) < before_epoch]

    def _load_best_score(self, best_path: Path, history: list[dict[str, Any]]) -> float:
        if best_path.exists():
            payload = torch.load(best_path, map_location="cpu")
            if isinstance(payload, dict):
                return _score_from_metrics(payload.get("metrics", {}))
        if history:
            return max(_score_from_metrics(record) for record in history)
        return -math.inf

    def _write_history(self, history: list[dict[str, Any]]) -> None:
        json_path = self.output_dir / "history.json"
        json_path.write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")

        csv_path = self.output_dir / "history.csv"
        fieldnames = sorted({key for record in history for key in record})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for record in history:
                writer.writerow(record)

    def _write_epoch_metrics(self, epoch: int, record: dict[str, Any]) -> None:
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        path = self.metrics_dir / f"epoch_{epoch:04d}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    def _write_checkpoint_manifest(
        self,
        stage: str,
        latest_path: Path,
        best_path: Path,
        epoch: int,
        metrics: dict[str, Any],
        best_score: float,
        checkpointed_epoch_path: Path | None,
    ) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "stage": stage,
            "latest_epoch": epoch,
            "latest_checkpoint": str(latest_path),
            "best_checkpoint": str(best_path),
            "best_score": best_score,
            "latest_metrics": metrics,
            "saved_epoch_checkpoints": sorted(path.name for path in self.checkpoint_dir.glob("epoch_*.pt")),
        }
        if checkpointed_epoch_path is not None:
            manifest["last_periodic_checkpoint"] = checkpointed_epoch_path.name
        (self.checkpoint_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    def _write_final_summary(
        self,
        stage: str,
        epochs: int,
        history: list[dict[str, Any]],
        best_score: float,
        best_checkpoint: Path,
        latest_checkpoint: Path,
    ) -> dict[str, Any]:
        latest = history[-1] if history else {}
        best_record = max(history, key=_score_from_metrics) if history else {}
        summary = {
            "stage": stage,
            "epochs_requested": epochs,
            "epochs_completed": int(latest.get("epoch", 0)) if latest else 0,
            "best_epoch": int(best_record.get("epoch", 0)) if best_record else 0,
            "best_score": best_score,
            "best_checkpoint": str(best_checkpoint),
            "latest_checkpoint": str(latest_checkpoint),
            "history_csv": str(self.output_dir / "history.csv"),
            "history_json": str(self.output_dir / "history.json"),
            "epoch_metrics_dir": str(self.metrics_dir),
            "latest_metrics": latest,
        }
        (self.output_dir / "final_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary

    def _print_final_summary(self, summary: dict[str, Any], latest: dict[str, Any] | None) -> None:
        print("\nTraining Summary")
        print("=" * 72)
        print(f"Stage: {summary['stage']} | epochs completed: {summary['epochs_completed']}/{summary['epochs_requested']}")
        print(f"Best epoch: {summary['best_epoch']} | best score: {summary['best_score']:.6f}")
        print(f"Latest checkpoint: {summary['latest_checkpoint']}")
        print(f"Best checkpoint:   {summary['best_checkpoint']}")
        if latest:
            keys = [
                "train_loss",
                "val_map50",
                "val_map75",
                "val_precision",
                "val_recall",
                "val_f1",
                "train_epoch_time_sec",
            ]
            for key in keys:
                if key in latest:
                    value = latest[key]
                    print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
        print(f"Metrics: {summary['history_csv']} and {summary['history_json']}")
