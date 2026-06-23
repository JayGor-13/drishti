"""DRISHTI-CORE Anti-UAV experiment runner."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import DRISHTIConfig, DRISHTIPipeline
from train import (
    AntiUAVDatasetPaths,
    AntiUAVRGBTVideoDataset,
    DRISHTICollator,
    DRISHTILossWeights,
    MODELSCOPE_ANTI_UAV_URL,
    ModelScopeAntiUAVCocoDataset,
    SyntheticAntiUAVDataset,
    configure_drishti_training_stage,
    detector_loss,
    detector_stage_loss,
    moe_stage_loss,
    scalar_metrics,
    stage_checkpoint_name,
)


@dataclass
class ExperimentConfig:
    smoke: bool = True
    dataset_url: str = MODELSCOPE_ANTI_UAV_URL
    data_root: str | None = None
    train_split: str = "train"
    val_split: str = "test"
    modality: str = "visible"
    train_image_root: str | None = None
    train_ann_file: str | None = None
    val_image_root: str | None = None
    val_ann_file: str | None = None
    results_dir: str = "results"
    seed: int = 42
    stage: str = "all"
    smoke_train_samples: int = 8
    smoke_val_samples: int = 4
    smoke_epochs: int = 1
    full_epochs: int = 15
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    num_frames: int = 5
    clip_stride: int = 4
    frame_stride: int = 1
    frame_height: int = 64
    frame_width: int = 64
    image_channels: int = 3
    box_format: str = "xywh"
    crop_size: int = 64
    num_crops: int = 8
    feature_dim: int = 64
    temporal_heads: int = 4
    temporal_layers: int = 1
    temporal_ffn_dim: int = 128
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_ffn_dim: int = 128
    eval_batches: int = 10

    @property
    def epochs(self) -> int:
        return self.smoke_epochs if self.smoke else self.full_epochs


DEFAULT_CONFIG = ExperimentConfig()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_config(config: ExperimentConfig) -> DRISHTIConfig:
    return DRISHTIConfig(
        image_channels=config.image_channels,
        temporal_window=config.num_frames,
        crop_size=config.crop_size,
        num_crops=config.num_crops,
        feature_dim=config.feature_dim,
        temporal_input_dim=config.feature_dim + 1,
        temporal_heads=config.temporal_heads,
        temporal_layers=config.temporal_layers,
        temporal_ffn_dim=config.temporal_ffn_dim,
        moe_num_experts=config.moe_num_experts,
        moe_top_k=config.moe_top_k,
        moe_ffn_dim=config.moe_ffn_dim,
    )


def _paths(config: ExperimentConfig) -> tuple[AntiUAVDatasetPaths, AntiUAVDatasetPaths]:
    return (
        AntiUAVDatasetPaths(config.train_image_root, config.train_ann_file),
        AntiUAVDatasetPaths(config.val_image_root, config.val_ann_file),
    )


def make_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    train_paths, val_paths = _paths(config)
    collator = DRISHTICollator()
    if config.data_root:
        train_dataset = AntiUAVRGBTVideoDataset(
            data_root=config.data_root,
            split=config.train_split,
            modality=config.modality,
            num_frames=config.num_frames,
            height=config.frame_height,
            width=config.frame_width,
            clip_stride=config.clip_stride,
            frame_stride=config.frame_stride,
            image_channels=config.image_channels,
            box_format=config.box_format,
        )
        val_dataset = AntiUAVRGBTVideoDataset(
            data_root=config.data_root,
            split=config.val_split,
            modality=config.modality,
            num_frames=config.num_frames,
            height=config.frame_height,
            width=config.frame_width,
            clip_stride=config.clip_stride,
            frame_stride=config.frame_stride,
            image_channels=config.image_channels,
            box_format=config.box_format,
        )
        source = f"antiuav_rgbt_{config.modality}_video"
    elif train_paths.is_complete and val_paths.is_complete:
        train_dataset = ModelScopeAntiUAVCocoDataset(
            root=str(train_paths.image_root),
            ann_file=str(train_paths.ann_file),
            num_frames=config.num_frames,
            height=config.frame_height,
            width=config.frame_width,
        )
        val_dataset = ModelScopeAntiUAVCocoDataset(
            root=str(val_paths.image_root),
            ann_file=str(val_paths.ann_file),
            num_frames=config.num_frames,
            height=config.frame_height,
            width=config.frame_width,
        )
        source = "modelscope_coco"
    elif config.smoke:
        train_dataset = SyntheticAntiUAVDataset(
            num_samples=config.smoke_train_samples,
            num_frames=config.num_frames,
            height=config.frame_height,
            width=config.frame_width,
            image_channels=config.image_channels,
        )
        val_dataset = SyntheticAntiUAVDataset(
            num_samples=config.smoke_val_samples,
            num_frames=config.num_frames,
            height=config.frame_height,
            width=config.frame_width,
            image_channels=config.image_channels,
        )
        source = "synthetic_smoke"
    else:
        raise ValueError(
            "Full DRISHTI training requires --data-root for Anti-UAV-RGBT or "
            "--train-image-root/--train-ann-file/--val-image-root/--val-ann-file "
            f"for COCO-format data from {config.dataset_url}."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        drop_last=False,
    )
    sizes = {
        "dataset_url": config.dataset_url,
        "source": source,
        "modality": config.modality,
        "image_channels": config.image_channels,
        "train": len(train_dataset),
        "val": len(val_dataset),
    }
    return train_loader, val_loader, sizes


def build_optimizer(model: DRISHTIPipeline, config: ExperimentConfig) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters for the selected stage")
    return torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)


def forward_and_loss(
    model: DRISHTIPipeline,
    batch: dict[str, Any],
    config: ExperimentConfig,
    weights: DRISHTILossWeights,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float], Any]:
    frames = batch["frames"].to(device)
    frame_targets = batch["frame_targets"]
    if config.stage == "detector":
        output = model.forward_detector(frames)
        parts = detector_stage_loss(output, frame_targets, weights)
        return parts["det"], scalar_metrics(parts), output
    if config.stage == "temporal":
        output = model.forward_temporal(frames)
        parts = detector_loss(output, frame_targets, weights)
        return parts["det"], scalar_metrics(parts), output

    output = model(frames)
    parts = moe_stage_loss(output, frame_targets, weights)
    return parts["loss"], scalar_metrics(parts), output


def output_diagnostics(output: Any) -> dict[str, float]:
    if hasattr(output, "router_probs"):
        confidence = torch.sigmoid(output.object_logits)
        return {
            "mean_confidence": float(confidence.mean().detach().cpu()),
            "mean_box_width": float(output.boxes[..., 2].mean().detach().cpu()),
            "mean_box_height": float(output.boxes[..., 3].mean().detach().cpu()),
            "load_balance": float(output.load_balance_loss.detach().cpu()),
        }
    confidence = torch.sigmoid(output.object_logits)
    return {
        "mean_confidence": float(confidence.mean().detach().cpu()),
        "mean_box_width": float(output.boxes[..., 2].mean().detach().cpu()),
        "mean_box_height": float(output.boxes[..., 3].mean().detach().cpu()),
        "load_balance": 0.0,
    }


def train_model(
    model: DRISHTIPipeline,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    results_dir: Path,
) -> list[dict[str, float]]:
    weights = DRISHTILossWeights()
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{config.epochs}")
        for step, batch in enumerate(progress, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics, output = forward_and_loss(model, batch, config, weights, device)
            loss.backward()
            if config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    config.grad_clip_norm,
                )
            optimizer.step()
            row = {
                "epoch": float(epoch + 1),
                "step": float(step),
                **metrics,
                **output_diagnostics(output),
            }
            history.append(row)
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.3f}")

        save_checkpoint(
            model,
            results_dir,
            optimizer=optimizer,
            config=config,
            epoch=epoch + 1,
            name=f"epoch_{epoch + 1:03d}.pt",
        )
    return history


def evaluate_model(
    model: DRISHTIPipeline,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    weights = DRISHTILossWeights()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="evaluate"), start=1):
            if config.smoke and batch_idx > config.eval_batches:
                break
            _, metrics, output = forward_and_loss(model, batch, config, weights, device)
            rows.append({**metrics, **output_diagnostics(output)})
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def write_history_csv(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    model: DRISHTIPipeline,
    results_dir: Path,
    optimizer: torch.optim.Optimizer | None,
    config: ExperimentConfig,
    epoch: int | None,
    name: str,
) -> Path:
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "experiment_config": asdict(config),
        "epoch": epoch,
    }
    path = checkpoint_dir / name
    torch.save(checkpoint, path)
    torch.save(checkpoint, checkpoint_dir / "latest.pt")
    return path


def load_checkpoint_if_requested(
    model: DRISHTIPipeline,
    checkpoint_path: str | None,
    device: torch.device,
) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", dest="smoke", action="store_true")
    parser.add_argument("--full", dest="smoke", action="store_false")
    parser.set_defaults(smoke=DEFAULT_CONFIG.smoke)
    parser.add_argument("--dataset-url", default=DEFAULT_CONFIG.dataset_url)
    parser.add_argument("--data-root", default=DEFAULT_CONFIG.data_root)
    parser.add_argument("--train-split", default=DEFAULT_CONFIG.train_split)
    parser.add_argument("--val-split", default=DEFAULT_CONFIG.val_split)
    parser.add_argument("--modality", choices=["infrared", "visible"], default=DEFAULT_CONFIG.modality)
    parser.add_argument("--train-image-root", default=DEFAULT_CONFIG.train_image_root)
    parser.add_argument("--train-ann-file", default=DEFAULT_CONFIG.train_ann_file)
    parser.add_argument("--val-image-root", default=DEFAULT_CONFIG.val_image_root)
    parser.add_argument("--val-ann-file", default=DEFAULT_CONFIG.val_ann_file)
    parser.add_argument("--results-dir", default=DEFAULT_CONFIG.results_dir)
    parser.add_argument("--stage", choices=["detector", "temporal", "moe", "all"], default=DEFAULT_CONFIG.stage)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_CONFIG.num_frames)
    parser.add_argument("--clip-stride", type=int, default=DEFAULT_CONFIG.clip_stride)
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_CONFIG.frame_stride)
    parser.add_argument("--height", type=int, default=DEFAULT_CONFIG.frame_height)
    parser.add_argument("--width", type=int, default=DEFAULT_CONFIG.frame_width)
    parser.add_argument("--image-channels", type=int, choices=[1, 3], default=None)
    parser.add_argument("--ir-repeat-rgb", action="store_true")
    parser.add_argument("--box-format", choices=["xywh", "xyxy"], default=DEFAULT_CONFIG.box_format)
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CONFIG.crop_size)
    parser.add_argument("--num-crops", type=int, default=DEFAULT_CONFIG.num_crops)
    parser.add_argument("--feature-dim", type=int, default=DEFAULT_CONFIG.feature_dim)
    parser.add_argument("--moe-num-experts", type=int, default=DEFAULT_CONFIG.moe_num_experts)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    image_channels = args.image_channels
    if args.ir_repeat_rgb:
        image_channels = 3
    elif image_channels is None:
        image_channels = 1 if args.data_root and args.modality == "infrared" else DEFAULT_CONFIG.image_channels

    config = ExperimentConfig(
        smoke=args.smoke,
        dataset_url=args.dataset_url,
        data_root=args.data_root,
        train_split=args.train_split,
        val_split=args.val_split,
        modality=args.modality,
        train_image_root=args.train_image_root,
        train_ann_file=args.train_ann_file,
        val_image_root=args.val_image_root,
        val_ann_file=args.val_ann_file,
        results_dir=args.results_dir,
        stage=args.stage,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        clip_stride=args.clip_stride,
        frame_stride=args.frame_stride,
        frame_height=args.height,
        frame_width=args.width,
        image_channels=image_channels,
        box_format=args.box_format,
        crop_size=args.crop_size,
        num_crops=args.num_crops,
        feature_dim=args.feature_dim,
        moe_num_experts=args.moe_num_experts,
    )
    if args.epochs is not None:
        if config.smoke:
            config.smoke_epochs = args.epochs
        else:
            config.full_epochs = args.epochs
    return config


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    set_seed(config.seed)
    device = torch.device(args.device)
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Running DRISHTI-CORE Anti-UAV detector "
        f"mode={'smoke' if config.smoke else 'full'} "
        f"stage={config.stage} epochs={config.epochs} "
        f"source={config.data_root or config.dataset_url}"
    )

    train_loader, val_loader, sizes = make_dataloaders(config)
    model = DRISHTIPipeline(build_model_config(config)).to(device)
    load_checkpoint_if_requested(model, args.resume_checkpoint, device)
    configure_drishti_training_stage(model, config.stage)
    optimizer = build_optimizer(model, config)

    history = train_model(model, train_loader, config, device, optimizer, results_dir)
    eval_summary = evaluate_model(model, val_loader, config, device)
    write_history_csv(history, results_dir / "train_history.csv")
    (results_dir / "config.json").write_text(
        json.dumps({**asdict(config), "dataset_sizes": sizes}, indent=2),
        encoding="utf-8",
    )
    (results_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    save_checkpoint(
        model,
        results_dir,
        optimizer=optimizer,
        config=config,
        epoch=config.epochs,
        name=stage_checkpoint_name(config.stage if config.stage != "all" else "moe"),
    )

    print("Experiment complete.")
    print(f"Results written to: {results_dir.resolve()}")
    print(json.dumps(eval_summary, indent=2))


if __name__ == "__main__":
    main()
