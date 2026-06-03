"""Anti-UAV experiment runner for the T-MoE detector."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import TMoEAntiDroneDetector, TMoEConfig
from train import (
    AntiUAVDatasetPaths,
    AntiUAVDetectionCollator,
    AntiUAVRGBTVideoDataset,
    MODELSCOPE_ANTI_UAV_URL,
    ModelScopeAntiUAVCocoDataset,
    SyntheticAntiUAVDataset,
    TMoELossWeights,
    cfcr_loss,
    detection_loss,
    expert_lora_similarity,
    load_balancing_loss,
    routing_entropy,
    semantic_alignment,
)


@dataclass
class ExperimentConfig:
    """Top-level Anti-UAV experiment configuration."""

    smoke: bool = True
    dataset_url: str = MODELSCOPE_ANTI_UAV_URL
    data_root: str | None = None
    train_split: str = "train"
    val_split: str = "test"
    modality: str = "infrared"
    train_image_root: str | None = None
    train_ann_file: str | None = None
    val_image_root: str | None = None
    val_ann_file: str | None = None
    results_dir: str = "results"
    seed: int = 42
    stage: str = "sparse"
    smoke_train_samples: int = 16
    smoke_val_samples: int = 8
    smoke_epochs: int = 1
    full_epochs: int = 15
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    num_frames: int = 9
    clip_stride: int = 4
    frame_stride: int = 1
    frame_height: int = 64
    frame_width: int = 64
    image_channels: int = 3
    box_format: str = "xywh"
    hidden_dim: int = 64
    ffn_dim: int = 128
    num_experts: int = 8
    top_k: int = 2
    num_layers: int = 1
    num_heads: int = 4
    patch_grid_size: int = 4
    motion_dim: int = 32
    cache_threshold: float = 0.15
    alpha_aux: float = 0.01
    beta_cfcr: float = 0.1
    cfcr_warmup_steps: int = 500
    lambda_box: float = 5.0
    lambda_giou: float = 2.0
    eval_batches: int = 25
    checkpoint_every_epoch: bool = True
    expert_noise_std: float = 0.0

    @property
    def epochs(self) -> int:
        return self.smoke_epochs if self.smoke else self.full_epochs

    @property
    def dense_routing(self) -> bool:
        return self.stage == "dense"

    @property
    def sparse_routing(self) -> bool:
        return self.stage == "sparse"


DEFAULT_CONFIG = ExperimentConfig()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_config(config: ExperimentConfig) -> TMoEConfig:
    return TMoEConfig(
        hidden_dim=config.hidden_dim,
        ffn_dim=config.ffn_dim,
        num_experts=config.num_experts,
        top_k=config.top_k,
        num_layers=config.num_layers,
        num_attention_heads=config.num_heads,
        patch_grid_size=config.patch_grid_size,
        image_channels=config.image_channels,
        motion_dim=config.motion_dim,
        cache_threshold=config.cache_threshold,
        dense_routing=config.dense_routing,
        use_temporal_cache=config.sparse_routing,
        num_classes=2,
        max_frames=max(config.num_frames, 9),
    )


def _paths(config: ExperimentConfig) -> tuple[AntiUAVDatasetPaths, AntiUAVDatasetPaths]:
    return (
        AntiUAVDatasetPaths(config.train_image_root, config.train_ann_file),
        AntiUAVDatasetPaths(config.val_image_root, config.val_ann_file),
    )


def make_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    train_paths, val_paths = _paths(config)
    collator = AntiUAVDetectionCollator(config.patch_grid_size)
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
            "Full Anti-UAV training requires either --data-root pointing at the extracted "
            "Anti-UAV-RGBT folder, or --train-image-root/--train-ann-file/"
            "--val-image-root/--val-ann-file pointing at COCO-format data from "
            f"{config.dataset_url}."
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


def loss_weights(config: ExperimentConfig) -> TMoELossWeights:
    return TMoELossWeights(
        alpha_aux=config.alpha_aux,
        beta_cfcr=config.beta_cfcr,
        lambda_box=config.lambda_box,
        lambda_giou=config.lambda_giou,
    )


def beta_for_step(config: ExperimentConfig, global_step: int) -> float:
    if not config.sparse_routing:
        return 0.0
    if config.cfcr_warmup_steps <= 0:
        return config.beta_cfcr
    return config.beta_cfcr * min(global_step / config.cfcr_warmup_steps, 1.0)


def compute_losses(
    model: TMoEAntiDroneDetector,
    output: Any,
    batch: dict[str, Any],
    weights: TMoELossWeights,
    beta_cfcr: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    class_targets = batch["class_targets"].to(output.class_logits.device)
    box_targets = batch["box_targets"].to(output.boxes.device)
    box_mask = batch["box_mask"].to(output.boxes.device)
    det_parts = detection_loss(
        output.class_logits,
        output.boxes,
        class_targets,
        box_targets,
        box_mask,
        weights=weights,
    )
    aux = torch.stack([load_balancing_loss(router.probs) for router in output.router_outputs]).mean()
    alignment = semantic_alignment(output.semantic_tokens)
    cfcr = torch.stack(
        [cfcr_loss(router.probs, output.motion_confidence, alignment) for router in output.router_outputs]
    ).mean()
    ortho = torch.stack(
        [expert_lora_similarity(block.moe.experts) for block in model.blocks]
    ).mean()
    total = det_parts["det"] + weights.alpha_aux * aux + beta_cfcr * cfcr
    total = total + weights.gamma_ortho * ortho
    return total, {
        "loss": float(total.detach().cpu()),
        "det": float(det_parts["det"].detach().cpu()),
        "cls": float(det_parts["cls"].detach().cpu()),
        "box_l1": float(det_parts["box_l1"].detach().cpu()),
        "giou": float(det_parts["giou"].detach().cpu()),
        "aux": float(aux.detach().cpu()),
        "cfcr": float(cfcr.detach().cpu()),
        "ortho": float(ortho.detach().cpu()),
        "beta_cfcr": beta_cfcr,
    }


def collect_diagnostics(model: TMoEAntiDroneDetector, output: Any, batch: dict[str, Any]) -> dict[str, Any]:
    cache_total = sum(stats.total_tokens for stats in output.moe_stats)
    cache_hit = sum(stats.cached_tokens for stats in output.moe_stats)
    expert_counts = []
    expert_probs = []
    entropy = []
    for router in output.router_outputs:
        counts = torch.bincount(
            router.topk_indices.reshape(-1),
            minlength=model.config.num_experts,
        ).float()
        expert_counts.append(counts.cpu().numpy())
        expert_probs.append(router.probs.mean(dim=(0, 1, 2)).detach().cpu().numpy())
        entropy.append(float(routing_entropy(router.probs).detach().cpu()))

    predictions = output.class_logits.argmax(dim=-1).detach().cpu()
    targets = batch["class_targets"].detach().cpu()
    positives = targets == 1
    predicted_positives = predictions == 1
    true_positives = (predicted_positives & positives).sum().item()
    return {
        "cache_efficiency": 100.0 * cache_hit / max(cache_total, 1),
        "cached_tokens": cache_hit,
        "executed_tokens": sum(stats.executed_tokens for stats in output.moe_stats),
        "total_tokens": cache_total,
        "routing_entropy": float(np.mean(entropy)),
        "expert_similarity": float(
            np.mean([expert_lora_similarity(block.moe.experts).item() for block in model.blocks])
        ),
        "patch_accuracy": float((predictions == targets).float().mean().item()),
        "positive_recall": float(true_positives / max(positives.sum().item(), 1)),
        "positive_precision": float(true_positives / max(predicted_positives.sum().item(), 1)),
        "expert_counts": np.stack(expert_counts),
        "expert_probs": np.stack(expert_probs),
        "motion_confidence": output.motion_confidence.detach().cpu().numpy(),
        "router_probs": output.router_outputs[0].probs.detach().cpu().numpy(),
    }


def build_optimizer(
    model: TMoEAntiDroneDetector,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def train_model(
    model: TMoEAntiDroneDetector,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    results_dir: Path | None = None,
    global_epoch_offset: int = 0,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    weights = loss_weights(config)
    optimizer = optimizer or build_optimizer(model, config)
    history: list[dict[str, float]] = []
    last_diag: dict[str, Any] = {}
    global_step = 0

    for epoch in range(config.epochs):
        model.train()
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{config.epochs}")
        for step, batch in enumerate(progress, start=1):
            global_step += 1
            frames = batch["frames"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(frames, reset_cache=True)
            beta = beta_for_step(config, global_step)
            loss, metrics = compute_losses(model, output, batch, weights, beta)
            loss.backward()
            if config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            diag = collect_diagnostics(model, output, batch)
            last_diag = diag
            row = {
                "epoch": float(epoch + 1),
                "global_epoch": float(global_epoch_offset + epoch + 1),
                "step": float(step),
                **metrics,
                "cache_efficiency": diag["cache_efficiency"],
                "patch_accuracy": diag["patch_accuracy"],
                "positive_recall": diag["positive_recall"],
                "routing_entropy": diag["routing_entropy"],
                "expert_similarity": diag["expert_similarity"],
            }
            history.append(row)
            progress.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                cache=f"{diag['cache_efficiency']:.1f}%",
                recall=f"{diag['positive_recall']:.2f}",
            )

        if config.checkpoint_every_epoch and results_dir is not None:
            save_checkpoint(
                model,
                results_dir,
                optimizer=optimizer,
                config=config,
                epoch=epoch + 1,
                global_epoch=global_epoch_offset + epoch + 1,
            )

    return history, last_diag


def evaluate_model(
    model: TMoEAntiDroneDetector,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any]]:
    model.eval()
    weights = loss_weights(config)
    metric_rows = []
    last_diag: dict[str, Any] = {}
    max_batches = config.eval_batches if config.smoke else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="evaluate"), start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            frames = batch["frames"].to(device)
            output = model(frames, reset_cache=True)
            _, metrics = compute_losses(model, output, batch, weights, beta_cfcr=0.0)
            diag = collect_diagnostics(model, output, batch)
            last_diag = diag
            metric_rows.append(
                {
                    **metrics,
                    "cache_efficiency": diag["cache_efficiency"],
                    "patch_accuracy": diag["patch_accuracy"],
                    "positive_recall": diag["positive_recall"],
                    "positive_precision": diag["positive_precision"],
                    "routing_entropy": diag["routing_entropy"],
                    "expert_similarity": diag["expert_similarity"],
                }
            )

    if not metric_rows:
        return {}, last_diag
    summary = {
        key: float(np.mean([row[key] for row in metric_rows]))
        for key in metric_rows[0].keys()
    }
    return summary, last_diag


def write_history_csv(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_loss_curves(history: list[dict[str, float]], results_dir: Path) -> None:
    if not history:
        return
    steps = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for key in ["loss", "det", "cls", "box_l1", "giou", "aux", "cfcr"]:
        axes[0].plot(steps, [row[key] for row in history], label=key)
    axes[0].set_title("Training Loss Components")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for key in ["cache_efficiency", "patch_accuracy", "positive_recall", "routing_entropy"]:
        axes[1].plot(steps, [row[key] for row in history], label=key)
    axes[1].set_title("Detection, Routing, And Cache Dynamics")
    axes[1].set_xlabel("Step")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(results_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def plot_heatmap(data: np.ndarray, title: str, path: Path, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_visualizations(
    history: list[dict[str, float]],
    eval_summary: dict[str, float],
    train_diag: dict[str, Any],
    eval_diag: dict[str, Any],
    results_dir: Path,
) -> None:
    plot_loss_curves(history, results_dir)

    if eval_summary:
        metrics = ["loss", "patch_accuracy", "positive_recall", "cache_efficiency"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(metrics, [eval_summary[key] for key in metrics], color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
        ax.set_title("Anti-UAV Evaluation Summary")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(results_dir / "evaluation_summary.png", dpi=180)
        plt.close(fig)

    if train_diag:
        plot_heatmap(
            train_diag["expert_counts"],
            "Top-k Expert Activation Counts By Layer",
            results_dir / "expert_activation_heatmap.png",
            "Expert",
            "Layer",
        )
        plot_heatmap(
            train_diag["expert_probs"],
            "Mean Routing Probability By Layer",
            results_dir / "routing_probability_heatmap.png",
            "Expert",
            "Layer",
        )

    diag = eval_diag or train_diag
    if diag:
        motion = diag["motion_confidence"][0]
        plot_heatmap(
            motion,
            "Motion Confidence By Frame And Patch",
            results_dir / "motion_confidence_heatmap.png",
            "Patch",
            "Frame",
        )
        router_probs = diag["router_probs"][0]
        flattened = router_probs.reshape(-1, router_probs.shape[-1])
        plot_heatmap(
            flattened,
            "Token-Level Routing Distribution",
            results_dir / "token_routing_heatmap.png",
            "Expert",
            "Frame x Patch Token",
        )


def save_checkpoint(
    model: TMoEAntiDroneDetector,
    results_dir: Path,
    optimizer: torch.optim.Optimizer | None = None,
    config: ExperimentConfig | None = None,
    epoch: int | None = None,
    global_epoch: int | None = None,
    name: str | None = None,
) -> Path:
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if name is None:
        checkpoint_epoch = global_epoch if global_epoch is not None else epoch
        name = f"epoch_{checkpoint_epoch:03d}.pt" if checkpoint_epoch is not None else "tmoe_antiuav.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "experiment_config": asdict(config) if config is not None else None,
        "epoch": epoch,
        "global_epoch": global_epoch,
    }
    path = checkpoint_dir / name
    torch.save(checkpoint, path)
    torch.save(checkpoint, checkpoint_dir / "latest.pt")
    return path


def load_checkpoint_if_requested(
    model: TMoEAntiDroneDetector,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | None,
    device: torch.device,
) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", dest="smoke", action="store_true", help="Run a synthetic smoke experiment")
    parser.add_argument("--full", dest="smoke", action="store_false", help="Run with real Anti-UAV data")
    parser.set_defaults(smoke=DEFAULT_CONFIG.smoke)
    parser.add_argument("--dataset-url", type=str, default=DEFAULT_CONFIG.dataset_url)
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_CONFIG.data_root,
        help="Extracted Anti-UAV-RGBT root, e.g. /content/drive/MyDrive/Anti-UAV-RGBT (Unzipped Files)",
    )
    parser.add_argument("--train-split", type=str, default=DEFAULT_CONFIG.train_split)
    parser.add_argument("--val-split", type=str, default=DEFAULT_CONFIG.val_split)
    parser.add_argument("--modality", choices=["infrared", "visible"], default=DEFAULT_CONFIG.modality)
    parser.add_argument("--train-image-root", type=str, default=DEFAULT_CONFIG.train_image_root)
    parser.add_argument("--train-ann-file", type=str, default=DEFAULT_CONFIG.train_ann_file)
    parser.add_argument("--val-image-root", type=str, default=DEFAULT_CONFIG.val_image_root)
    parser.add_argument("--val-ann-file", type=str, default=DEFAULT_CONFIG.val_ann_file)
    parser.add_argument("--results-dir", type=str, default=DEFAULT_CONFIG.results_dir)
    parser.add_argument("--stage", choices=["dense", "sparse"], default=DEFAULT_CONFIG.stage)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override smoke/full epoch count")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_CONFIG.num_frames)
    parser.add_argument("--clip-stride", type=int, default=DEFAULT_CONFIG.clip_stride)
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_CONFIG.frame_stride)
    parser.add_argument("--height", type=int, default=DEFAULT_CONFIG.frame_height)
    parser.add_argument("--width", type=int, default=DEFAULT_CONFIG.frame_width)
    parser.add_argument("--image-channels", type=int, choices=[1, 3], default=None)
    parser.add_argument(
        "--ir-repeat-rgb",
        action="store_true",
        help="Repeat IR grayscale frames to 3 channels instead of training a 1-channel stem",
    )
    parser.add_argument("--box-format", choices=["xywh", "xyxy"], default=DEFAULT_CONFIG.box_format)
    parser.add_argument("--patch-grid-size", type=int, default=DEFAULT_CONFIG.patch_grid_size)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG.hidden_dim)
    parser.add_argument("--ffn-dim", type=int, default=DEFAULT_CONFIG.ffn_dim)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
        patch_grid_size=args.patch_grid_size,
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
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
        "Running T-MoE Anti-UAV detector "
        f"mode={'smoke' if config.smoke else 'full'} "
        f"stage={config.stage} epochs={config.epochs} "
        f"source={config.data_root or config.dataset_url} "
        f"modality={config.modality} channels={config.image_channels}"
    )

    train_loader, val_loader, sizes = make_dataloaders(config)
    model = TMoEAntiDroneDetector(build_model_config(config)).to(device)
    if config.expert_noise_std > 0:
        model.add_expert_noise(config.expert_noise_std)
    optimizer = build_optimizer(model, config)
    checkpoint = load_checkpoint_if_requested(model, optimizer, args.resume_checkpoint, device)
    global_epoch_offset = int((checkpoint or {}).get("global_epoch") or 0)

    history, train_diag = train_model(
        model,
        train_loader,
        config,
        device,
        optimizer=optimizer,
        results_dir=results_dir,
        global_epoch_offset=global_epoch_offset,
    )
    eval_summary, eval_diag = evaluate_model(model, val_loader, config, device)

    write_history_csv(history, results_dir / "train_history.csv")
    (results_dir / "config.json").write_text(
        json.dumps({**asdict(config), "dataset_sizes": sizes}, indent=2),
        encoding="utf-8",
    )
    (results_dir / "eval_summary.json").write_text(
        json.dumps(eval_summary, indent=2),
        encoding="utf-8",
    )
    generate_visualizations(history, eval_summary, train_diag, eval_diag, results_dir)
    save_checkpoint(
        model,
        results_dir,
        optimizer=optimizer,
        config=config,
        epoch=config.epochs,
        global_epoch=global_epoch_offset + config.epochs,
        name="tmoe_antiuav_final.pt",
    )

    print("Experiment complete.")
    print(f"Results written to: {results_dir.resolve()}")
    print(json.dumps(eval_summary, indent=2))


if __name__ == "__main__":
    main()
