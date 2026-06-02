"""Unified ActivityNetQA experiment runner for T-MoE-LLaVA Micro-MoE."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import TMoEConfig, TMoELLaVAMicro
from train import (
    ActivityNetQACollator,
    ActivityNetQADataset,
    ActivityNetQARecord,
    SimpleQATokenizer,
    TMoELossWeights,
    autoregressive_loss,
    cfcr_loss,
    expert_lora_similarity,
    filter_records_with_available_videos,
    load_activitynetqa_records,
    load_balancing_loss,
    orthogonalization_loss,
    routing_entropy,
    split_records,
)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Set ``smoke=True`` to run one complete epoch on 5% of ActivityNetQA. Set
    ``smoke=False`` for the full configured run.
    """

    smoke: bool = False
    dataset_name: str = "lmms-lab/ActivityNetQA"
    dataset_split: str = "test"
    hf_token_env: str = "HF_TOKEN"
    video_root: str | None = None
    video_shards: tuple[int, ...] = ()
    shard_cache_dir: str = "hf_cache/activitynetqa"
    shard_extract_dir: str = "hf_cache/activitynetqa/extracted"
    keep_shard_zip: bool = False
    cleanup_extracted_shards: bool = False
    require_real_videos: bool = False
    resume_checkpoint: str | None = None
    results_dir: str = "results"
    seed: int = 42
    train_fraction: float = 0.8
    smoke_data_fraction: float = 0.05
    full_data_fraction: float = 1.0
    smoke_epochs: int = 1
    full_epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    num_frames: int = 4
    frame_height: int = 32
    frame_width: int = 32
    max_text_length: int = 64
    max_vocab_size: int = 4096
    hidden_dim: int = 64
    ffn_dim: int = 128
    num_experts: int = 8
    top_k: int = 2
    num_layers: int = 2
    num_heads: int = 4
    patch_grid_size: int = 2
    motion_dim: int = 32
    router_history_window: int = 2
    cache_threshold: float = 0.05
    lora_rank: int = 4
    lora_alpha: float = 8.0
    alpha_aux: float = 0.01
    beta_cfcr: float = 0.1
    gamma_ortho: float = 0.01
    eval_batches: int = 25
    checkpoint_every_epoch: bool = True

    @property
    def epochs(self) -> int:
        return self.smoke_epochs if self.smoke else self.full_epochs

    @property
    def data_fraction(self) -> float:
        return self.smoke_data_fraction if self.smoke else self.full_data_fraction


DEFAULT_CONFIG = ExperimentConfig(smoke=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_config(config: ExperimentConfig, vocab_size: int) -> TMoEConfig:
    return TMoEConfig(
        vocab_size=vocab_size,
        hidden_dim=config.hidden_dim,
        ffn_dim=config.ffn_dim,
        num_experts=config.num_experts,
        top_k=config.top_k,
        num_layers=config.num_layers,
        num_attention_heads=config.num_heads,
        patch_grid_size=config.patch_grid_size,
        motion_dim=config.motion_dim,
        router_history_window=config.router_history_window,
        cache_threshold=config.cache_threshold,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        max_text_length=config.max_text_length,
    )


def prepare_data_records(
    config: ExperimentConfig,
) -> tuple[
    list[ActivityNetQARecord],
    list[ActivityNetQARecord],
    SimpleQATokenizer,
    dict[str, int],
]:
    records = load_activitynetqa_records(
        dataset_name=config.dataset_name,
        split=config.dataset_split,
        hf_token_env=config.hf_token_env,
        limit_fraction=config.data_fraction,
        seed=config.seed,
    )
    train_records, test_records = split_records(
        records,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )
    tokenizer = SimpleQATokenizer.fit(train_records, max_vocab_size=config.max_vocab_size)
    sizes = {"records": len(records), "train": len(train_records), "test": len(test_records)}
    return train_records, test_records, tokenizer, sizes


def make_dataloaders_from_records(
    config: ExperimentConfig,
    train_records: list[ActivityNetQARecord],
    test_records: list[ActivityNetQARecord],
    tokenizer: SimpleQATokenizer,
    video_root: str | Path | None = None,
    require_real_videos: bool = False,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    active_video_root = str(video_root) if video_root is not None else config.video_root
    if require_real_videos:
        if active_video_root is None:
            raise ValueError("require_real_videos=True requires a video_root")
        train_records = filter_records_with_available_videos(train_records, active_video_root)
        test_records = filter_records_with_available_videos(test_records, active_video_root)
        if not train_records:
            raise RuntimeError(f"No training videos found under {active_video_root}")

    collator = ActivityNetQACollator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = ActivityNetQADataset(
        train_records,
        tokenizer,
        num_frames=config.num_frames,
        height=config.frame_height,
        width=config.frame_width,
        max_text_length=config.max_text_length,
        video_root=active_video_root,
        allow_proxy_videos=not require_real_videos,
    )
    test_dataset = ActivityNetQADataset(
        test_records,
        tokenizer,
        num_frames=config.num_frames,
        height=config.frame_height,
        width=config.frame_width,
        max_text_length=config.max_text_length,
        video_root=active_video_root,
        allow_proxy_videos=not require_real_videos,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        drop_last=False,
    )
    sizes = {
        "train": len(train_dataset),
        "test": len(test_dataset),
        "videos_required": int(require_real_videos),
    }
    return train_loader, test_loader, sizes


def make_dataloaders(
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader, SimpleQATokenizer, dict[str, int]]:
    train_records, test_records, tokenizer, metadata_sizes = prepare_data_records(config)
    train_loader, test_loader, loader_sizes = make_dataloaders_from_records(
        config,
        train_records,
        test_records,
        tokenizer,
        video_root=config.video_root,
        require_real_videos=config.require_real_videos,
    )
    return train_loader, test_loader, tokenizer, {**metadata_sizes, **loader_sizes}


def parse_video_shards(values: list[str] | None, all_shards: bool) -> tuple[int, ...]:
    """Parse CLI shard selections like ``1 2 3`` or ``1-4,7``."""

    if all_shards:
        return tuple(range(1, 29))
    if not values:
        return ()
    shards: set[int] = set()
    for value in values:
        for part in value.replace(",", " ").split():
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                if start > end:
                    raise ValueError(f"Invalid shard range: {part}")
                shards.update(range(start, end + 1))
            else:
                shards.add(int(part))
    invalid = [shard for shard in shards if shard < 1 or shard > 28]
    if invalid:
        raise ValueError(f"ActivityNetQA shard ids must be in 1..28, got {invalid}")
    return tuple(sorted(shards))


def load_hf_token(config: ExperimentConfig) -> str | None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()
    return os.getenv(config.hf_token_env) or os.getenv("HUGGINGFACE_HUB_TOKEN")


def download_video_shard(config: ExperimentConfig, shard_id: int) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install huggingface_hub before downloading video shards.") from exc

    filename = f"videos_chunked_{shard_id:02d}.zip"
    cache_dir = Path(config.shard_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=config.dataset_name,
            repo_type="dataset",
            filename=filename,
            token=load_hf_token(config),
            local_dir=str(cache_dir),
        )
    )


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination_root not in [target, *target.parents]:
                raise RuntimeError(f"Refusing to extract unsafe zip member: {member.filename}")
        archive.extractall(destination)


def prepare_video_shard(config: ExperimentConfig, shard_id: int) -> Path:
    shard_name = f"videos_chunked_{shard_id:02d}"
    shard_dir = Path(config.shard_extract_dir) / shard_name
    marker = shard_dir / ".extracted"
    if marker.exists():
        return shard_dir

    zip_path = download_video_shard(config, shard_id)
    safe_extract_zip(zip_path, shard_dir)
    marker.write_text(zip_path.name, encoding="utf-8")
    if not config.keep_shard_zip:
        try:
            zip_path.unlink()
        except OSError:
            pass
    return shard_dir


def compute_losses(
    model: TMoELLaVAMicro,
    output: Any,
    labels: torch.Tensor,
    weights: TMoELossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    ar = autoregressive_loss(output.logits, labels)
    aux = torch.stack([load_balancing_loss(router.probs) for router in output.router_outputs]).mean()
    cfcr = torch.stack(
        [cfcr_loss(router.probs, output.motion_confidence) for router in output.router_outputs]
    ).mean()
    ortho = torch.stack(
        [orthogonalization_loss(block.moe.experts) for block in model.blocks]
    ).mean()
    total = ar + weights.alpha_aux * aux + weights.beta_cfcr * cfcr
    total = total + weights.gamma_ortho * ortho
    return total, {
        "loss": float(total.detach().cpu()),
        "ar": float(ar.detach().cpu()),
        "aux": float(aux.detach().cpu()),
        "cfcr": float(cfcr.detach().cpu()),
        "ortho": float(ortho.detach().cpu()),
    }


def collect_diagnostics(model: TMoELLaVAMicro, output: Any) -> dict[str, Any]:
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

    return {
        "cache_efficiency": 100.0 * cache_hit / max(cache_total, 1),
        "cached_tokens": cache_hit,
        "executed_tokens": sum(stats.executed_tokens for stats in output.moe_stats),
        "total_tokens": cache_total,
        "routing_entropy": float(np.mean(entropy)),
        "expert_similarity": float(
            np.mean([expert_lora_similarity(block.moe.experts).item() for block in model.blocks])
        ),
        "expert_counts": np.stack(expert_counts),
        "expert_probs": np.stack(expert_probs),
        "motion_confidence": output.motion_confidence.detach().cpu().numpy(),
        "router_probs": output.router_outputs[0].probs.detach().cpu().numpy(),
    }


def build_optimizer(
    model: TMoELLaVAMicro,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def train_model(
    model: TMoELLaVAMicro,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    tokenizer: SimpleQATokenizer | None = None,
    results_dir: Path | None = None,
    shard_id: int | None = None,
    global_epoch_offset: int = 0,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    weights = TMoELossWeights(
        alpha_aux=config.alpha_aux,
        beta_cfcr=config.beta_cfcr,
        gamma_ortho=config.gamma_ortho,
    )
    optimizer = optimizer or build_optimizer(model, config)
    history: list[dict[str, float]] = []
    last_diag: dict[str, Any] = {}

    for epoch in range(config.epochs):
        model.train()
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{config.epochs}")
        for step, batch in enumerate(progress, start=1):
            frames = batch["frames"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(frames, input_ids, reset_cache=True)
            loss, metrics = compute_losses(model, output, labels, weights)
            loss.backward()
            if config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            diag = collect_diagnostics(model, output)
            last_diag = diag
            row = {
                "epoch": float(epoch + 1),
                "global_epoch": float(global_epoch_offset + epoch + 1),
                "step": float(step),
                **metrics,
                "cache_efficiency": diag["cache_efficiency"],
                "routing_entropy": diag["routing_entropy"],
                "expert_similarity": diag["expert_similarity"],
            }
            if shard_id is not None:
                row["shard"] = float(shard_id)
            history.append(row)
            progress.set_postfix(loss=f"{metrics['loss']:.3f}", cache=f"{diag['cache_efficiency']:.1f}%")

        if config.checkpoint_every_epoch and results_dir is not None:
            save_checkpoint(
                model,
                results_dir,
                optimizer=optimizer,
                config=config,
                tokenizer=tokenizer,
                epoch=epoch + 1,
                global_epoch=global_epoch_offset + epoch + 1,
                shard_id=shard_id,
            )

    return history, last_diag


def greedy_answer_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: SimpleQATokenizer,
) -> tuple[list[str], float]:
    shifted_predictions = logits[:, :-1].argmax(dim=-1)
    shifted_labels = labels[:, 1:]
    predictions: list[str] = []
    total = 0
    correct = 0
    for row_pred, row_label in zip(shifted_predictions, shifted_labels):
        mask = row_label != -100
        selected = row_pred[mask]
        target = row_label[mask]
        total += int(mask.sum().item())
        if selected.numel() > 0:
            correct += int((selected == target).sum().item())
        predictions.append(tokenizer.decode(selected))
    return predictions, correct / max(total, 1)


def evaluate_model(
    model: TMoELLaVAMicro,
    loader: DataLoader,
    tokenizer: SimpleQATokenizer,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    losses = []
    token_accuracies = []
    exact_matches = []
    samples: list[dict[str, Any]] = []
    last_diag: dict[str, Any] = {}
    max_batches = config.eval_batches if config.smoke else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="evaluate"), start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            frames = batch["frames"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            output = model(frames, input_ids, reset_cache=True)
            ar = autoregressive_loss(output.logits, labels)
            predictions, token_accuracy = greedy_answer_predictions(
                output.logits,
                labels,
                tokenizer,
            )
            diag = collect_diagnostics(model, output)
            last_diag = diag

            losses.append(float(ar.cpu()))
            token_accuracies.append(token_accuracy)
            for question, answer, pred, q_type, video in zip(
                batch["questions"],
                batch["answers"],
                predictions,
                batch["question_types"],
                batch["video_names"],
            ):
                exact = answer.strip().lower() == pred.strip().lower()
                exact_matches.append(float(exact))
                if len(samples) < 50:
                    samples.append(
                        {
                            "video_name": video,
                            "question_type": q_type,
                            "question": question,
                            "answer": answer,
                            "prediction": pred,
                            "exact_match": exact,
                        }
                    )

    summary = {
        "eval_ar_loss": float(np.mean(losses)) if losses else 0.0,
        "token_accuracy": float(np.mean(token_accuracies)) if token_accuracies else 0.0,
        "exact_match": float(np.mean(exact_matches)) if exact_matches else 0.0,
        "cache_efficiency": float(last_diag.get("cache_efficiency", 0.0)),
        "routing_entropy": float(last_diag.get("routing_entropy", 0.0)),
        "expert_similarity": float(last_diag.get("expert_similarity", 0.0)),
    }
    return summary, samples, last_diag


def write_history_csv(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_loss_curves(history: list[dict[str, float]], results_dir: Path) -> None:
    steps = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for key in ["loss", "ar", "aux", "cfcr", "ortho"]:
        axes[0].plot(steps, [row[key] for row in history], label=key)
    axes[0].set_title("Training Loss Components")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for key in ["cache_efficiency", "routing_entropy", "expert_similarity"]:
        axes[1].plot(steps, [row[key] for row in history], label=key)
    axes[1].set_title("Routing And Cache Dynamics")
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

    metrics = ["eval_ar_loss", "token_accuracy", "exact_match", "cache_efficiency"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(metrics, [eval_summary[key] for key in metrics], color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    ax.set_title("Evaluation Summary")
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
    model: TMoELLaVAMicro,
    results_dir: Path,
    optimizer: torch.optim.Optimizer | None = None,
    config: ExperimentConfig | None = None,
    tokenizer: SimpleQATokenizer | None = None,
    epoch: int | None = None,
    global_epoch: int | None = None,
    shard_id: int | None = None,
    name: str | None = None,
) -> Path:
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if name is None:
        if shard_id is None:
            checkpoint_epoch = global_epoch if global_epoch is not None else epoch
            name = f"epoch_{checkpoint_epoch:03d}.pt" if checkpoint_epoch is not None else "tmoe_micro.pt"
        else:
            name = f"shard_{shard_id:02d}_epoch_{epoch:03d}.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "experiment_config": asdict(config) if config is not None else None,
        "tokenizer_vocab": tokenizer.vocab if tokenizer is not None else None,
        "epoch": epoch,
        "global_epoch": global_epoch,
        "shard_id": shard_id,
    }
    path = checkpoint_dir / name
    torch.save(checkpoint, path)
    torch.save(checkpoint, checkpoint_dir / "latest.pt")
    return path


def load_checkpoint_if_requested(
    model: TMoELLaVAMicro,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any] | None:
    if config.resume_checkpoint is None:
        return None
    checkpoint = torch.load(config.resume_checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return checkpoint


def default_eval_summary() -> dict[str, float]:
    return {
        "eval_ar_loss": 0.0,
        "token_accuracy": 0.0,
        "exact_match": 0.0,
        "cache_efficiency": 0.0,
        "routing_entropy": 0.0,
        "expert_similarity": 0.0,
    }


def run_sharded_experiment(
    config: ExperimentConfig,
    device: torch.device,
    results_dir: Path,
) -> None:
    train_records, test_records, tokenizer, metadata_sizes = prepare_data_records(config)
    model = TMoELLaVAMicro(build_model_config(config, tokenizer.vocab_size)).to(device)
    optimizer = build_optimizer(model, config)
    checkpoint = load_checkpoint_if_requested(model, optimizer, config, device)
    global_epoch_offset = int((checkpoint or {}).get("global_epoch") or 0)

    all_history: list[dict[str, float]] = []
    shard_sizes: list[dict[str, Any]] = []
    train_diag: dict[str, Any] = {}
    eval_diag: dict[str, Any] = {}
    eval_summary: dict[str, float] = default_eval_summary()
    samples: list[dict[str, Any]] = []

    for shard_id in config.video_shards:
        print(f"Preparing ActivityNetQA video shard {shard_id:02d}...")
        shard_root = prepare_video_shard(config, shard_id)
        try:
            train_loader, test_loader, sizes = make_dataloaders_from_records(
                config,
                train_records,
                test_records,
                tokenizer,
                video_root=shard_root,
                require_real_videos=True,
            )
        except RuntimeError as exc:
            print(f"Skipping shard {shard_id:02d}: {exc}")
            if config.cleanup_extracted_shards:
                shutil.rmtree(shard_root, ignore_errors=True)
            continue

        sizes = {**sizes, "shard": shard_id, "video_root": str(shard_root)}
        shard_sizes.append(sizes)
        print(
            f"Training shard {shard_id:02d}: "
            f"train={sizes['train']} test={sizes['test']} epochs={config.epochs}"
        )
        history, train_diag = train_model(
            model,
            train_loader,
            config,
            device,
            optimizer=optimizer,
            tokenizer=tokenizer,
            results_dir=results_dir,
            shard_id=shard_id,
            global_epoch_offset=global_epoch_offset,
        )
        global_epoch_offset += config.epochs
        all_history.extend(history)
        save_checkpoint(
            model,
            results_dir,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            epoch=config.epochs,
            global_epoch=global_epoch_offset,
            shard_id=shard_id,
            name=f"after_shard_{shard_id:02d}.pt",
        )

        if len(test_loader.dataset) > 0:
            eval_summary, samples, eval_diag = evaluate_model(
                model,
                test_loader,
                tokenizer,
                config,
                device,
            )

        if config.cleanup_extracted_shards:
            shutil.rmtree(shard_root, ignore_errors=True)

    if not all_history:
        raise RuntimeError("No shard produced trainable examples. Check the shard ids and video filenames.")

    write_history_csv(all_history, results_dir / "train_history.csv")
    (results_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "dataset_sizes": metadata_sizes,
                "shard_sizes": shard_sizes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (results_dir / "eval_summary.json").write_text(
        json.dumps(eval_summary, indent=2),
        encoding="utf-8",
    )
    (results_dir / "sample_predictions.json").write_text(
        json.dumps(samples, indent=2),
        encoding="utf-8",
    )
    generate_visualizations(all_history, eval_summary, train_diag, eval_diag, results_dir)
    save_checkpoint(
        model,
        results_dir,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        epoch=config.epochs,
        global_epoch=global_epoch_offset,
        name="tmoe_micro_final.pt",
    )

    print("Sharded experiment complete.")
    print(f"Results written to: {results_dir.resolve()}")
    print(json.dumps(eval_summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", dest="smoke", action="store_true", help="Run 1 epoch on 5%% of ActivityNetQA")
    parser.add_argument("--full", dest="smoke", action="store_false", help="Run the full configured experiment")
    parser.set_defaults(smoke=DEFAULT_CONFIG.smoke)
    parser.add_argument("--results-dir", type=str, default=DEFAULT_CONFIG.results_dir)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_CONFIG.dataset_name)
    parser.add_argument("--dataset-split", type=str, default=DEFAULT_CONFIG.dataset_split)
    parser.add_argument("--video-root", type=str, default=DEFAULT_CONFIG.video_root)
    parser.add_argument(
        "--require-real-videos",
        action="store_true",
        help="Fail instead of using proxy frames when --video-root clips are missing or unreadable",
    )
    parser.add_argument(
        "--video-shards",
        nargs="*",
        default=None,
        help="Download/extract/train ActivityNetQA video shards, e.g. --video-shards 1 2 3 or 1-4,7",
    )
    parser.add_argument(
        "--all-video-shards",
        action="store_true",
        help="Train through all ActivityNetQA video shards, 1 through 28",
    )
    parser.add_argument("--shard-cache-dir", type=str, default=DEFAULT_CONFIG.shard_cache_dir)
    parser.add_argument("--shard-extract-dir", type=str, default=DEFAULT_CONFIG.shard_extract_dir)
    parser.add_argument("--keep-shard-zip", action="store_true")
    parser.add_argument(
        "--cleanup-extracted-shards",
        action="store_true",
        help="Delete each extracted shard folder after it finishes training",
    )
    parser.add_argument("--resume-checkpoint", type=str, default=DEFAULT_CONFIG.resume_checkpoint)
    parser.add_argument("--epochs", type=int, default=None, help="Override smoke/full epoch count")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig(
        smoke=args.smoke,
        results_dir=args.results_dir,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        video_root=args.video_root,
        video_shards=parse_video_shards(args.video_shards, args.all_video_shards),
        shard_cache_dir=args.shard_cache_dir,
        shard_extract_dir=args.shard_extract_dir,
        keep_shard_zip=args.keep_shard_zip,
        cleanup_extracted_shards=args.cleanup_extracted_shards,
        require_real_videos=args.require_real_videos,
        resume_checkpoint=args.resume_checkpoint,
        batch_size=args.batch_size,
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
        "Running T-MoE-LLaVA ActivityNetQA experiment "
        f"mode={'smoke' if config.smoke else 'full'} "
        f"epochs={config.epochs} data_fraction={config.data_fraction:.2f}"
    )
    if config.video_shards:
        print(f"Video shard mode enabled: {list(config.video_shards)}")
        run_sharded_experiment(config, device, results_dir)
        return

    train_loader, test_loader, tokenizer, sizes = make_dataloaders(config)
    model = TMoELLaVAMicro(build_model_config(config, tokenizer.vocab_size)).to(device)
    optimizer = build_optimizer(model, config)
    checkpoint = load_checkpoint_if_requested(model, optimizer, config, device)
    global_epoch_offset = int((checkpoint or {}).get("global_epoch") or 0)

    history, train_diag = train_model(
        model,
        train_loader,
        config,
        device,
        optimizer=optimizer,
        tokenizer=tokenizer,
        results_dir=results_dir,
        global_epoch_offset=global_epoch_offset,
    )
    eval_summary, samples, eval_diag = evaluate_model(
        model,
        test_loader,
        tokenizer,
        config,
        device,
    )

    write_history_csv(history, results_dir / "train_history.csv")
    (results_dir / "config.json").write_text(
        json.dumps({**asdict(config), "dataset_sizes": sizes}, indent=2),
        encoding="utf-8",
    )
    (results_dir / "eval_summary.json").write_text(
        json.dumps(eval_summary, indent=2),
        encoding="utf-8",
    )
    (results_dir / "sample_predictions.json").write_text(
        json.dumps(samples, indent=2),
        encoding="utf-8",
    )
    generate_visualizations(history, eval_summary, train_diag, eval_diag, results_dir)
    save_checkpoint(
        model,
        results_dir,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        epoch=config.epochs,
        global_epoch=global_epoch_offset + config.epochs,
        name="tmoe_micro_final.pt",
    )

    print("Experiment complete.")
    print(f"Results written to: {results_dir.resolve()}")
    print(json.dumps(eval_summary, indent=2))


if __name__ == "__main__":
    main()
