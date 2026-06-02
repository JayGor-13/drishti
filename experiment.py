"""Unified ActivityNetQA experiment runner for T-MoE-LLaVA Micro-MoE."""

from __future__ import annotations

import argparse
import csv
import json
import os
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
    SimpleQATokenizer,
    TMoELossWeights,
    autoregressive_loss,
    cfcr_loss,
    expert_lora_similarity,
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

    smoke: bool = True
    dataset_name: str = "lmms-lab/ActivityNetQA"
    dataset_split: str = "test"
    hf_token_env: str = "HF_TOKEN"
    video_root: str | None = None
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


def make_dataloaders(
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader, SimpleQATokenizer, dict[str, int]]:
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
    collator = ActivityNetQACollator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = ActivityNetQADataset(
        train_records,
        tokenizer,
        num_frames=config.num_frames,
        height=config.frame_height,
        width=config.frame_width,
        max_text_length=config.max_text_length,
        video_root=config.video_root,
    )
    test_dataset = ActivityNetQADataset(
        test_records,
        tokenizer,
        num_frames=config.num_frames,
        height=config.frame_height,
        width=config.frame_width,
        max_text_length=config.max_text_length,
        video_root=config.video_root,
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
    sizes = {"records": len(records), "train": len(train_dataset), "test": len(test_dataset)}
    return train_loader, test_loader, tokenizer, sizes


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


def train_model(
    model: TMoELLaVAMicro,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    weights = TMoELossWeights(
        alpha_aux=config.alpha_aux,
        beta_cfcr=config.beta_cfcr,
        gamma_ortho=config.gamma_ortho,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
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
                "step": float(step),
                **metrics,
                "cache_efficiency": diag["cache_efficiency"],
                "routing_entropy": diag["routing_entropy"],
                "expert_similarity": diag["expert_similarity"],
            }
            history.append(row)
            progress.set_postfix(loss=f"{metrics['loss']:.3f}", cache=f"{diag['cache_efficiency']:.1f}%")

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


def save_checkpoint(model: TMoELLaVAMicro, results_dir: Path) -> None:
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "tmoe_micro.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", dest="smoke", action="store_true", help="Run 1 epoch on 5% of ActivityNetQA")
    parser.add_argument("--full", dest="smoke", action="store_false", help="Run the full configured experiment")
    parser.set_defaults(smoke=DEFAULT_CONFIG.smoke)
    parser.add_argument("--results-dir", type=str, default=DEFAULT_CONFIG.results_dir)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_CONFIG.dataset_name)
    parser.add_argument("--dataset-split", type=str, default=DEFAULT_CONFIG.dataset_split)
    parser.add_argument("--video-root", type=str, default=DEFAULT_CONFIG.video_root)
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
    train_loader, test_loader, tokenizer, sizes = make_dataloaders(config)
    model = TMoELLaVAMicro(build_model_config(config, tokenizer.vocab_size)).to(device)

    history, train_diag = train_model(model, train_loader, config, device)
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
    save_checkpoint(model, results_dir)

    print("Experiment complete.")
    print(f"Results written to: {results_dir.resolve()}")
    print(json.dumps(eval_summary, indent=2))


if __name__ == "__main__":
    main()
