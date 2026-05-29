"""Run synthetic ablations for the Micro-MoE research scaffold."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from models import TMoEConfig, TMoELLaVAMicro
from train import (
    TMoELossWeights,
    TMoETrainer,
    TrainingConfig,
    expert_lora_similarity,
    routing_entropy,
)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    loss_weights: TMoELossWeights
    config_overrides: dict[str, object]


def default_loss_weights() -> TMoELossWeights:
    return TMoELossWeights(alpha_aux=0.01, beta_cfcr=0.1, gamma_ortho=0.01)


def build_specs() -> list[AblationSpec]:
    return [
        AblationSpec("Full (Control)", default_loss_weights(), {}),
        AblationSpec(
            "No Orthogonalization",
            TMoELossWeights(alpha_aux=0.01, beta_cfcr=0.1, gamma_ortho=0.0),
            {},
        ),
        AblationSpec(
            "No CFCR",
            TMoELossWeights(alpha_aux=0.01, beta_cfcr=0.0, gamma_ortho=0.01),
            {},
        ),
        AblationSpec(
            "No Router History",
            default_loss_weights(),
            {"router_history_window": 0},
        ),
        AblationSpec(
            "No Motion Routing",
            default_loss_weights(),
            {"use_motion_conditioning": False},
        ),
        AblationSpec("No Cache", default_loss_weights(), {"cache_threshold": 0.0}),
        AblationSpec(
            "Aggressive Cache",
            default_loss_weights(),
            {"cache_threshold": 0.1},
        ),
    ]


def build_config(args: argparse.Namespace, spec: AblationSpec) -> TMoEConfig:
    config = TMoEConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        patch_grid_size=args.patch_grid_size,
        motion_dim=args.motion_dim,
        router_history_window=args.router_history_window,
        cache_threshold=args.cache_threshold,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_text_length=args.text_length,
    )
    for name, value in spec.config_overrides.items():
        setattr(config, name, value)
    return config


def make_synthetic_batch(
    config: TMoEConfig,
    args: argparse.Namespace,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    frames = torch.randn(
        args.batch_size,
        args.frames,
        config.image_channels,
        args.height,
        args.width,
        generator=generator,
    )
    if args.frames > 1:
        jitter = 0.02 * torch.randn(
            args.batch_size,
            args.frames - 1,
            config.image_channels,
            args.height,
            args.width,
            generator=generator,
        )
        frames[:, 1:] = frames[:, :1] + jitter

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (args.batch_size, args.text_length),
        generator=generator,
    )
    labels = input_ids.clone()
    return frames, input_ids, labels


def average(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def run_ablation(spec: AblationSpec, args: argparse.Namespace) -> dict[str, float | str]:
    torch.manual_seed(args.seed)
    config = build_config(args, spec)
    model = TMoELLaVAMicro(config)
    trainer = TMoETrainer(
        model,
        TrainingConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            loss_weights=spec.loss_weights,
        ),
    )

    history: dict[str, list[float]] = {
        "loss": [],
        "ar": [],
        "aux": [],
        "cfcr": [],
        "ortho": [],
    }
    for step in range(args.steps):
        frames, input_ids, labels = make_synthetic_batch(
            config,
            args,
            seed=args.seed + 1000 + step,
        )
        metrics = trainer.train_step(frames, input_ids, labels)
        for name in history:
            history[name].append(metrics[name])

    frames, input_ids, _ = make_synthetic_batch(config, args, seed=args.seed + 9000)
    model.eval()
    with torch.no_grad():
        output = model(frames, input_ids, reset_cache=True)

    total_tokens = sum(stats.total_tokens for stats in output.moe_stats)
    cached_tokens = sum(stats.cached_tokens for stats in output.moe_stats)
    cache_efficiency = 100.0 * cached_tokens / max(total_tokens, 1)
    entropy = average(
        [routing_entropy(router.probs).item() for router in output.router_outputs]
    )
    cosine_sim = average(
        [expert_lora_similarity(block.moe.experts).item() for block in model.blocks]
    )

    return {
        "Configuration": spec.name,
        "Total Loss": average(history["loss"]),
        "AR Loss": average(history["ar"]),
        "Aux Loss": average(history["aux"]),
        "CFCR Loss": average(history["cfcr"]),
        "Ortho Loss": average(history["ortho"]),
        "Caching Efficiency (%)": cache_efficiency,
        "Routing Entropy": entropy,
        "Cosine Sim": cosine_sim,
    }


def print_markdown_table(rows: list[dict[str, float | str]]) -> None:
    headers = [
        "Configuration",
        "Total Loss",
        "AR Loss",
        "Aux Loss",
        "CFCR Loss",
        "Ortho Loss",
        "Caching Efficiency (%)",
        "Routing Entropy",
        "Cosine Sim",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        formatted = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                formatted.append(f"{value:.4f}")
            else:
                formatted.append(value)
        print("| " + " | ".join(formatted) + " |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--text-length", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--ffn-dim", type=int, default=64)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--patch-grid-size", type=int, default=2)
    parser.add_argument("--motion-dim", type=int, default=16)
    parser.add_argument("--router-history-window", type=int, default=2)
    parser.add_argument("--cache-threshold", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--lora-alpha", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [run_ablation(spec, args) for spec in build_specs()]
    print_markdown_table(rows)


if __name__ == "__main__":
    main()
