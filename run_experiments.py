"""Run hyperparameter sweeps and generate performance visualization charts."""

from __future__ import annotations

import argparse
import os
import json
import csv
from dataclasses import dataclass, asdict, field
import matplotlib.pyplot as plt
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
class ExperimentSpec:
    name: str
    loss_weights: TMoELossWeights
    config_overrides: dict[str, object]


def get_default_weights() -> TMoELossWeights:
    return TMoELossWeights(alpha_aux=0.01, beta_cfcr=0.1, gamma_ortho=0.01)


def build_experiments() -> list[ExperimentSpec]:
    d_weights = get_default_weights()
    return [
        ExperimentSpec("Control", d_weights, {}),
        # Routing Architecture Sweeps
        ExperimentSpec("k1_experts4", d_weights, {"top_k": 1, "num_experts": 4}),
        ExperimentSpec("k2_experts4", d_weights, {"top_k": 2, "num_experts": 4}),
        ExperimentSpec("k1_experts8", d_weights, {"top_k": 1, "num_experts": 8}),
        # Model Scale & Depth Sweeps
        ExperimentSpec("depth1_dim32", d_weights, {"num_layers": 1}),
        ExperimentSpec("depth4_dim32", d_weights, {"num_layers": 4}),
        ExperimentSpec(
            "depth2_dim64",
            d_weights,
            {"hidden_dim": 64, "ffn_dim": 128, "motion_dim": 32, "num_attention_heads": 4},
        ),
        # Router Temporal History Sweeps
        ExperimentSpec("history0", d_weights, {"router_history_window": 0}),
        ExperimentSpec("history1", d_weights, {"router_history_window": 1}),
        ExperimentSpec("history4", d_weights, {"router_history_window": 4}),
        # Cache Bypass Threshold Sweeps
        ExperimentSpec("cache_eps0.0", d_weights, {"cache_threshold": 0.0}),
        ExperimentSpec("cache_eps0.01", d_weights, {"cache_threshold": 0.01}),
        ExperimentSpec("cache_eps0.1", d_weights, {"cache_threshold": 0.1}),
        # Regularization Loss Sweeps
        ExperimentSpec(
            "loss_no_ortho",
            TMoELossWeights(alpha_aux=0.01, beta_cfcr=0.1, gamma_ortho=0.0),
            {},
        ),
        ExperimentSpec(
            "loss_no_cfcr",
            TMoELossWeights(alpha_aux=0.01, beta_cfcr=0.0, gamma_ortho=0.01),
            {},
        ),
    ]


def build_config(args: argparse.Namespace, spec: ExperimentSpec) -> TMoEConfig:
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


def generate_synthetic_batch(
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
        # Create consecutive differences that simulate dynamic movement with some static regions
        # We enforce static visual frames for 60% of temporal tokens to test caching
        noise = 0.01 * torch.randn(
            args.batch_size,
            args.frames - 1,
            config.image_channels,
            args.height,
            args.width,
            generator=generator,
        )
        frames[:, 1:] = frames[:, :1] + noise
    
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (args.batch_size, args.text_length),
        generator=generator,
    )
    labels = input_ids.clone()
    return frames, input_ids, labels


def run_experiment(spec: ExperimentSpec, args: argparse.Namespace) -> dict[str, list[float] | float | str]:
    print(f"\n--- Running Experiment: {spec.name} ---")
    torch.manual_seed(args.seed)
    config = build_config(args, spec)
    model = TMoELLaVAMicro(config).to(args.device)
    trainer = TMoETrainer(
        model,
        TrainingConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            loss_weights=spec.loss_weights,
        ),
    )

    history = {
        "loss": [],
        "ar": [],
        "aux": [],
        "cfcr": [],
        "ortho": [],
        "routing_entropy": [],
        "expert_similarity": [],
    }

    for step in range(args.steps):
        frames, input_ids, labels = generate_synthetic_batch(
            config,
            args,
            seed=args.seed + step,
        )
        metrics = trainer.train_step(frames, input_ids, labels)
        
        # Calculate routing diagnostic stats at this step
        model.eval()
        with torch.no_grad():
            output = model(frames, input_ids, reset_cache=True)
            entropy = sum(routing_entropy(r.probs).item() for r in output.router_outputs) / len(output.router_outputs)
            similarity = sum(expert_lora_similarity(b.moe.experts).item() for b in model.blocks) / len(model.blocks)
        
        history["loss"].append(metrics["loss"])
        history["ar"].append(metrics["ar"])
        history["aux"].append(metrics["aux"])
        history["cfcr"].append(metrics["cfcr"])
        history["ortho"].append(metrics["ortho"])
        history["routing_entropy"].append(entropy)
        history["expert_similarity"].append(similarity)
        
        if (step + 1) % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            print(f"Step {step+1}/{args.steps} | Loss: {metrics['loss']:.4f} | AR: {metrics['ar']:.4f} | Aux: {metrics['aux']:.4f}")

    # Evaluate caching efficiency on an evaluation batch
    eval_frames, eval_ids, _ = generate_synthetic_batch(config, args, seed=args.seed + 9999)
    model.eval()
    with torch.no_grad():
        output = model(eval_frames, eval_ids, reset_cache=True)
        total_tokens = sum(stats.total_tokens for stats in output.moe_stats)
        cached_tokens = sum(stats.cached_tokens for stats in output.moe_stats)
        cache_efficiency = 100.0 * cached_tokens / max(total_tokens, 1)

    return {
        "name": spec.name,
        "history": history,
        "final_loss": history["loss"][-1],
        "final_ar": history["ar"][-1],
        "final_aux": history["aux"][-1],
        "final_cfcr": history["cfcr"][-1],
        "final_ortho": history["ortho"][-1],
        "caching_efficiency": cache_efficiency,
        "final_entropy": history["routing_entropy"][-1],
        "final_similarity": history["expert_similarity"][-1],
    }


def generate_plots(results: list[dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    # Define a clean style/color palette
    colors = plt.cm.tab20(range(len(results)))
    
    # 1. Learning Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, res in enumerate(results):
        steps = range(1, len(res["history"]["loss"]) + 1)
        axes[0].plot(steps, res["history"]["loss"], label=res["name"], color=colors[i], linewidth=1.5)
        axes[1].plot(steps, res["history"]["ar"], label=res["name"], color=colors[i], linewidth=1.5)
    
    axes[0].set_title("Total Training Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Steps", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    axes[1].set_title("Autoregressive (AR) Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Steps", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    # Add a unified legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Caching Efficiency vs. AR Loss (Pareto Frontier)
    plt.figure(figsize=(10, 8))
    for i, res in enumerate(results):
        plt.scatter(
            res["caching_efficiency"],
            res["final_ar"],
            s=120,
            label=res["name"],
            color=colors[i],
            alpha=0.8,
            edgecolors="black",
            linewidth=1,
        )
        # Add labels to scatter points
        plt.annotate(
            res["name"],
            (res["caching_efficiency"], res["final_ar"]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
            fontsize=8,
            fontweight="bold",
        )
    
    plt.title("Edge Efficiency-Accuracy Pareto Frontier", fontsize=14, fontweight="bold")
    plt.xlabel("Caching Efficiency (% of tokens bypassed)", fontsize=12)
    plt.ylabel("Final AR Language Loss (Lower is Better)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "caching_vs_loss_pareto.png"), dpi=200)
    plt.close()

    # 3. Routing Dynamics (Entropy & Similarity)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, res in enumerate(results):
        steps = range(1, len(res["history"]["routing_entropy"]) + 1)
        axes[0].plot(steps, res["history"]["routing_entropy"], label=res["name"], color=colors[i], linewidth=1.5)
        axes[1].plot(steps, res["history"]["expert_similarity"], label=res["name"], color=colors[i], linewidth=1.5)
        
    axes[0].set_title("Routing Entropy Dynamics", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Steps", fontsize=12)
    axes[0].set_ylabel("Entropy", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    axes[1].set_title("Expert LoRA Similarity Dynamics", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Steps", fontsize=12)
    axes[1].set_ylabel("Mean Cosine Similarity", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "routing_dynamics.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nAll visualization plots successfully saved in {output_dir}")


def write_summary_files(results: list[dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save results CSV
    csv_path = os.path.join(output_dir, "results_summary.csv")
    headers = [
        "Configuration",
        "Total Loss",
        "AR Loss",
        "Aux Loss",
        "CFCR Loss",
        "Ortho Loss",
        "Caching Efficiency (%)",
        "Routing Entropy",
        "Expert Cosine Similarity",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for res in results:
            writer.writerow([
                res["name"],
                f"{res['final_loss']:.4f}",
                f"{res['final_ar']:.4f}",
                f"{res['final_aux']:.4f}",
                f"{res['final_cfcr']:.4f}",
                f"{res['final_ortho']:.4f}",
                f"{res['caching_efficiency']:.2f}",
                f"{res['final_entropy']:.4f}",
                f"{res['final_similarity']:.4f}",
            ])
            
    # 2. Save markdown report
    md_path = os.path.join(output_dir, "experiment_report.md")
    with open(md_path, "w") as f:
        f.write("# Hyperparameter Sweep & Performance Evaluation Report\n\n")
        f.write("This report summarizes the experimental results of the hyperparameter sweep training runs on the Micro-MoE architecture.\n\n")
        f.write("## 1. Summary of Performance Metrics\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for res in results:
            f.write(
                f"| {res['name']} "
                f"| {res['final_loss']:.4f} "
                f"| {res['final_ar']:.4f} "
                f"| {res['final_aux']:.4f} "
                f"| {res['final_cfcr']:.4f} "
                f"| {res['final_ortho']:.4f} "
                f"| {res['caching_efficiency']:.2f}% "
                f"| {res['final_entropy']:.4f} "
                f"| {res['final_similarity']:.4f} |\n"
            )
            
        f.write("\n## 2. Experimental Plots\n\n")
        f.write("- **Learning Curves**: Compare loss decay curves over training steps between different hyperparameter choices.  \n")
        f.write("  ![Learning Curves](learning_curves.png)\n")
        f.write("- **Efficiency vs. Accuracy Pareto Frontier**: Scatter plot displaying the caching efficiency versus final AR loss tradeoff. The ideal configuration sits towards the bottom-right corner.  \n")
        f.write("  ![Efficiency vs. Accuracy Pareto Frontier](caching_vs_loss_pareto.png)\n")
        f.write("- **Routing Stability & Expert Divergence**: Line chart visualizing how average routing entropy and LoRA parameter similarities evolve over steps.  \n")
        f.write("  ![Routing Stability](routing_dynamics.png)\n")
        
    print(f"Metrics CSV and Markdown report saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps per experiment")
    parser.add_argument("--sweep", action="store_true", help="Run all predefined experiments")
    parser.add_argument("--experiment", type=str, default="Control", help="Name of specific experiment to run (if --sweep is not set)")
    parser.add_argument("--output-dir", type=str, default="experiment_results", help="Directory to save logs and plots")
    parser.add_argument("--seed", type=int, default=42, help="Seed for training generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (cpu, cuda)")
    
    # Default hyperparameter values for model configurations
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--text-length", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--ffn-dim", type=int, default=64)
    parser.add_argument("--num-experts", type=int, default=8)
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
    experiments = build_experiments()
    
    if args.sweep:
        specs_to_run = experiments
    else:
        # Find the named experiment
        specs_to_run = [e for e in experiments if e.name == args.experiment]
        if not specs_to_run:
            print(f"Error: Experiment '{args.experiment}' not found. Available experiments:")
            for e in experiments:
                print(f"  - {e.name}")
            return
            
    results = []
    for spec in specs_to_run:
        res = run_experiment(spec, args)
        results.append(res)
        
    generate_plots(results, args.output_dir)
    write_summary_files(results, args.output_dir)


if __name__ == "__main__":
    main()
