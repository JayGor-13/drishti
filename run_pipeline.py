"""Smoke-run the T-MoE-LLaVA Micro-MoE scaffold."""

from __future__ import annotations

import argparse

import torch

from models import TMoEConfig, TMoELLaVAMicro


def build_config(args: argparse.Namespace) -> TMoEConfig:
    return TMoEConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        patch_grid_size=args.patch_grid_size,
        motion_dim=args.motion_dim,
        cache_threshold=args.cache_threshold,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_text_length=args.text_length,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--text-length", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--patch-grid-size", type=int, default=2)
    parser.add_argument("--motion-dim", type=int, default=16)
    parser.add_argument("--cache-threshold", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--lora-alpha", type=float, default=4.0)
    args = parser.parse_args()

    torch.manual_seed(42)
    config = build_config(args)
    model = TMoELLaVAMicro(config)
    frames = torch.randn(args.batch_size, args.frames, 3, args.height, args.width)
    frames[:, 1:] = frames[:, :1]
    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.text_length))

    with torch.no_grad():
        output = model(frames, input_ids, reset_cache=True)

    print("T-MoE-LLaVA Micro-MoE smoke run")
    print(f"logits: {tuple(output.logits.shape)}")
    print(f"next_token_logits: {tuple(output.next_token_logits.shape)}")
    print(f"multimodal_sequence: {tuple(output.multimodal_sequence.shape)}")
    for idx, stats in enumerate(output.moe_stats):
        print(
            f"block_{idx}: executed={stats.executed_tokens} "
            f"cached={stats.cached_tokens} total={stats.total_tokens}"
        )


if __name__ == "__main__":
    main()
