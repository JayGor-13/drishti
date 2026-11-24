"""Count parameters and per-frame FLOPs of the streaming detector, with honest MoE accounting.

Why this script exists
----------------------
The draft paper quotes parameter and GFLOP numbers that do not come from any script in this
repository. This script measures them with ``torch.utils.flop_counter`` on one streaming frame
(after warming the temporal buffer) and prints a per-module breakdown, plus the same model with
a dense MoE and without LDMI, so the "compute saved by sparse routing" claim is measured rather
than assumed.

Usage
-----
    python scripts/count_flops.py --config configs/default.yaml [--checkpoint path.pt] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import sys
from pathlib import Path

import torch

# Allow running as a plain script from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch.utils.flop_counter import FlopCounterMode

from drishti_v2.models import DRISHTIConfig, DRISHTIPipeline


def build(config: DRISHTIConfig, device: torch.device, checkpoint: str | None = None) -> DRISHTIPipeline:
    model = DRISHTIPipeline(config).to(device).eval()
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        own = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(compatible, strict=False)
    return model


@torch.no_grad()
def stream_flops(model: DRISHTIPipeline, config: DRISHTIConfig, device: torch.device) -> tuple[float, dict[str, float]]:
    frame = torch.rand(1, config.image_channels, config.image_height, config.image_width, device=device)
    model.reset_stream()
    for idx in range(config.temporal_window - 1):
        model.forward_stream(frame, idx)
    counter = FlopCounterMode(display=False)
    with counter:
        model.forward_stream(frame, config.temporal_window - 1)
    total = counter.get_total_flops() / 1e9
    per_module = {}
    for name, ops in counter.get_flop_counts().items():
        if name == "Global" or name.count(".") > 0:
            continue
        per_module[name] = round(sum(ops.values()) / 1e9, 4)
    return total, per_module


@torch.no_grad()
def stream_latency(model: DRISHTIPipeline, config: DRISHTIConfig, device: torch.device, iters: int = 20) -> float:
    frame = torch.rand(1, config.image_channels, config.image_height, config.image_width, device=device)
    model.reset_stream()
    for idx in range(5):
        model.forward_stream(frame, idx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for idx in range(iters):
        model.forward_stream(frame, 5 + idx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - started) / iters * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter and FLOP accounting for DRISHTI-CORE v2.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    base = DRISHTIConfig.from_yaml(args.config)
    # Pin the motion gate so the count is deterministic: threshold 0.0 = never dense (the normal
    # 8-crop operating point), threshold 1.0 = always the 16-crop dense fallback. With random or
    # untrained gate weights the mode would otherwise flip between runs.
    config = replace(base, motion_gate_threshold=0.0)
    model = build(config, device, args.checkpoint)

    report: dict[str, object] = {"device": str(device), "config": args.config}
    report["params_total"] = sum(p.numel() for p in model.parameters())
    report["params_by_module"] = {name: sum(p.numel() for p in child.parameters()) for name, child in model.named_children()}
    experts = sum(p.numel() for p in model.moe.experts.parameters())
    report["params_moe_experts_total"] = experts
    report["params_moe_experts_active_topk"] = int(experts * config.top_k / config.num_experts)
    report["params_active_per_frame"] = report["params_total"] - experts + report["params_moe_experts_active_topk"]

    total, per_module = stream_flops(model, config, device)
    report["gflops_per_frame_stream"] = round(total, 4)
    report["gflops_by_module"] = per_module

    dense = build(replace(config, dense_moe=True), device)
    dense_total, _ = stream_flops(dense, config, device)
    report["gflops_per_frame_dense_moe"] = round(dense_total, 4)
    report["moe_sparse_saving_percent_of_total"] = round(100.0 * (dense_total - total) / max(dense_total, 1e-9), 3)

    no_ldmi_cfg = replace(config, use_ldmi=False)
    no_ldmi = build(no_ldmi_cfg, device)
    no_ldmi_total, _ = stream_flops(no_ldmi, no_ldmi_cfg, device)
    report["gflops_per_frame_no_ldmi"] = round(no_ldmi_total, 4)

    dense_fallback_cfg = replace(base, motion_gate_threshold=1.0)
    dense_fallback = build(dense_fallback_cfg, device, args.checkpoint)
    fallback_total, _ = stream_flops(dense_fallback, dense_fallback_cfg, device)
    report["gflops_per_frame_dense_fallback_16_crops"] = round(fallback_total, 4)

    report["latency_ms_per_frame"] = round(stream_latency(model, config, device, args.iters), 2)
    report["fps"] = round(1000.0 / max(report["latency_ms_per_frame"], 1e-6), 2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
