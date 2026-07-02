from __future__ import annotations

import argparse
import time

import torch

from drishti_v2.models import DRISHTIConfig, DRISHTIPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DRISHTI forward latency.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = DRISHTIConfig.from_yaml(args.config)
    device = torch.device(args.device)
    model = DRISHTIPipeline(config).to(device).eval()
    frames = torch.rand(1, config.temporal_window, 3, config.image_height, config.image_width, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(frames)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            model(frames)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    fps = args.iters / max(elapsed, 1e-8)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({"iters": args.iters, "elapsed_sec": round(elapsed, 4), "fps": round(fps, 2), "params": params, "trainable": trainable})


if __name__ == "__main__":
    main()
