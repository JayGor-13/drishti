#!/usr/bin/env python
"""Sweep different clip-stride values and log results into separate output dirs.

Usage (Linux/remote GPU):
    python scripts/run_stride_sweep.py --config configs/default.yaml \
        --frames-root ~/dataset_frames --device cuda --epochs 1

Usage (Windows local):
    python scripts/run_stride_sweep.py --config configs/default.yaml \
        --synthetic --device cpu --epochs 1

Each stride setting writes its results to results/sweep_stride_<N>/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


STRIDES = [4, 8, 16, 32, 64]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep clip-stride values for DRISHTI-CORE v2.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--frames-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", default="stage1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--strides", nargs="+", type=int, default=STRIDES, help="Stride values to sweep.")
    args = parser.parse_args()

    results_summary: list[dict] = []
    for stride in args.strides:
        output_dir = f"results/sweep_stride_{stride}"
        print(f"\n{'=' * 60}")
        print(f"  Sweep: clip-stride = {stride}")
        print(f"  Output: {output_dir}")
        print(f"{'=' * 60}\n")

        cmd = [
            sys.executable, "-m", "drishti_v2.experiments.run_training",
            "--config", args.config,
            "--device", args.device,
            "--stage", args.stage,
            "--epochs", str(args.epochs),
            "--clip-stride", str(stride),
            "--output-dir", output_dir,
        ]
        if args.synthetic:
            cmd.append("--synthetic")
        if args.frames_root:
            cmd.extend(["--frames-root", args.frames_root])
        if args.data_root:
            cmd.extend(["--data-root", args.data_root])

        result = subprocess.run(cmd, capture_output=False, text=True)
        entry = {"clip_stride": stride, "exit_code": result.returncode, "output_dir": output_dir}
        results_summary.append(entry)
        if result.returncode != 0:
            print(f"[WARNING] Stride {stride} failed with exit code {result.returncode}")

    print(f"\n{'=' * 60}")
    print("  Sweep Summary")
    print(f"{'=' * 60}")
    for entry in results_summary:
        status = "OK" if entry["exit_code"] == 0 else f"FAILED ({entry['exit_code']})"
        print(f"  stride={entry['clip_stride']:>3}  →  {status}  →  {entry['output_dir']}")

    summary_path = Path("results") / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
