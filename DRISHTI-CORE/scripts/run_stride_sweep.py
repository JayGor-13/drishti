#!/usr/bin/env python
"""Sweep different clip-stride values and log results into separate output dirs.

Usage (Linux/remote GPU):
    python scripts/run_stride_sweep.py --config configs/default.yaml \
        --frames-root ~/dataset_frames --device cuda --epochs 80 --resume

Usage (Windows local):
    python scripts/run_stride_sweep.py --config configs/default.yaml \
        --synthetic --device cpu --epochs 1

Each stride setting writes its results to results/sweep_stride_<N>/.
Re-run with --resume to continue from each stride's latest checkpoint.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


STRIDES = [8, 16, 32, 64]


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
    parser.add_argument("--resume", action="store_true", help="Resume each stride from output-dir/checkpoints/latest.pt when present.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip a stride when final_summary.json already reached --epochs.")
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue to later strides if one stride fails.")
    args = parser.parse_args()

    results_summary: list[dict] = []
    for index, stride in enumerate(args.strides, start=1):
        output_dir = f"results/sweep_stride_{stride}"
        final_summary_path = Path(output_dir) / "final_summary.json"
        if args.skip_completed and final_summary_path.exists():
            summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
            if int(summary.get("epochs_completed", 0)) >= args.epochs:
                entry = {
                    "clip_stride": stride,
                    "exit_code": 0,
                    "status": "skipped_complete",
                    "output_dir": output_dir,
                    "final_summary": str(final_summary_path),
                    "best_score": summary.get("best_score"),
                    "best_epoch": summary.get("best_epoch"),
                }
                results_summary.append(entry)
                print(f"\nSkipping stride={stride}: already completed {summary.get('epochs_completed')} epochs.")
                continue

        print(f"\n{'=' * 60}")
        print(f"  Sweep {index}/{len(args.strides)}: clip-stride = {stride}")
        print(f"  Output: {output_dir}")
        if args.resume:
            print("  Resume: enabled")
        print(f"{'=' * 60}\n")

        cmd = [
            sys.executable,
            "-m",
            "drishti_v2.experiments.run_training",
            "--config",
            args.config,
            "--device",
            args.device,
            "--stage",
            args.stage,
            "--epochs",
            str(args.epochs),
            "--clip-stride",
            str(stride),
            "--output-dir",
            output_dir,
        ]
        if args.resume:
            cmd.append("--resume")
        if args.synthetic:
            cmd.append("--synthetic")
        if args.frames_root:
            cmd.extend(["--frames-root", args.frames_root])
        if args.data_root:
            cmd.extend(["--data-root", args.data_root])

        result = subprocess.run(cmd, capture_output=False, text=True)
        entry = {
            "clip_stride": stride,
            "exit_code": result.returncode,
            "status": "ok" if result.returncode == 0 else "failed",
            "output_dir": output_dir,
        }
        if final_summary_path.exists():
            summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
            entry.update(
                {
                    "final_summary": str(final_summary_path),
                    "epochs_completed": summary.get("epochs_completed"),
                    "best_epoch": summary.get("best_epoch"),
                    "best_score": summary.get("best_score"),
                    "latest_checkpoint": summary.get("latest_checkpoint"),
                    "best_checkpoint": summary.get("best_checkpoint"),
                }
            )
        results_summary.append(entry)
        if result.returncode != 0:
            print(f"[ERROR] Stride {stride} failed with exit code {result.returncode}")
            if not args.continue_on_failure:
                print("Stopping sweep. Re-run with --resume after fixing the issue, or use --continue-on-failure to keep going.")
                break

    print(f"\n{'=' * 60}")
    print("  Sweep Summary")
    print(f"{'=' * 60}")
    for entry in results_summary:
        status = "OK" if entry["exit_code"] == 0 else f"FAILED ({entry['exit_code']})"
        score = entry.get("best_score")
        score_text = f"  best={score:.6f}" if isinstance(score, (int, float)) else ""
        print(f"  stride={entry['clip_stride']:>3}  ->  {status}{score_text}  ->  {entry['output_dir']}")

    summary_path = Path("results") / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
