"""Cross-platform driver for the full four-stage training run plus the two evaluations.

Mirrors the student's ``scripts/train_all_ir.sh`` (stage1 -> stage2 -> stage3 -> finetune, each
stage starting from the previous stage's best checkpoint, resuming from ``<stage>_latest.pt`` when
present) but runs on Windows and Linux alike, and finishes with BOTH evaluators:

* ``run_eval``            the clip-level evaluator (comparable to the earlier run), and
* ``run_sequence_eval``   the honest sequence-level streaming evaluator, tracker on and off.

Usage
-----
    python scripts/run_all_stages.py --config configs/remote_ir_fast.yaml \
        --frames-root C:/data/dataset_frames --run-dir runs/cvpr_seed42 --seed 42

    # only some stages, or different epoch counts
    python scripts/run_all_stages.py ... --stages stage1 --epochs stage1=40
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["stage1", "stage2", "stage3", "finetune"]
DEFAULT_EPOCHS = {"stage1": 80, "stage2": 30, "stage3": 20, "finetune": 10, "joint": 40}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run(command: list[str], cwd: Path) -> None:
    log("RUN " + " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all DRISHTI training stages and evaluations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--modality", default="infrared")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None, help="Overrides config.seed via a copied config file.")
    parser.add_argument("--stages", nargs="+", default=None, choices=STAGES + ["joint"])
    parser.add_argument("--schedule", default="staged", choices=["staged", "joint"],
                        help="staged: stage1..finetune with freezing (original). joint: one stage, all modules train together.")
    parser.add_argument("--epochs", nargs="*", default=[], help="Overrides like stage1=40 finetune=5.")
    parser.add_argument("--init-checkpoint", default=None,
                        help="Weights to start the FIRST stage from (e.g. a model trained on another dataset). "
                             "Ignored for a stage that already has a <stage>_latest.pt to resume from.")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--seq-eval-args", default="",
                        help='Extra options for every sequence evaluation, e.g. "--memory --num-crops 12".')
    parser.add_argument("--eval-tag", default=None,
                        help="Write evaluation results to sequence_eval_<tag> (keeps earlier results).")
    parser.add_argument("--eval-only", action="store_true",
                        help="Train nothing; evaluate the last finished stage already in --run-dir.")
    parser.add_argument("--eval-split", default="val")
    # argparse refuses "--seq-eval-args --memory" (a value that looks like an option); glue it as
    # "--seq-eval-args=--memory" so both spellings work.
    argv = list(sys.argv[1:])
    for i, token in enumerate(argv[:-1]):
        if token == "--seq-eval-args":
            argv[i : i + 2] = [f"--seq-eval-args={argv[i + 1]}"]
            break
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs = dict(DEFAULT_EPOCHS)
    for item in args.epochs:
        key, value = item.split("=")
        epochs[key] = int(value)

    config_path = Path(args.config).resolve()
    if args.seed is not None:
        # Write a per-run copy of the config with the seed baked in so the run is self-describing.
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        raw["seed"] = int(args.seed)
        raw["frames_root"] = args.frames_root
        config_path = run_dir / ("config_eval.yaml" if args.eval_only else "config.yaml")
        config_path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")

    info_name = "eval_only_info.json" if args.eval_only else "run_info.json"  # keep the training run_info intact
    (run_dir / info_name).write_text(
        json.dumps({"config": str(config_path), "frames_root": args.frames_root, "epochs": epochs, "seed": args.seed,
                    "init_checkpoint": args.init_checkpoint,
                    "stages": args.stages, "argv": sys.argv, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2),
        encoding="utf-8",
    )

    previous_best: Path | None = Path(args.init_checkpoint).resolve() if args.init_checkpoint else None
    if previous_best is not None and not previous_best.exists():
        raise SystemExit(f"--init-checkpoint not found: {previous_best}")
    plan = ["joint"] if args.schedule == "joint" else STAGES
    if args.stages is None:
        args.stages = plan
    if args.eval_only:
        args.stages = []
    for stage in plan:
        stage_dir = run_dir / stage
        best = stage_dir / "checkpoints" / f"{stage}_best.pt"
        latest = stage_dir / "checkpoints" / f"{stage}_latest.pt"
        if stage not in args.stages:
            if best.exists():
                previous_best = best
            continue
        log(f"===== {stage}: {epochs[stage]} epochs -> {stage_dir}")
        command = [
            sys.executable, "-u", "-m", "drishti_v2.experiments.run_training",
            "--config", str(config_path), "--frames-root", args.frames_root, "--modality", args.modality,
            "--device", args.device, "--stage", stage, "--epochs", str(epochs[stage]), "--output-dir", str(stage_dir),
        ]
        if latest.exists():
            log(f"resuming from {latest}")
            command += ["--resume-checkpoint", str(latest)]
        elif previous_best is not None:
            log(f"starting from {previous_best}")
            command += ["--checkpoint", str(previous_best)]
        else:
            log("starting from initial weights")
        run(command, ROOT)
        if not best.exists():
            raise SystemExit(f"{stage} did not produce {best}")
        previous_best = best

    if args.skip_eval or previous_best is None:
        return
    final = previous_best
    log(f"===== evaluation of {final}")
    run([sys.executable, "-u", "-m", "drishti_v2.experiments.run_eval", "--config", str(config_path),
         "--frames-root", args.frames_root, "--modality", args.modality, "--split", args.eval_split,
         "--checkpoint", str(final), "--device", args.device,
         "--output", str(run_dir / "final_eval" / "metrics.json")], ROOT)
    eval_dir = run_dir / (f"sequence_eval_{args.eval_tag}" if args.eval_tag else "sequence_eval")
    seq_eval = [sys.executable, "-u", "-m", "drishti_v2.experiments.run_sequence_eval", "--config", str(config_path),
                "--frames-root", args.frames_root, "--modality", args.modality,
                "--checkpoint", str(final), "--device", args.device, "--output-dir", str(eval_dir)] + args.seq_eval_args.split()
    # Raw evaluation as before (all boxes, threshold from the config), tracker on and off.
    run(seq_eval + ["--split", args.eval_split], ROOT)
    run(seq_eval + ["--split", args.eval_split, "--no-tracker", "--save-raw"], ROOT)
    if args.eval_split != "val":
        # Choose NMS / top-1 / threshold on val only, then apply it once to the test raw predictions.
        run(seq_eval + ["--split", "val", "--no-tracker", "--save-raw"], ROOT)
        sweep_out = eval_dir / "postprocess_sweep.json"
        run([sys.executable, "-u", str(ROOT / "scripts" / "sweep_postprocess.py"), "--frames-root", args.frames_root,
             "--modality", args.modality,
             "--val-raw", str(eval_dir / "raw_val_no_tracker"),
             "--test-raw", str(eval_dir / f"raw_{args.eval_split}_no_tracker"),
             "--out", str(sweep_out)], ROOT)
        chosen = json.loads(sweep_out.read_text(encoding="utf-8"))["chosen_on_val"]
        # The tracker must never see duplicate boxes: always merge them and keep one box per frame.
        nms_iou = chosen["nms_iou"] if chosen["nms_iou"] is not None else 0.3
        tuned = ["--threshold", str(chosen["threshold"]), "--nms-iou", str(nms_iou), "--top1"]
        if chosen.get("smooth"):
            tuned += ["--smooth"]
        # Same post-processing without the tracker, run live (memory crops, if enabled, need a live run).
        run(seq_eval + ["--split", args.eval_split, "--no-tracker", "--tag", "no_tracker_tuned"] + tuned, ROOT)
        # Tracker feedback again, now with duplicates merged before they reach the tracker.
        run(seq_eval + ["--split", args.eval_split, "--tag", "tracker_tuned"] + tuned, ROOT)
    (run_dir / "ALL_STAGES_COMPLETE.txt").write_text(
        f"completed {time.strftime('%Y-%m-%dT%H:%M:%S')}\nfinal checkpoint {final}\n", encoding="utf-8")
    log("ALL DONE")


if __name__ == "__main__":
    main()
