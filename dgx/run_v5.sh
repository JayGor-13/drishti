#!/usr/bin/env bash
# Version 5 (option A), evaluation only: YOLO11n's detections + our Chain-of-Focus model + memory + 12 crops.
# Uses the trained Chain-of-Focus (fixed) checkpoint and the YOLO11n-640 predictions that already exist.
#
#   v5_guided        YOLO's top-3 boxes become guided crops for our model (fixes coverage); output = our boxes
#   v5_fused         + YOLO's boxes are also output, refined by our Chain-of-Focus head
#   v5_fused_noref   + YOLO's boxes output as they are (isolates what our refinement adds)
# Every job picks threshold / one-box-per-frame / merging / smoothing on val and applies it once to test,
# with and without the tracker, exactly like all other results.
#
# Usage (on the DGX):  bash code_v4/dgx/run_v5.sh 1          # 1 = the GPU to use
set -euo pipefail

GPU="${1:-1}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
RUN=/workspace/runs/a410_focus_fixed_s42
YOLO="/workspace/runs/baseline_yolo11n_640/raw_{split}"
cd "$ROOT"

for need in runs/baseline_yolo11n_640/raw_val runs/baseline_yolo11n_640/raw_test runs/a410_focus_fixed_s42/joint/checkpoints/joint_best.pt; do
    [[ -e "$need" ]] || { echo "missing $need"; exit 1; }
done

running() { docker ps --format '{{.Names}}' | grep -qx "$1"; }
start() {  # start <name> <seq-eval-args>
    if running "$1"; then echo "skip $1 (already running)"; return; fi
    bash "$JOB" "$GPU" "$1" scripts/run_all_stages.py --config configs/dgx_antiuav410_focus_fixed.yaml --frames-root "$DATA" \
        --run-dir "$RUN" --schedule joint --eval-only --eval-split test --eval-tag "$1" "--seq-eval-args=$2"
}

BASE="--memory --num-crops 12 --external-dets $YOLO --ext-topk 3 --ext-min-score 0.05"
start v5_guided      "$BASE"
start v5_fused       "$BASE --fuse-ext 1.0 --refine-ext"
start v5_fused_noref "$BASE --fuse-ext 1.0"

echo
echo "Started on GPU $GPU. Results: runs/a410_focus_fixed_s42/sequence_eval_v5_*/"
