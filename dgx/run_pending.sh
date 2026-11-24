#!/usr/bin/env bash
# Start the three pending items on ONE GPU, in the background, and return immediately.
#
#   1. YOLO11n at 448 and 320 (puts YOLO at our compute level: the accuracy-vs-compute curve)
#   2. Per-size / per-attribute comparison of our model and YOLO11 (where does each method win?)
#   3. Track-1 evaluation: our best setup (Chain-of-Focus fixed + memory + 12 crops) given the
#      first-frame box, for a like-for-like comparison with the 17 trackers
#
# Usage (on the DGX):  bash code_v4/dgx/run_pending.sh 1          # 1 = the GPU to use
# Logs: runs/logs/<job>.log. Re-running skips jobs that are already running.
set -euo pipefail

GPU="${1:-1}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
RUNS=/workspace/runs
cd "$ROOT"

running() { docker ps --format '{{.Names}}' | grep -qx "$1"; }
start() {  # start <name> <image> <python args...>
    local name="$1" image="$2"; shift 2
    if running "$name"; then echo "skip $name (already running)"; return; fi
    DRISHTI_IMAGE="$image" bash "$JOB" "$GPU" "$name" "$@"
}

# 1. YOLO11n at smaller inputs (reuses the YOLO-format data prepared by the 640 runs)
for size in 448 320; do
    start "baseline_yolo11n_$size" drishti-yolo:latest scripts/baselines/yolo_antiuav.py --stage all \
        --model yolo11n.pt --imgsz "$size" --epochs 40 --frames-root "$DATA" --yolo-root /workspace/data/yolo_a410 \
        --out "$RUNS/baseline_yolo11n_$size"
done

# 3. Track 1 on the best setup (memory crops are switched on automatically by --init-first-box)
start ff_eval_track1_c12 drishti:latest scripts/run_all_stages.py --config configs/dgx_antiuav410_focus_fixed.yaml \
    --frames-root "$DATA" --run-dir "$RUNS/a410_focus_fixed_s42" --schedule joint --eval-only --eval-split test \
    --eval-tag track1_c12 "--seq-eval-args=--init-first-box --num-crops 12"

# 2. Where each method wins (CPU only; uses predictions that already exist)
start analysis_by_attribute drishti:latest scripts/analyze_by_attribute.py --frames-root "$DATA" --split test \
    --method "ours_cof_mem_c12=$RUNS/a410_focus_fixed_s42/sequence_eval_mem_crops12/raw_test_no_tracker" \
    --method "ours_cof=$RUNS/a410_focus_fixed_s42/sequence_eval_base/raw_test_no_tracker" \
    --method "yolo11n_640=$RUNS/baseline_yolo11n_640/raw_test" \
    --method "yolo11s_640=$RUNS/baseline_yolo11s_640/raw_test" \
    --out "$RUNS/analysis_by_attribute.json"

echo
echo "All started on GPU $GPU. Check with:  docker ps   and   tail -c 2000 runs/logs/<job>.log | tr '\r' '\n' | tail -2"
