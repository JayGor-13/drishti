#!/usr/bin/env bash
# Version 7 (17 Oct): where is the ceiling, and a learned "second look" on frame-dynamics YOLO's boxes.
# Runs side by side with run_v6_dyn.sh on the same GPU (small jobs).
#
#   analyze_candidates      histogram of the best candidate overlap per frame (complete vs near misses),
#                           on the plain-YOLO candidates now, and on the frame-dynamics ones when they exist
#   v7_yolo_dyn_train       frame-dynamics YOLO11n predictions on the TRAIN split (the refiner learns from them)
#   v7_refiner              collect crops, train the refiner, refine val/test boxes, score with the usual protocol
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   nohup bash code_v4/dgx/run_v7.sh 1 > runs/logs/run_v7.log 2>&1 &
set -euo pipefail

GPU="${1:-1}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
DYN=/workspace/runs/baseline_yolo11n_640_dyn
cd "$ROOT"

for need in runs/baseline_yolo11n_640_dyn/train/weights/best.pt runs/baseline_yolo11n_640_dyn/raw_val runs/baseline_yolo11n_640_dyn/raw_test; do
    [[ -e "$need" ]] || { echo "missing $need"; exit 1; }
done
wait_for() { for n in "$@"; do docker wait "$n" > /dev/null 2>&1 || true; echo "$(date '+%F %T') finished $n"; done; }

echo "$(date '+%F %T') ceiling analysis on the plain-YOLO candidates (CPU, minutes)"
for split in val test; do
    bash "$JOB" "$GPU" "analyze_candidates_$split" scripts/analyze_candidates.py --candidates /workspace/runs/v6_candidates \
        --split $split --out "/workspace/runs/v6_candidates/analysis_$split.json"
done

echo "$(date '+%F %T') frame-dynamics YOLO on the train split (1 to 2 h)"
if [[ -d runs/baseline_yolo11n_640_dyn/raw_train && $(ls runs/baseline_yolo11n_640_dyn/raw_train | wc -l) -ge 200 ]]; then
    echo "train predictions already exist; skipping"
else
    DRISHTI_IMAGE=drishti-yolo:latest bash "$JOB" "$GPU" v7_yolo_dyn_train scripts/baselines/yolo_antiuav.py --stage predict \
        --frame-dynamics --predict-splits train --model yolo11n.pt --imgsz 640 --frames-root "$DATA" \
        --yolo-root /workspace/data/yolo_a410_dyn --out "$DYN"
    wait_for v7_yolo_dyn_train
fi
wait_for analyze_candidates_val analyze_candidates_test
[[ $(ls runs/baseline_yolo11n_640_dyn/raw_train 2>/dev/null | wc -l) -ge 200 ]] || { echo "train predictions missing (see runs/logs/v7_yolo_dyn_train.log); stopping"; exit 1; }

echo "$(date '+%F %T') refiner: collect (about 1 h), train (about 30 min), apply to val+test (about 2 h), score"
bash "$JOB" "$GPU" v7_refiner scripts/train_box_refiner.py --stage all --frames-root "$DATA" --det-root "$DYN" \
    --out /workspace/runs/baseline_yolo11n_640_dyn_refined
wait_for v7_refiner

if [[ -d runs/v6_candidates_dyn/test && $(ls runs/v6_candidates_dyn/test | wc -l) -ge 120 ]]; then
    echo "$(date '+%F %T') ceiling analysis on the frame-dynamics candidates"
    bash "$JOB" "$GPU" analyze_candidates_dyn_test scripts/analyze_candidates.py --candidates /workspace/runs/v6_candidates_dyn \
        --split test --out /workspace/runs/v6_candidates_dyn/analysis_test.json
    wait_for analyze_candidates_dyn_test
else
    echo "frame-dynamics candidates not complete yet; run later:"
    echo "  bash $JOB $GPU analyze_candidates_dyn_test scripts/analyze_candidates.py --candidates /workspace/runs/v6_candidates_dyn --split test --out /workspace/runs/v6_candidates_dyn/analysis_test.json"
fi

echo "$(date '+%F %T') ALL DONE"
grep -h "^TEST" runs/logs/v7_refiner.log || true
