#!/usr/bin/env bash
# Single-frame detector baselines, following the comparison table of Gaze-DETR (arXiv 2607.19040, Table III):
# the Ultralytics models of that table, at their input size 512, trained and scored exactly like our
# YOLO11n-640 baseline (same YOLO view, 40 epochs, val-chosen post-processing, same scoring code).
# Runs one after another on one GPU (small YOLOs about 4 to 6 h each, RT-DETR-l about 12 to 18 h at 20
# epochs; DETR-type models need few fine-tuning epochs, Gaze-DETR used 12). Safe to re-run: a finished
# model is skipped (its cost.json exists).
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   nohup bash code_v4/dgx/run_detector_baselines.sh 2 > runs/logs/run_detector_baselines.log 2>&1 &
# Two queues at once (about 1 day in total), e.g. both on GPU 2:
#   nohup bash code_v4/dgx/run_detector_baselines.sh 2 a > runs/logs/run_detector_baselines_a.log 2>&1 &
#   nohup bash code_v4/dgx/run_detector_baselines.sh 2 b > runs/logs/run_detector_baselines_b.log 2>&1 &
set -euo pipefail

GPU="${1:-1}"
QUEUE="${2:-all}"   # all | a | b
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
cd "$ROOT"

# model : batch : image : epochs     (queue a = RT-DETR + 2 small models, queue b = the other 4 small models)
QA="rtdetr-l.pt:16:drishti-yolo:20 yolo11n.pt:64:drishti-yolo:40 yolov8n.pt:64:drishti-yolo:40"
QB="yolov10n.pt:64:drishti-yolo:40 yolo12n.pt:64:drishti-yolo:40 yolov9t.pt:64:drishti-yolo:40 yolo26n.pt:64:drishti-yolo26:40"
case "$QUEUE" in a) MODELS="$QA" ;; b) MODELS="$QB" ;; *) MODELS="$QA $QB" ;; esac

for entry in $MODELS; do
    IFS=: read -r model batch image epochs <<< "$entry"
    name="baseline_${model%.pt}_512"
    name="${name//-/_}"
    if [[ -f "runs/$name/cost.json" ]]; then echo "skip $name (done)"; continue; fi
    if ! docker image inspect "$image:latest" > /dev/null 2>&1; then echo "skip $name: image $image not built"; continue; fi
    echo "$(date '+%F %T') start $name"
    DRISHTI_IMAGE="$image:latest" bash "$JOB" "$GPU" "$name" scripts/baselines/yolo_antiuav.py --stage all \
        --model "$model" --imgsz 512 --epochs "$epochs" --batch "$batch" \
        --frames-root /workspace/data/antiuav410_frames --yolo-root /workspace/data/yolo_a410 --out "/workspace/runs/$name"
    docker wait "$name" > /dev/null || true
    grep -q "ALL DONE" "runs/logs/$name.log" && echo "$(date '+%F %T') done $name" || echo "$(date '+%F %T') FAILED $name (see runs/logs/$name.log)"
done
echo "$(date '+%F %T') ALL DONE"
