#!/usr/bin/env bash
# Round 12 (19 Oct): paper material that does not depend on the decoder. Two YOLO queues for a free GPU.
#   queue a:  dyn_k12, dyn_k16            where does the gap curve peak? (k=8 is best so far)
#             dyn_k8_s1, dyn_k8_s2        seeds 1 and 2 of the best single model (mean +- std for the paper)
#   queue b:  yolo26n_dyn_k8              our input on the best plain detector of the 512 table (needs drishti-yolo26)
#             yolo11s_dyn_k8              our input on a larger model (does the gain survive scale?)
#             dyn_k5_s1                   one more seed of k=5 (k=5 and k=8 tie on tracking accuracy)
# Each run about 4 h (view once, train 2.5 h, predict 1.5 h). Both queues on one GPU: about 12 to 14 h.
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   G=3
#   nohup bash code_v4/dgx/run_extras.sh $G a > runs/logs/run_extras_a.log 2>&1 &
#   nohup bash code_v4/dgx/run_extras.sh $G b > runs/logs/run_extras_b.log 2>&1 &
set -euo pipefail

GPU="${1:-1}"
QUEUE="${2:-all}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
cd "$ROOT"

# name | model | image | extra arguments | yolo view
QA="baseline_yolo11n_640_dyn_k12|yolo11n.pt|drishti-yolo|--motion-encoding diff --dyn-gap 12|yolo_a410_dyn_k12
    baseline_yolo11n_640_dyn_k16|yolo11n.pt|drishti-yolo|--motion-encoding diff --dyn-gap 16|yolo_a410_dyn_k16
    baseline_yolo11n_640_dyn_k8_s1|yolo11n.pt|drishti-yolo|--motion-encoding diff --dyn-gap 8 --seed 1|yolo_a410_dyn_k8
    baseline_yolo11n_640_dyn_k8_s2|yolo11n.pt|drishti-yolo|--motion-encoding diff --dyn-gap 8 --seed 2|yolo_a410_dyn_k8"
QB="baseline_yolo26n_640_dyn_k8|yolo26n.pt|drishti-yolo26|--motion-encoding diff --dyn-gap 8|yolo_a410_dyn_k8
    baseline_yolo11s_640_dyn_k8|yolo11s.pt|drishti-yolo|--motion-encoding diff --dyn-gap 8|yolo_a410_dyn_k8
    baseline_yolo11n_640_dyn_k5_s1|yolo11n.pt|drishti-yolo|--motion-encoding diff --dyn-gap 5 --seed 1|yolo_a410_dyn_k5"
case "$QUEUE" in a) ENTRIES="$QA" ;; b) ENTRIES="$QB" ;; *) ENTRIES="$QA
$QB" ;; esac

while IFS='|' read -r name model image extra view; do
    name="$(echo "$name" | xargs)"; [[ -z "$name" ]] && continue
    if [[ -f "runs/$name/cost.json" ]]; then echo "skip $name (done)"; continue; fi
    if ! docker image inspect "$image:latest" > /dev/null 2>&1; then echo "skip $name: image $image not built"; continue; fi
    echo "$(date '+%F %T') start $name"
    # shellcheck disable=SC2086
    DRISHTI_IMAGE="$image:latest" bash "$JOB" "$GPU" "$name" scripts/baselines/yolo_antiuav.py --stage all \
        --model "$model" --imgsz 640 --epochs 40 $extra \
        --frames-root "$DATA" --yolo-root "/workspace/data/$view" --out "/workspace/runs/$name"
    docker wait "$name" > /dev/null || true
    grep -q "ALL DONE" "runs/logs/$name.log" && echo "$(date '+%F %T') done $name" || echo "$(date '+%F %T') FAILED $name (see runs/logs/$name.log)"
done <<< "$ENTRIES"
echo "$(date '+%F %T') ALL DONE"
