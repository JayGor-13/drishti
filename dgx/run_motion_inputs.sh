#!/usr/bin/env bash
# Motion-encoding ablation for YOLO11n-640 (17 Oct backlog ideas 3, 4, 2, 6): same detector, data and recipe
# as baseline_yolo11n_640_dyn; only the input channels or one loss weight change.
#
#   dyn_k5, dyn_k8   difference gap k = 5 / 8 (the dyn baseline is k = 3)
#   winner           [frame, signed diff t-1, signed diff t-2]: the 2025 challenge winner's exact input
#   flow             [frame, |diff t-1|, optical-flow magnitude]      (the winner found flow < differences)
#   ts               [frame, |diff t-1|, decaying motion trace]       (time surface, event-camera idea)
#   box15            k = 3 with box-loss weight 15 (default 7.5)      (re-uses the dyn view)
# Each: build its view (about 1 h, once), train (about 2.5 h), predict every val/test frame (1 to 2 h), score.
# Two queues side by side on one free GPU: about 12 to 15 h in total. Finished runs (cost.json) are skipped.
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   G=3
#   nohup bash code_v4/dgx/run_motion_inputs.sh $G a > runs/logs/run_motion_inputs_a.log 2>&1 &
#   nohup bash code_v4/dgx/run_motion_inputs.sh $G b > runs/logs/run_motion_inputs_b.log 2>&1 &
set -euo pipefail

GPU="${1:-1}"
QUEUE="${2:-all}"   # all | a | b
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
cd "$ROOT"

# tag | extra arguments | yolo view
QA="winner|--motion-encoding winner|yolo_a410_winner
    dyn_k5|--motion-encoding diff --dyn-gap 5|yolo_a410_dyn_k5
    box15|--motion-encoding diff --dyn-gap 3 --box-gain 15|yolo_a410_dyn"
QB="ts|--motion-encoding timesurface|yolo_a410_ts
    dyn_k8|--motion-encoding diff --dyn-gap 8|yolo_a410_dyn_k8
    flow|--motion-encoding flow|yolo_a410_flow"
case "$QUEUE" in a) ENTRIES="$QA" ;; b) ENTRIES="$QB" ;; *) ENTRIES="$QA
$QB" ;; esac

while IFS='|' read -r tag extra view; do
    tag="$(echo "$tag" | xargs)"; [[ -z "$tag" ]] && continue
    name="baseline_yolo11n_640_$tag"
    if [[ -f "runs/$name/cost.json" ]]; then echo "skip $name (done)"; continue; fi
    echo "$(date '+%F %T') start $name"
    # shellcheck disable=SC2086
    DRISHTI_IMAGE=drishti-yolo:latest bash "$JOB" "$GPU" "$name" scripts/baselines/yolo_antiuav.py --stage all \
        --model yolo11n.pt --imgsz 640 --epochs 40 $extra \
        --frames-root "$DATA" --yolo-root "/workspace/data/$view" --out "/workspace/runs/$name"
    docker wait "$name" > /dev/null || true
    grep -q "ALL DONE" "runs/logs/$name.log" && echo "$(date '+%F %T') done $name" || echo "$(date '+%F %T') FAILED $name (see runs/logs/$name.log)"
done <<< "$ENTRIES"
echo "$(date '+%F %T') ALL DONE"
python3 - <<'EOF'
import json, glob
print("TEST (val-chosen post-processing)          track  mAP50  mAP75")
for f in sorted(glob.glob("runs/baseline_yolo11n_640*/postprocess_sweep.json")):
    t = json.load(open(f))["chosen_on_val"]["test"]
    print(f"{f.split('/')[1]:36s} {100*t['antiuav_tracking_accuracy']:6.2f} {100*t['map50']:6.1f} {100*t['map75']:6.1f}")
EOF
