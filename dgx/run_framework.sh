#!/usr/bin/env bash
# Round 16 (22 Oct): the new-framework pieces (PLAN "Literature review, 21 Oct"), each on gap-8 YOLO11n-640 unless noted.
# Three queues for three GPUs; runs inside a queue share nothing, so they start together. About 6 to 8 h per queue.
#   g0:  k8_comp        ego-motion-compensated differencing               (the camera follows the drone)
#        k8_comp_nwd    compensation + NWD assignment/loss                 (first combination; after k8_comp, shares its view)
#   g1:  k8_comp_tprof  compensation + temporal-anomaly channel            (DeepPro at the input)
#        k8_comp_lcm    compensation + local-contrast channel
#   g2:  k8_nwd         NWD (Gaussian distance) assignment + box loss, plain gap-8 input
#        v8n_p2_k8      yolov8n with a stride-4 (P2) head, gap-8 input, COCO weights transferred
#        v8n_k8         yolov8n without P2, gap-8 input                     (the control for the P2 row)
# Reference: runs/baseline_yolo11n_640_dyn_k8/postprocess_sweep_low.json = 69.18 (seeds +- 0.18).
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   nohup bash code_v4/dgx/run_framework.sh 0 g0 > runs/logs/run_framework_g0.log 2>&1 &
#   nohup bash code_v4/dgx/run_framework.sh 1 g1 > runs/logs/run_framework_g1.log 2>&1 &
#   nohup bash code_v4/dgx/run_framework.sh 2 g2 > runs/logs/run_framework_g2.log 2>&1 &
set -uo pipefail

GPU="${1:-0}"
QUEUE="${2:-g0}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
cd "$ROOT"

run() {  # run <name> <args...>   (skips if postprocess_sweep.json exists)
    local name="$1"; shift
    if [[ -f "runs/$name/postprocess_sweep.json" ]]; then echo "skip $name (done)"; return; fi
    echo "$(date '+%F %T') start $name"
    DRISHTI_IMAGE=drishti-yolo:latest bash "$JOB" "$GPU" "$name" scripts/baselines/yolo_antiuav.py --stage all "$@" > /dev/null
    docker wait "$name" > /dev/null 2>&1 || true
    grep -q "ALL DONE" "runs/logs/$name.log" && echo "$(date '+%F %T') done $name" || echo "$(date '+%F %T') FAILED $name (runs/logs/$name.log)"
}
C="--epochs 40 --frames-root $DATA --imgsz 640 --dyn-gap 8"

case "$QUEUE" in
g0)
    run baseline_yolo11n_640_dyn_k8_comp $C --model yolo11n.pt --motion-encoding diff --compensate \
        --yolo-root /workspace/data/yolo_a410_dyn_k8_comp --out /workspace/runs/baseline_yolo11n_640_dyn_k8_comp
    run baseline_yolo11n_640_dyn_k8_comp_nwd $C --model yolo11n.pt --motion-encoding diff --compensate --nwd 1.0 \
        --yolo-root /workspace/data/yolo_a410_dyn_k8_comp --out /workspace/runs/baseline_yolo11n_640_dyn_k8_comp_nwd
    ;;
g1)
    run baseline_yolo11n_640_k8_comp_tprof $C --model yolo11n.pt --motion-encoding tprofile --compensate \
        --yolo-root /workspace/data/yolo_a410_k8_comp_tprof --out /workspace/runs/baseline_yolo11n_640_k8_comp_tprof &
    run baseline_yolo11n_640_k8_comp_lcm $C --model yolo11n.pt --motion-encoding lcm --compensate \
        --yolo-root /workspace/data/yolo_a410_k8_comp_lcm --out /workspace/runs/baseline_yolo11n_640_k8_comp_lcm &
    wait
    ;;
g2)
    run baseline_yolo11n_640_dyn_k8_nwd $C --model yolo11n.pt --motion-encoding diff --nwd 1.0 \
        --yolo-root /workspace/data/yolo_a410_dyn_k8 --out /workspace/runs/baseline_yolo11n_640_dyn_k8_nwd &
    run baseline_yolov8n_p2_640_dyn_k8 $C --model yolov8n-p2.yaml --pretrained yolov8n.pt --motion-encoding diff \
        --yolo-root /workspace/data/yolo_a410_dyn_k8 --out /workspace/runs/baseline_yolov8n_p2_640_dyn_k8 &
    run baseline_yolov8n_640_dyn_k8 $C --model yolov8n.pt --motion-encoding diff \
        --yolo-root /workspace/data/yolo_a410_dyn_k8 --out /workspace/runs/baseline_yolov8n_640_dyn_k8 &
    wait
    ;;
*) echo "unknown queue $QUEUE (g0 | g1 | g2)"; exit 1 ;;
esac
echo "$(date '+%F %T') ALL DONE ($QUEUE)"
python3 - <<'EOF'
import json, os
rows = ["baseline_yolo11n_640_dyn_k8", "baseline_yolo11n_640_dyn_k8_comp", "baseline_yolo11n_640_dyn_k8_comp_nwd",
        "baseline_yolo11n_640_k8_comp_tprof", "baseline_yolo11n_640_k8_comp_lcm", "baseline_yolo11n_640_dyn_k8_nwd",
        "baseline_yolov8n_640_dyn_k8", "baseline_yolov8n_p2_640_dyn_k8"]
print(f"{'run':40s} {'track':>7s} {'mAP50':>6s} {'mAP75':>6s}   (test, val-chosen, thresholds down to 0.001)")
for r in rows:
    for f in ("postprocess_sweep_low.json", "postprocess_sweep.json"):
        p = os.path.join("runs", r, f)
        if os.path.exists(p):
            t = json.load(open(p))["chosen_on_val"]["test"]
            print(f"{r:40s} {100*t['antiuav_tracking_accuracy']:7.2f} {100*t['map50']:6.1f} {100*t['map75']:6.1f}")
            break
EOF
