#!/usr/bin/env bash
# The untried shortlist ideas (PLAN backlog 14, 16, 17, 7, + higher resolution), 21 Oct. All on gap-8 YOLO11n-640
# (the best model), scored with the fixed protocol (thresholds down to 0.001), so compare against
# runs/baseline_yolo11n_640_dyn_k8/postprocess_sweep_low.json (from run_rescore.sh).
#   queue a:  delta     [frame, |speed|, |acceleration|] from a 9-frame least-squares fit       (idea 14)
#             blank     gap 8 + motion channels blanked in 20% of training images              (idea 16)
#             tta       gap-8 model as trained, test-time augmentation at prediction (no training) (idea 7)
#   queue b:  bgmed     [frame, |diff t-8|, |frame - median of last 16 frames|]                (idea 17)
#             res960    gap 8 at input size 960 (small drones get more pixels)
# About 10 to 12 h for both queues side by side on one GPU (H200: plenty of memory). Finished runs are skipped.
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   nohup bash code_v4/dgx/run_ideas.sh 1 a > runs/logs/run_ideas_a.log 2>&1 &
#   nohup bash code_v4/dgx/run_ideas.sh 1 b > runs/logs/run_ideas_b.log 2>&1 &
set -uo pipefail

GPU="${1:-1}"
QUEUE="${2:-all}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
K8=/workspace/runs/baseline_yolo11n_640_dyn_k8
cd "$ROOT"

yolo() {  # yolo <name> <python args...>
    local name="$1"; shift
    if [[ -f "runs/$name/postprocess_sweep.json" ]]; then echo "skip $name (done)"; return; fi
    echo "$(date '+%F %T') start $name"
    DRISHTI_IMAGE=drishti-yolo:latest bash "$JOB" "$GPU" "$name" scripts/baselines/yolo_antiuav.py "$@" > /dev/null
    docker wait "$name" > /dev/null 2>&1 || true
    grep -q "ALL DONE" "runs/logs/$name.log" && echo "$(date '+%F %T') done $name" || echo "$(date '+%F %T') FAILED $name (runs/logs/$name.log)"
}
common="--model yolo11n.pt --epochs 40 --frames-root $DATA"

if [[ "$QUEUE" == a || "$QUEUE" == all ]]; then
    yolo baseline_yolo11n_640_delta --stage all $common --imgsz 640 --motion-encoding delta \
        --yolo-root /workspace/data/yolo_a410_delta --out /workspace/runs/baseline_yolo11n_640_delta
    yolo baseline_yolo11n_640_dyn_k8_blank --stage all $common --imgsz 640 --motion-encoding diff --dyn-gap 8 --motion-blank-prob 0.2 \
        --yolo-root /workspace/data/yolo_a410_dyn_k8_blank --out /workspace/runs/baseline_yolo11n_640_dyn_k8_blank
    # TTA: prediction + scoring only, with the gap-8 model's weights
    if [[ ! -f runs/baseline_yolo11n_640_dyn_k8_tta/postprocess_sweep.json ]]; then
        yolo baseline_yolo11n_640_dyn_k8_tta_pred --stage predict $common --imgsz 640 --motion-encoding diff --dyn-gap 8 --tta \
            --weights $K8/train/weights/best.pt --out /workspace/runs/baseline_yolo11n_640_dyn_k8_tta
        DRISHTI_IMAGE=drishti-yolo:latest bash "$JOB" "$GPU" baseline_yolo11n_640_dyn_k8_tta scripts/baselines/yolo_antiuav.py --stage score \
            $common --imgsz 640 --out /workspace/runs/baseline_yolo11n_640_dyn_k8_tta > /dev/null
        docker wait baseline_yolo11n_640_dyn_k8_tta > /dev/null 2>&1 || true
        echo "$(date '+%F %T') done baseline_yolo11n_640_dyn_k8_tta"
    fi
fi
if [[ "$QUEUE" == b || "$QUEUE" == all ]]; then
    yolo baseline_yolo11n_640_bgmed --stage all $common --imgsz 640 --motion-encoding bgmed --dyn-gap 8 \
        --yolo-root /workspace/data/yolo_a410_bgmed --out /workspace/runs/baseline_yolo11n_640_bgmed
    yolo baseline_yolo11n_960_dyn_k8 --stage all $common --imgsz 960 --batch 32 --motion-encoding diff --dyn-gap 8 \
        --yolo-root /workspace/data/yolo_a410_dyn_k8 --out /workspace/runs/baseline_yolo11n_960_dyn_k8
fi
echo "$(date '+%F %T') ALL DONE ($QUEUE)"
python3 - <<'EOF'
import json, os
rows = ["baseline_yolo11n_640_dyn_k8", "baseline_yolo11n_640_delta", "baseline_yolo11n_640_dyn_k8_blank",
        "baseline_yolo11n_640_dyn_k8_tta", "baseline_yolo11n_640_bgmed", "baseline_yolo11n_960_dyn_k8"]
print(f"{'run':36s} {'track':>7s} {'mAP50':>6s} {'mAP75':>6s}   (test, val-chosen, thresholds down to 0.001)")
for r in rows:
    for f in ("postprocess_sweep_low.json", "postprocess_sweep.json"):
        p = os.path.join("runs", r, f)
        if os.path.exists(p):
            t = json.load(open(p))["chosen_on_val"]["test"]
            print(f"{r:36s} {100*t['antiuav_tracking_accuracy']:7.2f} {100*t['map50']:6.1f} {100*t['map75']:6.1f}   [{f}]")
            break
EOF
