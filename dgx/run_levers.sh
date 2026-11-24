#!/usr/bin/env bash
# Round 15 (21 Oct): the seven improvement levers, all on gap-8 YOLO11n-640, in two queues (about 15 h each).
# Compare every row with runs/baseline_yolo11n_640_dyn_k8/postprocess_sweep_low.json (69.18; seeds 69.17 +- 0.18).
#   queue a:  k8_all      train on EVERY training frame (4x the data; the others use every 4th)      ~10 h
#             k8_rect     rectangular native 640x512 training batches (no letterbox; mosaic off)      ~3 h
#             k8_zoom     zoom-in second look with the same model (no training)                     ~2 h
#   queue b:  diff2_3_12  two motion gaps at once: [frame, |diff t-3|, |diff t-12|]                   ~3 h
#             k8_800      input 800 (between 640 and 960)                                           ~4 h
#             k8_clahe    contrast-normalised frame channel (thermal crossover)                     ~3 h
#             k8_over     thermal-crossover and clutter frames oversampled 3x in training           ~4 h
#             k8_pred800  the existing 640 model predicting at 800 (no training)                    ~1 h
# Lever 3 (temporal box smoothing) runs on the laptop (scripts/smooth_boxes.py). Finished runs are skipped.
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   nohup bash code_v4/dgx/run_levers.sh 1 a > runs/logs/run_levers_a.log 2>&1 &
#   nohup bash code_v4/dgx/run_levers.sh 3 b > runs/logs/run_levers_b.log 2>&1 &
set -uo pipefail

GPU="${1:-1}"
QUEUE="${2:-all}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
K8=/workspace/runs/baseline_yolo11n_640_dyn_k8
cd "$ROOT"

run() {  # run <name> <python script> <args...>   (skips if postprocess_sweep.json exists)
    local name="$1"; shift
    if [[ -f "runs/$name/postprocess_sweep.json" ]]; then echo "skip $name (done)"; return; fi
    echo "$(date '+%F %T') start $name"
    DRISHTI_IMAGE=drishti-yolo:latest bash "$JOB" "$GPU" "$name" "$@" > /dev/null
    docker wait "$name" > /dev/null 2>&1 || true
    grep -q "ALL DONE" "runs/logs/$name.log" && echo "$(date '+%F %T') done $name" || echo "$(date '+%F %T') FAILED $name (runs/logs/$name.log)"
}
Y=scripts/baselines/yolo_antiuav.py
common="--model yolo11n.pt --epochs 40 --frames-root $DATA --motion-encoding diff --dyn-gap 8"

if [[ "$QUEUE" == a || "$QUEUE" == all ]]; then
    run baseline_yolo11n_640_dyn_k8_all $Y --stage all $common --imgsz 640 --train-stride 1 --val-stride 10 \
        --yolo-root /workspace/data/yolo_a410_dyn_k8_all --out /workspace/runs/baseline_yolo11n_640_dyn_k8_all
    run baseline_yolo11n_640_dyn_k8_rect $Y --stage all $common --imgsz 640 --rect \
        --yolo-root /workspace/data/yolo_a410_dyn_k8 --out /workspace/runs/baseline_yolo11n_640_dyn_k8_rect
    run baseline_yolo11n_640_dyn_k8_zoom scripts/zoom_refine.py --frames-root "$DATA" --dyn-gap 8 \
        --base $K8 --out /workspace/runs/baseline_yolo11n_640_dyn_k8_zoom
fi
if [[ "$QUEUE" == b || "$QUEUE" == all ]]; then
    run baseline_yolo11n_640_diff2_3_12 $Y --stage all --model yolo11n.pt --epochs 40 --frames-root $DATA --imgsz 640 \
        --motion-encoding diff2 --dyn-gap 3 --dyn-gap2 12 \
        --yolo-root /workspace/data/yolo_a410_diff2_3_12 --out /workspace/runs/baseline_yolo11n_640_diff2_3_12
    run baseline_yolo11n_800_dyn_k8 $Y --stage all $common --imgsz 800 --batch 48 \
        --yolo-root /workspace/data/yolo_a410_dyn_k8 --out /workspace/runs/baseline_yolo11n_800_dyn_k8
    run baseline_yolo11n_640_dyn_k8_clahe $Y --stage all $common --imgsz 640 --clahe \
        --yolo-root /workspace/data/yolo_a410_dyn_k8_clahe --out /workspace/runs/baseline_yolo11n_640_dyn_k8_clahe
    run baseline_yolo11n_640_dyn_k8_over $Y --stage all $common --imgsz 640 --oversample-attrs IC,DBC --oversample-factor 3 \
        --yolo-root /workspace/data/yolo_a410_dyn_k8_over --out /workspace/runs/baseline_yolo11n_640_dyn_k8_over
    if [[ ! -f runs/baseline_yolo11n_640_dyn_k8_pred800/postprocess_sweep.json ]]; then
        run baseline_yolo11n_640_dyn_k8_pred800_p $Y --stage predict $common --imgsz 800 --weights $K8/train/weights/best.pt \
            --out /workspace/runs/baseline_yolo11n_640_dyn_k8_pred800
        run baseline_yolo11n_640_dyn_k8_pred800 $Y --stage score $common --imgsz 800 --out /workspace/runs/baseline_yolo11n_640_dyn_k8_pred800
    fi
fi
echo "$(date '+%F %T') ALL DONE ($QUEUE)"
python3 - <<'EOF'
import json, os
rows = ["baseline_yolo11n_640_dyn_k8", "baseline_yolo11n_640_dyn_k8_all", "baseline_yolo11n_640_dyn_k8_rect", "baseline_yolo11n_640_dyn_k8_zoom",
        "baseline_yolo11n_640_diff2_3_12", "baseline_yolo11n_800_dyn_k8", "baseline_yolo11n_640_dyn_k8_clahe",
        "baseline_yolo11n_640_dyn_k8_over", "baseline_yolo11n_640_dyn_k8_pred800"]
print(f"{'run':40s} {'track':>7s} {'mAP50':>6s} {'mAP75':>6s}   (test, val-chosen, thresholds down to 0.001)")
for r in rows:
    for f in ("postprocess_sweep_low.json", "postprocess_sweep.json"):
        p = os.path.join("runs", r, f)
        if os.path.exists(p):
            t = json.load(open(p))["chosen_on_val"]["test"]
            print(f"{r:40s} {100*t['antiuav_tracking_accuracy']:7.2f} {100*t['map50']:6.1f} {100*t['map75']:6.1f}")
            break
EOF
