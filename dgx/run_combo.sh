#!/usr/bin/env bash
# Round 17 (22 Oct): combine the pieces that won (compensation +0.7, NWD +0.8, gap 12 +0.4, P2 +0.3) and sweep
# NWD's one constant. Each run ~3 to 4 h; the three queues run side by side. Reference gap 8 = 69.18 (+- 0.18).
#   c0:  comp_k12_nwd        compensation + gap 12 + NWD                (yolo11n)
#        comp_k12_nwd_s1     the same, seed 1
#   c1:  comp_k12_nwd_p2     compensation + gap 12 + NWD + P2 head      (yolov8n-p2, COCO weights transferred)
#        comp_k12_nwd_s2     compensation + gap 12 + NWD, seed 2
#   c2:  comp_nwd_c05        compensation + gap 8 + NWD with c = 0.5    (NWD constant sweep)
#        comp_nwd_c2         compensation + gap 8 + NWD with c = 2.0
# (compensation + gap 8 + NWD with c = 1.0 is round 16's k8_comp_nwd.)
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   nohup bash code_v4/dgx/run_combo.sh 0 c0 > runs/logs/run_combo_c0.log 2>&1 &
#   nohup bash code_v4/dgx/run_combo.sh 1 c1 > runs/logs/run_combo_c1.log 2>&1 &
#   nohup bash code_v4/dgx/run_combo.sh 3 c2 > runs/logs/run_combo_c2.log 2>&1 &
set -uo pipefail

GPU="${1:-0}"
QUEUE="${2:-c0}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
cd "$ROOT"

run() {  # run <name> <args...>
    local name="$1"; shift
    if [[ -f "runs/$name/postprocess_sweep.json" ]]; then echo "skip $name (done)"; return; fi
    echo "$(date '+%F %T') start $name"
    DRISHTI_IMAGE=drishti-yolo:latest bash "$JOB" "$GPU" "$name" scripts/baselines/yolo_antiuav.py --stage all "$@" > /dev/null
    docker wait "$name" > /dev/null 2>&1 || true
    grep -q "ALL DONE" "runs/logs/$name.log" && echo "$(date '+%F %T') done $name" || echo "$(date '+%F %T') FAILED $name (runs/logs/$name.log)"
}
# --patience 0: train the full 40 epochs. Early stopping on Ultralytics' val mAP (which saturates by epoch ~15) cut the
# first compensation + NWD run to 22 epochs and left it under-trained (22 Oct); every run here trains the whole schedule.
# --pick-weights val: Ultralytics' best.pt is chosen by its own val fitness, which saturates by epoch ~15, so a 40-epoch
# run can end up predicting with an epoch-12 checkpoint (22 Oct: a "full" run was bitwise identical to a 22-epoch one).
# Now best.pt and last.pt both predict val and the one with the higher val tracking accuracy is used for test.
C="--epochs 40 --patience 0 --pick-weights val --frames-root $DATA --imgsz 640 --motion-encoding diff --compensate"
V12=/workspace/data/yolo_a410_dyn_k12_comp
V8=/workspace/data/yolo_a410_dyn_k8_comp

case "$QUEUE" in
c0)
    run baseline_yolo11n_640_comp_k12_nwd    $C --model yolo11n.pt --dyn-gap 12 --nwd 1.0 --yolo-root $V12 --out /workspace/runs/baseline_yolo11n_640_comp_k12_nwd
    run baseline_yolo11n_640_comp_k12_nwd_s1 $C --model yolo11n.pt --dyn-gap 12 --nwd 1.0 --seed 1 --yolo-root $V12 --out /workspace/runs/baseline_yolo11n_640_comp_k12_nwd_s1
    ;;
c1)
    # wait for queue c0 to build the gap-12 compensated view (shared), then run in parallel
    while [[ ! -f "$ROOT/data/yolo_a410_dyn_k12_comp/antiuav410.yaml" ]]; do sleep 120; done
    run baseline_yolov8n_p2_640_comp_k12_nwd $C --model yolov8n-p2.yaml --pretrained yolov8n.pt --dyn-gap 12 --nwd 1.0 --yolo-root $V12 --out /workspace/runs/baseline_yolov8n_p2_640_comp_k12_nwd &
    run baseline_yolo11n_640_comp_k12_nwd_s2 $C --model yolo11n.pt --dyn-gap 12 --nwd 1.0 --seed 2 --yolo-root $V12 --out /workspace/runs/baseline_yolo11n_640_comp_k12_nwd_s2 &
    wait
    ;;
c2)
    # the gap-8 compensation + NWD run again, this time trained for all 40 epochs (the first one stopped at 22)
    run baseline_yolo11n_640_comp_nwd_full $C --model yolo11n.pt --dyn-gap 8 --nwd 1.0 --yolo-root $V8 --out /workspace/runs/baseline_yolo11n_640_comp_nwd_full &
    run baseline_yolo11n_640_comp_nwd_c05 $C --model yolo11n.pt --dyn-gap 8 --nwd 0.5 --yolo-root $V8 --out /workspace/runs/baseline_yolo11n_640_comp_nwd_c05 &
    run baseline_yolo11n_640_comp_nwd_c2  $C --model yolo11n.pt --dyn-gap 8 --nwd 2.0 --yolo-root $V8 --out /workspace/runs/baseline_yolo11n_640_comp_nwd_c2 &
    wait
    ;;
*) echo "unknown queue $QUEUE (c0 | c1 | c2)"; exit 1 ;;
esac
echo "$(date '+%F %T') ALL DONE ($QUEUE)"
python3 - <<'EOF'
import json, os
rows = ["baseline_yolo11n_640_dyn_k8", "baseline_yolo11n_640_dyn_k8_comp", "baseline_yolo11n_640_dyn_k8_nwd", "baseline_yolo11n_640_dyn_k8_comp_nwd",
        "baseline_yolo11n_640_comp_nwd_c05", "baseline_yolo11n_640_comp_nwd_c2", "baseline_yolo11n_640_comp_k12_nwd",
        "baseline_yolo11n_640_comp_k12_nwd_s1", "baseline_yolo11n_640_comp_k12_nwd_s2", "baseline_yolov8n_p2_640_comp_k12_nwd"]
print(f"{'run':42s} {'track':>7s} {'mAP50':>6s} {'mAP75':>6s}")
for r in rows:
    for f in ("postprocess_sweep_low.json", "postprocess_sweep.json"):
        p = os.path.join("runs", r, f)
        if os.path.exists(p):
            t = json.load(open(p))["chosen_on_val"]["test"]
            print(f"{r:42s} {100*t['antiuav_tracking_accuracy']:7.2f} {100*t['map50']:6.1f} {100*t['map75']:6.1f}")
            break
EOF
