#!/usr/bin/env bash
# Round 19 (23 Oct): the ablation table of the FINAL system (compensation + gap 12 + NWD, YOLO11n-640, 70.30 +- 0.09)
# and its transfer to another detector. One run each (~3 h), all in parallel on one GPU, then the table.
#   k12_nwd          final minus compensation         (gap 12 + NWD, raw differences)
#   comp_k12         final minus NWD                  (compensation + gap 12, IoU loss)
#   plain_nwd        final minus the motion input     (plain frames + NWD)
#   comp_k12_nwd_26  final recipe on YOLO26n          (does it transfer?)
# Rows that exist already: final (3 seeds), gap 12 alone, gap 8 alone, plain YOLO11n, comp at gap 8, NWD at gap 8, P2 variant.
# Then `dgx/run_rescore.sh` (CPU) adds the decoder row for every run.
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   nohup bash code_v4/dgx/run_ablation_final.sh 1 > runs/logs/run_ablation_final.log 2>&1 &
set -uo pipefail

GPU="${1:-1}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
cd "$ROOT"

run() {  # run <name> <image> <args...>
    local name="$1" image="$2"; shift 2
    if [[ -f "runs/$name/postprocess_sweep.json" ]]; then echo "skip $name (done)"; return; fi
    echo "$(date '+%F %T') start $name"
    DRISHTI_IMAGE="$image:latest" bash "$JOB" "$GPU" "$name" scripts/baselines/yolo_antiuav.py --stage all "$@" > /dev/null
    docker wait "$name" > /dev/null 2>&1 || true
    grep -q "ALL DONE" "runs/logs/$name.log" && echo "$(date '+%F %T') done $name" || echo "$(date '+%F %T') FAILED $name (runs/logs/$name.log)"
}
C="--epochs 40 --patience 0 --pick-weights val --frames-root $DATA --imgsz 640"

run baseline_yolo11n_640_k12_nwd drishti-yolo $C --model yolo11n.pt --motion-encoding diff --dyn-gap 12 --nwd 1.0 \
    --yolo-root /workspace/data/yolo_a410_dyn_k12 --out /workspace/runs/baseline_yolo11n_640_k12_nwd &
run baseline_yolo11n_640_comp_k12 drishti-yolo $C --model yolo11n.pt --motion-encoding diff --dyn-gap 12 --compensate \
    --yolo-root /workspace/data/yolo_a410_dyn_k12_comp --out /workspace/runs/baseline_yolo11n_640_comp_k12 &
run baseline_yolo11n_640_plain_nwd drishti-yolo $C --model yolo11n.pt --nwd 1.0 \
    --yolo-root /workspace/data/yolo_a410 --out /workspace/runs/baseline_yolo11n_640_plain_nwd &
run baseline_yolo26n_640_comp_k12_nwd drishti-yolo26 $C --model yolo26n.pt --motion-encoding diff --dyn-gap 12 --compensate --nwd 1.0 \
    --yolo-root /workspace/data/yolo_a410_dyn_k12_comp --out /workspace/runs/baseline_yolo26n_640_comp_k12_nwd &
wait
echo "$(date '+%F %T') ALL DONE (ablation)"
python3 - <<'EOF'
import json, os
rows = [("plain YOLO11n", "baseline_yolo11n_640"), ("plain + NWD", "baseline_yolo11n_640_plain_nwd"),
        ("gap 12", "baseline_yolo11n_640_dyn_k12"), ("gap 12 + NWD", "baseline_yolo11n_640_k12_nwd"),
        ("comp + gap 12", "baseline_yolo11n_640_comp_k12"), ("FINAL comp + gap 12 + NWD (s42)", "baseline_yolo11n_640_comp_k12_nwd"),
        ("final + P2 head", "baseline_yolov8n_p2_640_comp_k12_nwd"), ("final recipe on YOLO26n", "baseline_yolo26n_640_comp_k12_nwd"),
        ("YOLO26n plain (640)", "baseline_yolo26n_640")]
print(f"{'run':34s} {'track':>7s} {'mAP50':>6s} {'mAP75':>6s}")
for label, r in rows:
    for f in ("postprocess_sweep_low.json", "postprocess_sweep.json"):
        p = os.path.join("runs", r, f)
        if os.path.exists(p):
            t = json.load(open(p))["chosen_on_val"]["test"]
            print(f"{label:34s} {100*t['antiuav_tracking_accuracy']:7.2f} {100*t['map50']:6.1f} {100*t['map75']:6.1f}")
            break
EOF
