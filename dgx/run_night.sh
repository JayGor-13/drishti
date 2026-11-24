#!/usr/bin/env bash
# One command for everything that is left, on ONE GPU, in order (22 Oct). Start it and walk away; ~16 to 18 h.
#   zoom       zoom-in second look (prediction only, ~2 h), in the background beside the rest
#   c2         compensation + NWD at gap 8 for all 40 epochs, plus NWD c = 0.5 / 2.0      (~4 h, 3 runs in parallel)
#   c0         compensation + gap 12 + NWD, seeds 42 and 1                                  (~1 h view + 2 x 3 h)
#   c1         compensation + gap 12 + NWD seed 2, and the P2-head variant                 (~4 h, in parallel)
# Every finished run is skipped on a re-run, so if the machine hiccups just start it again.
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   nohup bash code_v4/dgx/run_night.sh 1 > runs/logs/run_night.log 2>&1 &
#   tail -f runs/logs/run_night.log
set -uo pipefail

GPU="${1:-1}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
CODE="$ROOT/$DRISHTI_CODE"
JOB="$CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
cd "$ROOT"
stamp() { date '+%F %T'; }

grep -q "patience 0" "$CODE/dgx/run_combo.sh" || { echo "code_v4 is older than the v20 zip (no --patience 0); unpack it first"; exit 1; }
grep -q "args.patience" "$CODE/scripts/baselines/yolo_antiuav.py" || { echo "scripts/baselines/yolo_antiuav.py is the old version (no --patience flag): training would still stop early. Unpack the current zip first"; exit 1; }

if [[ ! -f runs/baseline_yolo11n_640_dyn_k8_zoom/postprocess_sweep.json ]]; then
    echo "$(stamp) start zoom (background)"
    rm -rf runs/baseline_yolo11n_640_dyn_k8_zoom
    DRISHTI_IMAGE=drishti-yolo:latest bash "$JOB" "$GPU" baseline_yolo11n_640_dyn_k8_zoom scripts/zoom_refine.py \
        --frames-root "$DATA" --dyn-gap 8 --base /workspace/runs/baseline_yolo11n_640_dyn_k8 \
        --out /workspace/runs/baseline_yolo11n_640_dyn_k8_zoom > /dev/null
fi

for q in c2 c0 c1; do
    echo "$(stamp) ===== queue $q"
    bash "$CODE/dgx/run_combo.sh" "$GPU" "$q" 2>&1 | grep -E "start|done|FAILED|skip|ALL DONE"
done

docker wait baseline_yolo11n_640_dyn_k8_zoom > /dev/null 2>&1 || true
grep -q "ALL DONE" runs/logs/baseline_yolo11n_640_dyn_k8_zoom.log 2>/dev/null && echo "$(stamp) done zoom" || echo "$(stamp) zoom FAILED (runs/logs/baseline_yolo11n_640_dyn_k8_zoom.log)"

echo "$(stamp) ALL DONE (night)"
python3 - <<'EOF'
import json, os, statistics as st
rows = [("gap 8 (reference)", "baseline_yolo11n_640_dyn_k8"), ("compensation", "baseline_yolo11n_640_dyn_k8_comp"),
        ("NWD", "baseline_yolo11n_640_dyn_k8_nwd"), ("comp + NWD (22 epochs, old)", "baseline_yolo11n_640_dyn_k8_comp_nwd"),
        ("comp + NWD, full 40 epochs", "baseline_yolo11n_640_comp_nwd_full"), ("comp + NWD c=0.5", "baseline_yolo11n_640_comp_nwd_c05"),
        ("comp + NWD c=2.0", "baseline_yolo11n_640_comp_nwd_c2"), ("comp + gap12 + NWD s42", "baseline_yolo11n_640_comp_k12_nwd"),
        ("comp + gap12 + NWD s1", "baseline_yolo11n_640_comp_k12_nwd_s1"), ("comp + gap12 + NWD s2", "baseline_yolo11n_640_comp_k12_nwd_s2"),
        ("comp + gap12 + NWD + P2", "baseline_yolov8n_p2_640_comp_k12_nwd"), ("zoom second look", "baseline_yolo11n_640_dyn_k8_zoom"),
        ("temporal anomaly", "baseline_yolo11n_640_k8_comp_tprof")]
print(f"{'run':30s} {'track':>7s} {'mAP50':>6s} {'mAP75':>6s}")
seeds = []
for label, r in rows:
    for f in ("postprocess_sweep_low.json", "postprocess_sweep.json"):
        p = os.path.join("runs", r, f)
        if os.path.exists(p):
            t = json.load(open(p))["chosen_on_val"]["test"]
            print(f"{label:30s} {100*t['antiuav_tracking_accuracy']:7.2f} {100*t['map50']:6.1f} {100*t['map75']:6.1f}")
            if "gap12 + NWD s" in label: seeds.append(100*t['antiuav_tracking_accuracy'])
            break
if len(seeds) > 1: print(f"final system over {len(seeds)} seeds: {st.mean(seeds):.2f} +- {st.stdev(seeds):.2f}")
EOF
