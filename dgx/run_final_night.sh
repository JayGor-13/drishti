#!/usr/bin/env bash
# One command for the last GPU work (23 Oct): the ablation of the final system (4 runs in parallel, ~4 h), then the
# CPU re-scoring that adds the wide-threshold and decoder rows to every run (~1-2 h), then the final tables.
# Safe to re-run: finished pieces are skipped.
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   nohup bash code_v4/dgx/run_final_night.sh 1 > runs/logs/run_final_night.log 2>&1 &
set -uo pipefail

GPU="${1:-1}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
CODE="$ROOT/$DRISHTI_CODE"
cd "$ROOT"
stamp() { date '+%F %T'; }

grep -q "pick-weights" "$CODE/scripts/baselines/yolo_antiuav.py" || { echo "old code in code_v4; unpack the current zip first"; exit 1; }

echo "$(stamp) ===== ablation of the final system"
bash "$CODE/dgx/run_ablation_final.sh" "$GPU" 2>&1 | grep -E "start|done|FAILED|skip|ALL DONE"

echo "$(stamp) ===== re-scoring every run (wide thresholds + decoder rows, CPU)"
bash "$CODE/dgx/run_rescore.sh" "$GPU" 2>&1 | tail -n 60

echo "$(stamp) ALL DONE (final night)"
python3 - <<'EOF'
import json, os, glob
def get(d, f, *keys):
    try:
        x = json.load(open(os.path.join("runs", d, f)))
        for k in keys: x = x[k]
        return x
    except Exception:
        return None
print(f"\n{'run':40s} {'fair':>6s} {'+dec0':>6s} {'+dec15':>7s} {'mAP50':>6s} {'mAP75':>6s}")
for d in sorted(os.path.basename(p) for p in glob.glob("runs/baseline_*")):
    t = get(d, "postprocess_sweep_low.json", "chosen_on_val", "test") or get(d, "postprocess_sweep.json", "chosen_on_val", "test")
    if not t: continue
    d0 = get(d, "decode_raw.json", "decoder_lag0", "test", "antiuav_tracking_accuracy")
    d15 = get(d, "decode_raw.json", "decoder_lag15", "test", "antiuav_tracking_accuracy")
    f = lambda v: f"{100*v:6.2f}" if v is not None else "     -"
    print(f"{d:40s} {f(t['antiuav_tracking_accuracy'])} {f(d0)} {f(d15):>7s} {100*t['map50']:6.1f} {100*t['map75']:6.1f}")
EOF
