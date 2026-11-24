#!/usr/bin/env bash
# 21 Oct fix: re-score EVERY detector run with saved val/test predictions, fairly.
#   postprocess_sweep_low.json   the usual val-chosen post-processing, now also trying presence thresholds below
#                                0.1 (0.001 ... 0.05): the Anti-UAV tracking accuracy punishes frames with no output,
#                                so the old 0.1 floor handicapped every method
#   decode_raw.json              the sequence decoder applied to that detector ALONE (its top 10 boxes per frame),
#                                lags 0 and 15, chosen on val, test once
# CPU only (the GPU number is only passed to the job runner). Runs PARALLEL jobs at a time. Finished runs are skipped.
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   nohup bash code_v4/dgx/run_rescore.sh 1 > runs/logs/run_rescore.log 2>&1 &
set -uo pipefail

GPU="${1:-1}"
PARALLEL="${PARALLEL:-8}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
cd "$ROOT"

pending=()
for d in runs/baseline_*/; do
    d="${d%/}"; name="${d#runs/}"
    [[ -d "$d/raw_val" && -d "$d/raw_test" ]] || continue
    [[ $(ls "$d/raw_test" | wc -l) -ge 120 && $(ls "$d/raw_val" | wc -l) -ge 90 ]] || { echo "skip $name (predictions incomplete)"; continue; }
    pending+=("$name")
done
echo "$(date '+%F %T') ${#pending[@]} detector runs to re-score: ${pending[*]}"

running() { docker ps --format '{{.Names}}' | grep -cE '^(rescore|decraw)_' || true; }
throttle() { while [[ $(running) -ge $PARALLEL ]]; do sleep 30; done; }

for name in "${pending[@]}"; do
    short="${name#baseline_}"
    if [[ ! -f "runs/$name/postprocess_sweep_low.json" ]]; then
        throttle
        bash "$JOB" "$GPU" "rescore_$short" scripts/sweep_postprocess.py --frames-root "$DATA" \
            --val-raw "/workspace/runs/$name/raw_val" --test-raw "/workspace/runs/$name/raw_test" \
            --out "/workspace/runs/$name/postprocess_sweep_low.json" > /dev/null
    fi
    if [[ ! -f "runs/$name/decode_raw.json" ]]; then
        throttle
        bash "$JOB" "$GPU" "decraw_$short" scripts/decode_candidates.py --frames-root "$DATA" \
            --raw-val "/workspace/runs/$name/raw_val" --raw-test "/workspace/runs/$name/raw_test" --lags 0,15 \
            --out "/workspace/runs/$name/decode_raw.json" > /dev/null
    fi
done
while [[ $(running) -gt 0 ]]; do sleep 60; done
echo "$(date '+%F %T') ALL DONE"
python3 - <<'EOF'
import json, glob, os
print(f"{'run':34s} {'old (thr>=0.1)':>16s} {'new sweep':>10s} {'decoder lag0':>13s} {'decoder lag15':>14s}   (test tracking accuracy)")
for d in sorted(glob.glob("runs/baseline_*")):
    def g(f, *keys):
        try:
            x = json.load(open(os.path.join(d, f)))
            for k in keys: x = x[k]
            return f"{100 * x:.2f}"
        except Exception:
            return "-"
    print(f"{os.path.basename(d):34s} {g('postprocess_sweep.json','chosen_on_val','test','antiuav_tracking_accuracy'):>16s} "
          f"{g('postprocess_sweep_low.json','chosen_on_val','test','antiuav_tracking_accuracy'):>10s} "
          f"{g('decode_raw.json','decoder_lag0','test','antiuav_tracking_accuracy'):>13s} {g('decode_raw.json','decoder_lag15','test','antiuav_tracking_accuracy'):>14s}")
EOF
