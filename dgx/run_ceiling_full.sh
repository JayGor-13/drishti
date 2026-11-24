#!/usr/bin/env bash
# The decisive measurement for the next direction (18 Oct): are the frames that no candidate finds really lost,
# or are they in the detector's LOW-score boxes? Same candidate dump as version 6, but with the full detector
# pool (10 boxes per frame at score >= 0.001, instead of the top 3 at >= 0.05), on frame-dynamics YOLO.
# If the complete-miss rate drops a lot (8.3% -> a few %), sequence decoding over low-score boxes can recover
# them (backlog idea 11). If it stays, the misses need new proposals, not better selection.
#
# val and test in parallel on one GPU (about 2 GB each, 4 to 6 h), then the histogram (minutes).
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   nohup bash code_v4/dgx/run_ceiling_full.sh 1 > runs/logs/run_ceiling_full.log 2>&1 &
set -euo pipefail

GPU="${1:-1}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
RUN=/workspace/runs/a410_focus_fixed_s42
DYN=/workspace/runs/baseline_yolo11n_640_dyn
CAND=/workspace/runs/v6_candidates_dyn_full
cd "$ROOT"
wait_for() { for n in "$@"; do docker wait "$n" > /dev/null 2>&1 || true; echo "$(date '+%F %T') finished $n"; done; }

for split in val test; do
    bash "$JOB" "$GPU" "ceil_full_dump_$split" -m drishti_v2.experiments.dump_candidates \
        --config configs/dgx_antiuav410_focus_fixed.yaml --checkpoint $RUN/joint/checkpoints/joint_best.pt \
        --frames-root "$DATA" --split $split --ext-topk 10 --ext-min-score 0.001 --ext-dets "$DYN/raw_{split}" --out $CAND
done
# the histogram on the frame-dynamics top-3 pool (round 7's dump) can run right away
bash "$JOB" "$GPU" analyze_candidates_dyn_test scripts/analyze_candidates.py --candidates /workspace/runs/v6_candidates_dyn \
    --split test --out /workspace/runs/v6_candidates_dyn/analysis_test.json
wait_for ceil_full_dump_val ceil_full_dump_test analyze_candidates_dyn_test
for split in val test; do
    bash "$JOB" "$GPU" "analyze_candidates_full_$split" scripts/analyze_candidates.py --candidates $CAND --split $split \
        --out "$CAND/analysis_$split.json"
done
wait_for analyze_candidates_full_val analyze_candidates_full_test
echo "$(date '+%F %T') ALL DONE"
echo "== top-3 pool (round 7):"; grep -a -A4 "^source" runs/logs/analyze_candidates_dyn_test.log || true
echo "== full pool:";            grep -a -A4 "^source" runs/logs/analyze_candidates_full_test.log || true
