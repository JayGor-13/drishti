#!/usr/bin/env bash
# The full system for one (appearance eye, motion network) pair. Evaluation only, about 5 to 6 h on one GPU.
#   <variant>  which YOLO11n-640 run supplies the appearance boxes: dyn_k8, dyn_k8_s1, dyn_k8_s2, dyn, winner, ...
#   <net>      which trained motion network: a410_focus_fixed_s42 (default), a410_focus_fixed_s43, a410_joint448_s42, ...
#              (its own runs/<net>/config.yaml is used when present)
# Optional environment variables, for the ablation rows:
#   ABL_TAG     a name added to every output (e.g. nomem, crops8)
#   EVAL_FLAGS  our network's evaluation flags (default "--memory --num-crops 12")
#   DUMP_FLAGS  the candidate dump's flags (default "--num-crops 12"; add --no-memory to match EVAL_FLAGS)
#   SKIP_FUSED  1 = only candidates + ceiling + decoder (the ablation rows need no fused evaluation)
# Steps:
#   v5<id>_fused_noref   our network + memory + crops + that YOLO's boxes; val-chosen post-processing
#   fusion_sweep_<id>    calibrated fusion chosen on val
#   cand_<id>_{val,test} candidates with the FULL detector pool (10 boxes at >= 0.001)
#   ceiling_<id>_test    best-candidate histogram
#   decode_<id>          sequence decoder (lags 0, 5, 15), chosen on val, test scored once (CPU, ~20 min)
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   nohup bash code_v4/dgx/run_fusion_variant.sh 1 dyn_k8 > runs/logs/run_fusion_dyn_k8.log 2>&1 &
#   nohup bash code_v4/dgx/run_fusion_variant.sh 1 dyn_k8_s1 a410_focus_fixed_s43 > runs/logs/run_fusion_dyn_k8_s1_s43.log 2>&1 &
#   ABL_TAG=nomem EVAL_FLAGS="--num-crops 12" DUMP_FLAGS="--num-crops 12 --no-memory" SKIP_FUSED=1 \
#     nohup bash code_v4/dgx/run_fusion_variant.sh 3 dyn_k8 > runs/logs/run_fusion_dyn_k8_nomem.log 2>&1 &
set -euo pipefail

GPU="${1:-1}"
VARIANT="${2:-dyn_k8}"
NET="${3:-a410_focus_fixed_s42}"
ABL_TAG="${ABL_TAG:-}"
EVAL_FLAGS="${EVAL_FLAGS:---memory --num-crops 12}"
DUMP_FLAGS="${DUMP_FLAGS:---num-crops 12}"
SKIP_FUSED="${SKIP_FUSED:-0}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
RUN=/workspace/runs/$NET
CKPT=$RUN/joint/checkpoints/joint_best.pt
if [[ -f "runs/$NET/config.yaml" || -f "$ROOT/runs/$NET/config.yaml" ]]; then CFG=$RUN/config.yaml; else CFG=configs/dgx_antiuav410_focus_fixed.yaml; fi
DET=/workspace/runs/baseline_yolo11n_640_$VARIANT
if [[ "$NET" == "a410_focus_fixed_s42" ]]; then ID="$VARIANT"; else ID="${VARIANT}_${NET#a410_}"; ID="${ID/focus_fixed_/}"; fi
[[ -n "$ABL_TAG" ]] && ID="${ID}_$ABL_TAG"
CAND=/workspace/runs/candidates_$ID
TAG="v5${ID}_fused_noref"
cd "$ROOT"

for need in "runs/baseline_yolo11n_640_$VARIANT/raw_val" "runs/baseline_yolo11n_640_$VARIANT/raw_test" "runs/$NET/joint/checkpoints/joint_best.pt"; do
    [[ -e "$need" ]] || { echo "missing $need"; exit 1; }
done
wait_for() { for n in "$@"; do docker wait "$n" > /dev/null 2>&1 || true; echo "$(date '+%F %T') finished $n"; done; }

echo "$(date '+%F %T') start: $NET + $VARIANT (id $ID, config $CFG, eval '$EVAL_FLAGS', dump '$DUMP_FLAGS', skip fused $SKIP_FUSED)"
if [[ "$SKIP_FUSED" != "1" ]]; then
    bash "$JOB" "$GPU" "$TAG" scripts/run_all_stages.py --config $CFG --frames-root "$DATA" \
        --run-dir "$RUN" --schedule joint --eval-only --eval-split test --eval-tag "$TAG" \
        "--seq-eval-args=$EVAL_FLAGS --external-dets $DET/raw_{split} --ext-topk 3 --ext-min-score 0.05 --fuse-ext 1.0"
fi
for split in val test; do
    # shellcheck disable=SC2086
    bash "$JOB" "$GPU" "cand_${ID}_$split" -m drishti_v2.experiments.dump_candidates --config $CFG --checkpoint $CKPT \
        --frames-root "$DATA" --split $split --ext-topk 10 --ext-min-score 0.001 --ext-dets "$DET/raw_{split}" --out $CAND $DUMP_FLAGS
done
wait_for "cand_${ID}_val" "cand_${ID}_test"
bash "$JOB" "$GPU" "ceiling_${ID}_test" scripts/analyze_candidates.py --candidates $CAND --split test --out "$CAND/analysis_test.json"
bash "$JOB" "$GPU" "decode_${ID}" scripts/decode_candidates.py --candidates $CAND --out "$CAND/decode.json"

if [[ "$SKIP_FUSED" != "1" ]]; then
    wait_for "$TAG"
    bash "$JOB" "$GPU" "fusion_sweep_$ID" scripts/sweep_fusion.py --frames-root "$DATA" \
        --val-raw  "$RUN/sequence_eval_$TAG/raw_val_no_tracker" --test-raw "$RUN/sequence_eval_$TAG/raw_test_no_tracker" \
        --ext-dets "$DET/raw_{split}" --out "$RUN/fusion_sweep_$ID.json"
    wait_for "fusion_sweep_$ID"
fi
wait_for "ceiling_${ID}_test" "decode_${ID}"
echo "$(date '+%F %T') ALL DONE ($ID)"
grep -h "^TEST" "runs/logs/fusion_sweep_$ID.log" "runs/logs/decode_${ID}.log" 2>/dev/null || true
