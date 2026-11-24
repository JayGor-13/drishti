#!/usr/bin/env bash
# Decisive test (17 Oct): does our motion network still add anything once the detector itself sees motion?
# Same as version 5 / 6, but the appearance expert is the FRAME-DYNAMICS YOLO11n (baseline_yolo11n_640_dyn).
# Evaluation only (the small judge trains in minutes). About 5 to 6 h on one GPU.
#
#   v5dyn_fused_noref   our model + memory + 12 crops, frame-dynamics YOLO boxes as guided crops and output;
#                       threshold / one box / merging / smoothing chosen on val (like every other result)
#   fusion_sweep_dyn    calibrated fusion on those boxes, chosen on val
#   v6dyn_dump_*        candidates of val and test -> runs/v6_candidates_dyn
#   v6dyn_arbiter_*     judge fitted on half of val (mlp, attn); also reports the rules and the ceiling (oracle)
#
# Compare with frame-dynamics YOLO11n alone: 63.5 track, 87.8 mAP50, 43.4 mAP75 (test).
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   nohup bash code_v4/dgx/run_v6_dyn.sh 1 > runs/logs/run_v6_dyn.log 2>&1 &
set -euo pipefail

GPU="${1:-1}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
RUN=/workspace/runs/a410_focus_fixed_s42
CKPT=$RUN/joint/checkpoints/joint_best.pt
CFG=configs/dgx_antiuav410_focus_fixed.yaml
DYN=/workspace/runs/baseline_yolo11n_640_dyn
CAND=/workspace/runs/v6_candidates_dyn
cd "$ROOT"

for need in runs/baseline_yolo11n_640_dyn/raw_val runs/baseline_yolo11n_640_dyn/raw_test \
            runs/a410_focus_fixed_s42/joint/checkpoints/joint_best.pt; do
    [[ -e "$need" ]] || { echo "missing $need"; exit 1; }
done
wait_for() { for n in "$@"; do docker wait "$n" > /dev/null 2>&1 || true; echo "$(date '+%F %T') finished $n"; done; }

echo "$(date '+%F %T') start: v5 with frame-dynamics YOLO, and candidate dumps (in parallel)"
bash "$JOB" "$GPU" v5dyn_fused_noref scripts/run_all_stages.py --config $CFG --frames-root "$DATA" \
    --run-dir "$RUN" --schedule joint --eval-only --eval-split test --eval-tag v5dyn_fused_noref \
    "--seq-eval-args=--memory --num-crops 12 --external-dets $DYN/raw_{split} --ext-topk 3 --ext-min-score 0.05 --fuse-ext 1.0"
for split in val test; do
    bash "$JOB" "$GPU" "v6dyn_dump_$split" -m drishti_v2.experiments.dump_candidates --config $CFG --checkpoint $CKPT \
        --frames-root "$DATA" --split $split --ext-dets "$DYN/raw_{split}" --out $CAND
done

wait_for v6dyn_dump_val v6dyn_dump_test
echo "candidate files: val $(ls runs/v6_candidates_dyn/val | wc -l) (expect 90)  test $(ls runs/v6_candidates_dyn/test | wc -l) (expect 120)"
# the judge needs train/ only for --fit-on train; we fit on half of val (the valid protocol, see 17 Oct results)
mkdir -p runs/v6_candidates_dyn/train
for arch in mlp attn; do
    bash "$JOB" "$GPU" "v6dyn_arbiter_val_$arch" scripts/train_arbiter.py --candidates $CAND --fit-on val --arch $arch \
        --out "/workspace/runs/v6dyn_arbiter_val_$arch"
done

wait_for v5dyn_fused_noref
bash "$JOB" "$GPU" fusion_sweep_dyn scripts/sweep_fusion.py --frames-root "$DATA" \
    --val-raw  $RUN/sequence_eval_v5dyn_fused_noref/raw_val_no_tracker \
    --test-raw $RUN/sequence_eval_v5dyn_fused_noref/raw_test_no_tracker \
    --ext-dets "$DYN/raw_{split}" --out $RUN/fusion_sweep_dyn.json
wait_for v6dyn_arbiter_val_mlp v6dyn_arbiter_val_attn fusion_sweep_dyn

echo "$(date '+%F %T') ALL DONE. Frame-dynamics YOLO11n alone (test): track 63.51  mAP50 87.8  mAP75 43.4"
grep -h "^TEST" runs/logs/fusion_sweep_dyn.log runs/logs/v6dyn_arbiter_val_*.log || true
