#!/usr/bin/env bash
# Steps 1 and 2 of 17 Oct, one after the other, on one GPU. Safe to leave running; safe to re-run
# (every step skips what is already finished).
#
#   step 1  fusion_sweep     calibrated version-5 fusion chosen on val (CPU only, minutes)
#   step 2a v6_yolo_train    YOLO11n-640 predictions on the TRAIN split (needed to teach the arbiter)
#   step 2b v6_dump_*        our model + YOLO + memory run frame by frame; every candidate box and its
#                            cues saved to runs/v6_candidates/{train,val,test}   (train: every 2nd sequence)
#   step 2c v6_arbiter_*     the arbiter is trained four ways (fit on train / on half of val; mlp / attn);
#                            threshold chosen on val, test scored once
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   nohup bash code_v4/dgx/run_v6.sh 1 > runs/logs/run_v6.log 2>&1 &
#   tail -f runs/logs/run_v6.log
set -euo pipefail

GPU="${1:-1}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
JOB="$ROOT/$DRISHTI_CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
RUN=/workspace/runs/a410_focus_fixed_s42
CKPT=$RUN/joint/checkpoints/joint_best.pt
CFG=configs/dgx_antiuav410_focus_fixed.yaml
YOLO_OUT=/workspace/runs/baseline_yolo11n_640
CAND=/workspace/runs/v6_candidates
cd "$ROOT"

for need in runs/baseline_yolo11n_640/train/weights/best.pt runs/baseline_yolo11n_640/raw_val runs/baseline_yolo11n_640/raw_test \
            runs/a410_focus_fixed_s42/joint/checkpoints/joint_best.pt \
            runs/a410_focus_fixed_s42/sequence_eval_v5_fused_noref/raw_val_no_tracker; do
    [[ -e "$need" ]] || { echo "missing $need"; exit 1; }
done

wait_for() { for n in "$@"; do docker wait "$n" > /dev/null 2>&1 || true; echo "$(date '+%F %T') finished $n"; done; }
done_count() { ls "$ROOT/runs/v6_candidates/$1" 2>/dev/null | wc -l; }

echo "$(date '+%F %T') step 1: fusion sweep"
bash "$JOB" "$GPU" fusion_sweep scripts/sweep_fusion.py --frames-root "$DATA" \
    --val-raw  $RUN/sequence_eval_v5_fused_noref/raw_val_no_tracker \
    --test-raw $RUN/sequence_eval_v5_fused_noref/raw_test_no_tracker \
    --ext-dets "$YOLO_OUT/raw_{split}" --out $RUN/fusion_sweep.json

echo "$(date '+%F %T') step 2a: YOLO on train; 2b: candidates for val and test (in parallel)"
DRISHTI_IMAGE=drishti-yolo:latest bash "$JOB" "$GPU" v6_yolo_train scripts/baselines/yolo_antiuav.py --stage predict \
    --predict-splits train --model yolo11n.pt --imgsz 640 --frames-root "$DATA" --out "$YOLO_OUT"
for split in val test; do
    bash "$JOB" "$GPU" "v6_dump_$split" -m drishti_v2.experiments.dump_candidates --config $CFG --checkpoint $CKPT \
        --frames-root "$DATA" --split $split --ext-dets "$YOLO_OUT/raw_{split}" --out $CAND
done
wait_for v6_yolo_train

echo "$(date '+%F %T') step 2b: candidates for train (every 2nd sequence)"
bash "$JOB" "$GPU" v6_dump_train -m drishti_v2.experiments.dump_candidates --config $CFG --checkpoint $CKPT \
    --frames-root "$DATA" --split train --seq-stride 2 --ext-dets "$YOLO_OUT/raw_{split}" --out $CAND
wait_for fusion_sweep v6_dump_val v6_dump_test v6_dump_train
echo "candidate files: train $(done_count train) (expect 100)  val $(done_count val) (expect 90)  test $(done_count test) (expect 120)"

echo "$(date '+%F %T') step 2c: arbiters"
for fit in train val; do
    for arch in attn mlp; do
        bash "$JOB" "$GPU" "v6_arbiter_${fit}_$arch" scripts/train_arbiter.py --candidates $CAND --fit-on $fit --arch $arch \
            --out "/workspace/runs/v6_arbiter_${fit}_$arch"
    done
done
wait_for v6_arbiter_train_attn v6_arbiter_train_mlp v6_arbiter_val_attn v6_arbiter_val_mlp

echo "$(date '+%F %T') ALL DONE. Results:"
for f in runs/a410_focus_fixed_s42/fusion_sweep.json runs/v6_arbiter_*/arbiter_results.json; do echo "  $f"; done
grep -h "^TEST" runs/logs/fusion_sweep.log runs/logs/v6_arbiter_*.log || true
