#!/usr/bin/env bash
# Everything that is left, on our GPUs (20 Oct; the student runs nothing). Four modes:
#
#   ablate  the "inside our network" ablation rows (no memory, 8 crops, no Chain-of-Focus) + system cost, ~6 h
#   smoke   about 30-45 min on one GPU: each new baseline end to end on 2 sequences, 1 short epoch, to catch
#           install or format problems before the long runs. Run this first and check the summary at the end.
#   gpu3    DINO-4scale R50, Deformable-DETR R50 (MMDetection, 12 epochs each), then CHAL (CVPR'26)  ~ 2 days
#   gpu1    full system with seeds: (gap-8 YOLO seed 1 + our network seed 43), (seed 2 + seed 44), then SSTNet
#                                                                                                    ~ 1.5 days
# Every finished piece is skipped on a re-run. Each job has its own log in runs/logs/.
#
# Usage (on the DGX, from /raid/scratch/$USER/drishti):
#   nohup bash code_v4/dgx/run_remaining.sh smoke 1 > runs/logs/run_remaining_smoke.log 2>&1 &
#   nohup bash code_v4/dgx/run_remaining.sh gpu3 3  > runs/logs/run_remaining_gpu3.log 2>&1 &
#   nohup bash code_v4/dgx/run_remaining.sh gpu1 1  > runs/logs/run_remaining_gpu1.log 2>&1 &
#   nohup bash code_v4/dgx/run_remaining.sh ablate 3 > runs/logs/run_remaining_ablate.log 2>&1 &
set -euo pipefail

MODE="${1:?mode: smoke | gpu3 | gpu1 | ablate}"
GPU="${2:-1}"
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
export DRISHTI_CODE="${DRISHTI_CODE:-code_v4}"
CODE="$ROOT/$DRISHTI_CODE"
JOB="$CODE/dgx/run_job.sh"
DATA=/workspace/data/antiuav410_frames
cd "$ROOT"

ensure_image() {  # ensure_image <name> <dockerfile>
    if ! docker image inspect "$1:latest" > /dev/null 2>&1; then
        echo "$(date '+%F %T') building image $1 (a few minutes)"
        docker build -t "$1:latest" -f "$CODE/dgx/$2" "$CODE/dgx"
    fi
}
run_and_wait() {  # run_and_wait <image> <name> <python args...>
    local image="$1" name="$2"; shift 2
    if docker ps --format '{{.Names}}' | grep -qx "$name"; then echo "$name already running; waiting"; else
        echo "$(date '+%F %T') start $name"
        DRISHTI_IMAGE="$image:latest" bash "$JOB" "$GPU" "$name" "$@"
    fi
    docker wait "$name" > /dev/null 2>&1 || true
    if grep -q "ALL DONE" "runs/logs/$name.log"; then echo "$(date '+%F %T') done $name"; else echo "$(date '+%F %T') FAILED $name (see runs/logs/$name.log)"; fi
}
done_already() { [[ -f "runs/$1/cost.json" ]] && { echo "skip $1 (done)"; return 0; } || return 1; }

case "$MODE" in
smoke)
    ensure_image drishti-mmdet Dockerfile.mmdet
    ensure_image drishti-istd Dockerfile.istd
    run_and_wait drishti-mmdet smoke_dino scripts/baselines/mmdet_antiuav.py --stage all --model dino --frames-root "$DATA" \
        --coco-root /workspace/data/coco_a410_smoke --out /workspace/runs/smoke_dino --epochs 1 --train-stride 400 --val-stride 400 --max-seqs 2
    run_and_wait drishti-istd smoke_chal scripts/baselines/istd_antiuav.py --stage all --repo chal --frames-root "$DATA" \
        --view /workspace/data/istd_a410_smoke --out /workspace/runs/smoke_chal --epochs 1 --train-stride 400 --val-stride 400 --max-seqs 2
    run_and_wait drishti-istd smoke_sstnet scripts/baselines/istd_antiuav.py --stage all --repo sstnet --letterbox 1 --frames-root "$DATA" \
        --view /workspace/data/istd_a410_smoke --out /workspace/runs/smoke_sstnet --epochs 1 --train-stride 400 --val-stride 400 --max-seqs 2
    echo "== smoke summary"; grep -l "ALL DONE" runs/logs/smoke_*.log || true
    for n in smoke_dino smoke_chal smoke_sstnet; do grep -q "ALL DONE" "runs/logs/$n.log" || { echo "-- $n failed, last lines:"; tail -n 15 "runs/logs/$n.log"; }; done
    ;;
gpu3)
    ensure_image drishti-mmdet Dockerfile.mmdet
    ensure_image drishti-istd Dockerfile.istd
    for m in dino deformable_detr; do
        done_already "baseline_${m}_r50" || run_and_wait drishti-mmdet "baseline_${m}_r50" scripts/baselines/mmdet_antiuav.py --stage all \
            --model "$m" --frames-root "$DATA" --coco-root /workspace/data/coco_a410 --out "/workspace/runs/baseline_${m}_r50"
    done
    done_already baseline_chal || run_and_wait drishti-istd baseline_chal scripts/baselines/istd_antiuav.py --stage all --repo chal \
        --frames-root "$DATA" --view /workspace/data/istd_a410 --out /workspace/runs/baseline_chal
    ;;
gpu1)
    for pair in "dyn_k8_s1 a410_focus_fixed_s43" "dyn_k8_s2 a410_focus_fixed_s44"; do
        set -- $pair
        id="${1}_${2##*_}"
        if [[ -f "runs/candidates_$id/decode.json" ]]; then echo "skip system $id (done)"; continue; fi
        if [[ ! -f "runs/$2/joint/checkpoints/joint_best.pt" || ! -d "runs/baseline_yolo11n_640_$1/raw_test" ]]; then
            echo "skip system $id: $2 or baseline_yolo11n_640_$1 not finished"; continue
        fi
        echo "$(date '+%F %T') full system $id"
        bash "$CODE/dgx/run_fusion_variant.sh" "$GPU" "$1" "$2" > "runs/logs/run_fusion_$id.log" 2>&1 || echo "system $id FAILED"
    done
    # plain YOLO26n at 640, so "gap 8 on YOLO26n" is compared at the same resolution (only 512 existed)
    done_already baseline_yolo26n_640 || run_and_wait drishti-yolo26 baseline_yolo26n_640 scripts/baselines/yolo_antiuav.py --stage all \
        --model yolo26n.pt --imgsz 640 --epochs 40 --frames-root "$DATA" --yolo-root /workspace/data/yolo_a410 --out /workspace/runs/baseline_yolo26n_640
    ensure_image drishti-istd Dockerfile.istd
    done_already baseline_sstnet || run_and_wait drishti-istd baseline_sstnet scripts/baselines/istd_antiuav.py --stage all --repo sstnet \
        --letterbox 1 --frames-root "$DATA" --view /workspace/data/istd_a410 --out /workspace/runs/baseline_sstnet
    ;;
ablate)
    # "inside our network" rows of the ablation table, each the full system with one part removed (decoder output,
    # chosen on val), all three in parallel (~2 GB each, about 6 h), plus the cost of the full system (~20 min)
    row() {  # row <id> <net> <eval flags> <dump flags>
        if [[ -f "runs/candidates_$1/decode.json" ]]; then echo "skip ablation $1 (done)"; return; fi
        ABL_TAG="$5" EVAL_FLAGS="$3" DUMP_FLAGS="$4" SKIP_FUSED=1 \
            bash "$CODE/dgx/run_fusion_variant.sh" "$GPU" dyn_k8 "$2" > "runs/logs/run_ablation_$1.log" 2>&1 || echo "ablation $1 FAILED"
    }
    row dyn_k8_nomem          a410_focus_fixed_s42 "--num-crops 12"         "--num-crops 12 --no-memory" nomem &
    row dyn_k8_crops8         a410_focus_fixed_s42 "--memory --num-crops 8" "--num-crops 8"              crops8 &
    row dyn_k8_joint448_s42   a410_joint448_s42    "--memory --num-crops 12" "--num-crops 12"            "" &
    if [[ ! -f runs/system_cost.json ]]; then
        run_and_wait drishti-yolo system_cost scripts/time_system.py --frames-root "$DATA" \
            --yolo-run /workspace/runs/baseline_yolo11n_640_dyn_k8 --dyn-gap 8 \
            --config /workspace/runs/a410_focus_fixed_s42/config.yaml \
            --checkpoint /workspace/runs/a410_focus_fixed_s42/joint/checkpoints/joint_best.pt --out /workspace/runs/system_cost.json
    fi
    wait
    ;;
*) echo "unknown mode $MODE"; exit 1 ;;
esac
echo "$(date '+%F %T') ALL DONE ($MODE)"
