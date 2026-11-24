#!/usr/bin/env bash
# Start one DRISHTI job in its own Docker container, in the background, on one GPU.
#
# Usage:   bash code/dgx/run_job.sh <gpu-id> <job-name> <python arguments...>
# Example: bash code/dgx/run_job.sh 0 a410_resnet18 scripts/run_all_stages.py \
#              --config configs/dgx_antiuav410_enc_resnet18.yaml ...
#
# Folder layout on the DGX (ROOT is mounted as /workspace inside the container):
#   $ROOT/code        the repository        -> /workspace/code
#   $ROOT/data        datasets              -> /workspace/data
#   $ROOT/pretrained  resnet18 weights      -> /workspace/pretrained
#   $ROOT/runs        checkpoints and logs  -> /workspace/runs
# Log of the job: $ROOT/runs/logs/<job-name>.log
#
# Re-running the same command after a crash or `docker stop` resumes from the last saved epoch.
# CPU use is capped (shared machine): 16 cores and 4 threads per process by default;
# override with DRISHTI_CPUS=32 DRISHTI_THREADS=8 bash code/dgx/run_job.sh ...
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <gpu-id> <job-name> <python arguments...>" >&2
    exit 1
fi

GPU="$1"
NAME="$2"
shift 2
ROOT="${DRISHTI_ROOT:-/raid/scratch/$USER/drishti}"
IMAGE="${DRISHTI_IMAGE:-drishti:latest}"
# Code folder under ROOT. Use e.g. DRISHTI_CODE=code_v4 to run a new version while jobs started
# from the old folder are still running (they keep using the old code).
CODE="${DRISHTI_CODE:-code}"

mkdir -p "$ROOT/runs/logs"
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
        echo "A container named $NAME is already running. Stop it first: docker stop $NAME" >&2
        exit 1
    fi
    docker rm "$NAME" > /dev/null   # finished or stopped container with the same name
fi

ARGS=$(printf '%q ' "$@")
docker run -d \
    --name "$NAME" \
    --gpus "\"device=$GPU\"" \
    --ipc=host \
    --cpus="${DRISHTI_CPUS:-16}" \
    -e OMP_NUM_THREADS="${DRISHTI_THREADS:-4}" \
    -e MKL_NUM_THREADS="${DRISHTI_THREADS:-4}" \
    -u "$(id -u):$(id -g)" \
    -v "$ROOT:/workspace" \
    -w "/workspace/$CODE" \
    "$IMAGE" \
    bash -c "python -u $ARGS >> /workspace/runs/logs/$NAME.log 2>&1"

echo "started $NAME on GPU $GPU (code: $ROOT/$CODE)"
echo "log:    $ROOT/runs/logs/$NAME.log   (follow with: tail -f $ROOT/runs/logs/$NAME.log)"
echo "status: docker ps --filter name=$NAME"
