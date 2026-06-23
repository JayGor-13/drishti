#!/usr/bin/env bash
set -euo pipefail

MODE="--smoke"
RESULTS_DIR="results"
BATCH_SIZE="2"
EPOCHS=""
DEVICE=""
TRAIN_IMAGE_ROOT=""
TRAIN_ANN_FILE=""
VAL_IMAGE_ROOT=""
VAL_ANN_FILE=""
STAGE="all"
RESUME_CHECKPOINT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      MODE="--full"
      shift
      ;;
    --smoke)
      MODE="--smoke"
      shift
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --train-image-root)
      TRAIN_IMAGE_ROOT="$2"
      shift 2
      ;;
    --train-ann-file)
      TRAIN_ANN_FILE="$2"
      shift 2
      ;;
    --val-image-root)
      VAL_IMAGE_ROOT="$2"
      shift 2
      ;;
    --val-ann-file)
      VAL_ANN_FILE="$2"
      shift 2
      ;;
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --resume-checkpoint)
      RESUME_CHECKPOINT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

DEVICE_ARGS=()
if [[ -n "$DEVICE" ]]; then
  DEVICE_ARGS=(--device "$DEVICE")
fi
DATA_ARGS=()
if [[ -n "$TRAIN_IMAGE_ROOT" ]]; then
  DATA_ARGS+=(--train-image-root "$TRAIN_IMAGE_ROOT")
fi
if [[ -n "$TRAIN_ANN_FILE" ]]; then
  DATA_ARGS+=(--train-ann-file "$TRAIN_ANN_FILE")
fi
if [[ -n "$VAL_IMAGE_ROOT" ]]; then
  DATA_ARGS+=(--val-image-root "$VAL_IMAGE_ROOT")
fi
if [[ -n "$VAL_ANN_FILE" ]]; then
  DATA_ARGS+=(--val-ann-file "$VAL_ANN_FILE")
fi
EPOCH_ARGS=()
if [[ -n "$EPOCHS" ]]; then
  EPOCH_ARGS=(--epochs "$EPOCHS")
fi
RESUME_ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

python experiment.py "$MODE" --stage "$STAGE" --results-dir "$RESULTS_DIR" --batch-size "$BATCH_SIZE" "${EPOCH_ARGS[@]}" "${DEVICE_ARGS[@]}" "${DATA_ARGS[@]}" "${RESUME_ARGS[@]}"
