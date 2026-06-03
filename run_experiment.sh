#!/usr/bin/env bash
set -euo pipefail

MODE="--smoke"
RESULTS_DIR="results"
BATCH_SIZE="4"
EPOCHS=""
DEVICE=""
METADATA_FILE=""
VIDEO_ROOT=""
REQUIRE_REAL_VIDEOS="0"
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
    --metadata-file)
      METADATA_FILE="$2"
      shift 2
      ;;
    --video-root)
      VIDEO_ROOT="$2"
      shift 2
      ;;
    --require-real-videos)
      REQUIRE_REAL_VIDEOS="1"
      shift
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
VIDEO_ARGS=()
if [[ -n "$METADATA_FILE" ]]; then
  VIDEO_ARGS+=(--metadata-file "$METADATA_FILE")
fi
if [[ -n "$VIDEO_ROOT" ]]; then
  VIDEO_ARGS+=(--video-root "$VIDEO_ROOT")
fi
if [[ "$REQUIRE_REAL_VIDEOS" == "1" ]]; then
  VIDEO_ARGS+=(--require-real-videos)
fi
EPOCH_ARGS=()
if [[ -n "$EPOCHS" ]]; then
  EPOCH_ARGS=(--epochs "$EPOCHS")
fi
RESUME_ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

python experiment.py "$MODE" --results-dir "$RESULTS_DIR" --batch-size "$BATCH_SIZE" "${EPOCH_ARGS[@]}" "${DEVICE_ARGS[@]}" "${VIDEO_ARGS[@]}" "${RESUME_ARGS[@]}"
