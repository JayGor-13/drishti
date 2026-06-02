#!/usr/bin/env bash
set -euo pipefail

MODE="--smoke"
RESULTS_DIR="results"
BATCH_SIZE="4"
DEVICE=""
VIDEO_ROOT=""

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
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --video-root)
      VIDEO_ROOT="$2"
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
if [[ -n "$VIDEO_ROOT" ]]; then
  VIDEO_ARGS=(--video-root "$VIDEO_ROOT")
fi

python experiment.py "$MODE" --results-dir "$RESULTS_DIR" --batch-size "$BATCH_SIZE" "${DEVICE_ARGS[@]}" "${VIDEO_ARGS[@]}"
