#!/usr/bin/env bash
set -euo pipefail

MODE="--smoke"
RESULTS_DIR="results"
BATCH_SIZE="4"
EPOCHS=""
DEVICE=""
VIDEO_ROOT=""
VIDEO_SHARDS=""
ALL_VIDEO_SHARDS="0"
REQUIRE_REAL_VIDEOS="0"
KEEP_SHARD_ZIP="0"
CLEANUP_EXTRACTED_SHARDS="0"
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
    --video-root)
      VIDEO_ROOT="$2"
      shift 2
      ;;
    --require-real-videos)
      REQUIRE_REAL_VIDEOS="1"
      shift
      ;;
    --video-shards)
      VIDEO_SHARDS="$2"
      shift 2
      ;;
    --all-video-shards)
      ALL_VIDEO_SHARDS="1"
      shift
      ;;
    --keep-shard-zip)
      KEEP_SHARD_ZIP="1"
      shift
      ;;
    --cleanup-extracted-shards)
      CLEANUP_EXTRACTED_SHARDS="1"
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
if [[ -n "$VIDEO_ROOT" ]]; then
  VIDEO_ARGS=(--video-root "$VIDEO_ROOT")
fi
if [[ "$REQUIRE_REAL_VIDEOS" == "1" ]]; then
  VIDEO_ARGS+=(--require-real-videos)
fi
EPOCH_ARGS=()
if [[ -n "$EPOCHS" ]]; then
  EPOCH_ARGS=(--epochs "$EPOCHS")
fi
SHARD_ARGS=()
if [[ -n "$VIDEO_SHARDS" ]]; then
  SHARD_ARGS=(--video-shards "$VIDEO_SHARDS")
fi
if [[ "$ALL_VIDEO_SHARDS" == "1" ]]; then
  SHARD_ARGS+=(--all-video-shards)
fi
if [[ "$KEEP_SHARD_ZIP" == "1" ]]; then
  SHARD_ARGS+=(--keep-shard-zip)
fi
if [[ "$CLEANUP_EXTRACTED_SHARDS" == "1" ]]; then
  SHARD_ARGS+=(--cleanup-extracted-shards)
fi
RESUME_ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

python experiment.py "$MODE" --results-dir "$RESULTS_DIR" --batch-size "$BATCH_SIZE" "${EPOCH_ARGS[@]}" "${DEVICE_ARGS[@]}" "${VIDEO_ARGS[@]}" "${SHARD_ARGS[@]}" "${RESUME_ARGS[@]}"
