#!/usr/bin/env bash
# Run the released YOLOFT-L checkpoint on the official XS-VID Detection test.
set -euo pipefail
DATA="${1:-/workspace/data/XS-VID-v2}"
OUT="${2:-/workspace/runs/small_objects/yoloft-reference}"
MODELS="${3:-/workspace/data/YOLOFT-XSVID-v2-weights}"
cd /opt/YOLOFT
test -f "$DATA/legacy_yoloft.yaml" || {
    echo "Prepare XS-VID with the benchmark image using --dataset xsvid --stage data first." >&2
    exit 1
}
mkdir -p "$OUT"
if [[ ! -f "$MODELS/vid.pt" ]]; then
    if [[ "$(basename "$MODELS")" != YOLOFT-XSVID-v2-weights ]]; then
        echo "Expected the released vid.pt at $MODELS/vid.pt" >&2
        exit 1
    fi
    python tools/xsvid/download_release.py --hub hf --component model --destination "$(dirname "$MODELS")"
fi
python tools/xsvid/eval_vid.py --config config/xsvid/yoloft-l-temporal.yaml \
    --weights "$MODELS/vid.pt" --data "$DATA/legacy_yoloft.yaml" --cfg config/train/default.yaml \
    --project "$OUT" --name vid --device 0
python tools/xsvid/score_detection.py --gt "$DATA/annotations/protocols/paper_detection_test.json" \
    --pred "$OUT/vid/predictions.json" --output "$OUT/metrics.json"
