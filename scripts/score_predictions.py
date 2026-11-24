"""Score ANY method's saved predictions with exactly the metrics of run_sequence_eval.

Use this for every baseline so its numbers are computed by the same code as ours.

Prediction format: one JSON file per sequence, named ``<sequence_name>.json``::

    {"frames": {"0": {"boxes": [[x, y, w, h], ...], "scores": [0.91, ...]},
                "1": {"boxes": [], "scores": []}, ...}}

* boxes in ORIGINAL image pixels, top-left x, y, width, height (Anti-UAV / COCO convention);
  pass ``--normalized`` if they are already cx, cy, w, h in [0, 1].
* keep ALL boxes with their scores (do not threshold); mAP needs the full ranking.
* a missing frame counts as "no detection".

Usage
-----
    python scripts/score_predictions.py --frames-root ~/dataset_frames --split test \
        --pred-dir ~/baselines/deim/preds_test --name DEIM-D-FINE-S --out ~/baselines/deim/score_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drishti_v2.data.dataset import _box_exists, _read_antiuav_json, _target_from_antiuav_box  # noqa: E402
from drishti_v2.data.utils import list_image_files, xywh_to_cxcywh  # noqa: E402
from drishti_v2.evaluation.metrics import detection_metrics, temporal_detection_metrics  # noqa: E402


def score_sequence(seq_dir: Path, modality: str, pred: dict, normalized: bool, threshold: float) -> dict:
    frames = list_image_files(seq_dir / modality)
    boxes_gt, exists = _read_antiuav_json(seq_dir / f"{modality}.json")
    with Image.open(frames[0]) as im:
        size = (im.height, im.width)
    predictions, targets = [], []
    per_frame = pred.get("frames", {})
    for idx in range(len(frames)):
        raw = boxes_gt[idx] if idx < len(boxes_gt) else []
        present = exists[idx] if idx < len(exists) else _box_exists(raw)
        targets.append(_target_from_antiuav_box(raw, present, size, "xywh"))
        item = per_frame.get(str(idx), {"boxes": [], "scores": []})
        boxes = torch.tensor(item.get("boxes", []), dtype=torch.float32).view(-1, 4)
        scores = torch.tensor(item.get("scores", []), dtype=torch.float32).view(-1)
        if boxes.numel() and not normalized:
            boxes = xywh_to_cxcywh(boxes, image_size=size)
        predictions.append({"boxes": boxes, "scores": scores})
    metrics = detection_metrics(predictions, targets, score_threshold=threshold)
    metrics.update(temporal_detection_metrics(predictions, targets, threshold))
    metrics["frames"] = len(frames)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Score saved predictions with the DRISHTI sequence metrics.")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--modality", default="infrared")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--name", required=True, help="Method name written into the output.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold for P/R/F1 and tracking accuracy.")
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root, pred_dir = Path(args.frames_root).expanduser(), Path(args.pred_dir).expanduser()
    rows = []
    for seq_dir in sorted(p for p in (root / args.split).iterdir() if p.is_dir()):
        if not (seq_dir / f"{args.modality}.json").exists():
            continue
        pred_path = pred_dir / f"{seq_dir.name}.json"
        pred = json.loads(pred_path.read_text(encoding="utf-8")) if pred_path.exists() else {"frames": {}}
        if not pred_path.exists():
            print(f"WARNING: no predictions for {seq_dir.name}; scored as all-missed")
        rows.append({"sequence": seq_dir.name, **score_sequence(seq_dir, args.modality, pred, args.normalized, args.threshold)})
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    summary = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    summary.update({"method": args.name, "split": args.split, "num_sequences": len(rows), "threshold": args.threshold})
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_sequence": rows}, indent=2), encoding="utf-8")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
