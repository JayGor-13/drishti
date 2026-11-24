"""Choose the post-processing (NMS IoU, top-1, presence threshold) on VAL, then apply it once to TEST.

Works on the raw per-crop predictions that run_sequence_eval.py writes with ``--save-raw``,
so it needs no GPU and takes minutes. The test set is never used for choosing.

Usage
-----
    python scripts/sweep_postprocess.py --frames-root /workspace/data/antiuav410_frames \
        --val-raw  runs/a410_resnet18_s42/seq_raw/raw_val_no_tracker \
        --test-raw runs/a410_resnet18_s42/seq_raw/raw_test_no_tracker \
        --out runs/a410_resnet18_s42/postprocess_sweep.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drishti_v2.data.dataset import _box_exists, _read_antiuav_json, _target_from_antiuav_box  # noqa: E402
from drishti_v2.data.utils import list_image_files  # noqa: E402
from drishti_v2.evaluation.metrics import detection_metrics, temporal_detection_metrics  # noqa: E402
from drishti_v2.evaluation.postprocess import postprocess, smooth_sequence  # noqa: E402

KEYS = ["antiuav_tracking_accuracy", "map50", "map75", "precision", "recall", "f1", "false_positives_per_image"]


def load_split(frames_root: Path, split: str, modality: str, raw_dir: Path) -> list[tuple[list, list]]:
    data = []
    for seq_dir in sorted(p for p in (frames_root / split).iterdir() if p.is_dir()):
        raw_path = raw_dir / f"{seq_dir.name}.json"
        if not raw_path.exists():
            continue
        frames = list_image_files(seq_dir / modality)
        boxes, exists = _read_antiuav_json(seq_dir / f"{modality}.json")
        with Image.open(frames[0]) as im:
            size = (im.height, im.width)
        targets = []
        for idx in range(len(frames)):
            raw = boxes[idx] if idx < len(boxes) else []
            present = exists[idx] if idx < len(exists) else _box_exists(raw)
            targets.append(_target_from_antiuav_box(raw, present, size, "xywh"))
        pred = json.loads(raw_path.read_text(encoding="utf-8"))["frames"]
        raw = [pred.get(str(i), {"boxes": [], "scores": []}) for i in range(len(frames))]
        raw = [(torch.tensor(r["boxes"], dtype=torch.float32).view(-1, 4), torch.tensor(r["scores"], dtype=torch.float32).view(-1)) for r in raw]
        data.append((raw, targets))
    return data


def score(data, nms_iou, top1, threshold, smooth=False) -> dict[str, float]:
    rows = []
    for raw, targets in data:
        preds = []
        for boxes, scores in raw:
            b, s = postprocess(boxes, scores, nms_iou=nms_iou, top1=top1 or smooth)
            preds.append({"boxes": b, "scores": s})
        if smooth:
            preds = smooth_sequence(preds, threshold)
        m = detection_metrics(preds, targets, score_threshold=threshold)
        m.update(temporal_detection_metrics(preds, targets, threshold))
        rows.append(m)
    return {k: sum(r[k] for r in rows) / len(rows) for k in KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick post-processing on val, apply to test.")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--modality", default="infrared")
    parser.add_argument("--val-raw", required=True)
    parser.add_argument("--test-raw", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric", default="antiuav_tracking_accuracy", help="Val metric to maximise.")
    parser.add_argument("--thresholds", default="0.001,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5",
                        help="presence thresholds searched on val (before 21 Oct: 0.1 to 0.5 only)")
    args = parser.parse_args()
    root = Path(args.frames_root)
    val = load_split(root, "val", args.modality, Path(args.val_raw))
    test = load_split(root, "test", args.modality, Path(args.test_raw))
    print(f"loaded {len(val)} val and {len(test)} test sequences")

    # smoothing (idea C) only makes sense with one box per frame
    # thresholds below 0.1 matter: the Anti-UAV tracking accuracy punishes frames with no output (21 Oct fix)
    thresholds = [float(x) for x in args.thresholds.split(",")]
    grid = [(n, t, thr, False) for n, t, thr in itertools.product([None, 0.3], [False, True], thresholds)]
    grid += [(0.3, True, thr, True) for thr in thresholds]
    results = []
    for nms_iou, top1, thr, smooth in grid:
        r = score(val, nms_iou, top1, thr, smooth)
        results.append({"nms_iou": nms_iou, "top1": top1, "threshold": thr, "smooth": smooth, "val": r})
        print(f"val nms={nms_iou} top1={top1} smooth={smooth} thr={thr}: track {100*r['antiuav_tracking_accuracy']:.2f}  mAP50 {100*r['map50']:.1f}  P {100*r['precision']:.1f}  R {100*r['recall']:.1f}")
    # Tracking accuracy only looks at the top box, so NMS and top-1 tie with "all boxes". On a tie,
    # prefer the cleaner output (one box per frame, duplicates merged), which is what trackers report.
    best = max(results, key=lambda x: (round(x["val"][args.metric], 6), x["top1"], x["nms_iou"] is not None))
    baseline = next(x for x in results if x["nms_iou"] is None and not x["top1"] and x["threshold"] == 0.3)
    best["test"] = score(test, best["nms_iou"], best["top1"], best["threshold"], best["smooth"])
    baseline["test"] = score(test, None, False, 0.3)
    # the same choice WITHOUT smoothing, so the paper can show what smoothing adds
    if best["smooth"]:
        best["test_without_smoothing"] = score(test, best["nms_iou"], best["top1"], best["threshold"], False)
    out = {"chosen_on_val": best, "previous_default": baseline, "grid": results}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, row in [("previous default (no NMS, all boxes, thr 0.3)", baseline), ("chosen on val", best)]:
        t = row["test"]
        print(f"TEST {name}: nms={row['nms_iou']} top1={row['top1']} smooth={row.get('smooth', False)} thr={row['threshold']} -> "
              f"track {100*t['antiuav_tracking_accuracy']:.2f}  mAP50 {100*t['map50']:.1f}  mAP75 {100*t['map75']:.1f}  "
              f"P {100*t['precision']:.1f}  R {100*t['recall']:.1f}  FP/img {t['false_positives_per_image']:.3f}")


if __name__ == "__main__":
    main()
