"""Step 1 of 17 Oct: calibrated fusion of our boxes and a dense detector's boxes, chosen on VAL.

Works on the raw predictions already saved by the version-5 fused run (`--save-raw`, no tracker):
each frame's list holds our model's crop boxes first and then the detector's boxes (the detector's
predictions are re-read here to know how many of the last boxes are the detector's). No GPU needed.

Searched on val (tracking accuracy, ties broken by mAP@75), applied once to test:
  alpha      our boxes' scores are multiplied by alpha (detector scores unchanged): 1.0 = plain v5 fusion
  prefer     when our winning box overlaps a detector box (IoU >= 0.3), output the DETECTOR's box
             coordinates with the higher of the two scores (the detector's boxes are tighter)
  threshold  presence threshold
(duplicate merging 0.3 and one box per frame, as chosen for every model before)

Usage:
  python scripts/sweep_fusion.py --frames-root /workspace/data/antiuav410_frames \
     --val-raw  /workspace/runs/a410_focus_fixed_s42/sequence_eval_v5_fused_noref/raw_val_no_tracker \
     --test-raw /workspace/runs/a410_focus_fixed_s42/sequence_eval_v5_fused_noref/raw_test_no_tracker \
     --ext-dets "/workspace/runs/baseline_yolo11n_640/raw_{split}" \
     --out /workspace/runs/a410_focus_fixed_s42/fusion_sweep.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drishti_v2.data.utils import box_iou  # noqa: E402
from drishti_v2.evaluation.metrics import detection_metrics, temporal_detection_metrics  # noqa: E402
from drishti_v2.evaluation.postprocess import nms  # noqa: E402
from sweep_postprocess import KEYS, load_split  # noqa: E402


def ext_counts(ext_dir: Path, seq: str, n_frames: int, topk: int, min_score: float) -> list[int]:
    path = ext_dir / f"{seq}.json"
    frames = json.loads(path.read_text(encoding="utf-8"))["frames"] if path.exists() else {}
    out = []
    for i in range(n_frames):
        scores = frames.get(str(i), {"scores": []})["scores"]
        out.append(min(topk, sum(1 for s in scores if s >= min_score)))
    return out


def load(frames_root: Path, split: str, raw_dir: Path, ext_dir: Path, topk: int, min_score: float, modality: str):
    data = load_split(frames_root, split, modality, raw_dir)
    names = sorted(p.stem for p in raw_dir.glob("*.json"))
    seq_names = [p.name for p in sorted((frames_root / split).iterdir()) if p.is_dir() and p.name in set(names)]
    out = []
    for (raw, targets), name in zip(data, seq_names):
        counts = ext_counts(ext_dir, name, len(raw), topk, min_score)
        origin = []
        for (boxes, _), n in zip(raw, counts):
            o = torch.zeros(len(boxes), dtype=torch.bool)
            if n:
                o[-n:] = True  # the detector's boxes are appended last
            origin.append(o)
        out.append((raw, origin, targets))
    return out


def fuse(boxes, scores, is_ext, alpha, prefer):
    if boxes.numel() == 0:
        return boxes, scores
    scores = torch.where(is_ext, scores, scores * alpha)
    keep_boxes, keep_scores = nms(boxes, scores, 0.3)
    if keep_scores.numel() == 0:
        return keep_boxes, keep_scores
    best = int(keep_scores.argmax())
    box, score = keep_boxes[best : best + 1].clone(), keep_scores[best : best + 1].clone()
    if prefer and is_ext.any():
        ext_boxes, ext_scores = boxes[is_ext], scores[is_ext]
        ious = box_iou(box, ext_boxes)[0]
        j = int(ious.argmax())
        if float(ious[j]) >= 0.3:
            box = ext_boxes[j : j + 1].clone()
            score = torch.maximum(score, ext_scores[j : j + 1])
    return box, score


def evaluate(data, alpha, prefer, threshold):
    rows = []
    for raw, origin, targets in data:
        preds = []
        for (boxes, scores), o in zip(raw, origin):
            b, s = fuse(boxes, scores, o, alpha, prefer)
            preds.append({"boxes": b, "scores": s})
        m = detection_metrics(preds, targets, score_threshold=threshold)
        m.update(temporal_detection_metrics(preds, targets, threshold))
        rows.append(m)
    return {k: sum(r[k] for r in rows) / len(rows) for k in KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrated fusion of our boxes with a detector's, chosen on val.")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--modality", default="infrared")
    parser.add_argument("--val-raw", required=True)
    parser.add_argument("--test-raw", required=True)
    parser.add_argument("--ext-dets", required=True, help="detector predictions; '{split}' -> val / test")
    parser.add_argument("--ext-topk", type=int, default=3)
    parser.add_argument("--ext-min-score", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.frames_root)
    val = load(root, "val", Path(args.val_raw), Path(args.ext_dets.replace("{split}", "val")), args.ext_topk, args.ext_min_score, args.modality)
    test = load(root, "test", Path(args.test_raw), Path(args.ext_dets.replace("{split}", "test")), args.ext_topk, args.ext_min_score, args.modality)
    print(f"loaded {len(val)} val and {len(test)} test sequences")

    results = []
    for alpha, prefer, thr in itertools.product([1.0, 0.8, 0.6, 0.4, 0.2], [False, True], [0.1, 0.2, 0.3, 0.4]):
        r = evaluate(val, alpha, prefer, thr)
        results.append({"alpha": alpha, "prefer_detector_box": prefer, "threshold": thr, "val": r})
        print(f"val alpha={alpha} prefer={prefer} thr={thr}: track {100*r['antiuav_tracking_accuracy']:.2f}  "
              f"mAP50 {100*r['map50']:.1f}  mAP75 {100*r['map75']:.1f}", flush=True)
    best = max(results, key=lambda x: (round(x["val"]["antiuav_tracking_accuracy"], 4), x["val"]["map75"]))
    plain = next(x for x in results if x["alpha"] == 1.0 and not x["prefer_detector_box"] and x["threshold"] == 0.1)
    best["test"] = evaluate(test, best["alpha"], best["prefer_detector_box"], best["threshold"])
    plain["test"] = evaluate(test, 1.0, False, 0.1)
    Path(args.out).write_text(json.dumps({"chosen_on_val": best, "plain_v5": plain, "grid": results}, indent=2), encoding="utf-8")
    for name, row in [("plain v5 fusion", plain), ("calibrated, chosen on val", best)]:
        t = row["test"]
        print(f"TEST {name}: alpha={row['alpha']} prefer={row['prefer_detector_box']} thr={row['threshold']} -> "
              f"track {100*t['antiuav_tracking_accuracy']:.2f}  mAP50 {100*t['map50']:.1f}  mAP75 {100*t['map75']:.1f}  "
              f"P {100*t['precision']:.1f}  R {100*t['recall']:.1f}")


if __name__ == "__main__":
    main()
