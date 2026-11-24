"""Where does each method win? Per target-size and per-attribute breakdown on Anti-UAV410.

Reads saved raw predictions (normalized cx, cy, w, h; one JSON per sequence, as written by
run_sequence_eval --save-raw or the YOLO baseline), applies the same post-processing to every method
(duplicate merging 0.3, one box per frame, threshold 0.1 = what validation chose for all of them), and
reports, per group of frames:
  detect@0.5   share of visible-drone frames where the output box has IoU >= 0.5
  mean IoU     average IoU over visible-drone frames (a miss counts as 0): what the Anti-UAV
               tracking accuracy is mostly made of
  false alarm  share of no-drone frames where something was reported
Groups: target size from the true box in original pixels (tiny < 64 px^2 area, small < 256,
medium < 1024, large), and every per-frame attribute flag in the labels (OC occlusion, FM fast
motion, SV scale variation, IC thermal crossover, DBC dynamic background clutter, TS tiny scale,
VE out of view, LR low resolution).

Usage:
    python scripts/analyze_by_attribute.py --frames-root /workspace/data/antiuav410_frames --split test \
        --method ours=/workspace/runs/a410_focus_fixed_s42/sequence_eval_mem_crops12/raw_test_no_tracker \
        --method yolo11n_640=/workspace/runs/baseline_yolo11n_640/raw_test \
        --out /workspace/runs/analysis_by_attribute.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drishti_v2.data.utils import box_iou, list_image_files  # noqa: E402
from drishti_v2.evaluation.postprocess import postprocess  # noqa: E402

ATTRS = ["OC", "FM", "SV", "IC", "DBC", "TS", "VE", "LR"]
SIZES = [("tiny (<64 px2)", 0, 64), ("small (64-256)", 64, 256), ("medium (256-1024)", 256, 1024), ("large (>1024)", 1024, 1e12)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-size and per-attribute comparison of methods.")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--modality", default="infrared")
    parser.add_argument("--method", action="append", required=True, help="name=raw_prediction_dir (repeatable)")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    methods = dict(m.split("=", 1) for m in args.method)

    # stats[method][group] = [visible_frames, detected_at_0.5, iou_sum, absent_frames, false_alarms]
    stats = {name: defaultdict(lambda: [0, 0, 0.0, 0, 0]) for name in methods}
    root = Path(args.frames_root) / args.split
    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        labels = json.loads((seq_dir / f"{args.modality}.json").read_text(encoding="utf-8"))
        frames = list_image_files(seq_dir / args.modality)
        with Image.open(frames[0]) as im:
            width, height = im.size
        preds = {}
        for name, raw_dir in methods.items():
            path = Path(raw_dir) / f"{seq_dir.name}.json"
            preds[name] = json.loads(path.read_text(encoding="utf-8"))["frames"] if path.exists() else None
        for idx in range(len(frames)):
            exist = bool(labels["exist"][idx]) if idx < len(labels["exist"]) else False
            rect = labels["gt_rect"][idx] if idx < len(labels["gt_rect"]) else []
            visible = exist and len(rect) >= 4 and rect[2] > 0 and rect[3] > 0
            groups = ["all"]
            if visible:
                area = rect[2] * rect[3]
                groups += [label for label, lo, hi in SIZES if lo <= area < hi]
                groups += [a for a in ATTRS if isinstance(labels.get(a), list) and idx < len(labels[a]) and labels[a][idx]]
                gt = torch.tensor([[(rect[0] + rect[2] / 2) / width, (rect[1] + rect[3] / 2) / height, rect[2] / width, rect[3] / height]])
            for name in methods:
                if preds[name] is None:
                    continue
                item = preds[name].get(str(idx), {"boxes": [], "scores": []})
                boxes = torch.tensor(item["boxes"], dtype=torch.float32).view(-1, 4)
                scores = torch.tensor(item["scores"], dtype=torch.float32).view(-1)
                boxes, scores = postprocess(boxes, scores, nms_iou=args.nms_iou, top1=True)
                reported = scores.numel() > 0 and float(scores.max()) >= args.threshold
                for g in groups:
                    s = stats[name][g]
                    if visible:
                        iou = float(box_iou(boxes[:1], gt)[0, 0]) if reported else 0.0
                        s[0] += 1; s[1] += int(iou >= 0.5); s[2] += iou
                    else:
                        s[3] += 1; s[4] += int(reported)

    order = ["all"] + [label for label, _, _ in SIZES] + ATTRS
    result = {}
    for name in methods:
        result[name] = {}
        for g in order:
            if g not in stats[name]:
                continue
            vis, det, iou, absent, fa = stats[name][g]
            result[name][g] = {"visible_frames": vis, "detect_at_0.5": det / max(vis, 1), "mean_iou": iou / max(vis, 1),
                               "absent_frames": absent, "false_alarm": fa / max(absent, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    names = list(methods)
    print(f"{'group':20s} {'frames':>7s}  " + "  ".join(f"{n[:16]:>16s}" for n in names) + "   (detect@0.5 / mean IoU)")
    for g in order:
        if g not in result[names[0]]:
            continue
        vis = result[names[0]][g]["visible_frames"]
        cells = [f"{100 * result[n][g]['detect_at_0.5']:6.1f} / {result[n][g]['mean_iou']:.3f}" if g in result[n] else "n/a" for n in names]
        print(f"{g:20s} {vis:7d}  " + "  ".join(f"{c:>16s}" for c in cells))
    print("false alarm on no-drone frames: " + ", ".join(f"{n} {100 * result[n]['all']['false_alarm']:.1f}%" for n in names))


if __name__ == "__main__":
    main()
