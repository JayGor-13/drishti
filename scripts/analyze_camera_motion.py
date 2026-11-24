"""Measure target displacement and background residuals on Anti-UAV410."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.yolo_antiuav import MotionEncoder, sequences


def visible_boxes(labels, length):
    from drishti_v2.data.dataset import _box_exists, _candidate_boxes, _read_antiuav_json

    boxes, exists = _read_antiuav_json(labels)
    result = np.full((length, 4), np.nan)
    for i, value in enumerate(boxes[:length]):
        if not (exists[i] if i < len(exists) else _box_exists(value)):
            continue
        candidates = _candidate_boxes(value)
        if candidates and min(candidates[0][2:]) > 0:
            result[i] = candidates[0]
    return result


def exclude_box(mask, box):
    if not np.isfinite(box).all():
        return
    x, y, w, h = box
    height, width = mask.shape
    x0, y0 = max(0, math.floor(x) - 2), max(0, math.floor(y) - 2)
    x1, y1 = min(width, math.ceil(x + w) + 2), min(height, math.ceil(y + h) + 2)
    if x0 < x1 and y0 < y1:
        mask[y0:y1, x0:x1] = False


def background_residuals(encoder, frames, i, gap, shift, current_box, past_box):
    current = encoder._cached(frames, i)
    previous = encoder._cached(frames, i - gap)
    height, width = current.shape
    dx, dy = shift
    mask = np.ones(current.shape, dtype=bool)
    if dx or dy:
        # Use only the common image support, with a one-pixel interpolation margin.
        mask.fill(False)
        x0, x1 = math.ceil(max(0, dx)) + 1, math.floor(min(width, width + dx)) - 1
        y0, y1 = math.ceil(max(0, dy)) + 1, math.floor(min(height, height + dy)) - 1
        if x0 < x1 and y0 < y1:
            mask[y0:y1, x0:x1] = True
    for box in (current_box, past_box, past_box + (dx, dy, 0, 0)):
        exclude_box(mask, box)
    if mask.any():
        aligned = encoder._past(frames, i, gap)
        return (float(np.abs(current - previous)[mask].mean()),
                float(np.abs(current - aligned)[mask].mean()))
    return None


def collect(frames, labels, gaps, stride):
    boxes = visible_boxes(labels, len(frames))
    centers = boxes[:, :2] + boxes[:, 2:] / 2
    diagonals = np.linalg.norm(boxes[:, 2:], axis=1)
    missing = np.r_[0, np.cumsum(~np.isfinite(diagonals))]
    encoder = MotionEncoder("diff", gap=max(gaps), compensate=True)
    stats = defaultdict(list)
    for i in range(1, len(frames)):
        one_step = encoder._shift_between(frames, i - 1, i)
        stats["background_shift_px"].append(math.hypot(*one_step))
        if missing[i + 1] == missing[i - 1]:
            stats["target_step_px"].append(float(np.linalg.norm(centers[i] - centers[i - 1])))
        for gap in gaps:
            if i < gap:
                continue
            previous = i - gap
            shift = encoder._shift_between(frames, previous, i)
            if all(abs(value) < 0.25 for value in shift):
                shift = (0.0, 0.0)
            if missing[i + 1] == missing[previous]:
                delta = centers[i] - centers[previous]
                stats[f"target_shift_g{gap}_box_diagonals"].append(
                    float(np.linalg.norm(delta) / diagonals[i])
                )
                stats[f"aligned_target_shift_g{gap}_box_diagonals"].append(
                    float(np.linalg.norm(delta - shift) / diagonals[i])
                )
            if i % stride == 0:
                residuals = background_residuals(encoder, frames, i, gap, shift, boxes[i], boxes[previous])
                if residuals is not None:
                    raw, aligned = residuals
                    stats[f"background_raw_mae_g{gap}"].append(raw)
                    stats[f"background_aligned_mae_g{gap}"].append(aligned)
    return stats


def summarize(stats):
    result = {}
    for name, values in stats.items():
        samples = np.asarray(values)
        result[name] = {
            "count": len(samples), "median": float(np.median(samples)),
            "p25": float(np.percentile(samples, 25)), "p75": float(np.percentile(samples, 75)),
        }
        if name == "target_step_px":
            result[name]["fraction_below_1px"] = float(np.mean(samples < 1))
        if name == "background_shift_px":
            result[name]["fraction_zero"] = float(np.mean(samples == 0))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--gaps", nargs="+", type=int, default=[1, 8, 12])
    parser.add_argument("--residual-stride", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.residual_stride < 1 or min(args.gaps) < 1 or len(args.gaps) != len(set(args.gaps)):
        parser.error("Use distinct positive gaps and a positive residual stride")
    report = {"gaps": args.gaps, "residual_stride": args.residual_stride, "splits": {}}
    for split in args.splits:
        pooled = defaultdict(list)
        per_sequence = {}
        for name, frames, labels in sequences(args.frames_root, split, "infrared"):
            if len(frames) < 2:
                continue
            stats = collect(frames, labels, args.gaps, args.residual_stride)
            per_sequence[name] = summarize(stats)
            for metric, values in stats.items():
                pooled[metric].extend(values)
        if not per_sequence:
            raise ValueError(f"No usable infrared sequences in {args.frames_root / split}")
        report["splits"][split] = {"pooled": summarize(pooled), "per_sequence": per_sequence}
        print(f"{split}: measured {len(per_sequence)} sequences", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
