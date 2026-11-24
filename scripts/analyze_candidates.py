"""Where is the ceiling? Histogram of the BEST overlap any candidate box has with the true box, per frame.

Reads the candidate dumps of version 6 (drishti_v2/experiments/dump_candidates.py). For every frame where
the drone is visible it takes the best IoU among (a) all candidates, (b) our model's boxes only, (c) the
dense detector's boxes only, and bins it:
    complete miss  < 0.1       near miss  0.1-0.3 / 0.3-0.5       hit  0.5-0.75 / >= 0.75
A frame counts as "missed" when no candidate reaches 0.5. "Near misses" are what a box refiner could
rescue; "complete misses" need better proposals. Also split by target size (box diagonal in pixels,
the Anti-UAV410 convention: tiny < 10, small 10-30, medium 30-50, normal >= 50). No GPU.

Usage:
  python scripts/analyze_candidates.py --candidates /workspace/runs/v6_candidates --split test \
      --out /workspace/runs/v6_candidates/analysis_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drishti_v2.data.utils import box_iou  # noqa: E402

BINS = [(0.0, 0.1, "complete_miss_<0.1"), (0.1, 0.3, "near_miss_0.1-0.3"), (0.3, 0.5, "near_miss_0.3-0.5"),
        (0.5, 0.75, "hit_0.5-0.75"), (0.75, 1.01, "hit_>=0.75")]
SIZES = [(0, 10, "tiny_<10px"), (10, 30, "small_10-30px"), (30, 50, "medium_30-50px"), (50, 1e9, "normal_>=50px")]


def summarize(best: torch.Tensor) -> dict:
    n = max(len(best), 1)
    out = {name: round(float(((best >= lo) & (best < hi)).float().mean()), 4) for lo, hi, name in BINS}
    out["frames"] = int(len(best))
    out["missed_(best<0.5)"] = round(float((best < 0.5).float().mean()), 4)
    out["mean_best_iou"] = round(float(best.mean()), 4) if len(best) else 0.0
    out["mean_best_iou_of_hits"] = round(float(best[best >= 0.5].mean()), 4) if (best >= 0.5).any() else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Best-candidate IoU histogram (the ceiling of any selection).")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", default="640x512", help="WxH in pixels, for the size bins")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    width, height = (int(v) for v in args.image_size.lower().split("x"))

    best = {"all": [], "ours": [], "ext": []}
    diag = []
    files = sorted((Path(args.candidates) / args.split).glob("*.pt"))
    for f in files:
        d = torch.load(f, map_location="cpu")
        names = d["features"]
        is_ours = d["feats"][..., names.index("is_ours")].float() > 0.5
        for t in torch.nonzero(d["visible"]).flatten().tolist():
            mask = d["mask"][t]
            ious = box_iou(d["boxes"][t], d["gt"][t : t + 1])[:, 0] * mask
            best["all"].append(float(ious.max()) if mask.any() else 0.0)
            best["ours"].append(float((ious * is_ours[t]).max()) if (mask & is_ours[t]).any() else 0.0)
            best["ext"].append(float((ious * ~is_ours[t]).max()) if (mask & ~is_ours[t]).any() else 0.0)
            g = d["gt"][t]
            diag.append(float(((g[2] * width) ** 2 + (g[3] * height) ** 2) ** 0.5))
    best = {k: torch.tensor(v) for k, v in best.items()}
    diag = torch.tensor(diag)
    result = {"sequences": len(files), "split": args.split, "by_source": {k: summarize(v) for k, v in best.items()}, "by_size": {}}
    for lo, hi, name in SIZES:
        sel = (diag >= lo) & (diag < hi)
        result["by_size"][name] = {"share_of_frames": round(float(sel.float().mean()), 4), **{k: summarize(v[sel]) for k, v in best.items()}}
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"{args.split}: {len(files)} sequences, {len(diag)} visible frames")
    print(f"{'source':8s} " + " ".join(f"{name[:18]:>18s}" for _, _, name in BINS) + "   missed   mean IoU (hits)")
    for k, v in best.items():
        s = result["by_source"][k]
        print(f"{k:8s} " + " ".join(f"{100*s[name]:17.1f}%" for _, _, name in BINS) + f"   {100*s['missed_(best<0.5)']:5.1f}%   {s['mean_best_iou_of_hits']:.3f}")
    print("\nby target size (all candidates): share | missed | complete miss | near miss | mean IoU of hits")
    for _, _, name in SIZES:
        s = result["by_size"][name]
        a = s["all"]
        near = a["near_miss_0.1-0.3"] + a["near_miss_0.3-0.5"]
        print(f"  {name:16s} {100*s['share_of_frames']:5.1f}% | {100*a['missed_(best<0.5)']:5.1f}% | {100*a['complete_miss_<0.1']:5.1f}% | {100*near:5.1f}% | {a['mean_best_iou_of_hits']:.3f}")


if __name__ == "__main__":
    main()
