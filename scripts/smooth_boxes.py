"""Lever 3 (21 Oct): temporal smoothing of WHERE the box is, with confidence. No GPU, runs on saved boxes.

The decoder chooses which box; nothing so far smooths the box's position over time, and our boxes jitter
(overlap only 0.82 even on large drones). Here every frame outputs its top box (never abstain, the regime the
metric rewards) and the box coordinates are then filtered over time:
  ema      causal: state <- (1 - w) * state + w * box, with w = w0 * score^p; if the new box barely overlaps
           the state (IoU < gate) the drone jumped or re-appeared: reset to the box
  window   causal: score-weighted mean of the last N boxes that overlap the current one (IoU >= gate)
  centred  non-causal (decides L frames late): the same over frames t-L .. t+L
Centre and size can be smoothed with different strengths. Everything is chosen on val (coordinate search on
the exact tracking-accuracy formula); test is scored once with the official functions. Baseline row: the raw
top box with no smoothing.

Usage:
  python scripts/smooth_boxes.py --candidates ../runs/candidates_dyn_k8 --source ext --out ../runs/candidates_dyn_k8/smooth.json
  python scripts/smooth_boxes.py --frames-root <frames> --raw-val <dir> --raw-test <dir> --out <json>
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decode_candidates as dc  # noqa: E402


def top_boxes(seq: dict, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame: the top box of the chosen source (never abstain). Returns boxes [T,4], scores [T], has [T]."""
    s = seq["score"].copy()
    sel = seq["mask"] & ({"ext": ~seq["ours"], "ours": seq["ours"], "both": np.ones_like(seq["ours"])}[source])
    s[~sel] = -1.0
    j = s.argmax(1)
    has = s[np.arange(len(s)), j] > 0
    return seq["boxes"][np.arange(len(s)), j].astype(np.float64), np.clip(s[np.arange(len(s)), j], 0, 1), has


def smooth(boxes: np.ndarray, scores: np.ndarray, has: np.ndarray, p: dict) -> np.ndarray:
    T = len(boxes)
    out = boxes.copy()
    if p["method"] == "ema":
        state = None
        for t in range(T):
            if not has[t]:
                continue
            b = boxes[t]
            if state is None or dc.iou_one_to_many(b, state[None])[0] < p["gate"]:
                state = b.copy()
            else:
                w = min(p["w0"] * (scores[t] ** p["pow"]), 1.0)
                state = state.copy()
                state[:2] = (1 - w) * state[:2] + w * b[:2]
                ws = min(p["w0"] * p["size_mult"] * (scores[t] ** p["pow"]), 1.0)
                state[2:] = (1 - ws) * state[2:] + ws * b[2:]
            out[t] = state
        return out
    L = p["window"]
    for t in range(T):
        if not has[t]:
            continue
        lo, hi = (max(0, t - L), t + 1) if p["method"] == "window" else (max(0, t - L), min(T, t + L + 1))
        idx = [u for u in range(lo, hi) if has[u]]
        cand = boxes[idx]
        ious = dc.iou_one_to_many(boxes[t], cand)
        keep = ious >= p["gate"]
        w = scores[idx] * keep
        if w.sum() <= 0:
            continue
        mean = (cand * w[:, None]).sum(0) / w.sum()
        out[t, :2] = mean[:2]
        out[t, 2:] = p["size_mult"] * mean[2:] + (1 - p["size_mult"]) * boxes[t, 2:] if p["size_mult"] <= 1 else mean[2:]
    return out


def run(seqs, p, source) -> float:
    accs = []
    for seq in seqs:
        b, s, has = top_boxes(seq, source)
        sb = smooth(b, s, has, p) if p["method"] != "none" else b
        choice = np.where(has, 0, -1)
        accs.append(dc.fast_track_acc(seq, choice, sb.astype(np.float32)))
    return float(np.mean(accs))


def official(seqs, p, source) -> dict:
    import torch

    rows = []
    for seq in seqs:
        b, s, has = top_boxes(seq, source)
        sb = (smooth(b, s, has, p) if p["method"] != "none" else b).astype(np.float32)
        preds = [{"boxes": torch.from_numpy(sb[t : t + 1]), "scores": torch.tensor([max(float(s[t]), 1e-3)])} if has[t]
                 else {"boxes": torch.empty(0, 4), "scores": torch.empty(0)} for t in range(len(b))]
        targets = [{"boxes": torch.from_numpy(seq["gt"][t : t + 1]) if seq["visible"][t] else torch.empty(0, 4)} for t in range(len(b))]
        m = dc.detection_metrics(preds, targets, score_threshold=1e-4)
        m.update(dc.temporal_detection_metrics(preds, targets, 1e-4))
        rows.append(m)
    return {k: float(np.mean([r[k] for r in rows])) for k in dc.KEYS + ["flicker_rate"] if k in rows[0]}


def search(val, start, grid, source, rounds=2):
    best, best_acc = dict(start), run(val, start, source)
    for _ in range(rounds):
        for key, values in grid.items():
            for v in values:
                if v == best[key]:
                    continue
                trial = dict(best, **{key: v})
                acc = run(val, trial, source)
                if acc > best_acc + 1e-6:
                    best, best_acc = trial, acc
        print(f"  {best} val track {100*best_acc:.2f}", flush=True)
    return best, best_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal smoothing of box position with confidence.")
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--source", default="ext", choices=["ext", "ours", "both"])
    parser.add_argument("--frames-root", default=None)
    parser.add_argument("--raw-val", default=None)
    parser.add_argument("--raw-test", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    started = time.time()
    if args.candidates:
        val, test = dc.load(Path(args.candidates) / "val"), dc.load(Path(args.candidates) / "test")
        source = args.source
    else:
        fr = Path(args.frames_root)
        val, test = dc.load_raw(fr, "val", Path(args.raw_val)), dc.load_raw(fr, "test", Path(args.raw_test))
        source = "ext"
    print(f"loaded val {len(val)} / test {len(test)}", flush=True)
    results = {}

    def record(name, p):
        results[name] = {"params": p, "val_track": run(val, p, source), "test": official(test, p, source)}
        t = results[name]["test"]
        print(f"TEST {name:14s} {dc.fmt(t)}  (val {100*results[name]['val_track']:.2f})", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    record("raw_top_box", {"method": "none"})
    p, _ = search(val, {"method": "ema", "w0": 0.5, "pow": 1.0, "gate": 0.3, "size_mult": 1.0},
                  {"w0": [0.2, 0.35, 0.5, 0.7, 0.85, 1.0], "pow": [0.0, 0.5, 1.0, 2.0], "gate": [0.05, 0.1, 0.2, 0.3, 0.5],
                   "size_mult": [0.25, 0.5, 1.0, 1.5]}, source)
    record("ema", p)
    for method in ("window", "centred"):
        p, _ = search(val, {"method": method, "window": 2, "gate": 0.3, "size_mult": 1.0},
                      {"window": [1, 2, 3, 5, 8], "gate": [0.05, 0.1, 0.2, 0.3, 0.5], "size_mult": [0.0, 0.5, 1.0]}, source)
        record(method, p)
    print(f"saved {args.out} ({(time.time() - started) / 60:.0f} min)")


if __name__ == "__main__":
    main()
