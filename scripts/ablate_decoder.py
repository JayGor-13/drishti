"""Ablation rows that need no GPU: parts of the sequence decoder, and which candidates it sees.

Every variant is re-tuned on VAL with the same coordinate search as the full decoder (the removed part is
held fixed), and TEST is scored once with the official metric functions, so each row is a fair comparison.
  decoder parts   full | no absent state (a box is output in every frame) | no motion/size prior between
                  frames | no switch cost into/out of absent | lags 0 / 15
  candidate pool  our network only | dense detector only | both (per-frame rule and decoder for each)
Also writes the per-frame chosen boxes of the main variants as raw prediction folders, for
scripts/analyze_by_attribute.py (score 1 = decided present: the decoder has already made the decision).

Usage:
  python scripts/ablate_decoder.py --candidates ../runs/candidates_dyn_k8 --out ../runs/candidates_dyn_k8/ablation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decode_candidates as dc  # noqa: E402

GRID = {"alpha": [0.4, 0.6, 0.8, 1.0], "absent": [-20.0, -12.0, -9.0, -7.0, -5.5, -4.0, -3.0, -2.2, -1.5, -0.8, 0.0],
        "switch": [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0], "w_move": [0.0, 0.25, 0.5, 1.0, 2.0], "jump_cap": [2.0, 5.0, 10.0],
        "w_size": [0.0, 0.5, 1.0]}
START = {"alpha": 0.4, "absent": -4.0, "switch": 4.0, "w_move": 0.5, "jump_cap": 5.0, "w_size": 0.5, "merge": 0}
PF_GRID = {"alpha": [0.4, 0.6, 0.8, 1.0], "thr": [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]}


def restrict(seqs: list[dict], keep: str) -> list[dict]:
    out = []
    for s in seqs:
        s2 = dict(s)
        sel = s["ours"] if keep == "ours" else ~s["ours"]
        s2["mask"] = s["mask"] & sel
        out.append(s2)
    return out


def tune(val, start, grid, method, fixed=()):
    g = {k: v for k, v in grid.items() if k not in fixed}
    return dc.coordinate_search(val, start, g, method, rounds=2)


def write_raw(seqs, p, method, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for seq in seqs:
        choice, conf = method(seq, p)
        boxes = dc.output_boxes(seq, choice, conf, p)
        frames = {str(t): ({"boxes": [boxes[t].tolist()], "scores": [1.0]} if choice[t] >= 0 else {"boxes": [], "scores": []})
                  for t in range(len(choice))}
        (folder / f"{seq['name']}.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decoder and candidate-pool ablations (CPU).")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-seqs", type=int, default=None)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    val, test = dc.load(Path(args.candidates) / "val"), dc.load(Path(args.candidates) / "test")
    if args.max_seqs:
        val, test = val[: args.max_seqs], test[: args.max_seqs]
    rows = {}

    def record(name, p, method, v, t):
        res = dc.official(t, p, method)
        rows[name] = {"params": p, "test": res}
        print(f"TEST {name:34s} {dc.fmt(res)}", flush=True)
        (out / "ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return p

    # decoder parts (both sources)
    variants = [
        ("decoder lag 0 (full)", dict(START, lag=0), ()),
        ("decoder lag 15 (full)", dict(START, lag=15), ()),
        ("lag 0, no absent state", dict(START, lag=0, absent=-50.0), ("absent",)),
        ("lag 0, no motion/size prior", dict(START, lag=0, w_move=0.0, w_size=0.0), ("w_move", "w_size", "jump_cap")),
        ("lag 0, no switch cost", dict(START, lag=0, switch=0.0), ("switch",)),
    ]
    chosen = {}
    for name, start, fixed in variants:
        print(f"== {name}", flush=True)
        p, _ = tune(val, start, GRID, dc.decode, fixed)
        chosen[name] = record(name, p, dc.decode, val, test)

    # candidate pools
    for pool in ("ours", "ext"):
        v, t = restrict(val, pool), restrict(test, pool)
        print(f"== {pool} only, per-frame", flush=True)
        p, _ = tune(v, {"alpha": 1.0, "thr": 0.1, "merge": 0}, PF_GRID, dc.per_frame)
        chosen[f"{pool} only, per-frame"] = record(f"{pool} only, per-frame", p, dc.per_frame, v, t)
        print(f"== {pool} only, decoder lag 0", flush=True)
        p, _ = tune(v, dict(START, lag=0, alpha=1.0), GRID, dc.decode, ("alpha",))
        chosen[f"{pool} only, decoder lag 0"] = record(f"{pool} only, decoder lag 0", p, dc.decode, v, t)
    print("== both, per-frame", flush=True)
    p, _ = tune(val, {"alpha": 0.4, "thr": 0.05, "merge": 0}, PF_GRID, dc.per_frame)
    chosen["both, per-frame"] = record("both, per-frame", p, dc.per_frame, val, test)

    # raw folders for the per-size / per-attribute analysis
    for name, method, seqs in (("ext only, per-frame", dc.per_frame, restrict(test, "ext")),
                               ("ours only, per-frame", dc.per_frame, restrict(test, "ours")),
                               ("both, per-frame", dc.per_frame, test),
                               ("decoder lag 0 (full)", dc.decode, test)):
        folder = out / ("raw_" + name.split(",")[0].split(" (")[0].replace(" ", "_"))
        write_raw(seqs, chosen[name], method, folder)
        print(f"wrote {folder}", flush=True)
    print(f"done in {(time.time() - started) / 60:.0f} min -> {out / 'ablation.json'}")


if __name__ == "__main__":
    main()
