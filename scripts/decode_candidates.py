"""Temporal evidence accumulation: decode one box (or "absent") per frame over a whole sequence of candidates.

Borrowed from speech pitch tracking (pYIN: an HMM over per-frame pitch candidates plus an "unvoiced" state,
decoded with Viterbi) and from radar dynamic-programming track-before-detect. Here:
  states per frame   every candidate box (our network's crops + the dense detector's boxes, any score) + ABSENT
  emission           logit of the candidate's calibrated score (our scores times alpha), or a constant for ABSENT
  transition         candidate -> candidate: penalty for jumping (centre distance in units of box size, capped)
                     and for changing size; ABSENT <-> candidate: fixed switch costs (hysteresis, as in VAD)
  decoding           Viterbi with a fixed lag L: the box of frame t is decided at frame t+L from the best path
                     ending there (L = 0 is fully online; L > 0 waits L frames, like streaming speech recognition)
  merging            optional: the chosen box is averaged with same-frame candidates that overlap it (IoU >= 0.55),
                     weighted by score (the other source often has a tighter box)
A smooth path through low-score boxes can beat a single confident outlier: this is what recovers drones that
per-frame thresholding throws away. Every parameter is chosen on VAL (coordinate search on tracking accuracy);
TEST is scored once with the official metric functions. Also reports, on the same candidates, the per-frame
rule (highest calibrated score, no time) so the gain of decoding is isolated.

Usage:
  python scripts/decode_candidates.py --candidates ../runs/candidates_dyn_k8 --out ../runs/candidates_dyn_k8/decode.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drishti_v2.evaluation.metrics import detection_metrics, temporal_detection_metrics  # noqa: E402

KEYS = ["antiuav_tracking_accuracy", "map50", "map75", "precision", "recall", "f1", "false_positives_per_image"]


def load(split_dir: Path) -> list[dict]:
    seqs = []
    for f in sorted(split_dir.glob("*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        names = d["features"]
        feats = d["feats"].float().numpy()
        seqs.append({
            "name": f.stem,
            "boxes": d["boxes"].float().numpy(),                      # [T,K,4] cx cy w h, normalized
            "mask": d["mask"].numpy(),                                 # [T,K]
            "score": feats[..., names.index("score")],                 # [T,K]
            "ours": feats[..., names.index("is_ours")] > 0.5,          # [T,K]
            "gt": d["gt"].float().numpy(),                             # [T,4]
            "visible": d["visible"].numpy(),                           # [T]
        })
    return seqs


def load_raw(frames_root: Path, split: str, raw_dir: Path, modality: str = "infrared", topk: int = 10, min_score: float = 0.001) -> list[dict]:
    """Any detector's raw prediction folder (one JSON per sequence, normalized cx,cy,w,h) as decoder input:
    its top-k boxes per frame at score >= min_score are the candidates (no second source)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sweep_postprocess import load_split

    names = [p.name for p in sorted((frames_root / split).iterdir()) if p.is_dir() and (raw_dir / f"{p.name}.json").exists()]
    seqs = []
    for name, (raw, targets) in zip(names, load_split(frames_root, split, modality, raw_dir)):
        T = len(raw)
        boxes, score, mask = np.zeros((T, topk, 4), np.float32), np.zeros((T, topk), np.float32), np.zeros((T, topk), bool)
        gt, visible = np.zeros((T, 4), np.float32), np.zeros(T, bool)
        for t, ((b, s), tg) in enumerate(zip(raw, targets)):
            keep = s >= min_score
            b, s = b[keep], s[keep]
            order = torch.argsort(s, descending=True)[:topk]
            n = len(order)
            boxes[t, :n], score[t, :n], mask[t, :n] = b[order].numpy(), s[order].numpy(), True
            if tg["boxes"].numel():
                gt[t], visible[t] = tg["boxes"][0].numpy(), True
        seqs.append({"name": name, "boxes": boxes, "mask": mask, "score": score, "ours": np.zeros((T, topk), bool),
                     "gt": gt, "visible": visible})
    return seqs


def iou_one_to_many(b: np.ndarray, bs: np.ndarray) -> np.ndarray:
    ax0, ay0, ax1, ay1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    bx0, by0, bx1, by1 = bs[:, 0] - bs[:, 2] / 2, bs[:, 1] - bs[:, 3] / 2, bs[:, 0] + bs[:, 2] / 2, bs[:, 1] + bs[:, 3] / 2
    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    return inter / np.maximum(b[2] * b[3] + bs[:, 2] * bs[:, 3] - inter, 1e-9)


def calibrated(seq: dict, alpha: float) -> np.ndarray:
    s = np.where(seq["ours"], seq["score"] * alpha, seq["score"])
    return np.where(seq["mask"], s, 0.0)


def decode(seq: dict, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns chosen index per frame (-1 = absent) and its calibrated score."""
    s = calibrated(seq, p["alpha"])
    T, K = s.shape
    valid = seq["mask"] & (s > 0)
    emit = np.where(valid, np.log(np.clip(s, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - s, 1e-4, 1)), -np.inf)
    boxes = seq["boxes"]
    size = np.sqrt(np.clip(boxes[..., 2] * boxes[..., 3], 1e-8, None))      # [T,K]
    absent = K
    delta = np.full(K + 1, -np.inf)
    delta[:K] = emit[0]
    delta[absent] = p["absent"]
    deltas = np.empty((T, K + 1))
    psi = np.zeros((T, K + 1), dtype=np.int16)
    deltas[0] = delta
    for t in range(1, T):
        prev_c = boxes[t - 1]                                                 # [K,4]
        cur_c = boxes[t]
        d = np.sqrt(((cur_c[None, :, :2] - prev_c[:, None, :2]) ** 2).sum(-1))  # [Kprev,Kcur]
        scale = 0.5 * (size[t - 1][:, None] + size[t][None, :]) + 0.01
        jump = np.minimum(d / scale, p["jump_cap"])
        zoom = np.abs(np.log(np.clip(size[t][None, :], 1e-6, None) / np.clip(size[t - 1][:, None], 1e-6, None)))
        trans = -(p["w_move"] * jump + p["w_size"] * zoom)                    # [Kprev,Kcur]
        from_cand = delta[:K, None] + trans                                   # [Kprev,Kcur]
        best_prev = from_cand.argmax(0)
        best_val = from_cand[best_prev, np.arange(K)]
        from_abs = delta[absent] - p["switch"]
        use_abs = from_abs > best_val
        new = np.where(use_abs, from_abs, best_val) + emit[t]
        psi[t, :K] = np.where(use_abs, absent, best_prev)
        cand_to_abs = delta[:K] - p["switch"]
        j = int(cand_to_abs.argmax())
        if delta[absent] >= cand_to_abs[j]:
            new_abs, psi[t, absent] = delta[absent], absent
        else:
            new_abs, psi[t, absent] = cand_to_abs[j], j
        delta = np.append(new, new_abs + p["absent"])
        delta -= delta[np.isfinite(delta)].max()                              # keep numbers small
        deltas[t] = delta
    choice = np.full(T, -1, dtype=np.int64)
    lag = p["lag"]
    for t in range(T):
        end = min(t + lag, T - 1)
        state = int(deltas[end].argmax())
        for u in range(end, t, -1):
            state = int(psi[u, state])
        choice[t] = -1 if state == absent else state
    conf = np.array([s[t, c] if c >= 0 else 0.0 for t, c in enumerate(choice)])
    return choice, conf


def per_frame(seq: dict, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """No time: the highest calibrated score in each frame (what the calibrated fusion rule does)."""
    s = calibrated(seq, p["alpha"])
    choice = s.argmax(1)
    conf = s[np.arange(len(s)), choice]
    choice = np.where(conf >= p["thr"], choice, -1)
    return choice, np.where(choice >= 0, conf, 0.0)


def output_boxes(seq: dict, choice: np.ndarray, conf: np.ndarray, p: dict) -> np.ndarray:
    out = np.zeros((len(choice), 4), dtype=np.float32)
    s = calibrated(seq, p["alpha"])
    for t, c in enumerate(choice):
        if c < 0:
            continue
        b = seq["boxes"][t, c]
        if p.get("merge"):
            ok = seq["mask"][t] & (s[t] > 0)
            ious = iou_one_to_many(b, seq["boxes"][t])
            w = np.where(ok & (ious >= 0.55), s[t], 0.0)
            if w.sum() > 0:
                b = (seq["boxes"][t] * w[:, None]).sum(0) / w.sum()
        out[t] = b
    return out


def fast_track_acc(seq: dict, choice: np.ndarray, boxes: np.ndarray) -> float:
    """Exactly the official Anti-UAV tracking accuracy for one box per frame (fast numpy version)."""
    det = choice >= 0
    vis = seq["visible"]
    ious = np.zeros(len(choice))
    for t in np.nonzero(det & vis)[0]:
        ious[t] = iou_one_to_many(boxes[t], seq["gt"][t : t + 1])[0]
    term1 = (ious[vis].sum() + (~det[~vis]).sum()) / max(len(choice), 1)
    t_star = vis.sum()
    term2 = 0.2 * (((~det[vis]).sum() / t_star) ** 0.3) if t_star else 0.0
    return float(term1 - term2)


def run(seqs, p, method):
    accs = []
    for seq in seqs:
        choice, conf = method(seq, p)
        accs.append(fast_track_acc(seq, choice, output_boxes(seq, choice, conf, p)))
    return float(np.mean(accs))


def official(seqs, p, method) -> dict:
    rows = []
    for seq in seqs:
        choice, conf = method(seq, p)
        boxes = output_boxes(seq, choice, conf, p)
        preds = [{"boxes": torch.from_numpy(boxes[t : t + 1]), "scores": torch.tensor([max(float(conf[t]), 1e-3)])}
                 if choice[t] >= 0 else {"boxes": torch.empty(0, 4), "scores": torch.empty(0)} for t in range(len(choice))]
        targets = [{"boxes": torch.from_numpy(seq["gt"][t : t + 1]) if seq["visible"][t] else torch.empty(0, 4)} for t in range(len(choice))]
        m = detection_metrics(preds, targets, score_threshold=1e-4)
        m.update(temporal_detection_metrics(preds, targets, 1e-4))
        rows.append(m)
    return {k: float(np.mean([r[k] for r in rows])) for k in KEYS + ["flicker_rate"] if k in rows[0]}


def coordinate_search(val, start: dict, grid: dict, method, rounds: int = 2) -> tuple[dict, float]:
    best = dict(start)
    best_acc = run(val, best, method)
    for _ in range(rounds):
        for key, values in grid.items():
            for v in values:
                if v == best[key]:
                    continue
                trial = dict(best, **{key: v})
                acc = run(val, trial, method)
                if acc > best_acc + 1e-5:
                    best, best_acc = trial, acc
            print(f"  {key:9s} -> {best[key]!s:6s} val track {100*best_acc:.2f}", flush=True)
    return best, best_acc


def fmt(t: dict) -> str:
    return (f"track {100*t['antiuav_tracking_accuracy']:.2f}  mAP50 {100*t['map50']:.1f}  mAP75 {100*t['map75']:.1f}  "
            f"P {100*t['precision']:.1f}  R {100*t['recall']:.1f}" + (f"  flicker {100*t['flicker_rate']:.1f}" if "flicker_rate" in t else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Viterbi decoding of per-frame candidates with an absent state.")
    parser.add_argument("--candidates", default=None, help="candidate dumps (our network + a detector)")
    parser.add_argument("--frames-root", default=None, help="with --raw-val/--raw-test: decode any detector's raw output")
    parser.add_argument("--raw-val", default=None)
    parser.add_argument("--raw-test", default=None)
    parser.add_argument("--lags", default="0,5,15")
    parser.add_argument("--max-seqs", type=int, default=None, help="smoke test")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    started = time.time()
    if args.candidates:
        val, test = load(Path(args.candidates) / "val"), load(Path(args.candidates) / "test")
    else:
        fr = Path(args.frames_root)
        val, test = load_raw(fr, "val", Path(args.raw_val)), load_raw(fr, "test", Path(args.raw_test))
    if args.max_seqs:
        val, test = val[: args.max_seqs], test[: args.max_seqs]
    print(f"loaded val {len(val)} / test {len(test)} sequences in {time.time() - started:.0f}s", flush=True)
    results = {}

    print("per-frame rule (no time), chosen on val:", flush=True)
    pf, _ = coordinate_search(val, {"alpha": 0.6, "thr": 0.1, "merge": 0},
                              {"alpha": [0.4, 0.6, 0.8, 1.0], "thr": [0.0005, 0.001, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3], "merge": [0, 1]}, per_frame, rounds=1)
    results["per_frame"] = {"params": pf, "test": official(test, pf, per_frame)}
    print(f"TEST per-frame rule {pf}: {fmt(results['per_frame']['test'])}", flush=True)

    grid = {"alpha": [0.4, 0.6, 0.8, 1.0], "absent": [-20.0, -12.0, -9.0, -7.0, -5.5, -4.0, -3.0, -2.2, -1.5, -0.8, 0.0], "switch": [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0],
            "w_move": [0.0, 0.25, 0.5, 1.0, 2.0], "jump_cap": [2.0, 5.0, 10.0], "w_size": [0.0, 0.5, 1.0], "merge": [0, 1]}
    for lag in [int(x) for x in args.lags.split(",")]:
        print(f"decoder, lag {lag}, chosen on val:", flush=True)
        start = {"alpha": pf["alpha"], "absent": -2.2, "switch": 2.0, "w_move": 0.5, "jump_cap": 5.0, "w_size": 0.5, "merge": 0, "lag": lag}
        p, acc = coordinate_search(val, start, grid, decode)
        results[f"decoder_lag{lag}"] = {"params": p, "val_track": acc, "test": official(test, p, decode)}
        print(f"TEST decoder lag {lag} {p}: {fmt(results[f'decoder_lag{lag}']['test'])}", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {args.out} ({(time.time() - started) / 60:.0f} min)")


if __name__ == "__main__":
    main()
