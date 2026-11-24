"""How noisy are Anti-UAV410's own boxes? (backlog idea 8: the hard ceiling of box-tightness metrics)

Two proxies, no re-annotation needed, no GPU:
  smooth   fit a local quadratic trajectory (window 9 frames) to the labelled centre and size and compare the
           raw label with the fitted one. Drones move smoothly, so what is left is mostly annotation jitter.
           A predictor that followed the smooth trajectory *perfectly* would score this IoU against the labels.
  blob     inside a window around the label, threshold the thermal blob (mean + 1.5 std of the window) and take
           the tight box of bright pixels near the centre; compare with the label. Tells how loose or offset the
           hand boxes are relative to the visible target.
Both are reported by target size (box diagonal in pixels). The numbers to look at: the mean IoU (an upper
bound on tracking accuracy on visible frames for any predictor that matches the "true" target that well) and
the share with IoU >= 0.75 (an upper bound on mAP@75).

Usage:
  python scripts/check_annotation_noise.py --frames-root /workspace/data/antiuav410_frames --split test \
      --out /workspace/runs/annotation_noise_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "scripts" / "baselines"))
from drishti_v2.data.dataset import _box_exists, _candidate_boxes, _read_antiuav_json  # noqa: E402
from yolo_antiuav import sequences  # noqa: E402

SIZES = [(0, 10, "tiny_<10px"), (10, 30, "small_10-30px"), (30, 50, "medium_30-50px"), (50, 1e9, "normal_>=50px")]


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax0, ay0, ax1, ay1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx0, by0, bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    return inter / np.maximum(a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter, 1e-6)


def local_quadratic(values: np.ndarray, window: int = 9) -> np.ndarray:
    """Per-frame local quadratic fit over a centred window (Savitzky-Golay-like), handles gaps by index."""
    n = len(values)
    out = values.copy()
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        if hi - lo < 5:
            continue
        x = np.arange(lo, hi) - i
        coef = np.polyfit(x, values[lo:hi], 2)
        out[i] = coef[-1]
    return out


def blob_box(gray: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    cx, cy, w, h = box
    side = max(2.0 * max(w, h), 12.0)
    x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    x1, y1 = int(min(cx + side / 2, gray.shape[1])), int(min(cy + side / 2, gray.shape[0]))
    win = gray[y0:y1, x0:x1].astype(np.float32)
    if win.size < 16:
        return None
    thr = win.mean() + 1.5 * win.std()
    hot = win > thr
    if not hot.any():
        hot = win < win.mean() - 1.5 * win.std()  # cold drone on a warm background
        if not hot.any():
            return None
    ys, xs = np.nonzero(hot)
    # keep pixels in the connected-ish region around the label centre: within 1.5 box sizes of it
    keep = (np.abs(xs + x0 - cx) <= 1.5 * max(w, 2)) & (np.abs(ys + y0 - cy) <= 1.5 * max(h, 2))
    if not keep.any():
        return None
    xs, ys = xs[keep] + x0, ys[keep] + y0
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    return np.array([(xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2, bw, bh], dtype=np.float32)


def summarize(ious: np.ndarray) -> dict:
    if len(ious) == 0:
        return {"frames": 0}
    return {"frames": int(len(ious)), "mean_iou": round(float(ious.mean()), 4),
            "share_iou_ge_0.5": round(float((ious >= 0.5).mean()), 4), "share_iou_ge_0.75": round(float((ious >= 0.75).mean()), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotation-noise proxies for Anti-UAV410 boxes.")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--modality", default="infrared")
    parser.add_argument("--blob-stride", type=int, default=5, help="run the blob check on every Nth labelled frame")
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from PIL import Image

    smooth_ious, smooth_diag, blob_ious, blob_diag, size_resid = [], [], [], [], []
    seqs = list(sequences(Path(args.frames_root), args.split, args.modality))
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    for name, frames, label in seqs:
        boxes, exists = _read_antiuav_json(label)
        rows, idxs = [], []
        for idx in range(min(len(frames), len(boxes))):
            raw = boxes[idx]
            present = exists[idx] if idx < len(exists) else _box_exists(raw)
            if not present:
                continue
            for x, y, w, h in _candidate_boxes(raw):
                if w > 0 and h > 0:
                    rows.append([x + w / 2, y + h / 2, w, h])
                    idxs.append(idx)
                    break
        if len(rows) < 9:
            continue
        arr = np.array(rows, dtype=np.float32)
        idxs = np.array(idxs)
        # smooth within runs of consecutive labelled frames
        fitted = arr.copy()
        start = 0
        for k in range(1, len(idxs) + 1):
            if k == len(idxs) or idxs[k] != idxs[k - 1] + 1:
                if k - start >= 5:
                    for c in range(4):
                        fitted[start:k, c] = local_quadratic(arr[start:k, c])
                start = k
        fitted[:, 2:] = np.maximum(fitted[:, 2:], 1.0)
        ious = iou_xywh(arr, fitted)
        diag = np.sqrt(arr[:, 2] ** 2 + arr[:, 3] ** 2)
        smooth_ious.append(ious)
        smooth_diag.append(diag)
        size_resid.append(np.abs(arr[:, 2:] - fitted[:, 2:]))
        for j in range(0, len(idxs), max(args.blob_stride, 1)):
            with Image.open(frames[idxs[j]]) as im:
                gray = np.asarray(im.convert("L"))
            b = blob_box(gray, arr[j])
            if b is not None:
                blob_ious.append(float(iou_xywh(arr[j : j + 1], b[None])[0]))
                blob_diag.append(float(diag[j]))
    smooth_ious, smooth_diag = np.concatenate(smooth_ious), np.concatenate(smooth_diag)
    size_resid = np.concatenate(size_resid)
    blob_ious, blob_diag = np.array(blob_ious), np.array(blob_diag)
    result = {"split": args.split, "sequences": len(seqs),
              "smooth_trajectory": {"all": summarize(smooth_ious), "median_abs_size_residual_px": [round(float(v), 2) for v in np.median(size_resid, 0)]},
              "blob_refit": {"all": summarize(blob_ious)}}
    for lo, hi, nm in SIZES:
        result["smooth_trajectory"][nm] = summarize(smooth_ious[(smooth_diag >= lo) & (smooth_diag < hi)])
        result["blob_refit"][nm] = summarize(blob_ious[(blob_diag >= lo) & (blob_diag < hi)])
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{args.split}: {len(seqs)} sequences, {len(smooth_ious)} labelled frames")
    print("label vs smooth trajectory (upper bound for a perfect smooth predictor):")
    for key in ["all"] + [nm for _, _, nm in SIZES]:
        s = result["smooth_trajectory"][key]
        if s["frames"]:
            print(f"  {key:16s} frames {s['frames']:7d}  mean IoU {s['mean_iou']:.3f}  >=0.5 {100*s['share_iou_ge_0.5']:5.1f}%  >=0.75 {100*s['share_iou_ge_0.75']:5.1f}%")
    print(f"  median |size residual| (w, h) px: {result['smooth_trajectory']['median_abs_size_residual_px']}")
    print("label vs thermal blob re-fit:")
    for key in ["all"] + [nm for _, _, nm in SIZES]:
        s = result["blob_refit"][key]
        if s["frames"]:
            print(f"  {key:16s} frames {s['frames']:7d}  mean IoU {s['mean_iou']:.3f}  >=0.5 {100*s['share_iou_ge_0.5']:5.1f}%  >=0.75 {100*s['share_iou_ge_0.75']:5.1f}%")


if __name__ == "__main__":
    main()
