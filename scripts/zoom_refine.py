"""Lever 7 (21 Oct): a zoom-in second look with the SAME detector (no training).

For every frame: take the base run's top box, cut a window around it from the motion composite (frame + the two
difference channels, exactly what the detector was trained on), enlarge the window to the detector's input size
and run the detector again. A 10-pixel drone becomes a 50-pixel one, a size the detector knows well, so the box
edges can be placed more precisely. The crop detection closest to the original box replaces it (its position
mapped back to the full frame); the original score is kept, so presence decisions do not change and the only
thing measured is box quality. Then the usual val-chosen scoring.

Usage (inside the drishti-yolo image):
  python scripts/zoom_refine.py --frames-root /workspace/data/antiuav410_frames --dyn-gap 8 \
      --base /workspace/runs/baseline_yolo11n_640_dyn_k8 --out /workspace/runs/baseline_yolo11n_640_dyn_k8_zoom
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "scripts" / "baselines"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Zoom-in second look with the same YOLO.")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--modality", default="infrared")
    parser.add_argument("--base", required=True, help="base run folder (train/weights/best.pt, raw_val, raw_test)")
    parser.add_argument("--dyn-gap", type=int, default=8)
    parser.add_argument("--zoom", type=float, default=6.0, help="window side = zoom x max(box w, h)")
    parser.add_argument("--min-window", type=int, default=96)
    parser.add_argument("--max-window", type=int, default=320)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import cv2
    from PIL import Image
    from ultralytics import YOLO
    from yolo_antiuav import MotionEncoder, sequences

    model = YOLO(str(Path(args.base) / "train" / "weights" / "best.pt"))
    out = Path(args.out)
    stats = {"frames": 0, "refined": 0, "mean_shift_px": 0.0}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        raw_dir = out / f"raw_{split}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for k, (name, frames, _) in enumerate(sequences(Path(args.frames_root), split, args.modality)):
            if args.max_seqs is not None and k >= args.max_seqs:
                break
            target = raw_dir / f"{name}.json"
            src = Path(args.base) / f"raw_{split}" / f"{name}.json"
            if target.exists() or not src.exists():
                continue
            with Image.open(frames[0]) as im:
                W, H = im.size
            base = json.loads(src.read_text(encoding="utf-8"))["frames"]
            enc = MotionEncoder("diff", gap=args.dyn_gap)
            result = {}
            todo = []  # (frame index, top box index, crop image, x0, y0, side)
            for t in range(len(frames)):
                item = base.get(str(t), {"boxes": [], "scores": []})
                result[str(t)] = {"boxes": [list(b) for b in item["boxes"]], "scores": list(item["scores"])}
                if not item["scores"]:
                    continue
                j = int(np.argmax(item["scores"]))
                cx, cy, w, h = item["boxes"][j]
                side = int(round(float(np.clip(args.zoom * max(w * W, h * H), args.min_window, args.max_window))))
                x0, y0 = int(round(cx * W - side / 2)), int(round(cy * H - side / 2))  # integer window: no rounding mismatch
                img = enc(frames, t)
                crop = np.zeros((side, side, 3), np.uint8)
                sx0, sy0 = max(x0, 0), max(y0, 0)
                sx1, sy1 = min(x0 + side, W), min(y0 + side, H)
                if sx1 > sx0 and sy1 > sy0:
                    crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
                crop = cv2.resize(crop, (args.imgsz, args.imgsz), interpolation=cv2.INTER_CUBIC)[..., ::-1].copy()
                todo.append((t, j, crop, x0, y0, side))
                if len(todo) == args.batch or t == len(frames) - 1:
                    preds = model.predict(source=[c[2] for c in todo], imgsz=args.imgsz, conf=0.001, iou=0.6, max_det=10,
                                          device=0, verbose=False, half=True)
                    for (ft, fj, _, fx0, fy0, fside), r in zip(todo, preds):
                        stats["frames"] += 1
                        if not len(r.boxes):
                            continue
                        bx = r.boxes.xywhn.cpu().numpy()  # in crop units
                        ob = np.array(result[str(ft)]["boxes"][fj])
                        full = np.stack([(fx0 + bx[:, 0] * fside) / W, (fy0 + bx[:, 1] * fside) / H, bx[:, 2] * fside / W, bx[:, 3] * fside / H], 1)
                        d = np.hypot((full[:, 0] - ob[0]) * W, (full[:, 1] - ob[1]) * H)
                        i = int(np.argmin(d))
                        if d[i] <= max(1.5 * max(ob[2] * W, ob[3] * H), 6.0):  # the same object
                            stats["refined"] += 1
                            stats["mean_shift_px"] += float(d[i])
                            result[str(ft)]["boxes"][fj] = [float(v) for v in full[i]]
                    todo = []
            target.write_text(json.dumps({"frames": result}), encoding="utf-8")
        print(f"zoom-refined predictions for {split} -> {raw_dir}", flush=True)
    if stats["refined"]:
        stats["mean_shift_px"] /= stats["refined"]
    (out / "zoom_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(stats)
    py = sys.executable
    subprocess.run([py, str(CODE_ROOT / "scripts" / "sweep_postprocess.py"), "--frames-root", args.frames_root,
                    "--modality", args.modality, "--val-raw", str(out / "raw_val"), "--test-raw", str(out / "raw_test"),
                    "--out", str(out / "postprocess_sweep.json")], check=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
