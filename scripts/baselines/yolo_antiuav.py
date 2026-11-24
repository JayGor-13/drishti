"""YOLO11 baseline on Anti-UAV410 (Track-2 setting: a detector, no first-frame box), end to end.

Stages (run all with --stage all):
  prepare  build a YOLO-format view of the frames: images/<split>/<seq>__<frame>.jpg (symlinks) and
           labels/<split>/<seq>__<frame>.txt ("0 cx cy w h", empty file = no drone). Train/val are
           subsampled with --train-stride / --val-stride for speed; prediction always uses every frame.
  train    fine-tune COCO-pretrained YOLO11 on Anti-UAV410 train, select on val.
  predict  run the trained model on EVERY frame of every val and test sequence, keep all boxes with
           their scores, and save them in the same raw format as our own evaluator
           (normalized cx, cy, w, h), so the very same scripts score both methods.
  score    raw test score (scripts/score_predictions.py) and the val-chosen post-processing applied to
           test (scripts/sweep_postprocess.py): exactly the protocol used for our model.
  cost     GFLOPs at the input size and measured per-frame latency (batch 1).

--frame-dynamics (the "detector with frame dynamics" baseline, idea of the 2025 Anti-UAV challenge winner,
arXiv 2505.04917): the three input channels become
  [ current frame | |current - previous frame| | |current - frame k back| ]   (k = --dyn-gap, default 3)
so the detector sees motion directly at its input. Still 3 channels, so COCO weights load as usual.
Only past frames are used (causal, like our model). Training images are written as files
(--yolo-root must then be a separate folder, e.g. yolo_a410_dyn); prediction builds them on the fly.

--motion-encoding chooses what the two motion channels are (all causal, all 3 channels in total):
  diff         [frame, |diff t-1|, |diff t-k|]                  (= --frame-dynamics)
  flow         [frame, |diff t-1|, optical-flow magnitude t-1]  (the 2025 winner also fed flow)
  timesurface  [frame, |diff t-1|, motion history: max(decay * previous, |diff t-1|)]
               (a decaying trace of recent motion, as in event-camera time surfaces; helps slow or hovering drones)
  winner       [frame, (frame - t-1)/2 + 128, (frame - t-2)/2 + 128]  signed differences to the two previous
               frames: the exact input of the 2025 Anti-UAV challenge winner (arXiv 2505.04917, Eq. 1)
  delta        [frame, |speed|, |acceleration|]: per pixel, a quadratic least-squares fit over the last 9 frames
               (as speech's delta / delta-delta features); robust to one noisy frame, sees slow and turning drones
  bgmed        [frame, |diff t-k|, |frame - median of the last 16 frames|]: background subtraction
  diff2        [frame, |diff t-k1|, |diff t-k2|] with --dyn-gap k1 and --dyn-gap2 k2 (e.g. 3 and 12): fast and slow drones
--compensate            ego-motion compensation: the global background shift between past frames and the current
                        one is estimated (phase correlation at 1/4 resolution, accumulated frame to frame) and past
                        frames are warped before differencing. The camera follows the drone, so without this the
                        difference channels show the moving background more than the drone.
  tprofile     [frame, comp. |diff t-k|, temporal anomaly]: how unusual the pixel is against its own compensated
               history of --profile-len frames (robust z-score, median / MAD), the input-level form of DeepPro
  lcm          [frame, comp. |diff t-k|, local contrast]: |pixel - mean of the surrounding ring| (classic LCM prior)
--nwd C                 train with the normalized Gaussian Wasserstein distance instead of IoU in label assignment
                        and box loss (NWD, Wang et al. 2022), scale-adaptive: distance measured in units of the
                        target's own size, metric = exp(-d / (C * size)); use 0 for plain IoU
--pretrained W          for a model given as a .yaml (e.g. yolov8n-p2.yaml): transfer the weights of W into it
--pick-weights best|last|val   which checkpoint predicts: Ultralytics' best.pt (chosen by ITS val fitness, which
                        saturates by epoch ~15 here, so it is often an early checkpoint), last.pt (end of the
                        schedule), or val = predict the val split with both and keep the one with the higher
                        Anti-UAV tracking accuracy on val (the choice is recorded in cost.json)
--clahe                 contrast-normalise the frame channel (CLAHE) for thermal crossover
--rect                  rectangular training batches at the native 640x512 (no letterbox padding; mosaic is off)
--oversample-attrs A,B  duplicate every training frame flagged with any of these label attributes (IC = thermal
                        crossover, DBC = dynamic background clutter, ...) --oversample-factor times
--motion-blank-prob p   blank both motion channels in a share p of TRAINING images (SpecAugment-style), so the
                        detector still finds a drone that hovers and leaves no motion trace
--tta                   test-time augmentation at prediction (Ultralytics augment=True: flips + scales)
--weights               predict / cost with these weights instead of <out>/train/weights/best.pt
--box-gain is YOLO's box-loss weight (default 7.5; 15 = "care twice as much about tight boxes").

Usage (inside the drishti-yolo image, from the code folder):
    python scripts/baselines/yolo_antiuav.py --stage all --model yolo11s.pt --imgsz 640 \
        --frames-root /workspace/data/antiuav410_frames --yolo-root /workspace/data/yolo_a410 \
        --out /workspace/runs/baseline_yolo11s_640
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
CODE_ROOT = Path(__file__).resolve().parents[2]


def dynamics_image(frames: list, idx: int, gap: int, gain: float = 4.0):
    """HxWx3 uint8 RGB: current gray frame, amplified |difference| to the previous frame and to `gap` frames back."""
    import numpy as np
    from PIL import Image

    def gray(i):
        with Image.open(frames[max(i, 0)]) as im:
            return np.asarray(im.convert("L"), dtype=np.float32)

    cur = gray(idx)
    d1 = np.clip(np.abs(cur - gray(idx - 1)) * gain, 0, 255) if idx > 0 else np.zeros_like(cur)
    dk = np.clip(np.abs(cur - gray(idx - gap)) * gain, 0, 255) if idx > 0 else np.zeros_like(cur)
    return np.stack([cur, d1, dk], axis=-1).astype(np.uint8)


class MotionEncoder:
    """Builds the 3-channel input of one frame. Call it in frame order within a sequence (reset() between
    sequences): 'timesurface' carries state from frame to frame, the others do not."""

    def __init__(self, encoding: str, gap: int = 3, gain: float = 4.0, decay: float = 0.85, flow_gain: float = 24.0,
                 gap2: int | None = None, clahe: bool = False, compensate: bool = False, profile_len: int = 16) -> None:
        self.encoding, self.gap, self.gain, self.decay, self.flow_gain = encoding, gap, gain, decay, flow_gain
        self.gap2, self.clahe, self.compensate, self.profile_len = gap2, clahe, compensate, profile_len
        # compensation accumulates frame-to-frame shifts, so the encoder must see every frame in order
        self.stateful = encoding == "timesurface" or compensate or encoding in ("tprofile", "lcm")
        self.reset()

    def reset(self) -> None:
        self.surface = None
        self.prev = None
        self.prev_idx = None
        self.cache = {}
        self.shift = {}      # frame index -> shift (dx, dy) of that frame relative to the previous one
        self.last_idx = None

    def _shift_between(self, frames, a: int, b: int):
        """Global translation that maps frame a onto frame b (b > a), accumulated from consecutive shifts."""
        import numpy as np

        dx = dy = 0.0
        for i in range(a + 1, b + 1):
            if i not in self.shift:
                self.shift[i] = self._consecutive_shift(frames, i)
            dx += self.shift[i][0]
            dy += self.shift[i][1]
        return dx, dy

    def _consecutive_shift(self, frames, i: int):
        import cv2
        import numpy as np

        if i <= 0:
            return 0.0, 0.0
        a = cv2.resize(self._cached(frames, i - 1), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        b = cv2.resize(self._cached(frames, i), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), win)
        if response < 0.05 or abs(dx) > a.shape[1] / 4 or abs(dy) > a.shape[0] / 4:
            return 0.0, 0.0  # textureless sky or an unreliable estimate: assume a still background
        return 4.0 * dx, 4.0 * dy

    def _past(self, frames, idx: int, back: int):
        """Frame idx-back as seen from frame idx: warped by the accumulated background shift if compensating."""
        import cv2
        import numpy as np

        j = max(idx - back, 0)
        img = self._cached(frames, j)
        if not self.compensate or j == idx:
            return img
        dx, dy = self._shift_between(frames, j, idx)
        if abs(dx) < 0.25 and abs(dy) < 0.25:
            return img
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        cur = self._cached(frames, idx)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_TRANSPARENT, dst=cur.copy())

    def _cached(self, frames, i):
        i = max(i, 0)
        if i not in self.cache:
            self.cache[i] = self._gray(frames, i)
            if len(self.cache) > 40:
                for old in sorted(self.cache)[: len(self.cache) - 40]:
                    del self.cache[old]
        return self.cache[i]

    def _gray(self, frames, i):
        import numpy as np
        from PIL import Image

        with Image.open(frames[max(i, 0)]) as im:
            return np.asarray(im.convert("L"), dtype=np.float32)

    def __call__(self, frames: list, idx: int):
        image = self._encode(frames, idx)
        if self.clahe:
            import cv2

            image = image.copy()
            image[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image[..., 0])
        return image

    def _encode(self, frames: list, idx: int):
        import numpy as np

        if self.encoding == "diff" and not self.compensate:
            return dynamics_image(frames, idx, self.gap, self.gain)
        if self.encoding in ("diff", "diff2", "tprofile", "lcm"):
            cur = self._cached(frames, idx)
            zero = np.zeros_like(cur)

            def adiff(back):
                return np.clip(np.abs(cur - self._past(frames, idx, back)) * self.gain, 0, 255) if idx > 0 else zero

            if self.encoding == "diff":
                return np.stack([cur, adiff(1), adiff(self.gap)], axis=-1).astype(np.uint8)
            if self.encoding == "diff2":
                return np.stack([cur, adiff(self.gap), adiff(self.gap2 or 12)], axis=-1).astype(np.uint8)
            if self.encoding == "tprofile":
                n = min(self.profile_len, idx)
                if n >= 4:
                    hist = np.stack([self._past(frames, idx, b) for b in range(1, n + 1)], 0)
                    med = np.median(hist, axis=0)
                    mad = np.median(np.abs(hist - med), axis=0) * 1.4826 + 2.0  # +2 grey levels: sensor noise floor
                    z = np.abs(cur - med) / mad
                    third = np.clip(z * 32.0, 0, 255)  # z = 8 saturates
                else:
                    third = zero
                return np.stack([cur, adiff(self.gap), third], axis=-1).astype(np.uint8)
            if self.encoding == "lcm":
                import cv2

                inner = cv2.blur(cur, (5, 5))
                outer = cv2.blur(cur, (15, 15))
                ring = (outer * 225.0 - inner * 25.0) / 200.0
                third = np.clip(np.abs(inner - ring) * self.gain, 0, 255)
                return np.stack([cur, adiff(self.gap), third], axis=-1).astype(np.uint8)
        if self.encoding == "delta":
            n = 9
            x = np.arange(-(n - 1), 1, dtype=np.float64)                       # t-8 ... t
            design = np.stack([np.ones_like(x), x, x ** 2], 1)
            w = np.linalg.pinv(design)                                          # rows: offset, speed, acceleration
            stack = np.stack([self._cached(frames, idx + int(k)) for k in x], 0)
            speed = np.tensordot(w[1], stack, axes=1)
            accel = np.tensordot(w[2], stack, axes=1)
            cur = stack[-1]
            return np.stack([cur, np.clip(np.abs(speed) * 8 * self.gain, 0, 255),
                             np.clip(np.abs(accel) * 64 * self.gain, 0, 255)], axis=-1).astype(np.uint8)
        if self.encoding == "bgmed":
            cur = self._cached(frames, idx)
            dk = np.clip(np.abs(cur - self._cached(frames, idx - self.gap)) * self.gain, 0, 255) if idx > 0 else np.zeros_like(cur)
            if idx > 0:
                bg = np.median(np.stack([self._cached(frames, idx - j) for j in range(1, 17)], 0), axis=0)
                fg = np.clip(np.abs(cur - bg) * self.gain, 0, 255)
            else:
                fg = np.zeros_like(cur)
            return np.stack([cur, dk, fg], axis=-1).astype(np.uint8)
        if self.encoding == "winner":
            cur = self._gray(frames, idx)
            d1 = (cur - self._gray(frames, idx - 1)) / 2 + 128 if idx > 0 else np.full_like(cur, 128)
            d2 = (cur - self._gray(frames, idx - 2)) / 2 + 128 if idx > 1 else np.full_like(cur, 128)
            return np.stack([cur, np.clip(d1, 0, 255), np.clip(d2, 0, 255)], axis=-1).astype(np.uint8)
        cur = self._gray(frames, idx)
        prev = self.prev if (self.prev is not None and self.prev_idx == idx - 1) else self._gray(frames, idx - 1)
        d1 = np.clip(np.abs(cur - prev) * self.gain, 0, 255) if idx > 0 else np.zeros_like(cur)
        if self.encoding == "flow":
            import cv2

            small_prev = cv2.resize(prev, None, fx=0.5, fy=0.5).astype(np.uint8)
            small_cur = cv2.resize(cur, None, fx=0.5, fy=0.5).astype(np.uint8)
            if idx > 0:
                flow = cv2.calcOpticalFlowFarneback(small_prev, small_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.sqrt((flow ** 2).sum(-1)) * 2.0  # back to full-resolution pixels
                third = cv2.resize(np.clip(mag * self.flow_gain, 0, 255), (cur.shape[1], cur.shape[0]))
            else:
                third = np.zeros_like(cur)
        elif self.encoding == "timesurface":
            if self.surface is None or self.prev_idx != idx - 1:
                self.surface = np.zeros_like(cur)
            self.surface = np.maximum(self.surface * self.decay, d1)
            third = self.surface
        else:
            raise ValueError(self.encoding)
        self.prev, self.prev_idx = cur, idx
        return np.stack([cur, d1, third], axis=-1).astype(np.uint8)


def make_encoder(args):
    if args.motion_encoding == "none":
        return None
    return MotionEncoder(args.motion_encoding, args.dyn_gap, decay=args.ts_decay, gap2=getattr(args, "dyn_gap2", None),
                         clahe=getattr(args, "clahe", False), compensate=getattr(args, "compensate", False),
                         profile_len=getattr(args, "profile_len", 16))


def predict_source(frames: list, idx: int, args, encoder=None):
    """What the detector is fed for one frame: the file path, or (motion encodings) the composite as a BGR array,
    which is exactly what Ultralytics gets when it reads the training JPEG (written as RGB by PIL) with OpenCV."""
    if encoder is None:
        return str(frames[idx])
    return encoder(frames, idx)[..., ::-1].copy()


def install_nwd(c_rel: float, *, assign_only: bool = False, fixed_scale: bool = False) -> None:
    """Replace Ultralytics' bbox_iou (used by the task-aligned assigner and BboxLoss, training only) with a
    scale-adaptive normalized Gaussian Wasserstein distance: boxes as Gaussians N(centre, diag(w^2/4, h^2/4)),
    metric = exp(-W2 / (c_rel * sqrt(w2 * h2))). Evaluation (box_iou, NMS) is untouched.
    The fixed-scale control is assignment-only, where coordinates are in pixels.
    """
    if fixed_scale and not assign_only:
        raise ValueError("Fixed-scale NWD requires pixel coordinates and is supported in assignment only")
    import torch
    import ultralytics.utils.loss as uloss
    import ultralytics.utils.tal as utal

    def nwd(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        if xywh:
            c1, w1, h1 = box1[..., :2], box1[..., 2:3], box1[..., 3:4]
            c2, w2, h2 = box2[..., :2], box2[..., 2:3], box2[..., 3:4]
        else:
            c1 = (box1[..., :2] + box1[..., 2:4]) / 2
            w1, h1 = (box1[..., 2:3] - box1[..., 0:1]), (box1[..., 3:4] - box1[..., 1:2])
            c2 = (box2[..., :2] + box2[..., 2:4]) / 2
            w2, h2 = (box2[..., 2:3] - box2[..., 0:1]), (box2[..., 3:4] - box2[..., 1:2])
        w2sq = ((c1 - c2) ** 2).sum(-1, keepdim=True) + ((w1 - w2) / 2) ** 2 + ((h1 - h2) / 2) ** 2
        size = torch.sqrt((w2.clamp_min(eps) * h2.clamp_min(eps)))
        scale = c_rel if fixed_scale else c_rel * size
        return torch.exp(-torch.sqrt(w2sq.clamp_min(0) + eps) / (scale + eps))

    if not assign_only:
        uloss.bbox_iou = nwd
    utal.bbox_iou = nwd
    where = "the assigner" if assign_only else "the assigner and the box loss"
    scale = f"{c_rel} pixels" if fixed_scale else f"{c_rel} x target size"
    print(f"NWD installed (c = {scale}) in {where}")


def load_model(weights: str):
    """YOLO family (yolov8n, yolov9t, yolov10n, yolo11n, yolo12n, yolo26n) or RT-DETR (rtdetr-l)."""
    if "rtdetr" in Path(str(weights)).name:
        from ultralytics import RTDETR

        return RTDETR(str(weights))
    from ultralytics import YOLO

    return YOLO(str(weights))


def trained_model(args, which: str | None = None):
    """The fine-tuned weights of this run (or --weights), loaded with the right class for --model."""
    which = which or getattr(args, "chosen_weights", None) or ("last" if getattr(args, "pick_weights", "best") == "last" else "best")
    best = Path(args.weights) if getattr(args, "weights", None) else Path(args.out) / "train" / "weights" / f"{which}.pt"
    if "rtdetr" in Path(args.model).name:
        from ultralytics import RTDETR

        return RTDETR(str(best))
    return load_model(str(best))


def sequences(frames_root: Path, split: str, modality: str):
    from drishti_v2.data.utils import list_image_files

    for seq_dir in sorted(p for p in (frames_root / split).iterdir() if p.is_dir()):
        frame_dir, label = seq_dir / modality, seq_dir / f"{modality}.json"
        if frame_dir.exists() and label.exists():
            yield seq_dir.name, list_image_files(frame_dir), label


def prepare(args) -> Path:
    from PIL import Image

    from drishti_v2.data.dataset import _box_exists, _candidate_boxes, _read_antiuav_json

    root = Path(args.yolo_root)
    counts = {}
    for split, stride in (("train", args.train_stride), ("val", args.val_stride)):
        img_dir, lab_dir = root / "images" / split, root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        encoder = make_encoder(args)
        over_attrs = [a.strip() for a in args.oversample_attrs.split(",") if a.strip()] if split == "train" else []
        for name, frames, label in sequences(Path(args.frames_root), split, args.modality):
            boxes, exists = _read_antiuav_json(label)
            flags = json.loads(Path(label).read_text(encoding="utf-8")) if over_attrs else {}
            with Image.open(frames[0]) as im:
                width, height = im.size
            stride = max(stride, 1)
            wanted = list(range(0, len(frames), stride))
            if encoder is not None:
                encoder.reset()
            # a stateful encoding must see every frame in order, even those not written
            all_written = all((img_dir / f"{name}__{i:06d}.jpg").exists() for i in wanted)
            indices = range(len(frames)) if (encoder is not None and encoder.stateful and not all_written) else wanted
            for idx in indices:
                image = encoder(frames, idx) if (encoder is not None and not all_written) else None
                if idx % stride:
                    continue
                stem = f"{name}__{idx:06d}"
                link = img_dir / (f"{stem}.jpg" if encoder is not None else f"{stem}{frames[idx].suffix}")
                if encoder is not None:
                    if not link.exists():
                        if split == "train" and args.motion_blank_prob > 0 and ((__import__('zlib').crc32(f'{name}{idx}'.encode()) % 1000) / 1000.0) < args.motion_blank_prob:
                            image = image.copy()
                            image[..., 1:] = 0  # no motion trace in this training image
                        Image.fromarray(image).save(link, quality=95)
                elif not link.exists():
                    try:
                        os.symlink(frames[idx].resolve(), link)
                    except OSError:  # e.g. Windows without symlink rights
                        shutil.copy(frames[idx], link)
                raw = boxes[idx] if idx < len(boxes) else []
                present = exists[idx] if idx < len(exists) else _box_exists(raw)
                lines = []
                if present:
                    for x, y, w, h in _candidate_boxes(raw):
                        if w > 0 and h > 0:
                            cx, cy = (x + w / 2) / width, (y + h / 2) / height
                            lines.append(f"0 {min(max(cx, 0), 1):.6f} {min(max(cy, 0), 1):.6f} {min(w / width, 1):.6f} {min(h / height, 1):.6f}")
                (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                n += 1
                flagged = any(isinstance(flags.get(a), list) and idx < len(flags[a]) and flags[a][idx] for a in over_attrs)
                for dup in range(1, args.oversample_factor if flagged else 1):
                    dlink = img_dir / f"{stem}_x{dup}{link.suffix}"
                    if not dlink.exists():
                        shutil.copy(link, dlink)
                    (lab_dir / f"{stem}_x{dup}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    n += 1
        counts[split] = n
    data_yaml = root / "antiuav410.yaml"
    data_yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: uav\n", encoding="utf-8")
    print(f"prepared YOLO view: {counts} images -> {data_yaml}")
    return data_yaml


def train(args) -> Path:
    out = Path(args.out)
    if args.nwd > 0:
        install_nwd(args.nwd)
    model = load_model(args.model)
    if str(args.model).endswith(".yaml") and args.pretrained:
        model.load(args.pretrained)
        print(f"weights of {args.pretrained} transferred into {args.model}")
    model.train(data=str(Path(args.yolo_root) / "antiuav410.yaml"), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                device=args.device, workers=args.workers, project=str(out), name="train", exist_ok=True, seed=args.seed,
                plots=False, patience=(args.patience if args.patience > 0 else args.epochs + 1), box=args.box_gain, rect=args.rect)
    best = out / "train" / "weights" / "best.pt"
    print(f"best weights: {best}")
    return best


def val_track_accuracy(args, raw_dir: Path) -> float:
    """Anti-UAV tracking accuracy on the val split for one raw folder (top box, duplicates merged, threshold 0.001)."""
    sys.path.insert(0, str(CODE_ROOT / "scripts"))
    from sweep_postprocess import load_split, score

    data = load_split(Path(args.frames_root), "val", args.modality, raw_dir)
    return float(score(data, 0.3, True, 0.001)["antiuav_tracking_accuracy"])


def pick_weights_on_val(args) -> str:
    """Predict val with best.pt and last.pt, keep the checkpoint with the higher val tracking accuracy."""
    out = Path(args.out)
    scores = {}
    for which in ("best", "last"):
        if not (out / "train" / "weights" / f"{which}.pt").exists():
            continue
        args.chosen_weights = which
        args.predict_splits_override = "val"
        predict_split(args, "val", out / f"raw_val_{which}")
        scores[which] = val_track_accuracy(args, out / f"raw_val_{which}")
        print(f"val tracking accuracy with {which}.pt: {100 * scores[which]:.2f}", flush=True)
    chosen = max(scores, key=scores.get)
    (out / "weights_choice.json").write_text(json.dumps({"chosen": chosen, "val_tracking_accuracy": scores}, indent=2), encoding="utf-8")
    # the chosen val predictions become raw_val (no second pass needed)
    raw_val = out / "raw_val"
    if raw_val.exists():
        shutil.rmtree(raw_val)
    shutil.copytree(out / f"raw_val_{chosen}", raw_val)
    print(f"chosen weights: {chosen}.pt", flush=True)
    return chosen


def predict_split(args, split: str, raw_dir: Path) -> None:
    model = trained_model(args)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for k, (name, frames, _) in enumerate(sequences(Path(args.frames_root), split, args.modality)):
        if args.max_seqs is not None and k >= args.max_seqs:
            break
        target = raw_dir / f"{name}.json"
        if target.exists():
            continue
        result = {}
        encoder = make_encoder(args)
        for start in range(0, len(frames), 64):
            batch = [predict_source(frames, i, args, encoder) for i in range(start, min(start + 64, len(frames)))]
            for k2, r in enumerate(model.predict(source=batch, imgsz=args.imgsz, conf=0.001, iou=0.6,
                                                 max_det=10, device=args.device, stream=True, verbose=False, half=True,
                                                 augment=args.tta)):
                result[str(start + k2)] = {"boxes": r.boxes.xywhn.cpu().tolist(), "scores": r.boxes.conf.cpu().tolist()}
        target.write_text(json.dumps({"frames": result}), encoding="utf-8")
    print(f"predictions for {split} -> {raw_dir}", flush=True)


def predict(args) -> None:
    out = Path(args.out)
    splits = [x.strip() for x in args.predict_splits.split(",") if x.strip()]
    if getattr(args, "pick_weights", "best") == "val" and not getattr(args, "weights", None):
        args.chosen_weights = pick_weights_on_val(args)
        splits = [s for s in splits if s != "val"]
    for split in splits:
        predict_split(args, split, out / f"raw_{split}")
    return


def _predict_old(args) -> None:  # kept for reference; superseded by predict() above
    out = Path(args.out)
    model = trained_model(args)
    for split in [x.strip() for x in args.predict_splits.split(",") if x.strip()]:
        raw_dir = out / f"raw_{split}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for k, (name, frames, _) in enumerate(sequences(Path(args.frames_root), split, args.modality)):
            if args.max_seqs is not None and k >= args.max_seqs:
                break
            target = raw_dir / f"{name}.json"
            if target.exists():
                continue
            result = {}
            encoder = make_encoder(args)
            for start in range(0, len(frames), 64):
                batch = [predict_source(frames, i, args, encoder) for i in range(start, min(start + 64, len(frames)))]
                for k, r in enumerate(model.predict(source=batch, imgsz=args.imgsz, conf=0.001, iou=0.6,
                                                    max_det=10, device=args.device, stream=True, verbose=False, half=True,
                                                    augment=args.tta)):
                    result[str(start + k)] = {"boxes": r.boxes.xywhn.cpu().tolist(), "scores": r.boxes.conf.cpu().tolist()}
            target.write_text(json.dumps({"frames": result}), encoding="utf-8")
        print(f"predictions for {split} -> {raw_dir}")


def score(args) -> None:
    out = Path(args.out)
    py = sys.executable
    subprocess.run([py, str(CODE_ROOT / "scripts" / "score_predictions.py"), "--frames-root", args.frames_root, "--split", "test",
                    "--modality", args.modality, "--pred-dir", str(out / "raw_test"), "--name", Path(args.model).stem,
                    "--normalized", "--out", str(out / "score_test_raw.json")], check=True)
    subprocess.run([py, str(CODE_ROOT / "scripts" / "sweep_postprocess.py"), "--frames-root", args.frames_root,
                    "--modality", args.modality, "--val-raw", str(out / "raw_val"), "--test-raw", str(out / "raw_test"),
                    "--out", str(out / "postprocess_sweep.json")], check=True)


def cost(args) -> None:
    import torch
    from ultralytics.utils.torch_utils import get_flops

    out = Path(args.out)
    model = trained_model(args)
    gflops = float(get_flops(model.model, imgsz=args.imgsz))
    params = sum(p.numel() for p in model.model.parameters())
    frames = next(sequences(Path(args.frames_root), "test", args.modality))[1][:220]
    encoder = make_encoder(args)
    for i in range(20):
        model.predict(source=predict_source(frames, i, args, encoder), imgsz=args.imgsz, device=args.device, verbose=False, half=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    for i in range(20, len(frames)):  # with --frame-dynamics this includes building the difference images
        model.predict(source=predict_source(frames, i, args, encoder), imgsz=args.imgsz, device=args.device, verbose=False, half=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - started) / max(len(frames) - 20, 1) * 1000
    choice = Path(args.out) / "weights_choice.json"
    info = {"weights": json.loads(choice.read_text(encoding="utf-8")) if choice.exists() else getattr(args, "pick_weights", "best"),
            "motion_encoding": args.motion_encoding, "dyn_gap": args.dyn_gap, "box_gain": args.box_gain, "seed": args.seed, "model": args.model, "imgsz": args.imgsz, "params": params, "gflops": gflops, "latency_ms_batch1": ms, "fps": 1000 / ms}
    (out / "cost.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(info)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11 baseline on Anti-UAV410, scored with the DRISHTI protocol.")
    parser.add_argument("--stage", default="all", choices=["prepare", "train", "predict", "score", "cost", "all"])
    parser.add_argument("--model", default="yolo11s.pt", help="COCO-pretrained: yolov8n.pt yolov9t.pt yolov10n.pt yolo11n.pt yolo12n.pt yolo26n.pt rtdetr-l.pt ...")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0", help="inside the container the chosen GPU is always 0")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--modality", default="infrared")
    parser.add_argument("--yolo-root", default=None, help="where the YOLO-format view is written (needed for prepare and train)")
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--val-stride", type=int, default=10)
    parser.add_argument("--out", required=True)
    parser.add_argument("--predict-splits", default="val,test", help="splits to predict, e.g. train,val,test (version 6 needs train)")
    parser.add_argument("--frame-dynamics", action="store_true", help="input = [frame, |diff to t-1|, |diff to t-k|]")
    parser.add_argument("--dyn-gap", type=int, default=3, help="k for the second difference channel")
    parser.add_argument("--motion-encoding", default="none", choices=["none", "diff", "diff2", "flow", "timesurface", "winner", "delta", "bgmed", "tprofile", "lcm"],
                        help="motion channels at the input (see the module docstring); --frame-dynamics = diff")
    parser.add_argument("--ts-decay", type=float, default=0.85, help="timesurface: decay per frame of the motion trace")
    parser.add_argument("--box-gain", type=float, default=7.5, help="YOLO box-loss weight (Ultralytics default 7.5)")
    parser.add_argument("--seed", type=int, default=42, help="training seed (42 = all runs so far)")
    parser.add_argument("--patience", type=int, default=0, help="early-stopping patience; 0 = train all epochs (default since 22 Oct: the val mAP used for stopping saturates early and stopped runs are under-trained; before, patience was epochs//4)")
    parser.add_argument("--motion-blank-prob", type=float, default=0.0, help="share of training images with blank motion channels")
    parser.add_argument("--dyn-gap2", type=int, default=None, help="diff2: the second (longer) gap")
    parser.add_argument("--clahe", action="store_true", help="contrast-normalise the frame channel")
    parser.add_argument("--compensate", action="store_true", help="ego-motion compensation before differencing")
    parser.add_argument("--profile-len", type=int, default=16, help="tprofile: frames of history")
    parser.add_argument("--nwd", type=float, default=0.0, help="NWD assignment + box loss with this c (0 = IoU)")
    parser.add_argument("--pretrained", default=None, help="weights to transfer into a .yaml model")
    parser.add_argument("--pick-weights", default="best", choices=["best", "last", "val"], help="which checkpoint predicts (see docstring)")
    parser.add_argument("--force-train", action="store_true", help="retrain even if train/weights/best.pt exists")
    parser.add_argument("--rect", action="store_true", help="rectangular (native 640x512) training batches")
    parser.add_argument("--oversample-attrs", default="", help="e.g. IC,DBC: duplicate flagged training frames")
    parser.add_argument("--oversample-factor", type=int, default=3)
    parser.add_argument("--tta", action="store_true", help="test-time augmentation at prediction")
    parser.add_argument("--weights", default=None, help="predict/cost with these weights instead of <out>/train/weights/best.pt")
    parser.add_argument("--max-seqs", type=int, default=None, help="smoke test: predict only the first N sequences per split")
    args = parser.parse_args()
    if args.yolo_root is None and args.stage in ("prepare", "train", "all"):
        parser.error("--yolo-root is required for --stage prepare, train or all")
    if args.frame_dynamics and args.motion_encoding == "none":
        args.motion_encoding = "diff"
    args.frame_dynamics = args.motion_encoding != "none"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    stages = ["prepare", "train", "predict", "score", "cost"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "prepare" and (Path(args.yolo_root) / "antiuav410.yaml").exists() and args.stage == "all":
            print("YOLO view already prepared; skipping")
            continue
        if stage == "train" and args.stage == "all" and not args.force_train and (Path(args.out) / "train" / "weights" / "best.pt").exists():
            print("trained weights exist; skipping training (use --force-train to retrain)")
            continue
        print(f"===== {stage}", flush=True)
        globals()[stage](args)
    print("ALL DONE")


if __name__ == "__main__":
    main()
