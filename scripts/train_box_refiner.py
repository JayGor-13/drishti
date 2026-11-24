"""Version 7: a small box REFINER trained on the dense detector's own boxes (a learned "second look").

Why: with frame-dynamics YOLO the detector finds the drone in about 90% of frames, but its boxes are loose
(the oracle over all candidates reaches only 66.7 tracking accuracy). Chain-of-Focus refined OUR boxes
well but hurt YOLO's, because it never saw YOLO's boxes in training. This refiner is trained exactly on
the detector's boxes: crop a window around the box from the frame-dynamics image (frame + two motion
channels), and predict how to move and resize the box to match the true box.

Stages (--stage all): collect  crops from the TRAIN split (detector boxes with IoU >= 0.3 to the true box,
                               plus jittered true boxes), capped at --max-samples
                      train    small CNN, 4 outputs (dx, dy, dlog w, dlog h), smooth-L1 + GIoU loss
                      apply    refine every box of the detector's raw val/test predictions
                               (scores unchanged) -> <out>/raw_{split}, same format as before
                      score    val-chosen post-processing applied to test (scripts/sweep_postprocess.py),
                               so the number is directly comparable with the unrefined detector

Usage:
  python scripts/train_box_refiner.py --stage all --frames-root /workspace/data/antiuav410_frames \
      --det-root /workspace/runs/baseline_yolo11n_640_dyn --out /workspace/runs/baseline_yolo11n_640_dyn_refined
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "scripts" / "baselines"))
from drishti_v2.data.dataset import _box_exists, _candidate_boxes, _read_antiuav_json  # noqa: E402
from drishti_v2.data.utils import box_iou  # noqa: E402
from yolo_antiuav import dynamics_image, sequences  # noqa: E402

CROP = 96          # refiner input size (pixels)
MIN_WINDOW = 48    # window side = clamp(4 * max(w, h), MIN_WINDOW, MAX_WINDOW) pixels
MAX_WINDOW = 256


def window_of(box_px: np.ndarray) -> tuple[float, float, float]:
    cx, cy, w, h = box_px
    side = float(np.clip(4 * max(w, h), MIN_WINDOW, MAX_WINDOW))
    return cx, cy, side


def crop_window(image: np.ndarray, cx: float, cy: float, side: float) -> np.ndarray:
    """HxWx3 uint8 -> CROPxCROPx3 uint8 window centred at (cx, cy), zero padded at the border."""
    from PIL import Image

    h, w = image.shape[:2]
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))
    out = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return np.asarray(Image.fromarray(out).resize((CROP, CROP), Image.BILINEAR))


def encode(box_px: np.ndarray, gt_px: np.ndarray, side: float) -> np.ndarray:
    return np.array([(gt_px[0] - box_px[0]) / side, (gt_px[1] - box_px[1]) / side,
                     np.log(max(gt_px[2], 1) / max(box_px[2], 1)), np.log(max(gt_px[3], 1) / max(box_px[3], 1))], dtype=np.float32)


def decode(box: torch.Tensor, delta: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """box [N,4] cx,cy,w,h px; delta [N,4]; side [N] -> refined [N,4] px."""
    return torch.stack([box[:, 0] + delta[:, 0] * side, box[:, 1] + delta[:, 1] * side,
                        box[:, 2] * torch.exp(delta[:, 2].clamp(-2, 2)), box[:, 3] * torch.exp(delta[:, 3].clamp(-2, 2))], dim=1)


class Refiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        chans = [3, 32, 64, 128, 256]
        layers = []
        for a, b in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                       nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(256 + 2, 128), nn.ReLU(inplace=True), nn.Linear(128, 4))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)  # starts as "leave the box as it is"

    def forward(self, crops: torch.Tensor, rel_size: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.features(crops), rel_size], dim=1))


def to_input(crops_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(crops_u8).permute(0, 3, 1, 2).float().div_(255.0)


def gt_px_for(boxes, exists, idx) -> np.ndarray | None:
    raw = boxes[idx] if idx < len(boxes) else []
    present = exists[idx] if idx < len(exists) else _box_exists(raw)
    if not present:
        return None
    for x, y, w, h in _candidate_boxes(raw):
        if w > 0 and h > 0:
            return np.array([x + w / 2, y + h / 2, w, h], dtype=np.float32)
    return None


def collect(args) -> None:
    rng = random.Random(args.seed)
    det_dir = Path(args.det_root) / f"raw_{args.train_split}"
    seqs = list(sequences(Path(args.frames_root), args.train_split, args.modality))[:: max(args.seq_stride, 1)]
    crops, targets, rel, meta = [], [], [], []
    for name, frames, label in tqdm(seqs, desc="collect"):
        path = det_dir / f"{name}.json"
        if not path.exists():
            continue
        dets = json.loads(path.read_text(encoding="utf-8"))["frames"]
        boxes, exists = _read_antiuav_json(label)
        from PIL import Image

        with Image.open(frames[0]) as im:
            width, height = im.size
        for idx in range(0, len(frames), max(args.frame_stride, 1)):
            gt = gt_px_for(boxes, exists, idx)
            if gt is None:
                continue
            item = dets.get(str(idx), {"boxes": [], "scores": []})
            cands = []
            for (cx, cy, w, h), s in zip(item["boxes"], item["scores"]):
                if s < args.min_score:
                    continue
                b = np.array([cx * width, cy * height, w * width, h * height], dtype=np.float32)
                iou = float(box_iou(torch.tensor(b).view(1, 4) / torch.tensor([width, height, width, height]),
                                    torch.tensor(gt).view(1, 4) / torch.tensor([width, height, width, height]))[0, 0])
                if iou >= args.min_iou:
                    cands.append(b)
            cands = cands[: args.ext_topk]
            for _ in range(args.jitter_per_frame):  # jittered true boxes: covers offsets the detector makes elsewhere
                sc = np.exp(rng.uniform(-0.45, 0.45))
                cands.append(np.array([gt[0] + rng.uniform(-0.35, 0.35) * gt[2], gt[1] + rng.uniform(-0.35, 0.35) * gt[3],
                                       gt[2] * sc, gt[3] * np.exp(rng.uniform(-0.45, 0.45))], dtype=np.float32))
            if not cands:
                continue
            image = dynamics_image(frames, idx, args.dyn_gap)
            for b in cands:
                cx, cy, side = window_of(b)
                crops.append(crop_window(image, cx, cy, side))
                targets.append(encode(b, gt, side))
                rel.append(np.array([b[2] / side, b[3] / side], dtype=np.float32))
                meta.append((name, idx))
            if len(crops) >= args.max_samples * 1.5:
                break
        if len(crops) >= args.max_samples * 1.5:
            break
    if not crops:
        raise SystemExit(f"no training samples: no detector predictions found in {det_dir} (predict the train split first)")
    idx = list(range(len(crops)))
    rng.shuffle(idx)
    idx = idx[: args.max_samples]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"crops": torch.from_numpy(np.stack([crops[i] for i in idx])), "targets": torch.from_numpy(np.stack([targets[i] for i in idx])),
                "rel": torch.from_numpy(np.stack([rel[i] for i in idx]))}, out / "refiner_samples.pt")
    print(f"collected {len(idx)} samples from {len(set(meta[i][0] for i in idx))} sequences -> {out / 'refiner_samples.pt'}")


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """pred/target [N,4] cx,cy,w,h in window units."""
    def xyxy(b):
        return torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    p, t = xyxy(pred), xyxy(target)
    lt, rb = torch.maximum(p[:, :2], t[:, :2]), torch.minimum(p[:, 2:], t[:, 2:])
    inter = (rb - lt).clamp_min(0).prod(1)
    union = pred[:, 2] * pred[:, 3] + target[:, 2] * target[:, 3] - inter
    iou = inter / union.clamp_min(1e-6)
    lt2, rb2 = torch.minimum(p[:, :2], t[:, :2]), torch.maximum(p[:, 2:], t[:, 2:])
    enclose = (rb2 - lt2).clamp_min(0).prod(1)
    return (1 - (iou - (enclose - union) / enclose.clamp_min(1e-6))).mean()


def train(args) -> None:
    torch.manual_seed(args.seed)
    out = Path(args.out)
    data = torch.load(out / "refiner_samples.pt")
    crops, targets, rel = data["crops"], data["targets"], data["rel"]
    n = len(crops)
    perm = torch.randperm(n)
    n_val = max(n // 20, 1)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    device = torch.device(args.device)
    model = Refiner().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    base = torch.tensor([0.0, 0.0, 1.0, 1.0])  # the input box, in window units: centre 0,0 with size rel

    def batch_boxes(t, r):  # decode targets/preds into window units for the GIoU term
        return torch.stack([t[:, 0], t[:, 1], r[:, 0] * torch.exp(t[:, 2].clamp(-2, 2)), r[:, 1] * torch.exp(t[:, 3].clamp(-2, 2))], 1)

    def evaluate(idx):
        model.eval()
        ious_before, ious_after = [], []
        with torch.no_grad():
            for i in range(0, len(idx), 512):
                b = idx[i : i + 512]
                x, r, t = to_input(crops[b].numpy()).to(device), rel[b].to(device), targets[b].to(device)
                pred = model(x, r)
                gt_b = batch_boxes(t, r)
                def iou_of(bx):
                    p = torch.stack([bx[:, 0] - bx[:, 2] / 2, bx[:, 1] - bx[:, 3] / 2, bx[:, 0] + bx[:, 2] / 2, bx[:, 1] + bx[:, 3] / 2], 1)
                    g = torch.stack([gt_b[:, 0] - gt_b[:, 2] / 2, gt_b[:, 1] - gt_b[:, 3] / 2, gt_b[:, 0] + gt_b[:, 2] / 2, gt_b[:, 1] + gt_b[:, 3] / 2], 1)
                    inter = (torch.minimum(p[:, 2:], g[:, 2:]) - torch.maximum(p[:, :2], g[:, :2])).clamp_min(0).prod(1)
                    return inter / (bx[:, 2] * bx[:, 3] + gt_b[:, 2] * gt_b[:, 3] - inter).clamp_min(1e-6)
                ious_before.append(iou_of(torch.cat([torch.zeros(len(b), 2, device=device), r], 1)))
                ious_after.append(iou_of(batch_boxes(pred, r)))
        return float(torch.cat(ious_before).mean()), float(torch.cat(ious_after).mean())

    best = (-1.0, None)
    for epoch in range(args.epochs):
        model.train()
        order = tr_idx[torch.randperm(len(tr_idx))]
        total = 0.0
        for i in range(0, len(order), args.batch):
            b = order[i : i + args.batch]
            x = to_input(crops[b].numpy())
            if args.augment:  # horizontal flip, consistent with the target offset
                flip = torch.rand(len(b)) < 0.5
                x[flip] = x[flip].flip(-1)
            x, r, t = x.to(device), rel[b].to(device), targets[b].clone().to(device)
            if args.augment:
                t[flip.to(device), 0] = -t[flip.to(device), 0]
            pred = model(x, r)
            loss = nn.functional.smooth_l1_loss(pred, t, beta=0.05) + 0.5 * giou_loss(batch_boxes(pred, r), batch_boxes(t, r))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
        sched.step()
        before, after = evaluate(val_idx)
        print(f"epoch {epoch + 1}/{args.epochs}: loss {total / len(order):.4f}  held-out mean IoU before {before:.3f} -> after {after:.3f}", flush=True)
        if after > best[0]:
            best = (after, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    torch.save({"state_dict": best[1], "held_out_iou": best[0], "crop": CROP}, out / "refiner.pt")
    print(f"saved refiner (held-out mean IoU {best[0]:.3f})")


@torch.no_grad()
def apply(args) -> None:
    device = torch.device(args.device)
    model = Refiner().to(device)
    model.load_state_dict(torch.load(Path(args.out) / "refiner.pt", map_location="cpu")["state_dict"])
    model.eval()
    for split in [s.strip() for s in args.apply_splits.split(",") if s.strip()]:
        det_dir = Path(args.det_root) / f"raw_{split}"
        out_dir = Path(args.out) / f"raw_{split}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, frames, _ in tqdm(list(sequences(Path(args.frames_root), split, args.modality)), desc=f"apply ({split})"):
            target = out_dir / f"{name}.json"
            src = det_dir / f"{name}.json"
            if target.exists() or not src.exists():
                continue
            from PIL import Image

            with Image.open(frames[0]) as im:
                width, height = im.size
            dets = json.loads(src.read_text(encoding="utf-8"))["frames"]
            result = {}
            for idx in range(len(frames)):
                item = dets.get(str(idx), {"boxes": [], "scores": []})
                order = sorted(range(len(item["scores"])), key=lambda j: -item["scores"][j])
                keep = [j for j in order if item["scores"][j] >= args.min_score][: args.ext_topk]
                if not keep:
                    result[str(idx)] = {"boxes": [], "scores": []}
                    continue
                image = dynamics_image(frames, idx, args.dyn_gap)
                boxes_px, sides, crops, rel = [], [], [], []
                for j in keep:
                    cx, cy, w, h = item["boxes"][j]
                    b = np.array([cx * width, cy * height, w * width, h * height], dtype=np.float32)
                    wcx, wcy, side = window_of(b)
                    boxes_px.append(b)
                    sides.append(side)
                    crops.append(crop_window(image, wcx, wcy, side))
                    rel.append([b[2] / side, b[3] / side])
                pred = model(to_input(np.stack(crops)).to(device), torch.tensor(rel, device=device)).cpu()
                refined = decode(torch.tensor(np.stack(boxes_px)), pred, torch.tensor(sides))
                refined[:, 0::2] = refined[:, 0::2].clamp(0, width)
                refined[:, 1::2] = refined[:, 1::2].clamp(0, height)
                result[str(idx)] = {"boxes": (refined / torch.tensor([width, height, width, height])).tolist(),
                                    "scores": [item["scores"][j] for j in keep]}
            target.write_text(json.dumps({"frames": result}), encoding="utf-8")
        print(f"refined predictions for {split} -> {out_dir}")


def score(args) -> None:
    out = Path(args.out)
    py = sys.executable
    subprocess.run([py, str(CODE_ROOT / "scripts" / "score_predictions.py"), "--frames-root", args.frames_root, "--split", "test",
                    "--modality", args.modality, "--pred-dir", str(out / "raw_test"), "--name", "refined",
                    "--normalized", "--out", str(out / "score_test_raw.json")], check=True)
    subprocess.run([py, str(CODE_ROOT / "scripts" / "sweep_postprocess.py"), "--frames-root", args.frames_root,
                    "--modality", args.modality, "--val-raw", str(out / "raw_val"), "--test-raw", str(out / "raw_test"),
                    "--out", str(out / "postprocess_sweep.json")], check=True)
    ref = json.loads((Path(args.det_root) / "postprocess_sweep.json").read_text(encoding="utf-8"))["chosen_on_val"]["test"] \
        if (Path(args.det_root) / "postprocess_sweep.json").exists() else None
    new = json.loads((out / "postprocess_sweep.json").read_text(encoding="utf-8"))["chosen_on_val"]["test"]
    for label, t in (("detector alone", ref), ("detector + refiner", new)):
        if t:
            print(f"TEST {label:20s}: track {100*t['antiuav_tracking_accuracy']:.2f}  mAP50 {100*t['map50']:.1f}  "
                  f"mAP75 {100*t['map75']:.1f}  P {100*t['precision']:.1f}  R {100*t['recall']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and apply a box refiner on a dense detector's boxes.")
    parser.add_argument("--stage", default="all", choices=["collect", "train", "apply", "score", "all"])
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--modality", default="infrared")
    parser.add_argument("--det-root", required=True, help="detector run folder with raw_train, raw_val, raw_test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--apply-splits", default="val,test")
    parser.add_argument("--seq-stride", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=120000)
    parser.add_argument("--jitter-per-frame", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--min-iou", type=float, default=0.3, help="detector boxes with less overlap are not refinable; skipped in training")
    parser.add_argument("--ext-topk", type=int, default=3)
    parser.add_argument("--dyn-gap", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--augment", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    stages = ["collect", "train", "apply", "score"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if args.stage == "all" and stage == "collect" and (Path(args.out) / "refiner_samples.pt").exists():
            print("samples already collected; skipping")
            continue
        if args.stage == "all" and stage == "train" and (Path(args.out) / "refiner.pt").exists():
            print("refiner already trained; skipping")
            continue
        print(f"===== {stage}", flush=True)
        globals()[stage](args)
    print("ALL DONE")


if __name__ == "__main__":
    main()
