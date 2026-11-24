"""Which layer of a trained DRISHTI model knows best WHERE the drone is?

For every crop that contains the drone (the crop the loss assigns to it), this collects features from
five places and fits a ridge-regression probe from each to the drone's box inside the crop:

    enc_layer1  ResNet-18 layer1 (16x16, shallow, high resolution), pooled to 4x4
    enc_layer2  ResNet-18 layer2 (8x8, deeper), pooled to 4x4
    enc_out     encoder output after projection (what the temporal module receives)
    temporal    temporal-fusion output
    moe         MoE output (what the detection head reads in the original design)

The probe is fitted on half of the clips and scored on the other half (mean IoU of the predicted box
with the true box, and R^2 of the box centre). If an early layer locates better than the MoE output,
later modules are losing location information, which motivates the layer-mix head (version 3).
No training of the model itself; about one pass over the validation split.

Usage
-----
    python scripts/probe_layers.py --config <run>/config.yaml --checkpoint <run>/<stage>/checkpoints/<stage>_best.pt \
        --frames-root /workspace/data/antiuav410_frames --split val --out <run>/layer_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drishti_v2.experiments.common import build_loader, build_model, load_config, resolve_device  # noqa: E402
from drishti_v2.training.stage_losses import assign_crops  # noqa: E402

LAYERS = ["enc_layer1", "enc_layer2", "enc_out", "temporal", "moe"]


def box_iou_pairs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a1, a2 = a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2
    b1, b2 = b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2
    inter = (torch.minimum(a2, b2) - torch.maximum(a1, b1)).clamp_min(0).prod(-1)
    return inter / (a[:, 2:].clamp_min(0).prod(-1) + b[:, 2:].clamp_min(0).prod(-1) - inter).clamp_min(1e-9)


def ridge(train_x, train_y, test_x, lam):
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-6)
    xs, xt = (train_x - mean) / std, (test_x - mean) / std
    xs = torch.cat([xs, torch.ones(len(xs), 1)], 1)
    xt = torch.cat([xt, torch.ones(len(xt), 1)], 1)
    eye = torch.eye(xs.shape[1]); eye[-1, -1] = 0
    w = torch.linalg.solve(xs.T @ xs + lam * len(xs) * eye, xs.T @ train_y)
    return xt @ w


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probes of box location per layer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config, args.device)
    model = build_model(config, checkpoint=args.checkpoint, device=device).eval()
    if not hasattr(model.encoder, "layer1"):
        raise SystemExit("The probe needs the resnet18 crop encoder (encoder_backbone: resnet18).")
    loader = build_loader(config, None, args.split, config.eval_batch_size, shuffle=False, frames_root=args.frames_root, augment=False)

    captured: dict[str, torch.Tensor] = {}
    hooks = [
        model.encoder.layer1.register_forward_hook(lambda m, i, o: captured.__setitem__("enc_layer1", F.adaptive_avg_pool2d(o, 4).flatten(1))),
        model.encoder.layer2.register_forward_hook(lambda m, i, o: captured.__setitem__("enc_layer2", F.adaptive_avg_pool2d(o, 4).flatten(1))),
        model.encoder.proj.register_forward_hook(lambda m, i, o: captured.__setitem__("enc_out", o)),
        model.temporal.register_forward_hook(lambda m, i, o: captured.__setitem__("temporal", o.flatten(0, 1))),
        model.moe.register_forward_hook(lambda m, i, o: captured.__setitem__("moe", o[0].flatten(0, 1))),
    ]
    feats = {k: [] for k in LAYERS}
    targets_all = []
    for b_idx, batch in enumerate(loader):
        if args.max_batches is not None and b_idx >= args.max_batches:
            break
        frames = batch["frames"].to(device)
        out = model(frames)  # hooks keep the LAST call of each module = the last frame of the clip
        last = [clip[-1] for clip in batch["targets"]]
        labels, box_targets, _ = assign_crops(out.proposal_centers, out.objectness_logits, out.crop_boxes, last,
                                              out.crop_scale, tuple(out.heatmap.shape[-2:]))
        positive = labels.squeeze(-1).flatten() > 0.5
        if not positive.any():
            continue
        # keep only positives whose true box lies inside the crop (the head can only predict inside it)
        tgt = box_targets.flatten(0, 1)
        inside = positive & ((tgt[:, :2] - 0.5).abs() < 0.5).all(-1)
        if not inside.any():
            continue
        for k in LAYERS:
            feats[k].append(captured[k].detach().float().cpu()[inside.cpu()])
        targets_all.append(tgt[inside].detach().float().cpu())
    for h in hooks:
        h.remove()

    y = torch.cat(targets_all)
    n = len(y)
    if n < 50:
        raise SystemExit(f"Only {n} usable positive crops; run on more data.")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, te = perm[: n // 2], perm[n // 2 :]
    results = {"num_samples": n, "split": args.split, "checkpoint": args.checkpoint, "layers": {}}
    print(f"{n} positive crops ({len(te)} held out)")
    for k in LAYERS:
        x = torch.cat(feats[k])
        best = None
        for lam in (1e-3, 1e-2, 1e-1):
            pred = ridge(x[tr], y[tr], x[te], lam)
            pred = torch.cat([pred[:, :2], pred[:, 2:].clamp_min(1e-4)], 1)
            iou = box_iou_pairs(pred, y[te]).mean().item()
            r2 = 1 - ((pred[:, :2] - y[te, :2]) ** 2).sum() / ((y[te, :2] - y[tr, :2].mean(0)) ** 2).sum()
            if best is None or iou > best["mean_iou"]:
                best = {"mean_iou": iou, "center_r2": r2.item(), "ridge_lambda": lam, "dims": x.shape[1]}
        results["layers"][k] = best
        print(f"  {k:11s} dims {best['dims']:5d}  mean IoU {best['mean_iou']:.3f}  centre R^2 {best['center_r2']:.3f}")
    if getattr(model, "layer_mix", None) is not None:
        results["layer_mix_weights"] = dict(zip(["enc_out", "temporal", "moe"], model.layer_mix.weights().tolist()))
        print("  learned layer-mix weights:", results["layer_mix_weights"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
