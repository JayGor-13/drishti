"""Version 6, learning step: train an ARBITER that picks, per frame, which candidate box to trust.

Input: the candidate dumps from drishti_v2/experiments/dump_candidates.py (train, val, test).
Each frame has up to 15 candidates (12 from our model's crops, 3 from the dense detector), each
with ~25 cues (source, own score, heatmap evidence, agreement between sources, agreement with the
previous frame, distance to the memory's prediction, ...).

The arbiter scores every candidate: "is this the drone?" (label: IoU with the true box >= 0.5).
  --arch mlp   each candidate on its own (small MLP)
  --arch attn  the candidates of a frame also look at each other (one self-attention layer), so the
               arbiter can reason "the detector and the motion branch agree here" (default)
--fit-on train  fit on the train split, choose the threshold on val (default)
--fit-on val    fit on a random half of the val sequences, choose the threshold on the other half.
                Use this because our model and YOLO were both TRAINED on the train split, so their scores
                there are over-confident; on val they behave as they will on test.
Test is never touched until the very end and is scored once.
The output per frame is the single highest-scoring candidate (the Anti-UAV protocol: one box per frame).

Also reports, on the same candidates, the hand-written rules for comparison:
  max_score   plain version-5 fusion (highest raw score wins)
  ours_only   our model's best box, ext_only  the detector's best box
  oracle      the best candidate in hindsight (upper bound of any arbiter on these candidates)

Usage:
  python scripts/train_arbiter.py --candidates /workspace/runs/v6_candidates --out /workspace/runs/v6_arbiter
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drishti_v2.data.utils import box_iou  # noqa: E402
from drishti_v2.evaluation.metrics import detection_metrics, temporal_detection_metrics  # noqa: E402

KEYS = ["antiuav_tracking_accuracy", "map50", "map75", "precision", "recall", "f1", "false_positives_per_image"]
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]


def load_dir(path: Path) -> list[dict]:
    seqs = []
    for f in sorted(path.glob("*.pt")):
        d = torch.load(f, map_location="cpu")
        n, k, _ = d["feats"].shape
        iou = torch.zeros(n, k)
        for t in range(n):
            if d["visible"][t]:
                iou[t] = box_iou(d["boxes"][t], d["gt"][t : t + 1])[:, 0]
        d["iou"] = iou * d["mask"]
        d["name"] = f.stem
        d["feats"] = d["feats"].float()
        seqs.append(d)
    return seqs


class Arbiter(nn.Module):
    def __init__(self, num_features: int, hidden: int = 64, attn: bool = True) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(num_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True) if attn else None
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """feats [B, K, F], mask [B, K] (True = real candidate) -> logits [B, K] (padding = -inf)."""
        h = self.inp(feats)
        if self.attn is not None:
            safe = mask.clone()
            safe[~mask.any(1), 0] = True  # frames with no candidate: avoid an all-masked softmax
            a, _ = self.attn(h, h, h, key_padding_mask=~safe)
            h = self.norm(h + a)
        return self.out(h).squeeze(-1).masked_fill(~mask, float("-inf"))


def frame_tensors(seqs: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (torch.cat([s["feats"] for s in seqs]), torch.cat([s["mask"] for s in seqs]), torch.cat([s["iou"] for s in seqs]))


def pick(seq: dict, scores: torch.Tensor) -> list[dict]:
    """scores [T, K] (masked = -inf or ignored) -> one box per frame."""
    preds = []
    for t in range(scores.shape[0]):
        if not seq["mask"][t].any():
            preds.append({"boxes": torch.empty(0, 4), "scores": torch.empty(0)})
            continue
        s = scores[t].masked_fill(~seq["mask"][t], float("-inf"))
        j = int(s.argmax())
        preds.append({"boxes": seq["boxes"][t, j : j + 1].float(), "scores": s[j : j + 1].float()})
    return preds


def metrics(seqs: list[dict], scores: list[torch.Tensor], threshold: float) -> dict:
    rows = []
    for seq, sc in zip(seqs, scores):
        preds = pick(seq, sc)
        targets = [{"boxes": seq["gt"][t : t + 1].float() if seq["visible"][t] else torch.empty(0, 4)} for t in range(len(preds))]
        m = detection_metrics(preds, targets, score_threshold=threshold)
        m.update(temporal_detection_metrics(preds, targets, threshold))
        rows.append(m)
    return {k: sum(r[k] for r in rows) / len(rows) for k in KEYS}


def rule_scores(seq: dict, rule: str, feature_names: list[str]) -> torch.Tensor:
    f = seq["feats"]
    score = f[..., feature_names.index("score")]
    ours = f[..., feature_names.index("is_ours")] > 0.5
    if rule == "max_score":
        return score
    if rule == "ours_only":
        return score.masked_fill(~ours, -1.0)
    if rule == "ext_only":
        return score.masked_fill(ours, -1.0)
    if rule == "oracle":  # best candidate in hindsight, scored 1 when it is a hit, else its raw score
        hit = seq["iou"] >= 0.5
        return torch.where(hit, 1.0 + seq["iou"], score * 0.0)
    raise ValueError(rule)


def choose_and_test(name: str, val: list[dict], test: list[dict], val_scores, test_scores) -> dict:
    grid = [(thr, metrics(val, val_scores, thr)) for thr in THRESHOLDS]
    thr, best_val = max(grid, key=lambda x: (round(x[1]["antiuav_tracking_accuracy"], 4), x[1]["map75"]))
    result = {"threshold": thr, "val": best_val, "test": metrics(test, test_scores, thr)}
    t = result["test"]
    print(f"TEST {name:12s} thr={thr:.2f}: track {100*t['antiuav_tracking_accuracy']:.2f}  mAP50 {100*t['map50']:.1f}  "
          f"mAP75 {100*t['map75']:.1f}  P {100*t['precision']:.1f}  R {100*t['recall']:.1f}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the version-6 arbiter on dumped candidates.")
    parser.add_argument("--candidates", required=True, help="folder with train/ val/ test/ candidate dumps")
    parser.add_argument("--arch", choices=["mlp", "attn"], default="attn")
    parser.add_argument("--fit-on", choices=["train", "val"], default="train")
    parser.add_argument("--val-every", type=int, default=3, help="check val every N epochs (keep the best)")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4096, help="frames per step")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    root, out = Path(args.candidates), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    test = load_dir(root / "test")
    if args.fit_on == "train":
        train, val = load_dir(root / "train"), load_dir(root / "val")
    else:
        pool = load_dir(root / "val")
        random.Random(args.seed).shuffle(pool)
        train, val = pool[: len(pool) // 2], pool[len(pool) // 2 :]
    names = train[0]["features"]
    print(f"fit on {args.fit_on}; sequences: fit {len(train)}  choose threshold {len(val)}  test {len(test)}; {len(names)} cues per candidate", flush=True)
    feats, mask, iou = frame_tensors(train)
    real = feats[mask]
    mean, std = real.mean(0), real.std(0).clamp_min(1e-3)
    norm = lambda x: (x - mean) / std  # noqa: E731
    label = (iou >= 0.5).float()
    pos = float(label[mask].mean())
    print(f"train frames {feats.shape[0]}, candidates {int(mask.sum())}, positives {100*pos:.1f}%", flush=True)

    device = torch.device(args.device)
    model = Arbiter(len(names), args.hidden, attn=args.arch == "attn").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor((1 - pos) / max(pos, 1e-4)).clamp(max=20).to(device))
    xs, ms, ys, ious = norm(feats).to(device), mask.to(device), label.to(device), iou.to(device)

    def score_all(seqs):
        model.eval()
        with torch.no_grad():
            return [torch.sigmoid(model(norm(s["feats"]).to(device), s["mask"].to(device))).cpu() for s in seqs]

    best = (-1.0, None)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(xs.shape[0], device=device)
        total = 0.0
        for i in range(0, len(order), args.batch):
            b = order[i : i + args.batch]
            logits = model(xs[b], ms[b])
            safe = logits.masked_fill(~ms[b], 0.0)
            # per-candidate "is it the drone" + a soft term that prefers the tighter of two hits
            loss = bce(safe, ys[b]) + 0.5 * nn.functional.binary_cross_entropy_with_logits(safe, ious[b], reduction="none")
            loss = (loss * ms[b]).sum() / ms[b].sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
        sched.step()
        if (epoch + 1) % args.val_every and epoch + 1 != args.epochs:
            print(f"epoch {epoch + 1}/{args.epochs}: loss {total / len(order):.4f}", flush=True)
            continue
        val_track = max(metrics(val, score_all(val), thr)["antiuav_tracking_accuracy"] for thr in (0.2, 0.3, 0.4, 0.5))
        print(f"epoch {epoch + 1}/{args.epochs}: loss {total / len(order):.4f}  val track (coarse) {100*val_track:.2f}", flush=True)
        if val_track > best[0]:
            best = (val_track, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    torch.save({"state_dict": best[1], "mean": mean, "std": std, "features": names, "arch": args.arch, "hidden": args.hidden}, out / "arbiter.pt")

    results = {"arbiter": choose_and_test(f"arbiter-{args.arch}", val, test, score_all(val), score_all(test))}
    for rule in ("max_score", "ours_only", "ext_only", "oracle"):
        results[rule] = choose_and_test(rule, val, test, [rule_scores(s, rule, names) for s in val], [rule_scores(s, rule, names) for s in test])
    results["settings"] = vars(args)
    (out / "arbiter_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out / 'arbiter_results.json'}")


if __name__ == "__main__":
    main()
