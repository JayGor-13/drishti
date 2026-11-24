"""Version 6, data step: dump per-frame CANDIDATES with arbitration cues, for training a learned arbiter.

Runs our trained model frame by frame exactly like version 5 (YOLO's top boxes as guided crops, memory
crops, 12 crops) and, for every frame, saves up to K_OURS + K_EXT candidate boxes:
  * our model's box for every crop (K_OURS = num_crops), and
  * the dense detector's top boxes (K_EXT = --ext-topk, padded),
each with a small vector of cues (see FEATURES), plus the true box for labels. Nothing is learned here.

One file per sequence: <out>/<split>/<sequence>.pt with
  feats  float16 [T, K, F]   cues per candidate        boxes [T, K, 4]  normalized cx, cy, w, h
  mask   bool    [T, K]      real candidate or padding gt    [T, 4]      true box (zeros if absent)
  visible bool   [T]         drone visible in the frame

Usage:
  python -m drishti_v2.experiments.dump_candidates --config <cfg> --checkpoint <ckpt> --frames-root <frames> \
      --split train --ext-dets "/workspace/runs/baseline_yolo11n_640/raw_{split}" --out /workspace/runs/v6_candidates
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from drishti_v2.data.dataset import AntiUAVExtractedFrameDataset, _box_exists, _pil_to_tensor, _resize_chw, _target_from_antiuav_box
from drishti_v2.data.utils import box_iou
from drishti_v2.experiments.common import build_model, load_config, resolve_device
from drishti_v2.tracker import TargetMemory

FEATURES = [
    "is_ours", "is_ext", "score", "score_rank_in_source",
    "src_motion", "src_edge", "src_grid", "src_guided", "src_pad",
    "cx", "cy", "w", "h", "log_area",
    "heat_at_center", "heat_max_frame", "gate_confidence",
    "best_iou_other_source", "score_of_best_other",
    "iou_prev_top_ours", "iou_prev_top_ext",
    "dist_memory_pred", "memory_active",
    "num_ext_dets", "top_ext_score",
]


def _iou_row(box: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    if others.numel() == 0:
        return torch.zeros(1)
    return box_iou(box.view(1, 4), others.view(-1, 4))[0]


@torch.no_grad()
def dump_sequence(model, config, sequence, device, external_frames: dict, ext_topk: int, ext_min_score: float,
                  use_memory: bool = True) -> dict:
    model.eval()
    model.reset_stream()
    memory = TargetMemory(threshold=config.objectness_threshold, max_age=30)  # same as the v5 run
    k_ours = config.num_crops
    k_total = k_ours + ext_topk
    n = sequence.num_frames
    feats = torch.zeros(n, k_total, len(FEATURES))
    boxes_all = torch.zeros(n, k_total, 4)
    mask = torch.zeros(n, k_total, dtype=torch.bool)
    gt = torch.zeros(n, 4)
    visible = torch.zeros(n, dtype=torch.bool)
    prev_top_ours = prev_top_ext = None
    for idx in range(n):
        frame, size = _pil_to_tensor(sequence.frame_paths[idx], config.image_channels)
        frame = _resize_chw(frame, config.image_height, config.image_width).unsqueeze(0).to(device)
        raw_box = sequence.boxes[idx] if idx < len(sequence.boxes) else []
        exists = sequence.exists[idx] if idx < len(sequence.exists) else _box_exists(raw_box)
        target = _target_from_antiuav_box(raw_box, exists, size, config.box_format)
        if target["boxes"].numel():
            gt[idx] = target["boxes"][0]
            visible[idx] = True

        item = external_frames.get(str(idx), {"boxes": [], "scores": []})
        eb = torch.tensor(item["boxes"], dtype=torch.float32).view(-1, 4)
        es = torch.tensor(item["scores"], dtype=torch.float32).view(-1)
        keep = es >= ext_min_score
        eb, es = eb[keep], es[keep]
        order = torch.argsort(es, descending=True)[:ext_topk]
        eb, es = eb[order], es[order]
        num_ext_all = int(keep.sum())

        guided = eb[:, :2].clamp(0, 1).unsqueeze(0) if eb.numel() else None
        mem_centers = memory.guided_centers() if use_memory else None
        if mem_centers is not None:
            guided = mem_centers if guided is None else torch.cat([guided, mem_centers], dim=1)
        out = model.forward_stream(frame, idx, guided.to(device) if guided is not None else None)

        ob = out.boxes[0].cpu()[:k_ours]
        os_ = torch.sigmoid(out.objectness_logits[0, :, 0]).cpu()[:k_ours]
        src = out.proposal_sources[0].cpu()[:k_ours]
        heat = out.heatmap[0, 0].cpu()
        hh, hw = heat.shape
        gate = float(out.motion_gate_confidence.reshape(-1)[0]) if out.motion_gate_confidence is not None else 1.0
        mem_pred = mem_centers[0, 0] if mem_centers is not None else None

        cands = [(ob[i], float(os_[i]), "ours", int(src[i])) for i in range(len(ob))]
        cands += [(eb[j], float(es[j]), "ext", -1) for j in range(len(eb))]
        ours_scores = sorted((c[1] for c in cands if c[2] == "ours"), reverse=True)
        ext_scores = sorted((c[1] for c in cands if c[2] == "ext"), reverse=True)
        for slot, (box, score, kind, source) in enumerate(cands):
            col = slot if kind == "ours" else k_ours + (slot - len(ob))
            others = eb if kind == "ours" else ob
            other_scores = es if kind == "ours" else os_
            ious = _iou_row(box, others)
            j = int(ious.argmax()) if others.numel() else 0
            cx, cy, w, h = [float(v) for v in box]
            f = {
                "is_ours": float(kind == "ours"), "is_ext": float(kind == "ext"), "score": score,
                "score_rank_in_source": float((ours_scores if kind == "ours" else ext_scores).index(score)),
                "src_motion": float(source == 0), "src_edge": float(source == 1), "src_grid": float(source == 2),
                "src_guided": float(source == 3), "src_pad": float(source == 4),
                "cx": cx, "cy": cy, "w": w, "h": h, "log_area": float(torch.log(torch.tensor(max(w * h, 1e-8)))),
                "heat_at_center": float(heat[min(max(int(round(cy * (hh - 1))), 0), hh - 1), min(max(int(round(cx * (hw - 1))), 0), hw - 1)]),
                "heat_max_frame": float(heat.max()), "gate_confidence": gate,
                "best_iou_other_source": float(ious[j]) if others.numel() else 0.0,
                "score_of_best_other": float(other_scores[j]) if others.numel() else 0.0,
                "iou_prev_top_ours": float(_iou_row(box, prev_top_ours)[0]) if prev_top_ours is not None else 0.0,
                "iou_prev_top_ext": float(_iou_row(box, prev_top_ext)[0]) if prev_top_ext is not None else 0.0,
                "dist_memory_pred": float((box[:2] - mem_pred).norm()) if mem_pred is not None else 1.0,
                "memory_active": float(mem_pred is not None),
                "num_ext_dets": float(num_ext_all), "top_ext_score": float(es[0]) if es.numel() else 0.0,
            }
            feats[idx, col] = torch.tensor([f[name] for name in FEATURES])
            boxes_all[idx, col] = box
            mask[idx, col] = True

        prev_top_ours = ob[int(os_.argmax())] if os_.numel() else None
        prev_top_ext = eb[0] if eb.numel() else None
        # memory follows the plain version-5 rule (best of ours and the detector), so the cues are arbiter-independent
        all_boxes = torch.cat([ob, eb]) if eb.numel() else ob
        all_scores = torch.cat([os_, es]) if es.numel() else os_
        memory.update(all_boxes, all_scores)
    return {"feats": feats.half(), "boxes": boxes_all, "mask": mask, "gt": gt, "visible": visible, "features": FEATURES}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump arbitration candidates for version 6.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--ext-dets", required=True, help="detector predictions; '{split}' is replaced")
    parser.add_argument("--ext-topk", type=int, default=3)
    parser.add_argument("--ext-min-score", type=float, default=0.05)
    parser.add_argument("--num-crops", type=int, default=12)
    parser.add_argument("--no-memory", action="store_true", help="ablation: no memory crops (and no memory cue)")
    parser.add_argument("--seq-stride", type=int, default=1, help="use every Nth sequence (e.g. 2 on train for speed)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config.num_crops = args.num_crops
    device = resolve_device(config, args.device)
    model = build_model(config, checkpoint=args.checkpoint, device=device)
    dataset = AntiUAVExtractedFrameDataset(frames_root=args.frames_root, split=args.split, modality=config.modality,
                                           num_frames=config.temporal_window, height=config.image_height,
                                           width=config.image_width, clip_stride=config.clip_stride, frame_stride=1,
                                           image_channels=config.image_channels, box_format=config.box_format)
    ext_dir = Path(args.ext_dets.replace("{split}", args.split))
    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    sequences = dataset.sequences[:: max(args.seq_stride, 1)]
    for seq in tqdm(sequences, desc=f"candidates ({args.split})"):
        target = out_dir / f"{seq.name}.pt"
        if target.exists():
            continue
        path = ext_dir / f"{seq.name}.json"
        ext = json.loads(path.read_text(encoding="utf-8"))["frames"] if path.exists() else {}
        torch.save(dump_sequence(model, config, seq, device, ext, args.ext_topk, args.ext_min_score, not args.no_memory), target)
    print(f"done: {len(sequences)} sequences -> {out_dir}")


if __name__ == "__main__":
    main()
