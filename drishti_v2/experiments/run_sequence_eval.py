"""Sequence-level streaming evaluation (the honest protocol).

Why this script exists
----------------------
``DRISHTIEvaluator`` scores only the last frame of every 5-frame clip, samples clips every
``clip_stride`` frames and never feeds tracker predictions back as guided crops. That is fine
for picking the best epoch, but it is not how Anti-UAV numbers are reported. This script walks
every frame of every sequence in order through ``forward_stream`` (one frame in, one result out,
no look-ahead), optionally feeds the tracker's predicted centres back as guided crops, and
computes every metric per sequence before averaging over sequences.

It also reports where detections come from (motion peak, edge, grid, guided crop) and how often
the target actually lies inside any crop window, so the motion-guidance claim can be checked.

Usage
-----
    python -m drishti_v2.experiments.run_sequence_eval --config configs/default.yaml \
        --frames-root /path/to/frames --split test --checkpoint results/.../finetune_best.pt

    # ablation row: same model, no tracker feedback
    python -m drishti_v2.experiments.run_sequence_eval ... --no-tracker
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from drishti_v2.data.dataset import (
    AntiUAVExtractedFrameDataset,
    _box_exists,
    _pil_to_tensor,
    _resize_chw,
    _target_from_antiuav_box,
)
from drishti_v2.evaluation.evaluator import (
    SOURCE_NAMES,
    attribute_detection,
    crop_half_extent,
    source_attribution_metrics,
)
from drishti_v2.evaluation.metrics import detection_metrics, heatmap_peak_metrics, temporal_detection_metrics
from drishti_v2.evaluation.postprocess import postprocess, smooth_sequence
from drishti_v2.experiments.common import add_common_args, build_model, load_config, resolve_device, seed_everything
from drishti_v2.models import DRISHTIConfig
from drishti_v2.tracker import SimpleTracker, TargetMemory


@torch.no_grad()
def evaluate_sequence(
    model: torch.nn.Module,
    config: DRISHTIConfig,
    sequence,
    device: torch.device,
    use_tracker: bool = True,
    max_frames: int | None = None,
    frame_stride: int = 1,
    threshold: float | None = None,
    nms_iou: float | None = None,
    top1: bool = False,
    raw_out: list | None = None,
    memory: bool = False,
    memory_max_age: int = 30,
    smooth: bool = False,
    smooth_hold: int = 8,
    init_first_box: bool = False,
    external: dict | None = None,
    ext_topk: int = 3,
    ext_min_score: float = 0.05,
    fuse_ext: float = 0.0,
    refine_ext: bool = False,
) -> dict[str, Any]:
    """Run one sequence frame by frame and return its metrics.

    ``nms_iou`` merges duplicate boxes from overlapping crops (also before they reach the tracker),
    ``top1`` keeps only the best box per frame. ``raw_out`` (a list) receives the raw per-crop
    predictions so post-processing can be swept later without the GPU (scripts/sweep_postprocess.py).
    """

    threshold = config.objectness_threshold if threshold is None else threshold
    model.eval()
    model.reset_stream()
    tracker = SimpleTracker(config.tracker_dist_threshold, config.tracker_max_coast, config.tracker_birth_threshold)
    # Track-1 mode (``init_first_box``): like the trackers, we are given the drone's box in the first frame.
    # We output it for that frame and seed the memory (and tracker) with it as a location prior.
    target_memory = TargetMemory(threshold=threshold, max_age=memory_max_age) if (memory or init_first_box) else None
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    heatmaps: list[torch.Tensor] = []
    source_counts = {name: 0 for name in SOURCE_NAMES.values()}
    tp_source_counts = {name: 0 for name in SOURCE_NAMES.values()}
    gt_visible_frames = 0
    gt_covered_frames = 0
    focus_windows = 0.0
    half_extent = crop_half_extent(model)

    num_frames = sequence.num_frames if max_frames is None else min(sequence.num_frames, max_frames)
    ext_frames = (external or {}).get(sequence.name, {})
    started = time.perf_counter()
    for stream_idx, frame_idx in enumerate(range(0, num_frames, max(frame_stride, 1))):
        frame, original_size = _pil_to_tensor(sequence.frame_paths[frame_idx], config.image_channels)
        frame = _resize_chw(frame, config.image_height, config.image_width).unsqueeze(0).to(device)
        raw_box = sequence.boxes[frame_idx] if frame_idx < len(sequence.boxes) else []
        exists = sequence.exists[frame_idx] if frame_idx < len(sequence.exists) else _box_exists(raw_box)
        target = _target_from_antiuav_box(raw_box, exists, original_size, config.box_format)

        # Version 5 (option A): a dense detector's boxes for this frame (precomputed), best first.
        ext_boxes = torch.empty(0, 4)
        ext_scores = torch.empty(0)
        if external is not None:
            item = ext_frames.get(str(frame_idx), {"boxes": [], "scores": []})
            eb = torch.tensor(item["boxes"], dtype=torch.float32).view(-1, 4)
            es = torch.tensor(item["scores"], dtype=torch.float32).view(-1)
            keep = es >= ext_min_score
            eb, es = eb[keep], es[keep]
            order = torch.argsort(es, descending=True)[:ext_topk]
            ext_boxes, ext_scores = eb[order], es[order]

        guided = None
        if use_tracker:
            tracker.predict()
            guided = tracker.get_guided_centers()
        if target_memory is not None:
            remembered = target_memory.guided_centers()
            if remembered is not None:
                # memory crops first: they get the guided slots before tracker tracks
                guided = remembered if guided is None else torch.cat([remembered, guided.cpu()], dim=1)
        if ext_boxes.numel() > 0:
            # detector boxes get the first guided crops, so the drone is inside a crop whenever YOLO saw it
            ext_centers = ext_boxes[:, :2].clamp(0, 1).unsqueeze(0)
            guided = ext_centers if guided is None else torch.cat([ext_centers, guided.cpu()], dim=1)
        if guided is not None:
            guided = guided.to(device)
        output = model.forward_stream(frame, stream_idx, guided)
        if output.focus_windows is not None:
            focus_windows += float(output.focus_windows.sum())
        logits = output.objectness_logits[0]
        raw_scores = torch.sigmoid(logits[:, 0]).detach().cpu()
        raw_boxes = output.boxes[0].detach().cpu()
        model_boxes, model_scores = raw_boxes, raw_scores  # one per crop; used for source attribution
        if ext_boxes.numel() > 0 and fuse_ext > 0:
            fused_boxes = ext_boxes
            if refine_ext and getattr(model, "focus", None) is not None:
                ext_logits = torch.logit(ext_scores.clamp(1e-4, 1 - 1e-4)).view(1, -1, 1).to(device)
                fused_boxes, _ = model.focus.refine(frame, ext_boxes.view(1, -1, 4).to(device), ext_logits)
                fused_boxes = fused_boxes[0].detach().cpu()
            raw_boxes = torch.cat([raw_boxes, fused_boxes], dim=0)
            raw_scores = torch.cat([raw_scores, (fuse_ext * ext_scores).clamp(0, 1)], dim=0)
        if init_first_box and stream_idx == 0 and target["boxes"].numel() > 0:
            raw_boxes = target["boxes"][:1].float().cpu()
            raw_scores = torch.ones(1)
            logits = torch.full((1, 1), 10.0, device=logits.device)
        if raw_out is not None:
            raw_out.append({"boxes": raw_boxes.tolist(), "scores": raw_scores.tolist()})
        boxes, scores = postprocess(raw_boxes, raw_scores, nms_iou=nms_iou, top1=top1)
        if target_memory is not None:
            target_memory.update(raw_boxes, raw_scores)
        if use_tracker:
            # Feed the de-duplicated boxes, so one drone does not spawn several tracks (and guided crops).
            tracker_logits = torch.logit(scores.clamp(1e-6, 1 - 1e-6)).unsqueeze(-1).to(logits.device)
            tracker.update(boxes.to(logits.device), tracker_logits)

        predictions.append({"boxes": boxes, "scores": scores})
        targets.append(target)
        heatmaps.append(output.heatmap[0].detach().cpu())
        attribution = attribute_detection(
            model_boxes,
            model_scores,
            output.proposal_sources[0].detach().cpu(),
            output.proposal_centers[0].detach().cpu(),
            target,
            threshold,
            half_extent,
        )
        if attribution["top_source"] is not None:
            source_counts[attribution["top_source"]] += 1
            if attribution["top_is_tp"]:
                tp_source_counts[attribution["top_source"]] += 1
        if attribution["gt_visible"]:
            gt_visible_frames += 1
            gt_covered_frames += int(attribution["gt_covered"])
    elapsed = max(time.perf_counter() - started, 1e-8)
    if smooth:
        # smoothing needs one box per frame
        predictions = smooth_sequence(
            [dict(zip(("boxes", "scores"), postprocess(p["boxes"], p["scores"], top1=True))) for p in predictions],
            threshold, hold=smooth_hold,
        )

    metrics: dict[str, Any] = detection_metrics(predictions, targets, score_threshold=threshold)
    metrics.update(temporal_detection_metrics(predictions, targets, threshold))
    metrics.update(heatmap_peak_metrics(heatmaps, targets))
    metrics.update(source_attribution_metrics(source_counts, tp_source_counts, gt_visible_frames, gt_covered_frames))
    metrics["frames"] = len(predictions)
    metrics["visible_frames"] = gt_visible_frames
    metrics["fps"] = len(predictions) / elapsed
    metrics["focus_windows_per_frame"] = focus_windows / max(len(predictions), 1)
    return metrics


def average_over_sequences(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Unweighted mean over sequences of every scalar metric (the Anti-UAV convention)."""

    keys = [key for key in rows[0] if isinstance(rows[0][key], (int, float)) and key != "sequence"]
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming, sequence-level DRISHTI evaluation.")
    add_common_args(parser)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None, help="Dataset split to evaluate (default: config.eval_split).")
    parser.add_argument("--no-tracker", action="store_true", help="Disable tracker feedback (guided crops).")
    parser.add_argument("--max-sequences", type=int, default=None, help="Evaluate only the first N sequences (smoke runs).")
    parser.add_argument("--max-frames", type=int, default=None, help="Evaluate only the first N frames of each sequence.")
    parser.add_argument("--threshold", type=float, default=None, help="Objectness threshold (default: config).")
    parser.add_argument("--nms-iou", type=float, default=None, help="Merge duplicate boxes overlapping more than this IoU (e.g. 0.3).")
    parser.add_argument("--top1", action="store_true", help="Keep only the best box per frame.")
    parser.add_argument("--save-raw", action="store_true",
                        help="Also save raw per-crop predictions (raw_<split>/<seq>.json) for scripts/sweep_postprocess.py.")
    parser.add_argument("--tag", default=None, help="Name used in the output files (default: tracker / no_tracker).")
    parser.add_argument("--memory", action="store_true", help="Idea A: keep crops at the drone's predicted position after it was last seen.")
    parser.add_argument("--memory-max-age", type=int, default=30, help="Frames the memory crops stay after the last confident detection.")
    parser.add_argument("--num-crops", type=int, default=None, help="Idea B: crop budget at inference (default: config, 8).")
    parser.add_argument("--smooth", action="store_true", help="Idea C: temporal smoothing of the one-box-per-frame output.")
    parser.add_argument("--smooth-hold", type=int, default=8, help="Frames a box is held through a dropout.")
    parser.add_argument("--external-dets", default=None,
                        help="Version 5: folder of a dense detector's raw predictions per sequence (normalized cx,cy,w,h); "
                             "'{split}' is replaced by the split, e.g. /workspace/runs/baseline_yolo11n_640/raw_{split}")
    parser.add_argument("--ext-topk", type=int, default=3, help="Detector boxes per frame used as guided crops / fused.")
    parser.add_argument("--ext-min-score", type=float, default=0.05)
    parser.add_argument("--fuse-ext", type=float, default=0.0,
                        help="Also output the detector's boxes, scores multiplied by this weight (0 = guided crops only).")
    parser.add_argument("--refine-ext", action="store_true", help="Refine the detector's boxes with our Chain-of-Focus head.")
    parser.add_argument("--init-first-box", action="store_true",
                        help="Track-1 mode: given the first-frame box (as the trackers are); used as a location prior.")
    parser.add_argument("--output-dir", default="results/sequence_eval")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.modality:
        config.modality = args.modality
    if args.box_format:
        config.box_format = args.box_format
    seed_everything(config.seed)
    device = resolve_device(config, args.device)
    if args.num_crops:
        config.num_crops = args.num_crops  # the crop engine reads this at run time
    checkpoint = args.checkpoint or config.checkpoint
    model = build_model(config, checkpoint=checkpoint, device=device)

    frames_root = args.frames_root or config.frames_root
    if not frames_root:
        raise ValueError("Provide --frames-root (extracted frame directories are required for sequence evaluation).")
    split = args.split or config.eval_split
    dataset = AntiUAVExtractedFrameDataset(
        frames_root=frames_root,
        split=split,
        modality=config.modality,
        num_frames=config.temporal_window,
        height=config.image_height,
        width=config.image_width,
        clip_stride=config.clip_stride,
        frame_stride=1,
        image_channels=config.image_channels,
        box_format=config.box_format,
        augment=False,
    )
    sequences = dataset.sequences[: args.max_sequences] if args.max_sequences else dataset.sequences
    frame_stride = args.frame_stride or 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("no_tracker" if args.no_tracker else "tracker")
    raw_dir = output_dir / f"raw_{split}_{tag}"
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
    external = None
    if args.external_dets:
        ext_dir = Path(args.external_dets.replace("{split}", split))
        external = {}
        for seq in sequences:
            path = ext_dir / f"{seq.name}.json"
            external[seq.name] = json.loads(path.read_text(encoding="utf-8"))["frames"] if path.exists() else {}
        missing = sum(1 for v in external.values() if not v)
        print(f"external detections from {ext_dir} ({missing} of {len(sequences)} sequences missing)")
    rows: list[dict[str, Any]] = []
    for sequence in tqdm(sequences, desc=f"sequences ({split})"):
        raw: list | None = [] if args.save_raw else None
        metrics = evaluate_sequence(
            model,
            config,
            sequence,
            device,
            use_tracker=not args.no_tracker,
            max_frames=args.max_frames,
            frame_stride=frame_stride,
            threshold=args.threshold,
            nms_iou=args.nms_iou,
            top1=args.top1,
            raw_out=raw,
            memory=args.memory,
            memory_max_age=args.memory_max_age,
            smooth=args.smooth,
            smooth_hold=args.smooth_hold,
            init_first_box=args.init_first_box,
            external=external,
            ext_topk=args.ext_topk,
            ext_min_score=args.ext_min_score,
            fuse_ext=args.fuse_ext,
            refine_ext=args.refine_ext,
        )
        rows.append({"sequence": sequence.name, **metrics})
        if raw is not None:
            # normalized cx, cy, w, h: score with scripts/score_predictions.py --normalized
            frames = {str(i): item for i, item in enumerate(raw)}
            (raw_dir / f"{sequence.name}.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")

    summary = average_over_sequences(rows)
    summary["num_sequences"] = len(rows)
    summary["total_frames"] = int(sum(row["frames"] for row in rows))
    summary["tracker_feedback"] = 0.0 if args.no_tracker else 1.0
    summary["checkpoint"] = str(checkpoint)
    summary["split"] = split
    summary["nms_iou"] = args.nms_iou
    summary["top1"] = bool(args.top1)
    summary["memory"] = bool(args.memory)
    summary["num_crops"] = config.num_crops
    summary["smooth"] = bool(args.smooth)
    summary["init_first_box"] = bool(args.init_first_box)
    summary["external_dets"] = args.external_dets
    summary["fuse_ext"] = args.fuse_ext
    summary["refine_ext"] = bool(args.refine_ext)
    summary["threshold"] = args.threshold if args.threshold is not None else config.objectness_threshold
    with (output_dir / f"per_sequence_{split}_{tag}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / f"summary_{split}_{tag}.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    headline = [
        "map50",
        "map75",
        "precision",
        "recall",
        "antiuav_tracking_accuracy",
        "heatmap_peak_within_5px",
        "gt_inside_some_crop",
        "top_box_source_motion",
        "top_box_source_edge",
        "top_box_source_grid",
        "top_box_source_guided",
        "fps",
    ]
    print(json.dumps({key: round(summary[key], 4) for key in headline if key in summary}, indent=2))
    print(f"Per-sequence results written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
