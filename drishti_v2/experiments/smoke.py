from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from tqdm import tqdm

from drishti_v2.data.dataset import _read_antiuav_json, _target_from_antiuav_box
from drishti_v2.evaluation.metrics import (
    detection_metrics,
    heatmap_motion_direction_metrics,
    heatmap_peak_metrics,
    temporal_detection_metrics,
)
from drishti_v2.evaluation.visualize import draw_boxes_on_image, save_metrics_bar, save_moe_diagnostics, save_training_curves
from drishti_v2.models import DRISHTIConfig, DRISHTIPipeline
from drishti_v2.training import StageLossFactory, apply_training_stage
from drishti_v2.tracker import SimpleTracker


def find_sequence_dir(root: str | Path, modality: str = "visible") -> Path:
    """Resolve either the sequence directory itself or a parent containing one."""

    root = Path(root)
    if (root / f"{modality}.mp4").exists() and (root / f"{modality}.json").exists():
        return root
    candidates = [
        path
        for path in sorted(root.rglob(f"{modality}.mp4"))
        if (path.parent / f"{modality}.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No {modality}.mp4/json pair found under {root}")
    return candidates[0].parent


def run_smoke(
    config: DRISHTIConfig,
    sequence_root: str | Path,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Overfit briefly on one Anti-UAV sequence and render a prediction video."""

    import cv2

    config.validate()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sequence_dir = find_sequence_dir(sequence_root, config.modality)
    output_dir = Path(output_dir or Path(config.smoke_output_video).parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, targets, fps = load_sequence(sequence_dir, config)
    if frames.shape[0] < 1:
        raise RuntimeError(f"No frames loaded from {sequence_dir}")

    model = DRISHTIPipeline(config).to(device)
    if config.checkpoint:
        payload = torch.load(config.checkpoint, map_location=device)
        model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload)

    history = overfit_sequence(model, config, frames, targets, device)
    video_path, predictions, heatmaps, moe_metrics = render_sequence_video(model, config, frames, targets, output_dir, fps, device)

    metrics = detection_metrics(predictions, targets, score_threshold=config.visualization_threshold)
    metrics.update(heatmap_peak_metrics([h.squeeze(0) for h in heatmaps], targets))
    metrics.update(heatmap_motion_direction_metrics(heatmaps, [targets]))
    metrics.update(temporal_detection_metrics(predictions, targets, config.visualization_threshold))
    metrics.update(moe_metrics)

    summary = {
        "sequence_dir": str(sequence_dir.resolve()),
        "video_path": str(video_path.resolve()),
        "num_frames": int(frames.shape[0]),
        "train_steps": int(config.smoke_train_steps),
        "metrics": metrics,
    }
    (output_dir / "smoke_summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
    save_metrics_bar({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, output_dir / "smoke_metrics.png")
    save_moe_diagnostics(metrics, output_dir / "smoke_moe_diagnostics.png")
    save_training_curves(history, output_dir / "smoke_training_curves.png")
    return summary


def load_sequence(sequence_dir: Path, config: DRISHTIConfig) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], float]:
    """Load and resize a direct Anti-UAV mp4/json sequence."""

    import cv2

    video_path = sequence_dir / f"{config.modality}.mp4"
    annotation_path = sequence_dir / f"{config.modality}.json"
    raw_boxes, exists = _read_antiuav_json(annotation_path)

    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 12.0)
    frames = []
    targets = []
    max_frames = config.smoke_max_frames if config.smoke_max_frames > 0 else int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in range(max_frames):
        ok, frame = capture.read()
        if not ok:
            break
        original_size = (int(frame.shape[0]), int(frame.shape[1]))
        tensor = _frame_to_tensor(frame, config.image_channels)
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(config.image_height, config.image_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        raw_box = raw_boxes[frame_idx] if frame_idx < len(raw_boxes) else []
        present = exists[frame_idx] if frame_idx < len(exists) else False
        frames.append(tensor)
        targets.append(_target_from_antiuav_box(raw_box, present, original_size, config.box_format))
    capture.release()
    return torch.stack(frames, dim=0), targets, fps


def overfit_sequence(
    model: DRISHTIPipeline,
    config: DRISHTIConfig,
    frames: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> list[dict[str, float]]:
    """Run a tiny end-to-end overfit loop on one sequence."""

    if config.smoke_train_steps <= 0:
        return []

    apply_training_stage(model, "stage4")
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=config.smoke_lr)
    loss_fn = StageLossFactory.make_loss("stage4", config=config)
    history = []
    window = min(config.temporal_window, frames.shape[0])
    max_start = max(frames.shape[0] - window, 0)

    progress = tqdm(range(config.smoke_train_steps), desc="smoke overfit", leave=False)
    for step in progress:
        start = step % (max_start + 1)
        clip = frames[start : start + window].unsqueeze(0).to(device)
        clip_targets = targets[start : start + window]
        output = model(clip)
        losses = loss_fn(output, targets=[clip_targets], all_heatmaps=output.all_heatmaps)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        row = {
            "epoch": float(step + 1),
            "train_loss": float(losses["loss"].detach().cpu()),
            "train_grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
        }
        history.append(row)
        progress.set_postfix(loss=f"{row['train_loss']:.4f}", grad=f"{row['train_grad_norm']:.2f}")
    return history


@torch.no_grad()
def render_sequence_video(
    model: DRISHTIPipeline,
    config: DRISHTIConfig,
    frames: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    output_dir: Path,
    fps: float,
    device: torch.device,
) -> tuple[Path, list[dict[str, torch.Tensor]], list[torch.Tensor], dict[str, Any]]:
    import cv2

    model.eval()
    model.reset_stream()
    tracker = SimpleTracker(config.tracker_dist_threshold, config.tracker_max_coast, config.tracker_birth_threshold)
    video_path = Path(config.smoke_output_video)
    if not video_path.is_absolute():
        video_path = output_dir / video_path.name
    video_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = int(frames.shape[-2]), int(frames.shape[-1])
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (width, height))
    predictions: list[dict[str, torch.Tensor]] = []
    heatmaps: list[torch.Tensor] = []
    expert_utilization = []
    expert_reuse = []
    moe_scalars = {"load_balance_cv": [], "router_entropy": [], "token_drop_rate": [], "expert_overlap": []}

    for frame_idx, frame in enumerate(tqdm(frames, desc="render boxes", leave=False)):
        tracker.predict()
        guided = tracker.get_guided_centers()
        guided = guided.to(device) if guided is not None else None
        output = model.forward_stream(frame.unsqueeze(0).to(device), frame_idx, guided)
        scores = torch.sigmoid(output.objectness_logits[0, :, 0]).detach().cpu()
        boxes = output.boxes[0].detach().cpu()
        tracker.update(boxes, output.objectness_logits[0].detach().cpu())
        predictions.append({"boxes": boxes, "scores": scores})
        heatmaps.append(output.heatmap.detach().cpu())

        diagnostics = output.moe_diagnostics
        expert_utilization.append(diagnostics.expert_utilization.detach().cpu())
        expert_reuse.append(diagnostics.expert_reuse_frequency.detach().cpu())
        for key in moe_scalars:
            moe_scalars[key].append(float(getattr(diagnostics, key).detach().cpu()))

        overlay = draw_boxes_on_image(
            frame,
            boxes,
            scores,
            threshold=config.visualization_threshold,
            top_k=1,
            gt_boxes=targets[frame_idx].get("boxes"),
        )
        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    writer.release()

    moe_metrics: dict[str, Any] = {}
    if expert_utilization:
        moe_metrics["expert_utilization"] = torch.stack(expert_utilization).mean(dim=0).tolist()
        moe_metrics["expert_reuse_frequency"] = torch.stack(expert_reuse).mean(dim=0).tolist()
        for key, values in moe_scalars.items():
            moe_metrics[key] = float(sum(values) / max(len(values), 1))
    return video_path, predictions, heatmaps, moe_metrics


def _frame_to_tensor(frame: Any, channels: int) -> torch.Tensor:
    import cv2

    if channels == 1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return torch.from_numpy(gray.copy()).float().div(255.0).unsqueeze(0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).float().div(255.0).permute(2, 0, 1).contiguous()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)
