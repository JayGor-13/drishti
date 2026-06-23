"""Losses and staged training controls for the DRISHTI procedure."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.drishti import DetectorStageOutput, DRISHTIOutput, DRISHTIPipeline


@dataclass
class DRISHTILossWeights:
    heatmap: float = 1.0
    cls: float = 1.0
    bbox: float = 2.0
    moe_balance: float = 1.0


STAGE_CHECKPOINTS = {
    "detector": "detector_best.pt",
    "temporal": "temporal_best.pt",
    "moe": "moe_best.pt",
}


def _extract_boxes(target: dict[str, Tensor] | None, device: torch.device, dtype: torch.dtype) -> Tensor:
    if target is None:
        return torch.zeros(0, 4, device=device, dtype=dtype)
    boxes = target.get("boxes")
    if boxes is None or boxes.numel() == 0:
        return torch.zeros(0, 4, device=device, dtype=dtype)
    return boxes.to(device=device, dtype=dtype).reshape(-1, 4).clamp(0.0, 1.0)


def heatmap_targets_from_frame_targets(
    frame_targets: list[list[dict[str, Tensor]]],
    heatmap_shape: tuple[int, int, int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create center-point heatmap labels for Stage 1+2 MotionCNN."""

    batch, time, channels, heat_h, heat_w = heatmap_shape
    if channels != 1:
        raise ValueError("DRISHTI heatmaps must have one channel")
    heatmaps = torch.zeros(batch, time, 1, heat_h, heat_w, device=device, dtype=dtype)
    for batch_idx, sequence_targets in enumerate(frame_targets):
        if not sequence_targets:
            continue
        for time_idx in range(time):
            target_index = len(sequence_targets) - 1 if time == 1 else min(time_idx, len(sequence_targets) - 1)
            boxes = _extract_boxes(sequence_targets[target_index], device, dtype)
            for box in boxes:
                x = min(heat_w - 1, max(0, int(float(box[0]) * heat_w)))
                y = min(heat_h - 1, max(0, int(float(box[1]) * heat_h)))
                heatmaps[batch_idx, time_idx, 0, y, x] = 1.0
    return heatmaps


def _detector_parts(
    heatmaps: Tensor,
    proposal_boxes: Tensor,
    object_logits: Tensor,
    boxes: Tensor,
    frame_targets: list[list[dict[str, Tensor]]],
    weights: DRISHTILossWeights,
) -> dict[str, Tensor]:
    heatmap_target = heatmap_targets_from_frame_targets(
        frame_targets=frame_targets,
        heatmap_shape=tuple(heatmaps.shape),
        device=heatmaps.device,
        dtype=heatmaps.dtype,
    )
    class_targets, box_targets, box_mask = assign_proposal_targets(
        proposal_boxes,
        frame_targets,
    )
    heatmap = F.mse_loss(heatmaps, heatmap_target)
    cls = F.binary_cross_entropy_with_logits(object_logits, class_targets)
    if box_mask.any():
        bbox = F.smooth_l1_loss(boxes[box_mask], box_targets[box_mask])
    else:
        bbox = boxes.new_zeros(())
    total = weights.heatmap * heatmap + weights.cls * cls + weights.bbox * bbox
    return {
        "det": total,
        "heatmap": heatmap,
        "cls": cls,
        "bbox": bbox,
    }


def assign_proposal_targets(
    proposal_boxes: Tensor,
    frame_targets: list[list[dict[str, Tensor]]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Assign final-frame target boxes to the nearest crop proposal center."""

    if proposal_boxes.ndim != 3 or proposal_boxes.shape[-1] != 4:
        raise ValueError("proposal_boxes must have shape [batch, crops, 4]")
    batch, crops, _ = proposal_boxes.shape
    class_targets = proposal_boxes.new_zeros(batch, crops, 1)
    box_targets = proposal_boxes.new_zeros(batch, crops, 4)
    box_mask = torch.zeros(batch, crops, device=proposal_boxes.device, dtype=torch.bool)
    proposal_centers = proposal_boxes[..., :2]

    for batch_idx, sequence_targets in enumerate(frame_targets):
        final_target = sequence_targets[-1] if sequence_targets else None
        boxes = _extract_boxes(final_target, proposal_boxes.device, proposal_boxes.dtype)
        used: set[int] = set()
        for box in boxes:
            distances = torch.norm(proposal_centers[batch_idx] - box[:2], dim=-1)
            order = torch.argsort(distances)
            slot = int(order[0])
            for candidate in order.tolist():
                if candidate not in used:
                    slot = candidate
                    break
            used.add(slot)
            class_targets[batch_idx, slot, 0] = 1.0
            box_targets[batch_idx, slot] = box
            box_mask[batch_idx, slot] = True
    return class_targets, box_targets, box_mask


def detector_loss(
    output: DRISHTIOutput,
    frame_targets: list[list[dict[str, Tensor]]],
    weights: DRISHTILossWeights | None = None,
) -> dict[str, Tensor]:
    """Stage 1+2+3 detector loss from ``procedure.md``.

    Total: MSE heatmap + BCE objectness + 2 * SmoothL1 bbox on positives.
    """

    weights = weights or DRISHTILossWeights()
    return _detector_parts(
        heatmaps=output.heatmaps,
        proposal_boxes=output.proposal_boxes,
        object_logits=output.object_logits,
        boxes=output.boxes,
        frame_targets=frame_targets,
        weights=weights,
    )


drishti_detector_loss = detector_loss


def detector_stage_loss(
    output: DetectorStageOutput,
    frame_targets: list[list[dict[str, Tensor]]],
    weights: DRISHTILossWeights | None = None,
) -> dict[str, Tensor]:
    """Detector-only loss for Stage 1 training."""

    weights = weights or DRISHTILossWeights()
    return _detector_parts(
        heatmaps=output.proposal.heatmap.unsqueeze(1),
        proposal_boxes=output.proposal.boxes,
        object_logits=output.object_logits,
        boxes=output.boxes,
        frame_targets=frame_targets,
        weights=weights,
    )


def moe_stage_loss(
    output: DRISHTIOutput,
    frame_targets: list[list[dict[str, Tensor]]],
    weights: DRISHTILossWeights | None = None,
) -> dict[str, Tensor]:
    """Stage 5 loss: detector loss plus MoE load balancing."""

    weights = weights or DRISHTILossWeights()
    parts = detector_loss(output, frame_targets, weights)
    total = parts["det"] + weights.moe_balance * output.load_balance_loss
    return {
        "loss": total,
        **parts,
        "load_balance": output.load_balance_loss,
    }


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def configure_drishti_training_stage(model: DRISHTIPipeline, stage: str) -> None:
    """Freeze/unfreeze modules according to DRISHTI-CORE stage order."""

    normalized = stage.lower().strip()
    aliases = {
        "stage1": "detector",
        "stage1+2+3": "detector",
        "stage1_2_3": "detector",
        "stage2": "temporal",
        "stage4": "temporal",
        "stage3": "moe",
        "stage5": "moe",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"detector", "temporal", "moe", "all"}:
        raise ValueError("stage must be detector, temporal, moe, or all")

    _set_trainable(model, False)
    model.crop_encoder.requires_grad_(False)
    model.crop_encoder.frozen = True

    if normalized == "detector":
        _set_trainable(model.motion_proposer, True)
        _set_trainable(model.detection_head, True)
    elif normalized == "temporal":
        _set_trainable(model.temporal_fusion, True)
    elif normalized == "moe":
        _set_trainable(model.moe, True)
    elif normalized == "all":
        _set_trainable(model, True)
        model.crop_encoder.requires_grad_(False)
        model.crop_encoder.frozen = True


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def stage_checkpoint_name(stage: str) -> str:
    normalized = stage.lower().strip()
    return STAGE_CHECKPOINTS.get(normalized, f"{normalized}_best.pt")


def scalar_metrics(parts: dict[str, Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in parts.items()}
