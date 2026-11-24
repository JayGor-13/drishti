from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from drishti_v2.assignment import linear_sum_assignment
from drishti_v2.models.config import DRISHTIConfig
from drishti_v2.models.pipeline import PipelineOutput
from drishti_v2.training.ciou_loss import ciou_loss
from drishti_v2.training.focal_loss import heatmap_focal_loss, sigmoid_focal_loss
from drishti_v2.training.motion_loss import motion_displacement_loss
from drishti_v2.training.temporal_loss import temporal_consistency_loss, trajectory_smoothness_loss


def last_targets(targets: list) -> list[dict]:
    return [clip[-1] for clip in targets] if targets and isinstance(targets[0], list) else targets


def make_gt_heatmaps(targets: list[dict], heatmap_size: tuple[int, int], device: torch.device) -> Tensor:
    batch = len(targets)
    height, width = heatmap_size
    heatmaps = torch.zeros(batch, 1, height, width, device=device)
    for batch_idx, target in enumerate(targets):
        boxes = target.get("boxes", torch.empty(0, 4)).to(device)
        if boxes.numel() == 0:
            continue
        dtype = boxes.dtype if boxes.is_floating_point() else torch.float32
        boxes = boxes.to(dtype=dtype)
        x = torch.arange(width, device=device, dtype=dtype).view(1, 1, width)
        y = torch.arange(height, device=device, dtype=dtype).view(1, height, 1)
        # Round the centre to an integer heatmap pixel so the Gaussian peak is exactly 1.0
        # there. ``heatmap_focal_loss`` treats only ``target >= 1.0`` as positives; with a
        # fractional centre no pixel ever reached 1.0, the loss saw zero positives and the
        # MotionCNN collapsed to an all-zero heatmap (CenterNet draws at the integer pixel and
        # leaves the sub-pixel residual to the offset head, which is what our offset head is for).
        center_x = (boxes[:, 0].clamp(0, 1) * max(width - 1, 1)).round().view(-1, 1, 1)
        center_y = (boxes[:, 1].clamp(0, 1) * max(height - 1, 1)).round().view(-1, 1, 1)
        half_w = boxes[:, 2].clamp_min(0.0) * max(width - 1, 1) / 2.0
        half_h = boxes[:, 3].clamp_min(0.0) * max(height - 1, 1) / 2.0
        sigma = torch.maximum(half_w, half_h).div(3.0).clamp_min(1.0).view(-1, 1, 1)
        gaussian = torch.exp(-((x - center_x).square() + (y - center_y).square()) / (2.0 * sigma.square()))
        heatmaps[batch_idx, 0] = gaussian.amax(dim=0).to(heatmaps.dtype)
    return heatmaps.clamp(0.0, 1.0)


def assign_crops(
    proposal_centers: Tensor,
    objectness_logits: Tensor,
    crop_boxes: Tensor,
    targets: list[dict],
    crop_scale: tuple[float, float],
    heatmap_size: tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor]:
    """Globally assign distinct crops to GTs and build prediction-independent targets."""

    batch, num_crops, _ = proposal_centers.shape
    labels = objectness_logits.new_zeros(batch, num_crops, 1)
    box_targets = crop_boxes.new_zeros(batch, num_crops, 4)
    offset_targets = crop_boxes.new_zeros(batch, num_crops, 2)
    scale = crop_boxes.new_tensor(crop_scale)
    heatmap_h, heatmap_w = heatmap_size
    heatmap_span = crop_boxes.new_tensor([max(heatmap_w - 1, 1), max(heatmap_h - 1, 1)])

    for batch_idx, target in enumerate(targets):
        boxes = target.get("boxes", torch.empty(0, 4)).detach().to(
            device=proposal_centers.device,
            dtype=proposal_centers.dtype,
        )
        if boxes.numel() == 0:
            continue

        # GT rows and crop columns produce a one-to-one assignment whenever K >= N_gt.
        cost = torch.cdist(boxes[:, :2].detach(), proposal_centers[batch_idx].detach())
        gt_indices, crop_indices = linear_sum_assignment(cost)
        for gt_idx, crop_idx in zip(gt_indices.tolist(), crop_indices.tolist()):
            gt = boxes[gt_idx]
            center = proposal_centers[batch_idx, crop_idx].detach()
            labels[batch_idx, crop_idx, 0] = 1.0

            target_offset = ((gt[:2] - center) * heatmap_span).clamp(-0.5, 0.5)
            corrected_target_center = center + target_offset / heatmap_span
            relative_xy = (gt[:2] - corrected_target_center) / scale + 0.5
            relative_wh = gt[2:] / scale
            box_targets[batch_idx, crop_idx] = torch.cat([relative_xy, relative_wh])
            offset_targets[batch_idx, crop_idx] = target_offset

    return labels, box_targets, offset_targets


class DetectionLossMixin:
    focal_gamma: float
    focal_alpha: float
    heatmap_alpha: float
    heatmap_beta: float

    def detection_terms(self, output: PipelineOutput, targets: list) -> dict[str, Tensor]:
        heatmaps = output.all_heatmaps or [output.heatmap]
        centers_seq = output.proposal_centers_seq or [output.proposal_centers]
        logits_seq = output.objectness_logits_seq or [output.objectness_logits]
        crop_boxes_seq = output.crop_boxes_seq or [output.crop_boxes]
        offsets_seq = output.center_offsets_seq or [output.center_offsets]
        steps = min(len(heatmaps), len(centers_seq), len(logits_seq), len(crop_boxes_seq), len(offsets_seq))

        heatmap_losses: list[Tensor] = []
        cls_losses: list[Tensor] = []
        bbox_losses: list[Tensor] = []
        offset_losses: list[Tensor] = []
        labels_seq: list[Tensor] = []

        for time_idx in range(steps):
            step_targets = _targets_at_time(targets, time_idx)
            heatmap_size = tuple(heatmaps[time_idx].shape[-2:])
            gt_heatmap = make_gt_heatmaps(
                step_targets,
                heatmap_size,
                heatmaps[time_idx].device,
            ).to(heatmaps[time_idx].dtype)
            heatmap_losses.append(
                heatmap_focal_loss(heatmaps[time_idx], gt_heatmap, self.heatmap_alpha, self.heatmap_beta)
            )

            labels, box_targets, offset_targets = assign_crops(
                centers_seq[time_idx],
                logits_seq[time_idx],
                crop_boxes_seq[time_idx],
                step_targets,
                output.crop_scale,
                heatmap_size,
            )
            labels_seq.append(labels.squeeze(-1))
            cls_losses.append(
                sigmoid_focal_loss(logits_seq[time_idx], labels, self.focal_gamma, self.focal_alpha)
            )
            positive = labels.squeeze(-1) > 0.5
            if positive.any():
                bbox_losses.append(ciou_loss(crop_boxes_seq[time_idx][positive], box_targets[positive]))
                offset_losses.append(F.smooth_l1_loss(offsets_seq[time_idx][positive], offset_targets[positive]))
            else:
                zero = logits_seq[time_idx].sum() * 0.0
                bbox_losses.append(zero)
                offset_losses.append(zero)

        if not heatmap_losses:
            raise ValueError("Pipeline output did not contain any supervised time steps")
        return {
            "heatmap": torch.stack(heatmap_losses).mean(),
            "cls": torch.stack(cls_losses).mean(),
            "bbox": torch.stack(bbox_losses).mean(),
            "offset": torch.stack(offset_losses).mean(),
            "labels": labels_seq[-1].unsqueeze(-1),
            "labels_seq": labels_seq,
        }


def _targets_at_time(targets: list, time_idx: int) -> list[dict]:
    if not targets or not isinstance(targets[0], list):
        return targets
    result = []
    for clip in targets:
        if not clip:
            result.append({"boxes": torch.empty(0, 4)})
        else:
            result.append(clip[min(time_idx, len(clip) - 1)])
    return result


class Stage1Loss(nn.Module, DetectionLossMixin):
    def __init__(
        self,
        w_hm: float = 1.0,
        w_cls: float = 1.0,
        w_box: float = 2.0,
        w_offset: float = 1.0,
        w_motion: float = 0.5,
        w_gate: float = 0.01,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        heatmap_alpha: float = 2.0,
        heatmap_beta: float = 4.0,
        motion_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.w_hm = w_hm
        self.w_cls = w_cls
        self.w_box = w_box
        self.w_offset = w_offset
        self.w_motion = w_motion
        self.w_gate = w_gate
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.heatmap_alpha = heatmap_alpha
        self.heatmap_beta = heatmap_beta
        self.motion_temperature = motion_temperature

    def forward(self, output: PipelineOutput, targets: list, all_heatmaps: list[Tensor] | None = None) -> dict[str, Tensor]:
        terms = self.detection_terms(output, targets)
        all_heatmaps = all_heatmaps or output.all_heatmaps
        motion = (
            motion_displacement_loss(all_heatmaps, targets, self.motion_temperature)
            if all_heatmaps is not None and targets and isinstance(targets[0], list)
            else output.objectness_logits.sum() * 0.0
        )
        gate = (1.0 - output.motion_gate_confidence).mean()
        total = (
            self.w_hm * terms["heatmap"]
            + self.w_cls * terms["cls"]
            + self.w_box * terms["bbox"]
            + self.w_offset * terms["offset"]
            + self.w_motion * motion
            + self.w_gate * gate
        )
        return {
            "loss": total,
            "heatmap": terms["heatmap"],
            "cls": terms["cls"],
            "bbox": terms["bbox"],
            "offset": terms["offset"],
            "motion_disp": motion,
            "gate": gate,
            "balance": output.balance_loss,
        }


class Stage2Loss(nn.Module, DetectionLossMixin):
    def __init__(
        self,
        w_hm: float = 0.5,
        w_cls: float = 1.0,
        w_box: float = 2.0,
        w_offset: float = 1.0,
        w_tc: float = 0.3,
        w_sm: float = 0.1,
        sigma_spatial: float = 0.1,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        heatmap_alpha: float = 2.0,
        heatmap_beta: float = 4.0,
    ) -> None:
        super().__init__()
        self.w_hm = w_hm
        self.w_cls = w_cls
        self.w_box = w_box
        self.w_offset = w_offset
        self.w_tc = w_tc
        self.w_sm = w_sm
        self.sigma_spatial = sigma_spatial
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.heatmap_alpha = heatmap_alpha
        self.heatmap_beta = heatmap_beta

    def forward(
        self,
        output: PipelineOutput,
        targets: list,
        logits_seq: list[Tensor] | None = None,
        centers_seq: list[Tensor] | None = None,
        boxes_seq: list[Tensor] | None = None,
    ) -> dict[str, Tensor]:
        terms = self.detection_terms(output, targets)
        logits_seq = logits_seq or output.objectness_logits_seq
        centers_seq = centers_seq or output.proposal_centers_seq
        boxes_seq = boxes_seq or output.boxes_seq
        zero = output.objectness_logits.sum() * 0.0
        temporal = (
            temporal_consistency_loss(logits_seq, centers_seq, self.sigma_spatial)
            if logits_seq is not None and centers_seq is not None
            else zero
        )
        smooth = zero
        if boxes_seq is not None:
            labels_seq = terms["labels_seq"]
            smooth = trajectory_smoothness_loss(boxes_seq, labels_seq[-len(boxes_seq) :])

        total = (
            self.w_hm * terms["heatmap"]
            + self.w_cls * terms["cls"]
            + self.w_box * terms["bbox"]
            + self.w_offset * terms["offset"]
            + self.w_tc * temporal
            + self.w_sm * smooth
        )
        return {
            "loss": total,
            "heatmap": terms["heatmap"],
            "cls": terms["cls"],
            "bbox": terms["bbox"],
            "offset": terms["offset"],
            "temporal_consist": temporal,
            "traj_smooth": smooth,
            "balance": output.balance_loss,
        }


class Stage3Loss(nn.Module, DetectionLossMixin):
    def __init__(
        self,
        w_cls: float = 1.0,
        w_box: float = 2.0,
        w_offset: float = 1.0,
        w_bal: float = 0.01,
        w_zloss: float = 0.001,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        heatmap_alpha: float = 2.0,
        heatmap_beta: float = 4.0,
    ) -> None:
        super().__init__()
        self.w_cls = w_cls
        self.w_box = w_box
        self.w_offset = w_offset
        self.w_bal = w_bal
        self.w_zloss = w_zloss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.heatmap_alpha = heatmap_alpha
        self.heatmap_beta = heatmap_beta

    def forward(self, output: PipelineOutput, targets: list) -> dict[str, Tensor]:
        terms = self.detection_terms(output, targets)
        z_loss = output.moe_diagnostics.router_z_loss
        total = (
            self.w_cls * terms["cls"]
            + self.w_box * terms["bbox"]
            + self.w_offset * terms["offset"]
            + self.w_bal * output.balance_loss
            + self.w_zloss * z_loss
        )
        return {
            "loss": total,
            "cls": terms["cls"],
            "bbox": terms["bbox"],
            "offset": terms["offset"],
            "balance": output.balance_loss,
            "z_loss": z_loss,
        }


class Stage4Loss(nn.Module):
    def __init__(self, config: DRISHTIConfig | None = None) -> None:
        super().__init__()
        self.stage1 = Stage1Loss(
            w_motion=0.3,
            w_gate=(config.w_gate_sparsity if config else 0.01),
            w_offset=(config.w_subpixel_offset if config else 1.0),
            focal_gamma=(config.focal_gamma if config else 2.0),
            focal_alpha=(config.focal_alpha if config else 0.25),
        )
        self.stage2 = Stage2Loss(
            w_hm=0.5,
            w_tc=0.15,
            w_sm=0.05,
            w_offset=(config.w_subpixel_offset if config else 1.0),
            sigma_spatial=(config.sigma_spatial_consist if config else 0.1),
        )
        self.stage3 = Stage3Loss(
            w_offset=(config.w_subpixel_offset if config else 1.0),
            w_bal=(config.moe_balance_weight if config else 0.01),
            w_zloss=(config.router_z_loss_weight if config else 0.001),
        )

    def forward(self, output: PipelineOutput, targets: list, **kwargs: Any) -> dict[str, Tensor]:
        s1 = self.stage1(output, targets, kwargs.get("all_heatmaps"))
        s2 = self.stage2(output, targets, kwargs.get("logits_seq"), kwargs.get("centers_seq"), kwargs.get("boxes_seq"))
        s3 = self.stage3(output, targets)
        total = (
            s1["loss"]
            + self.stage2.w_tc * s2["temporal_consist"]
            + self.stage2.w_sm * s2["traj_smooth"]
            + self.stage3.w_bal * s3["balance"]
            + self.stage3.w_zloss * s3["z_loss"]
        )
        return {
            "loss": total,
            "heatmap": s1["heatmap"],
            "cls": s1["cls"],
            "bbox": s1["bbox"],
            "offset": s1["offset"],
            "motion_disp": s1["motion_disp"],
            "temporal_consist": s2["temporal_consist"],
            "traj_smooth": s2["traj_smooth"],
            "balance": s3["balance"],
            "z_loss": s3["z_loss"],
        }


class StageLossFactory:
    @staticmethod
    def make_loss(stage: str, config: DRISHTIConfig | None = None, **overrides: Any) -> nn.Module:
        stage = stage.lower()
        defaults = StageLossFactory._defaults(config)
        defaults.update(overrides)

        if stage in {"stage1", "detector"}:
            return Stage1Loss(**_pick(defaults, Stage1Loss))
        if stage in {"stage2", "temporal"}:
            return Stage2Loss(**_pick(defaults, Stage2Loss))
        if stage in {"stage3", "moe"}:
            return Stage3Loss(**_pick(defaults, Stage3Loss))
        if stage in {"stage4", "finetune", "e2e", "all", "joint"}:
            return Stage4Loss(config)
        raise ValueError(f"Unknown stage: {stage}")

    @staticmethod
    def _defaults(config: DRISHTIConfig | None) -> dict[str, Any]:
        if config is None:
            return {}
        return {
            "focal_gamma": config.focal_gamma,
            "focal_alpha": config.focal_alpha,
            "heatmap_alpha": config.heatmap_focal_alpha,
            "heatmap_beta": config.heatmap_focal_beta,
            "w_motion": config.w_motion_displacement,
            "w_offset": config.w_subpixel_offset,
            "w_gate": config.w_gate_sparsity,
            "w_tc": config.w_temporal_consistency,
            "w_sm": config.w_trajectory_smoothness,
            "sigma_spatial": config.sigma_spatial_consist,
            "w_bal": config.moe_balance_weight,
            "w_zloss": config.router_z_loss_weight,
        }


def _pick(values: dict[str, Any], cls: type) -> dict[str, Any]:
    names = cls.__init__.__code__.co_varnames
    return {key: value for key, value in values.items() if key in names}
