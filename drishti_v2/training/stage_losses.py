from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from drishti_v2.models.config import DRISHTIConfig
from drishti_v2.models.motion_cnn import MotionCNN
from drishti_v2.models.pipeline import PipelineOutput
from drishti_v2.training.ciou_loss import ciou_loss
from drishti_v2.training.focal_loss import heatmap_focal_loss, sigmoid_focal_loss
from drishti_v2.training.motion_loss import motion_displacement_loss
from drishti_v2.training.temporal_loss import temporal_consistency_loss, trajectory_smoothness_loss


def last_targets(targets: list) -> list[dict]:
    return [clip[-1] for clip in targets] if targets and isinstance(targets[0], list) else targets


def make_gt_heatmaps(targets: list[dict], heatmap_size: tuple[int, int], device: torch.device) -> Tensor:
    heatmaps = []
    for target in targets:
        boxes = target.get("boxes", torch.empty(0, 4)).to(device)
        heatmaps.append(MotionCNN.make_gt_heatmap(boxes, heatmap_size))
    return torch.stack(heatmaps, dim=0)


def assign_crops(output: PipelineOutput, targets: list[dict]) -> tuple[Tensor, Tensor]:
    """Assign each GT box to the nearest crop center."""

    batch, num_crops, _ = output.proposal_centers.shape
    labels = output.objectness_logits.new_zeros(batch, num_crops, 1)
    box_targets = output.crop_boxes.detach().new_zeros(batch, num_crops, 4)

    for batch_idx, target in enumerate(targets):
        boxes = target.get("boxes", torch.empty(0, 4)).to(output.proposal_centers.device)
        if boxes.numel() == 0:
            continue
        distances = torch.cdist(output.proposal_centers[batch_idx], boxes[:, :2])
        for crop_idx in distances.argmin(dim=0).unique():
            gt_idx = distances[crop_idx].argmin()
            gt = boxes[gt_idx]
            labels[batch_idx, crop_idx, 0] = 1.0

            global_size = output.boxes[batch_idx, crop_idx, 2:].clamp_min(1e-6)
            crop_size = output.crop_boxes[batch_idx, crop_idx, 2:].clamp_min(1e-6)
            crop_scale = global_size / crop_size
            rel_xy = (gt[:2] - output.proposal_centers[batch_idx, crop_idx]) / crop_scale + 0.5
            rel_wh = gt[2:] / crop_scale
            box_targets[batch_idx, crop_idx] = torch.cat([rel_xy, rel_wh]).clamp(0.0, 1.0)

    return labels, box_targets


class DetectionLossMixin:
    focal_gamma: float
    focal_alpha: float
    heatmap_alpha: float
    heatmap_beta: float

    def detection_terms(self, output: PipelineOutput, targets: list) -> dict[str, Tensor]:
        targets = last_targets(targets)
        heatmap_size = tuple(output.heatmap.shape[-2:])
        gt_heatmap = make_gt_heatmaps(targets, heatmap_size, output.heatmap.device).to(output.heatmap.dtype)
        heatmap = heatmap_focal_loss(output.heatmap, gt_heatmap, self.heatmap_alpha, self.heatmap_beta)

        labels, box_targets = assign_crops(output, targets)
        cls = sigmoid_focal_loss(output.objectness_logits, labels, self.focal_gamma, self.focal_alpha)
        positive = labels.squeeze(-1) > 0.5
        bbox = (
            ciou_loss(output.crop_boxes[positive], box_targets[positive])
            if positive.any()
            else output.objectness_logits.sum() * 0.0
        )
        return {"heatmap": heatmap, "cls": cls, "bbox": bbox, "labels": labels}


class Stage1Loss(nn.Module, DetectionLossMixin):
    def __init__(
        self,
        w_hm: float = 1.0,
        w_cls: float = 1.0,
        w_box: float = 2.0,
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
            + self.w_motion * motion
            + self.w_gate * gate
        )
        return {
            "loss": total,
            "heatmap": terms["heatmap"],
            "cls": terms["cls"],
            "bbox": terms["bbox"],
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
        zero = output.objectness_logits.sum() * 0.0
        temporal = (
            temporal_consistency_loss(logits_seq, centers_seq, self.sigma_spatial)
            if logits_seq is not None and centers_seq is not None
            else zero
        )
        smooth = zero
        if boxes_seq is not None:
            labels = terms["labels"].squeeze(-1)
            smooth = trajectory_smoothness_loss(boxes_seq, [labels] * len(boxes_seq))

        total = (
            self.w_hm * terms["heatmap"]
            + self.w_cls * terms["cls"]
            + self.w_box * terms["bbox"]
            + self.w_tc * temporal
            + self.w_sm * smooth
        )
        return {
            "loss": total,
            "heatmap": terms["heatmap"],
            "cls": terms["cls"],
            "bbox": terms["bbox"],
            "temporal_consist": temporal,
            "traj_smooth": smooth,
            "balance": output.balance_loss,
        }


class Stage3Loss(nn.Module, DetectionLossMixin):
    def __init__(
        self,
        w_cls: float = 1.0,
        w_box: float = 2.0,
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
        self.w_bal = w_bal
        self.w_zloss = w_zloss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.heatmap_alpha = heatmap_alpha
        self.heatmap_beta = heatmap_beta

    def forward(self, output: PipelineOutput, targets: list) -> dict[str, Tensor]:
        terms = self.detection_terms(output, targets)
        z_loss = output.moe_diagnostics.router_z_loss
        total = self.w_cls * terms["cls"] + self.w_box * terms["bbox"] + self.w_bal * output.balance_loss + self.w_zloss * z_loss
        return {
            "loss": total,
            "cls": terms["cls"],
            "bbox": terms["bbox"],
            "balance": output.balance_loss,
            "z_loss": z_loss,
        }


class Stage4Loss(nn.Module):
    def __init__(self, config: DRISHTIConfig | None = None) -> None:
        super().__init__()
        self.stage1 = Stage1Loss(
            w_motion=0.3,
            w_gate=(config.w_gate_sparsity if config else 0.01),
            focal_gamma=(config.focal_gamma if config else 2.0),
            focal_alpha=(config.focal_alpha if config else 0.25),
        )
        self.stage2 = Stage2Loss(
            w_hm=0.5,
            w_tc=0.15,
            w_sm=0.05,
            sigma_spatial=(config.sigma_spatial_consist if config else 0.1),
        )
        self.stage3 = Stage3Loss(
            w_bal=(config.moe_balance_weight if config else 0.01),
            w_zloss=(config.router_z_loss_weight if config else 0.001),
        )

    def forward(self, output: PipelineOutput, targets: list, **kwargs: Any) -> dict[str, Tensor]:
        s1 = self.stage1(output, targets, kwargs.get("all_heatmaps"))
        s2 = self.stage2(output, targets, kwargs.get("logits_seq"), kwargs.get("centers_seq"), kwargs.get("boxes_seq"))
        s3 = self.stage3(output, targets)
        total = s1["loss"] + s2["temporal_consist"] + s2["traj_smooth"] + s3["z_loss"]
        return {
            "loss": total,
            "heatmap": s1["heatmap"],
            "cls": s1["cls"],
            "bbox": s1["bbox"],
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
        if stage in {"stage4", "finetune", "e2e", "all"}:
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
