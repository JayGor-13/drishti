from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from drishti_v2.models.motion_cnn import MotionCNN
from drishti_v2.models.pipeline import PipelineOutput


class DRISHTILoss(nn.Module):
    """Combined heatmap, objectness, bbox, and MoE balance loss."""

    def __init__(
        self,
        w_heatmap: float = 1.0,
        w_cls: float = 1.0,
        w_bbox: float = 2.0,
        w_balance: float = 0.01,
    ) -> None:
        super().__init__()
        self.w_heatmap = w_heatmap
        self.w_cls = w_cls
        self.w_bbox = w_bbox
        self.w_balance = w_balance

    def _last_targets(self, targets: list) -> list[dict]:
        if targets and isinstance(targets[0], list):
            return [clip[-1] for clip in targets]
        return targets

    def _make_heatmaps(self, targets: list[dict], heatmap_size: tuple[int, int], device: torch.device) -> Tensor:
        maps = []
        for target in targets:
            boxes = target.get("boxes", torch.empty(0, 4)).to(device)
            maps.append(MotionCNN.make_gt_heatmap(boxes, heatmap_size))
        return torch.stack(maps, dim=0)

    def _assign_crops(self, output: PipelineOutput, targets: list[dict]) -> tuple[Tensor, Tensor]:
        batch, num_crops, _ = output.proposal_centers.shape
        labels = output.objectness_logits.new_zeros(batch, num_crops, 1)
        box_targets = output.crop_boxes.detach().new_zeros(batch, num_crops, 4)
        crop_w = output.boxes[..., 2].new_tensor(1.0)
        for b_idx, target in enumerate(targets):
            boxes = target.get("boxes", torch.empty(0, 4)).to(output.proposal_centers.device)
            if boxes.numel() == 0:
                continue
            centers = output.proposal_centers[b_idx]
            distances = torch.cdist(centers, boxes[:, :2])
            crop_indices = distances.argmin(dim=0).unique()
            for crop_idx in crop_indices:
                gt_idx = distances[crop_idx].argmin()
                gt = boxes[gt_idx]
                labels[b_idx, crop_idx, 0] = 1.0
                # Convert global gt box into crop-relative coordinates.
                # This uses the proposal center and predicted global scale ratio from the crop geometry.
                global_pred_size = output.boxes[b_idx, crop_idx, 2:].clamp_min(1e-6)
                crop_scale = global_pred_size / output.crop_boxes[b_idx, crop_idx, 2:].clamp_min(1e-6)
                rel_xy = (gt[:2] - centers[crop_idx]) / crop_scale + 0.5
                rel_wh = gt[2:] / crop_scale
                box_targets[b_idx, crop_idx] = torch.cat([rel_xy, rel_wh]).clamp(0.0, 1.0)
        return labels, box_targets

    def forward(self, output: PipelineOutput, targets: list, heatmap_size: tuple[int, int] | None = None) -> dict[str, Tensor]:
        last_targets = self._last_targets(targets)
        heatmap_size = heatmap_size or tuple(output.heatmap.shape[-2:])
        gt_heatmap = self._make_heatmaps(last_targets, heatmap_size, output.heatmap.device).to(output.heatmap.dtype)
        heatmap_loss = F.mse_loss(output.heatmap, gt_heatmap)

        labels, box_targets = self._assign_crops(output, last_targets)
        cls_loss = F.binary_cross_entropy_with_logits(output.objectness_logits, labels)
        positive = labels.squeeze(-1) > 0.5
        if positive.any():
            bbox_loss = F.smooth_l1_loss(output.crop_boxes[positive], box_targets[positive])
        else:
            bbox_loss = output.objectness_logits.sum() * 0.0
        balance = output.balance_loss
        total = (
            self.w_heatmap * heatmap_loss
            + self.w_cls * cls_loss
            + self.w_bbox * bbox_loss
            + self.w_balance * balance
        )
        return {"loss": total, "heatmap": heatmap_loss, "cls": cls_loss, "bbox": bbox_loss, "balance": balance}
