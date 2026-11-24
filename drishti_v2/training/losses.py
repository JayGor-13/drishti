from __future__ import annotations

import warnings

from torch import Tensor, nn

from drishti_v2.models.pipeline import PipelineOutput
from drishti_v2.training.stage_losses import DetectionLossMixin


class DRISHTILoss(nn.Module, DetectionLossMixin):
    """Combined heatmap, objectness, bbox, and MoE balance loss."""

    def __init__(
        self,
        w_heatmap: float = 1.0,
        w_cls: float = 1.0,
        w_bbox: float = 2.0,
        w_balance: float = 0.01,
        w_offset: float = 1.0,
    ) -> None:
        warnings.warn(
            "DRISHTILoss is kept for compatibility. Prefer StageLossFactory.make_loss(stage, config=config).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self.w_heatmap = w_heatmap
        self.w_cls = w_cls
        self.w_bbox = w_bbox
        self.w_balance = w_balance
        self.w_offset = w_offset
        self.focal_gamma = 2.0
        self.focal_alpha = 0.25
        self.heatmap_alpha = 2.0
        self.heatmap_beta = 4.0

    def forward(self, output: PipelineOutput, targets: list, heatmap_size: tuple[int, int] | None = None) -> dict[str, Tensor]:
        del heatmap_size
        terms = self.detection_terms(output, targets)
        balance = output.balance_loss
        total = (
            self.w_heatmap * terms["heatmap"]
            + self.w_cls * terms["cls"]
            + self.w_bbox * terms["bbox"]
            + self.w_offset * terms["offset"]
            + self.w_balance * balance
        )
        return {
            "loss": total,
            "heatmap": terms["heatmap"],
            "cls": terms["cls"],
            "bbox": terms["bbox"],
            "offset": terms["offset"],
            "balance": balance,
        }
