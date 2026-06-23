"""DRISHTI-CORE detector-only pipeline."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import DRISHTIConfig
from .crop_encoder import FrozenCropEncoder
from .detection_head import DRISHTIDetectionHead
from .moe import DRISHTIMoE
from .motion_proposal import MotionCropProposal
from .temporal_fusion import TemporalFusion
from .types import DetectorStageOutput, DRISHTIOutput


class DRISHTIPipeline(nn.Module):
    """Stages 1-6 for Anti-UAV bounding-box prediction."""

    def __init__(self, config: DRISHTIConfig | None = None) -> None:
        super().__init__()
        self.config = config or DRISHTIConfig()
        self.motion_proposer = MotionCropProposal(
            image_channels=self.config.image_channels,
            crop_size=self.config.crop_size,
            num_crops=self.config.num_crops,
        )
        self.crop_encoder = FrozenCropEncoder(
            feature_dim=self.config.feature_dim,
            image_channels=self.config.image_channels,
            frozen=True,
        )
        self.temporal_fusion = TemporalFusion(self.config)
        self.moe = DRISHTIMoE(self.config)
        self.detection_head = DRISHTIDetectionHead(self.config.feature_dim)

    def triplet_at(self, frames: Tensor, center_index: int) -> Tensor:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [batch, time, channels, height, width]")
        _, time, channels, _, _ = frames.shape
        if channels != self.config.image_channels:
            raise ValueError(f"expected {self.config.image_channels} channels, got {channels}")
        indices = [
            max(0, min(time - 1, center_index - 1)),
            max(0, min(time - 1, center_index)),
            max(0, min(time - 1, center_index + 1)),
        ]
        return torch.cat([frames[:, idx] for idx in indices], dim=1)

    def final_triplet(self, frames: Tensor) -> Tensor:
        return self.triplet_at(frames, frames.shape[1] - 1)

    def encode_triplet(self, triplet: Tensor) -> DetectorStageOutput:
        proposal = self.motion_proposer(triplet)
        batch = triplet.shape[0]
        crop_features = self.crop_encoder(proposal.crops).reshape(
            batch,
            self.config.num_crops,
            self.config.feature_dim,
        )
        object_logits, boxes, _ = self.detection_head(crop_features)
        return DetectorStageOutput(
            proposal=proposal,
            crop_features=crop_features,
            object_logits=object_logits,
            boxes=boxes,
        )

    def encode_temporal(self, frames: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [batch, time, channels, height, width]")
        batch, time, _, _, _ = frames.shape
        if time < 1:
            raise ValueError("frames must contain at least one frame")
        start = max(0, time - self.config.temporal_window)
        center_indices = list(range(start, time))
        if len(center_indices) < self.config.temporal_window:
            pad = [center_indices[0]] * (self.config.temporal_window - len(center_indices))
            center_indices = pad + center_indices

        temporal_inputs = []
        heatmaps = []
        proposal_boxes = []
        motion_scores = []
        crop_features = []
        for center_index in center_indices:
            stage = self.encode_triplet(self.triplet_at(frames, center_index))
            scores = stage.proposal.motion_scores.reshape(batch, self.config.num_crops, 1)
            temporal_inputs.append(torch.cat([stage.crop_features, scores], dim=-1))
            heatmaps.append(stage.proposal.heatmap)
            proposal_boxes.append(stage.proposal.boxes)
            motion_scores.append(scores.squeeze(-1))
            crop_features.append(stage.crop_features)

        return (
            torch.stack(temporal_inputs, dim=1),
            torch.stack(heatmaps, dim=1),
            torch.stack(proposal_boxes, dim=1),
            torch.stack(motion_scores, dim=1),
            torch.stack(crop_features, dim=1),
        )

    def forward_detector(self, frames: Tensor) -> DetectorStageOutput:
        """Run only Stage 1+2+3 plus detection head for detector training."""

        return self.encode_triplet(self.final_triplet(frames))

    def forward_temporal(self, frames: Tensor) -> DRISHTIOutput:
        """Run through temporal fusion and detect before the MoE stage."""

        temporal_inputs, heatmaps, proposal_boxes, motion_scores, crop_features = self.encode_temporal(frames)
        temporal = self.temporal_fusion(temporal_inputs)
        object_logits, boxes, predictions = self.detection_head(temporal.fused_features)
        router_probs = temporal.fused_features.new_zeros(
            temporal.fused_features.shape[:2] + (self.config.moe_num_experts,)
        )
        router_topk = torch.zeros(
            temporal.fused_features.shape[:2] + (self.config.moe_top_k,),
            device=temporal.fused_features.device,
            dtype=torch.long,
        )
        return DRISHTIOutput(
            heatmaps=heatmaps,
            proposal_boxes=proposal_boxes[:, -1],
            motion_scores=motion_scores,
            crop_features=crop_features,
            temporal_tokens=temporal.temporal_tokens,
            temporal_features=temporal.fused_features,
            moe_features=temporal.fused_features,
            object_logits=object_logits,
            boxes=boxes,
            predictions=predictions,
            router_probs=router_probs,
            router_topk=router_topk,
            load_balance_loss=temporal.fused_features.new_zeros(()),
        )

    def forward(self, frames: Tensor) -> DRISHTIOutput:
        temporal_inputs, heatmaps, proposal_boxes, motion_scores, crop_features = self.encode_temporal(frames)
        temporal = self.temporal_fusion(temporal_inputs)
        moe = self.moe(temporal.fused_features)
        object_logits, boxes, predictions = self.detection_head(moe.hidden_states)
        return DRISHTIOutput(
            heatmaps=heatmaps,
            proposal_boxes=proposal_boxes[:, -1],
            motion_scores=motion_scores,
            crop_features=crop_features,
            temporal_tokens=temporal.temporal_tokens,
            temporal_features=temporal.fused_features,
            moe_features=moe.hidden_states,
            object_logits=object_logits,
            boxes=boxes,
            predictions=predictions,
            router_probs=moe.router_probs,
            router_topk=moe.topk_indices,
            load_balance_loss=moe.load_balance_loss,
        )
