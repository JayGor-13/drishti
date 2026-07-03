from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from drishti_v2.models.config import DRISHTIConfig
from drishti_v2.models.crop_encoder import CropEncoder
from drishti_v2.models.crop_proposal import CropProposalEngine
from drishti_v2.models.detection_head import DetectionHead
from drishti_v2.models.ldmi import LocalDifferentialMotion
from drishti_v2.models.moe import MoEDiagnostics, SparseMoE
from drishti_v2.models.motion_cnn import MotionCNN
from drishti_v2.models.temporal_fusion import CausalTemporalFusion


@dataclass
class PipelineOutput:
    heatmap: Tensor
    proposal_centers: Tensor
    proposal_scores: Tensor
    proposal_sources: Tensor
    crop_features: Tensor
    fused_features: Tensor
    moe_features: Tensor
    objectness_logits: Tensor
    crop_boxes: Tensor
    boxes: Tensor
    balance_loss: Tensor
    moe_diagnostics: MoEDiagnostics | None = None


class DRISHTIPipeline(nn.Module):
    """Full DRISHTI-CORE v2 causal detector."""

    def __init__(self, config: DRISHTIConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.ldmi = LocalDifferentialMotion(config.image_channels, config.ldmi_scales)
        self.motion_cnn = MotionCNN(config.image_channels, config.motion_cnn_channels)
        self.crop_engine = CropProposalEngine(config)
        self.encoder = CropEncoder(config.encoder_feature_dim, in_channels=config.image_channels)
        self.temporal = CausalTemporalFusion(
            feature_dim=config.encoder_feature_dim + 1,
            out_dim=config.encoder_feature_dim,
            nhead=config.temporal_heads,
            num_layers=config.temporal_layers,
            ffn_dim=config.temporal_ffn_dim,
            dropout=config.temporal_dropout,
            max_seq_len=config.temporal_window,
        )
        self.moe = SparseMoE(
            d_model=config.encoder_feature_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            ffn_dim=config.expert_ffn_dim,
            dropout=config.moe_dropout,
            dense=config.dense_moe,
        )
        self.head = DetectionHead(config.encoder_feature_dim, config.head_hidden_dim)
        self._stream_buffer: list[Tensor] = []
        self._stream_feature_buffer: list[Tensor] = []
        if config.encoder_frozen:
            self.encoder.freeze()

    def _make_triplet(self, frames: Tensor, t_idx: int) -> Tensor:
        t0 = max(0, t_idx - 2)
        t1 = max(0, t_idx - 1)
        return torch.cat([frames[:, t0], frames[:, t1], frames[:, t_idx]], dim=1)

    def _boxes_to_global(self, crop_boxes: Tensor, centers: Tensor, frame_shape: tuple[int, int]) -> Tensor:
        height, width = frame_shape
        crop_w = self.config.crop_size / float(width)
        crop_h = self.config.crop_size / float(height)
        global_boxes = crop_boxes.clone()
        global_boxes[..., 0] = centers[..., 0] + (crop_boxes[..., 0] - 0.5) * crop_w
        global_boxes[..., 1] = centers[..., 1] + (crop_boxes[..., 1] - 0.5) * crop_h
        global_boxes[..., 2] = crop_boxes[..., 2] * crop_w
        global_boxes[..., 3] = crop_boxes[..., 3] * crop_h
        return global_boxes.clamp(0.0, 1.0)

    def _forward_single(
        self,
        frame: Tensor,
        triplet: Tensor,
        frame_index: int,
        guided_centers: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        filtered = self.ldmi(triplet) if self.config.use_ldmi else triplet
        heatmap = self.motion_cnn(filtered)
        proposal = self.crop_engine(frame, heatmap, frame_index, guided_centers)
        encoded = self.encoder(proposal.crops).reshape(frame.shape[0], self.config.num_crops, -1)
        augmented = torch.cat([encoded, proposal.scores.unsqueeze(-1)], dim=-1)
        return heatmap, proposal.centers, proposal.scores, proposal.source_labels, encoded, augmented

    def forward(self, frames: Tensor, frame_index: int = 0, guided_centers: Tensor | None = None) -> PipelineOutput:
        if frames.ndim != 5:
            raise ValueError(f"Expected [B, T, C, H, W], got {tuple(frames.shape)}")
        batch, time, _, height, width = frames.shape
        features: list[Tensor] = []
        last: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None = None

        for t_idx in range(time):
            guided = guided_centers if t_idx == time - 1 else None
            heatmap, centers, scores, sources, encoded, augmented = self._forward_single(
                frames[:, t_idx],
                self._make_triplet(frames, t_idx),
                frame_index + t_idx,
                guided,
            )
            features.append(augmented)
            last = (heatmap, centers, scores, sources, encoded)

        assert last is not None
        sequence = torch.stack(features[-self.config.temporal_window :], dim=1)
        fused = self.temporal(sequence)
        moe_features, moe_diag = self.moe(fused)
        logits, crop_boxes = self.head(moe_features)
        heatmap, centers, scores, sources, encoded = last
        global_boxes = self._boxes_to_global(crop_boxes, centers, (height, width))
        return PipelineOutput(
            heatmap=heatmap,
            proposal_centers=centers,
            proposal_scores=scores,
            proposal_sources=sources,
            crop_features=encoded,
            fused_features=fused,
            moe_features=moe_features,
            objectness_logits=logits,
            crop_boxes=crop_boxes,
            boxes=global_boxes,
            balance_loss=moe_diag.balance_loss,
            moe_diagnostics=moe_diag,
        )

    @torch.no_grad()
    def forward_stream(
        self,
        frame: Tensor,
        frame_index: int,
        guided_centers: Tensor | None = None,
    ) -> PipelineOutput:
        if frame.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(frame.shape)}")
        self.eval()
        frames_for_triplet = [item[:, : self.config.image_channels] for item in self._stream_buffer[-2:]]
        while len(frames_for_triplet) < 2:
            frames_for_triplet.insert(0, frame)
        triplet = torch.cat([frames_for_triplet[-2], frames_for_triplet[-1], frame], dim=1)
        heatmap, centers, scores, sources, encoded, augmented = self._forward_single(
            frame, triplet, frame_index, guided_centers
        )
        self._stream_buffer.append(frame.detach())
        self._stream_buffer = self._stream_buffer[-self.config.temporal_window :]
        self._stream_feature_buffer.append(augmented.detach())
        self._stream_feature_buffer = self._stream_feature_buffer[-self.config.temporal_window :]
        seq = self._stream_feature_buffer
        if len(seq) < self.config.temporal_window:
            seq = [seq[0]] * (self.config.temporal_window - len(seq)) + seq
        sequence = torch.stack(seq[-self.config.temporal_window :], dim=1)
        fused = self.temporal(sequence)
        moe_features, moe_diag = self.moe(fused)
        logits, crop_boxes = self.head(moe_features)
        global_boxes = self._boxes_to_global(crop_boxes, centers, frame.shape[-2:])
        return PipelineOutput(
            heatmap, centers, scores, sources, encoded, fused, moe_features,
            logits, crop_boxes, global_boxes, moe_diag.balance_loss, moe_diag,
        )

    def reset_stream(self) -> None:
        self._stream_buffer.clear()
        self._stream_feature_buffer.clear()
