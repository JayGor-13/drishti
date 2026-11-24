from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from drishti_v2.assignment import linear_sum_assignment
from drishti_v2.models.config import DRISHTIConfig
from drishti_v2.models.crop_encoder import build_crop_encoder
from drishti_v2.models.focus import FocusRefiner
from drishti_v2.models.crop_proposal import CropProposalEngine
from drishti_v2.models.detection_head import DetectionHead, LayerMix
from drishti_v2.models.ldmi import LocalDifferentialMotion
from drishti_v2.models.moe import MoEDiagnostics, SparseMoE
from drishti_v2.models.motion_cnn import MotionCNN
from drishti_v2.models.motion_gate import MotionGate
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
    center_offsets: Tensor
    boxes: Tensor
    crop_scale: tuple[float, float]
    balance_loss: Tensor
    moe_diagnostics: MoEDiagnostics
    motion_gate_confidence: Tensor
    used_dense_mode: bool
    all_heatmaps: list[Tensor] | None = None
    proposal_centers_seq: list[Tensor] | None = None
    proposal_sources_seq: list[Tensor] | None = None
    objectness_logits_seq: list[Tensor] | None = None
    crop_boxes_seq: list[Tensor] | None = None
    center_offsets_seq: list[Tensor] | None = None
    boxes_seq: list[Tensor] | None = None
    frame_last: Tensor | None = None  # the frame the boxes belong to (for Chain-of-Focus)
    focus_windows: Tensor | None = None  # focus crops used per frame at inference


class DRISHTIPipeline(nn.Module):
    """Full DRISHTI-CORE v2 causal detector."""

    def __init__(self, config: DRISHTIConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.ldmi = LocalDifferentialMotion(
            config.image_channels,
            config.ldmi_scales,
            use_sobel_edge=config.use_sobel_edge,
        )
        self.motion_cnn = MotionCNN(
            config.image_channels,
            config.motion_cnn_channels,
            in_channels=config.motion_input_channels,
        )
        self.motion_gate = (
            MotionGate(config.motion_gate_hidden, config.motion_gate_active_threshold)
            if config.use_motion_gate
            else None
        )
        self.crop_engine = CropProposalEngine(config)
        self.encoder = build_crop_encoder(config)
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
            num_sources=5,
            use_source_bias=True,
        )
        self.head = DetectionHead(config.encoder_feature_dim, config.head_hidden_dim)
        dim = config.encoder_feature_dim
        self.layer_mix = LayerMix(3, dim) if getattr(config, "head_layer_mix", False) else None
        self.fine_proj = (
            nn.Sequential(nn.LayerNorm(64 * 16), nn.Linear(64 * 16, dim))
            if getattr(config, "head_fine_features", False) and hasattr(self.encoder, "forward_multi")
            else None
        )
        self.focus = FocusRefiner(config, self.encoder) if getattr(config, "focus_mode", "off") != "off" else None
        self._stream_buffer: list[Tensor] = []
        self._stream_feature_buffer: list[Tensor] = []
        self._stream_center_buffer: list[Tensor] = []
        self._stream_source_buffer: list[Tensor] = []
        self._last_fine: Tensor | None = None
        self._stream_prev_heatmap: Tensor | None = None
        if config.encoder_frozen:
            self.encoder.freeze()

    def _make_triplet(self, frames: Tensor, t_idx: int) -> Tensor:
        t0 = max(0, t_idx - 2)
        t1 = max(0, t_idx - 1)
        return torch.cat([frames[:, t0], frames[:, t1], frames[:, t_idx]], dim=1)

    def _crop_scale(self, frame_shape: tuple[int, int]) -> tuple[float, float]:
        height, width = frame_shape
        crop_w = max(self.config.crop_size - 1, 1) / float(max(width - 1, 1))
        crop_h = max(self.config.crop_size - 1, 1) / float(max(height - 1, 1))
        return crop_w, crop_h

    def _apply_head(self, encoded: Tensor, fused: Tensor, moe_out: Tensor, fine: Tensor | None):
        """Detection head, optionally on a learned layer mix and with fine features for the box branch."""

        features = moe_out if self.layer_mix is None else self.layer_mix([encoded, fused, moe_out])
        box_features = None
        if self.fine_proj is not None and fine is not None:
            box_features = features + self.fine_proj(fine)
        return self.head(features, box_features)

    def _boxes_to_global(
        self,
        crop_boxes: Tensor,
        centers: Tensor,
        frame_shape: tuple[int, int],
        center_offsets: Tensor | None = None,
        heatmap_size: tuple[int, int] | None = None,
    ) -> Tensor:
        crop_w, crop_h = self._crop_scale(frame_shape)
        corrected_centers = centers
        if center_offsets is not None:
            if heatmap_size is None:
                height, width = frame_shape
                heatmap_size = (max((height + 3) // 4, 1), max((width + 3) // 4, 1))
            heatmap_h, heatmap_w = heatmap_size
            correction = torch.stack(
                [
                    center_offsets[..., 0] / float(max(heatmap_w - 1, 1)),
                    center_offsets[..., 1] / float(max(heatmap_h - 1, 1)),
                ],
                dim=-1,
            )
            corrected_centers = centers + correction

        global_boxes = crop_boxes.clone()
        global_boxes[..., 0] = corrected_centers[..., 0] + (crop_boxes[..., 0] - 0.5) * crop_w
        global_boxes[..., 1] = corrected_centers[..., 1] + (crop_boxes[..., 1] - 0.5) * crop_h
        global_boxes[..., 2] = crop_boxes[..., 2] * crop_w
        global_boxes[..., 3] = crop_boxes[..., 3] * crop_h
        return global_boxes.clamp(0.0, 1.0)

    def _motion_step(self, triplet: Tensor, prev_heatmap: Tensor | None = None) -> tuple[Tensor, Tensor]:
        filtered = self.ldmi(triplet) if self.config.use_ldmi else triplet
        if getattr(self.config, "heatmap_recurrent", False):
            if prev_heatmap is None:
                prev = filtered.new_zeros(filtered.shape[0], 1, *filtered.shape[-2:])
            else:  # previous frame's heatmap, detached, upsampled to frame size
                prev = F.interpolate(prev_heatmap.detach(), size=filtered.shape[-2:], mode="bilinear", align_corners=False)
            filtered = torch.cat([filtered, prev], dim=1)
        heatmap = self.motion_cnn(filtered)
        if self.motion_gate is None:
            confidence = heatmap.new_ones(heatmap.shape[0])
        else:
            confidence = self.motion_gate(heatmap)
        return heatmap, confidence

    def _use_dense_mode(self, confidence: Tensor) -> bool:
        if self.motion_gate is None:
            return False
        return bool((confidence < self.config.motion_gate_threshold).any().detach().cpu().item())

    def _align_temporal_history(
        self,
        features: list[Tensor],
        centers: list[Tensor],
        sources: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Align historical proposal slots to the newest proposal layout."""

        reference_centers = centers[-1]
        reference_sources = sources[-1]
        batch, reference_crops, _ = reference_centers.shape
        aligned_features: list[Tensor] = []
        aligned_centers: list[Tensor] = []
        aligned_sources: list[Tensor] = []
        padding_steps: list[Tensor] = []

        for time_idx, (step_features, step_centers, step_sources) in enumerate(zip(features, centers, sources)):
            if time_idx == len(features) - 1:
                aligned_features.append(step_features)
                aligned_centers.append(step_centers)
                aligned_sources.append(step_sources)
                padding_steps.append(torch.zeros(batch, reference_crops, dtype=torch.bool, device=step_features.device))
                continue

            batch_features = []
            batch_centers = []
            batch_sources = []
            batch_padding = []
            for batch_idx in range(batch):
                spatial_cost = torch.cdist(reference_centers[batch_idx].detach(), step_centers[batch_idx].detach())
                source_cost = (
                    reference_sources[batch_idx, :, None] != step_sources[batch_idx, None, :]
                ).to(spatial_cost.dtype)
                row_indices, col_indices = linear_sum_assignment(spatial_cost + 2.0 * source_cost)
                matches = {int(row): int(col) for row, col in zip(row_indices.tolist(), col_indices.tolist())}

                feature_rows = []
                center_rows = []
                source_rows = []
                padded_rows = []
                for row in range(reference_crops):
                    if row in matches:
                        col = matches[row]
                        feature_rows.append(step_features[batch_idx, col])
                        center_rows.append(step_centers[batch_idx, col])
                        source_rows.append(step_sources[batch_idx, col])
                        padded_rows.append(False)
                    else:
                        feature_rows.append(step_features.new_zeros(step_features.shape[-1]))
                        center_rows.append(reference_centers[batch_idx, row])
                        source_rows.append(step_sources.new_tensor(self.crop_engine.PAD))
                        padded_rows.append(True)
                batch_features.append(torch.stack(feature_rows))
                batch_centers.append(torch.stack(center_rows))
                batch_sources.append(torch.stack(source_rows))
                batch_padding.append(torch.tensor(padded_rows, dtype=torch.bool, device=step_features.device))

            aligned_features.append(torch.stack(batch_features))
            aligned_centers.append(torch.stack(batch_centers))
            aligned_sources.append(torch.stack(batch_sources))
            padding_steps.append(torch.stack(batch_padding))

        padding_mask = torch.stack(padding_steps, dim=1)
        if not padding_mask.any():
            padding_mask = None
        return (
            torch.stack(aligned_features, dim=1),
            torch.stack(aligned_centers, dim=1),
            torch.stack(aligned_sources, dim=1),
            padding_mask,
        )

    def _forward_single(
        self,
        frame: Tensor,
        heatmap: Tensor,
        frame_index: int,
        guided_centers: Tensor | None,
        dense: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        proposal = self.crop_engine(frame, heatmap, frame_index, guided_centers, dense=dense)
        num_crops = proposal.centers.shape[1]
        fine = None
        if self.fine_proj is not None:
            encoded, fine = self.encoder.forward_multi(proposal.crops)
            encoded = encoded.reshape(frame.shape[0], num_crops, -1)
            fine = fine.reshape(frame.shape[0], num_crops, -1)
        else:
            encoded = self.encoder(proposal.crops).reshape(frame.shape[0], num_crops, -1)
        self._last_fine = fine
        augmented = torch.cat([encoded, proposal.scores.unsqueeze(-1)], dim=-1)
        return heatmap, proposal.centers, proposal.scores, proposal.source_labels, encoded, augmented

    def forward(self, frames: Tensor, frame_index: int = 0, guided_centers: Tensor | None = None) -> PipelineOutput:
        if frames.ndim != 5:
            raise ValueError(f"Expected [B, T, C, H, W], got {tuple(frames.shape)}")
        batch, time, _, height, width = frames.shape
        heatmaps: list[Tensor] = []
        confidences: list[Tensor] = []
        for t_idx in range(time):
            heatmap, confidence = self._motion_step(self._make_triplet(frames, t_idx), heatmaps[-1] if heatmaps else None)
            heatmaps.append(heatmap)
            confidences.append(confidence)

        gate_confidence = torch.stack(confidences, dim=1)
        dense_mode = self._use_dense_mode(gate_confidence)
        features: list[Tensor] = []
        centers_history: list[Tensor] = []
        sources_history: list[Tensor] = []
        encoded_history: list[Tensor] = []
        fine_history: list[Tensor | None] = []
        scores_history: list[Tensor] = []

        for t_idx in range(time):
            guided = guided_centers if t_idx == time - 1 else None
            heatmap, centers, scores, sources, encoded, augmented = self._forward_single(
                frames[:, t_idx],
                heatmaps[t_idx],
                frame_index + t_idx,
                guided,
                dense=dense_mode,
            )
            features.append(augmented)
            centers_history.append(centers)
            sources_history.append(sources)
            encoded_history.append(encoded)
            fine_history.append(self._last_fine)
            scores_history.append(scores)

        fused_history: list[Tensor] = []
        moe_history: list[Tensor] = []
        diagnostics_history: list[MoEDiagnostics] = []
        logits_history: list[Tensor] = []
        crop_boxes_history: list[Tensor] = []
        offsets_history: list[Tensor] = []
        boxes_history: list[Tensor] = []

        for t_idx in range(time):
            start = max(0, t_idx + 1 - self.config.temporal_window)
            sequence, centers_seq, sources_seq, padding_mask = self._align_temporal_history(
                features[start : t_idx + 1],
                centers_history[start : t_idx + 1],
                sources_history[start : t_idx + 1],
            )
            fused_step = self.temporal(
                sequence,
                centers_seq=centers_seq,
                source_labels_seq=sources_seq,
                padding_mask=padding_mask,
            )
            moe_step, diagnostics = self.moe(fused_step, source_labels=sources_history[t_idx])
            logits_step, crop_boxes_step, offsets_step = self._apply_head(
                encoded_history[t_idx], fused_step, moe_step, fine_history[t_idx]
            )
            boxes_step = self._boxes_to_global(
                crop_boxes_step,
                centers_history[t_idx],
                (height, width),
                offsets_step,
                tuple(heatmaps[t_idx].shape[-2:]),
            )
            fused_history.append(fused_step)
            moe_history.append(moe_step)
            diagnostics_history.append(diagnostics)
            logits_history.append(logits_step)
            crop_boxes_history.append(crop_boxes_step)
            offsets_history.append(offsets_step)
            boxes_history.append(boxes_step)

        heatmap = heatmaps[-1]
        centers = centers_history[-1]
        scores = scores_history[-1]
        sources = sources_history[-1]
        encoded = encoded_history[-1]
        fused = fused_history[-1]
        moe_features = moe_history[-1]
        moe_diagnostics = diagnostics_history[-1]
        logits = logits_history[-1]
        crop_boxes = crop_boxes_history[-1]
        center_offsets = offsets_history[-1]
        global_boxes = boxes_history[-1]
        balance_loss = torch.stack([item.balance_loss for item in diagnostics_history]).mean()
        focus_windows = None
        if self.focus is not None and not torch.is_grad_enabled():
            global_boxes, focus_windows = self.focus.refine(frames[:, -1], global_boxes, logits)
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
            center_offsets=center_offsets,
            boxes=global_boxes,
            crop_scale=self._crop_scale((height, width)),
            balance_loss=balance_loss,
            moe_diagnostics=moe_diagnostics,
            motion_gate_confidence=gate_confidence[:, -1],
            used_dense_mode=dense_mode,
            all_heatmaps=heatmaps,
            proposal_centers_seq=centers_history,
            proposal_sources_seq=sources_history,
            objectness_logits_seq=logits_history,
            crop_boxes_seq=crop_boxes_history,
            center_offsets_seq=offsets_history,
            boxes_seq=boxes_history,
            frame_last=frames[:, -1],
            focus_windows=focus_windows,
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
            pad_frame = frames_for_triplet[0] if frames_for_triplet else frame
            frames_for_triplet.insert(0, pad_frame)
        triplet = torch.cat([frames_for_triplet[-2], frames_for_triplet[-1], frame], dim=1)
        heatmap, confidence = self._motion_step(triplet, self._stream_prev_heatmap)
        self._stream_prev_heatmap = heatmap.detach()
        dense_mode = self._use_dense_mode(confidence)
        heatmap, centers, scores, sources, encoded, augmented = self._forward_single(
            frame, heatmap, frame_index, guided_centers, dense=dense_mode
        )
        self._stream_buffer.append(frame.detach())
        self._stream_buffer = self._stream_buffer[-self.config.temporal_window :]
        self._stream_feature_buffer.append(augmented.detach())
        self._stream_feature_buffer = self._stream_feature_buffer[-self.config.temporal_window :]
        self._stream_center_buffer.append(centers.detach())
        self._stream_center_buffer = self._stream_center_buffer[-self.config.temporal_window :]
        self._stream_source_buffer.append(sources.detach())
        self._stream_source_buffer = self._stream_source_buffer[-self.config.temporal_window :]
        sequence, centers_seq, sources_seq, padding_mask = self._align_temporal_history(
            self._stream_feature_buffer,
            self._stream_center_buffer,
            self._stream_source_buffer,
        )
        fused = self.temporal(
            sequence,
            centers_seq=centers_seq,
            source_labels_seq=sources_seq,
            padding_mask=padding_mask,
        )
        moe_features, moe_diagnostics = self.moe(fused, source_labels=sources)
        logits, crop_boxes, center_offsets = self._apply_head(encoded, fused, moe_features, self._last_fine)
        global_boxes = self._boxes_to_global(
            crop_boxes,
            centers,
            frame.shape[-2:],
            center_offsets,
            tuple(heatmap.shape[-2:]),
        )
        focus_windows = None
        if self.focus is not None:
            global_boxes, focus_windows = self.focus.refine(frame, global_boxes, logits)
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
            center_offsets=center_offsets,
            boxes=global_boxes,
            crop_scale=self._crop_scale(frame.shape[-2:]),
            balance_loss=moe_diagnostics.balance_loss,
            moe_diagnostics=moe_diagnostics,
            motion_gate_confidence=confidence,
            used_dense_mode=dense_mode,
            all_heatmaps=[heatmap],
            proposal_centers_seq=[centers],
            proposal_sources_seq=[sources],
            objectness_logits_seq=[logits],
            crop_boxes_seq=[crop_boxes],
            center_offsets_seq=[center_offsets],
            boxes_seq=[global_boxes],
            frame_last=frame,
            focus_windows=focus_windows,
        )

    def reset_stream(self) -> None:
        self._stream_prev_heatmap = None
        self._stream_buffer.clear()
        self._stream_feature_buffer.clear()
        self._stream_center_buffer.clear()
        self._stream_source_buffer.clear()
