"""Typed outputs shared across DRISHTI-CORE stages."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class CropProposalOutput:
    crops: Tensor
    motion_scores: Tensor
    boxes: Tensor
    centers: Tensor
    heatmap: Tensor


@dataclass
class DetectorStageOutput:
    proposal: CropProposalOutput
    crop_features: Tensor
    object_logits: Tensor
    boxes: Tensor


@dataclass
class TemporalFusionOutput:
    fused_features: Tensor
    temporal_tokens: Tensor


@dataclass
class DRISHTIMoEOutput:
    hidden_states: Tensor
    router_logits: Tensor
    router_probs: Tensor
    topk_indices: Tensor
    topk_scores: Tensor
    load_balance_loss: Tensor


@dataclass
class DRISHTIOutput:
    heatmaps: Tensor
    proposal_boxes: Tensor
    motion_scores: Tensor
    crop_features: Tensor
    temporal_tokens: Tensor
    temporal_features: Tensor
    moe_features: Tensor
    object_logits: Tensor
    boxes: Tensor
    predictions: Tensor
    router_probs: Tensor
    router_topk: Tensor
    load_balance_loss: Tensor
