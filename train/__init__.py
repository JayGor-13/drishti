"""Training utilities for T-MoE Anti-UAV detection."""

from .antiuav import (
    AntiUAVDatasetPaths,
    AntiUAVDetectionCollator,
    AntiUAVRGBTVideoDataset,
    MODELSCOPE_ANTI_UAV_URL,
    ModelScopeAntiUAVCocoDataset,
    SyntheticAntiUAVDataset,
)
from .loss import (
    TMoELossWeights,
    cfcr_loss,
    detection_loss,
    expert_lora_similarity,
    focal_classification_loss,
    load_balancing_loss,
    orthogonalization_loss,
    routing_entropy,
    semantic_alignment,
    total_antiuav_loss,
    total_tmoe_loss,
)
from .trainer import TMoETrainer, TrainingConfig

__all__ = [
    "AntiUAVDatasetPaths",
    "AntiUAVDetectionCollator",
    "AntiUAVRGBTVideoDataset",
    "MODELSCOPE_ANTI_UAV_URL",
    "ModelScopeAntiUAVCocoDataset",
    "SyntheticAntiUAVDataset",
    "TMoELossWeights",
    "TMoETrainer",
    "TrainingConfig",
    "cfcr_loss",
    "detection_loss",
    "expert_lora_similarity",
    "focal_classification_loss",
    "load_balancing_loss",
    "orthogonalization_loss",
    "routing_entropy",
    "semantic_alignment",
    "total_antiuav_loss",
    "total_tmoe_loss",
]
