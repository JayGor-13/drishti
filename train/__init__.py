"""Training utilities for DRISHTI-CORE Anti-UAV detection."""

from .antiuav import (
    AntiUAVDatasetPaths,
    AntiUAVDetectionCollator,
    AntiUAVRGBTVideoDataset,
    DRISHTICollator,
    MODELSCOPE_ANTI_UAV_URL,
    ModelScopeAntiUAVCocoDataset,
    SyntheticAntiUAVDataset,
)
from .drishti_loss import (
    DRISHTILossWeights,
    STAGE_CHECKPOINTS,
    assign_proposal_targets,
    configure_drishti_training_stage,
    detector_loss,
    detector_stage_loss,
    drishti_detector_loss,
    heatmap_targets_from_frame_targets,
    moe_stage_loss,
    scalar_metrics,
    stage_checkpoint_name,
    trainable_parameter_names,
)

__all__ = [
    "AntiUAVDatasetPaths",
    "AntiUAVDetectionCollator",
    "AntiUAVRGBTVideoDataset",
    "DRISHTICollator",
    "DRISHTILossWeights",
    "MODELSCOPE_ANTI_UAV_URL",
    "ModelScopeAntiUAVCocoDataset",
    "STAGE_CHECKPOINTS",
    "SyntheticAntiUAVDataset",
    "assign_proposal_targets",
    "configure_drishti_training_stage",
    "detector_loss",
    "detector_stage_loss",
    "drishti_detector_loss",
    "heatmap_targets_from_frame_targets",
    "moe_stage_loss",
    "scalar_metrics",
    "stage_checkpoint_name",
    "trainable_parameter_names",
]
