"""Configuration for DRISHTI-CORE."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DRISHTIConfig:
    """Detector-only configuration for Anti-UAV experiments."""

    image_channels: int = 3
    temporal_window: int = 5
    crop_size: int = 64
    num_crops: int = 8
    feature_dim: int = 256
    temporal_input_dim: int = 257
    temporal_heads: int = 4
    temporal_layers: int = 2
    temporal_ffn_dim: int = 512
    temporal_dropout: float = 0.1
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_ffn_dim: int = 512
    load_balance_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.temporal_input_dim != self.feature_dim + 1:
            raise ValueError("temporal_input_dim must equal feature_dim + motion score")
        if self.moe_top_k < 1 or self.moe_top_k > self.moe_num_experts:
            raise ValueError("moe_top_k must be in [1, moe_num_experts]")
