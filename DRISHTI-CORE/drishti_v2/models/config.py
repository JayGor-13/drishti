from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DRISHTIConfig:
    """Master configuration for the DRISHTI-CORE v2 pipeline."""

    image_channels: int = 3
    image_height: int = 448
    image_width: int = 448

    ldmi_scales: tuple[int, ...] = (15, 31)
    use_ldmi: bool = True

    motion_cnn_channels: tuple[int, ...] = (32, 64, 64)

    num_crops: int = 8
    crop_size: int = 64
    border_width_frac: float = 0.07
    scan_period: int = 4
    use_edge_crops: bool = True
    use_grid_crops: bool = True
    use_motion_crops: bool = True
    use_guided_crops: bool = True

    encoder_feature_dim: int = 256
    encoder_frozen: bool = True

    temporal_window: int = 5
    temporal_heads: int = 4
    temporal_layers: int = 2
    temporal_ffn_dim: int = 512
    temporal_dropout: float = 0.1

    num_experts: int = 8
    top_k: int = 2
    expert_ffn_dim: int = 512
    moe_dropout: float = 0.1
    moe_balance_weight: float = 0.01
    dense_moe: bool = False

    head_hidden_dim: int = 256
    objectness_threshold: float = 0.3

    tracker_dist_threshold: float = 0.15
    tracker_max_coast: int = 15
    tracker_birth_threshold: float = 0.3

    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DRISHTIConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - valid)
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {unknown}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=True), encoding="utf-8")

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.image_height, self.image_width

    def validate(self) -> None:
        if self.num_crops < 1:
            raise ValueError("num_crops must be positive")
        if self.crop_size < 2:
            raise ValueError("crop_size must be at least 2")
        if not 0.0 < self.border_width_frac < 0.5:
            raise ValueError("border_width_frac must be in (0, 0.5)")
        if self.scan_period < 1:
            raise ValueError("scan_period must be positive")
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
