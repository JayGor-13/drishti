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

    ldmi_scales: tuple[int, ...] = (7, 15, 31, 63)
    use_ldmi: bool = True

    motion_cnn_channels: tuple[int, ...] = (32, 64, 64)
    use_motion_gate: bool = True
    motion_gate_hidden: int = 16
    motion_gate_threshold: float = 0.5
    motion_gate_active_threshold: float = 0.5

    num_crops: int = 8
    crop_size: int = 64
    dense_grid_size: int = 4
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
    router_z_loss_weight: float = 0.001
    dense_moe: bool = False

    head_hidden_dim: int = 256
    objectness_threshold: float = 0.3

    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    heatmap_focal_alpha: float = 2.0
    heatmap_focal_beta: float = 4.0
    w_motion_displacement: float = 0.5
    w_gate_sparsity: float = 0.01
    w_temporal_consistency: float = 0.3
    w_trajectory_smoothness: float = 0.1
    sigma_spatial_consist: float = 0.1

    tracker_dist_threshold: float = 0.15
    tracker_max_coast: int = 15
    tracker_birth_threshold: float = 0.3

    dataset_name: str = "anti_uav"
    data_root: str | None = None
    frames_root: str | None = None
    modality: str = "visible"
    box_format: str = "xywh"
    clip_stride: int = 4
    frame_stride: int = 1
    eval_split: str = "val"
    output_dir: str = "results"
    checkpoint: str | None = None
    smoke_sequence_dir: str | None = None
    smoke_max_frames: int = 24
    smoke_train_steps: int = 2
    smoke_lr: float = 1e-4
    smoke_output_video: str = "results/smokerun/bounding_boxes.mp4"
    visualization_threshold: float = 0.3

    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        self.ldmi_scales = tuple(self.ldmi_scales)
        self.motion_cnn_channels = tuple(self.motion_cnn_channels)

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

    @property
    def motion_input_channels(self) -> int:
        return self.image_channels * (5 if self.use_ldmi else 3)

    @property
    def dense_num_crops(self) -> int:
        return self.dense_grid_size * self.dense_grid_size

    def validate(self) -> None:
        if self.num_crops < 1:
            raise ValueError("num_crops must be positive")
        if self.dense_grid_size < 1:
            raise ValueError("dense_grid_size must be positive")
        if self.crop_size < 2:
            raise ValueError("crop_size must be at least 2")
        if not 0.0 < self.border_width_frac < 0.5:
            raise ValueError("border_width_frac must be in (0, 0.5)")
        if self.scan_period < 1:
            raise ValueError("scan_period must be positive")
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if self.temporal_window < 1:
            raise ValueError("temporal_window must be positive")
        if not 0.0 <= self.motion_gate_threshold <= 1.0:
            raise ValueError("motion_gate_threshold must be in [0, 1]")
        if not 0.0 <= self.motion_gate_active_threshold <= 1.0:
            raise ValueError("motion_gate_active_threshold must be in [0, 1]")
        if self.modality not in {"visible", "infrared"}:
            raise ValueError("modality must be 'visible' or 'infrared'")
        if self.box_format not in {"xywh", "xyxy"}:
            raise ValueError("box_format must be 'xywh' or 'xyxy'")
