from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(**({"slots": True} if sys.version_info >= (3, 10) else {}))
class DRISHTIConfig:
    """Master configuration for the DRISHTI-CORE v2 pipeline."""

    image_channels: int = 1
    image_height: int = 448
    image_width: int = 448

    ldmi_scales: tuple[int, ...] = (7, 15, 31, 63)
    use_ldmi: bool = True
    use_sobel_edge: bool = True

    motion_cnn_channels: tuple[int, ...] = (32, 64, 64)
    use_motion_gate: bool = True
    motion_gate_hidden: int = 16
    motion_gate_threshold: float = 0.5
    motion_gate_active_threshold: float = 0.5

    num_crops: int = 8
    crop_size: int = 64
    crop_scales: tuple[float, ...] = (1.0, 2.0, 4.0)
    dense_grid_size: int = 4
    border_width_frac: float = 0.07
    scan_period: int = 4
    use_edge_crops: bool = True
    use_grid_crops: bool = True
    use_motion_crops: bool = True
    use_guided_crops: bool = True

    encoder_feature_dim: int = 256
    encoder_frozen: bool = True
    # Train the crop encoder together with the MotionCNN and head in stage 1 (it is still frozen in
    # stages 2-3 and fine-tuned in stage 4). False reproduces the original runs, in which the encoder
    # kept its random initial weights for the whole of stages 1-3.
    train_encoder_in_stage1: bool = False
    encoder_backbone: str = "simple"  # "simple" (original small CNN) or "resnet18" (stem + layer1 + layer2)
    encoder_pretrained_path: str | None = None  # ImageNet resnet18 .pth for encoder_backbone "resnet18"

    # Idea D: the motion heatmap also sees its own previous heatmap, so it can keep firing on a drone that
    # stops moving (hovering, or a camera following it). Adds one input channel to the MotionCNN.
    heatmap_recurrent: bool = False

    # Version 3 (layer use): head reads a learned mix of encoder / temporal / MoE outputs, and the box
    # branch also gets the encoder's shallower, higher-resolution layer1 features (resnet18 encoder only).
    head_layer_mix: bool = False
    head_fine_features: bool = False

    # Chain-of-Focus refinement (drishti_v2/models/focus.py). "off" reproduces earlier runs.
    focus_mode: str = "off"  # "off" | "fixed" (1 step, version 1) | "adaptive" (learned stop, version 2)
    focus_topk: int = 2  # candidates per frame that get a closer look
    focus_max_steps: int = 1  # 1 for fixed; up to 3 for adaptive
    focus_expand: float = 3.0  # focus window = expand x box size (pixels)
    focus_min_window_px: float = 12.0
    focus_max_window_px: float = 96.0
    focus_halt_iou: float = 0.7  # adaptive: "stop" target = IoU(current box, GT) >= this
    focus_train_windows: int = 4  # jittered GT boxes per clip during training (plus the model's own top box)
    w_focus: float = 1.0
    w_focus_halt: float = 0.5

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
    w_subpixel_offset: float = 1.0
    sigma_spatial_consist: float = 0.1

    tracker_dist_threshold: float = 0.15
    tracker_max_coast: int = 15
    tracker_birth_threshold: float = 0.3

    dataset_name: str = "anti_uav"
    data_root: str | None = None
    frames_root: str | None = None
    modality: str = "infrared"
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
        self.crop_scales = tuple(float(scale) for scale in self.crop_scales)

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
        if not self.use_ldmi:
            return 3 * self.image_channels + int(self.heatmap_recurrent)
        return 3 * self.image_channels + 6 + int(self.use_sobel_edge) + int(self.heatmap_recurrent)

    @property
    def encoder_in_channels(self) -> int:
        return self.image_channels * len(self.crop_scales)

    @property
    def dense_num_crops(self) -> int:
        return self.dense_grid_size * self.dense_grid_size

    def validate(self) -> None:
        if self.image_channels not in {1, 3}:
            raise ValueError("image_channels must be 1 or 3")
        if self.num_crops < 1:
            raise ValueError("num_crops must be positive")
        if self.dense_grid_size < 1:
            raise ValueError("dense_grid_size must be positive")
        if self.crop_size < 2:
            raise ValueError("crop_size must be at least 2")
        if not self.crop_scales or any(scale <= 0.0 for scale in self.crop_scales):
            raise ValueError("crop_scales must contain positive values")
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
