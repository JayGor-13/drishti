from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from drishti_v2.data import AntiUAVExtractedFrameDataset, AntiUAVRGBTVideoDataset, DRISHTICollator, SyntheticAntiUAVDataset
from drishti_v2.models import DRISHTIConfig, DRISHTIPipeline


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(config: DRISHTIConfig, requested: str | None = None) -> torch.device:
    value = requested or config.device
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_config(path: str | Path | None) -> DRISHTIConfig:
    return DRISHTIConfig.from_yaml(path) if path else DRISHTIConfig()


def build_loader(
    config: DRISHTIConfig,
    data_root: str | None,
    split: str,
    batch_size: int,
    synthetic: bool = False,
    shuffle: bool = False,
    frames_root: str | None = None,
    modality: str | None = None,
    clip_stride: int | None = None,
    frame_stride: int | None = None,
    box_format: str | None = None,
) -> DataLoader:
    data_root = data_root or config.data_root
    frames_root = frames_root or config.frames_root
    modality = modality or config.modality
    clip_stride = config.clip_stride if clip_stride is None else clip_stride
    frame_stride = config.frame_stride if frame_stride is None else frame_stride
    box_format = box_format or config.box_format

    if frames_root:
        dataset = AntiUAVExtractedFrameDataset(
            frames_root=frames_root,
            split=split,
            modality=modality,
            num_frames=config.temporal_window,
            height=config.image_height,
            width=config.image_width,
            clip_stride=clip_stride,
            frame_stride=frame_stride,
            image_channels=config.image_channels,
            box_format=box_format,
        )
    elif data_root:
        dataset = AntiUAVRGBTVideoDataset(
            data_root=data_root,
            split=split,
            modality=modality,
            num_frames=config.temporal_window,
            height=config.image_height,
            width=config.image_width,
            clip_stride=clip_stride,
            frame_stride=frame_stride,
            image_channels=config.image_channels,
            box_format=box_format,
        )
    elif synthetic:
        smoke_height = min(config.image_height, 64)
        smoke_width = min(config.image_width, 64)
        dataset = SyntheticAntiUAVDataset(
            num_samples=16,
            num_frames=config.temporal_window,
            height=smoke_height,
            width=smoke_width,
            image_channels=config.image_channels,
        )
    else:
        raise ValueError("Provide --frames-root, --data-root, or --synthetic.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0 if synthetic else config.num_workers,
        collate_fn=DRISHTICollator(),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(not synthetic and config.num_workers > 0),
        drop_last=False,
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--frames-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--modality", default=None, choices=["visible", "infrared"])
    parser.add_argument("--clip-stride", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--box-format", default=None, choices=["xywh", "xyxy"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Use generated data for smoke runs.")


def build_model(config: DRISHTIConfig, checkpoint: str | None = None, device: torch.device | str = "cpu") -> DRISHTIPipeline:
    model = DRISHTIPipeline(config).to(device)
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload)
    return model
