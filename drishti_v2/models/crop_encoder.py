from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class _FreezableEncoder(nn.Module):
    """Shared freeze/unfreeze behaviour: a frozen encoder stays in eval mode even if a parent calls train()."""

    _frozen: bool

    def freeze(self) -> None:
        self._frozen = True
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        self._frozen = False
        self.train(True)
        for parameter in self.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True):
        super().train(False if self._frozen else mode)
        return self


class CropEncoder(_FreezableEncoder):
    """Small CNN patch encoder (the original DRISHTI encoder, random init)."""

    def __init__(self, out_dim: int = 256, in_channels: int = 3) -> None:
        super().__init__()
        self._frozen = False
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        spatial_dim = 256 * 7 * 7
        self.proj = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(spatial_dim),
            nn.Linear(spatial_dim, out_dim),
        )

    def forward(self, crops: Tensor) -> Tensor:
        return self.proj(self.features(crops))


class _BasicBlock(nn.Module):
    """torchvision-compatible ResNet BasicBlock (parameter names match the official checkpoint)."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18CropEncoder(_FreezableEncoder):
    """ResNet-18 stem + layer1 + layer2 (ImageNet-pretrained when weights are given).

    A 64x64 crop becomes an 8x8x128 map, pooled to 4x4 and projected to ``out_dim``.
    For infrared input with three crop scales, the three scales fill the three RGB input channels,
    so the pretrained first convolution is used as-is. For other channel counts its weights are
    averaged over RGB and repeated.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, out_dim: int = 256, in_channels: int = 3, pretrained_path: str | None = None) -> None:
        super().__init__()
        self._frozen = False
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = nn.Sequential(_BasicBlock(64, 64), _BasicBlock(64, 64))
        self.layer2 = nn.Sequential(_BasicBlock(64, 128, stride=2), _BasicBlock(128, 128))
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Sequential(nn.Flatten(1), nn.LayerNorm(128 * 16), nn.Linear(128 * 16, out_dim))
        if in_channels == 3:
            mean, std = self.IMAGENET_MEAN, self.IMAGENET_STD
        else:  # grey or multi-scale grey input: use the RGB-averaged statistics on every channel
            mean = (sum(self.IMAGENET_MEAN) / 3,) * in_channels
            std = (sum(self.IMAGENET_STD) / 3,) * in_channels
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1), persistent=False)
        self.pretrained = False
        if pretrained_path:
            self.load_imagenet(pretrained_path)

    def load_imagenet(self, path: str | Path) -> None:
        state = torch.load(path, map_location="cpu")
        wanted = {k: v for k, v in state.items() if k.startswith(("conv1.", "bn1.", "layer1.", "layer2."))}
        conv = wanted["conv1.weight"]
        if conv.shape[1] != self.conv1.in_channels:
            wanted["conv1.weight"] = conv.mean(dim=1, keepdim=True).repeat(1, self.conv1.in_channels, 1, 1)
        missing, unexpected = self.load_state_dict(wanted, strict=False)
        leftover = [k for k in missing if not k.startswith("proj.")]
        if leftover or unexpected:
            raise RuntimeError(f"ImageNet weights did not fit: missing={leftover} unexpected={unexpected}")
        self.pretrained = True

    def forward(self, crops: Tensor) -> Tensor:
        return self.forward_multi(crops)[0]

    def forward_multi(self, crops: Tensor) -> tuple[Tensor, Tensor]:
        """Return (main feature [N, out_dim], fine feature [N, 64*4*4]).

        The fine feature is layer1 (16x16 for a 64 px crop) pooled to 4x4: shallower, higher-resolution
        features that keep more of a few-pixel target's position than the deeper layer2 output.
        """

        x = (crops - self.mean) / self.std
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        fine = self.layer1(x)
        x = self.layer2(fine)
        return self.proj(self.pool(x)), self.pool(fine).flatten(1)


def build_crop_encoder(config) -> nn.Module:
    """Pick the crop encoder from the config (``encoder_backbone``: 'simple' or 'resnet18')."""

    backbone = getattr(config, "encoder_backbone", "simple")
    if backbone == "simple":
        return CropEncoder(config.encoder_feature_dim, in_channels=config.encoder_in_channels)
    if backbone == "resnet18":
        return ResNet18CropEncoder(
            config.encoder_feature_dim,
            in_channels=config.encoder_in_channels,
            pretrained_path=getattr(config, "encoder_pretrained_path", None),
        )
    raise ValueError(f"Unknown encoder_backbone: {backbone!r}")
