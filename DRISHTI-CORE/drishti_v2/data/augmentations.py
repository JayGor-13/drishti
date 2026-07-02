from __future__ import annotations

import random

import torch
from torch import Tensor
import torch.nn.functional as F


class VideoAugmentation:
    """Consistent clip-level augmentations for video windows."""

    def __init__(self, train: bool = True) -> None:
        self.train = train

    def __call__(self, frames: list[Tensor], targets: list[dict]) -> tuple[list[Tensor], list[dict]]:
        if not self.train:
            return frames, targets

        if random.random() < 0.5:
            frames = [torch.flip(frame, dims=(-1,)) for frame in frames]
            for target in targets:
                if target["boxes"].numel() > 0:
                    target["boxes"][:, 0] = 1.0 - target["boxes"][:, 0]

        brightness = 1.0 + random.uniform(-0.3, 0.3)
        contrast = 1.0 + random.uniform(-0.3, 0.3)
        frames = [((frame - 0.5) * contrast + 0.5).mul(brightness).clamp(0, 1) for frame in frames]

        if random.random() < 0.2:
            frames = [self._blur(frame) for frame in frames]

        if random.random() < 0.2:
            frames = self._erase(frames)
        return frames, targets

    @staticmethod
    def _blur(frame: Tensor) -> Tensor:
        kernel = frame.new_tensor([1.0, 2.0, 1.0])
        kernel = (kernel[:, None] * kernel[None, :]).div(16.0)
        kernel = kernel.expand(frame.shape[0], 1, 3, 3)
        return F.conv2d(frame.unsqueeze(0), kernel, padding=1, groups=frame.shape[0]).squeeze(0)

    @staticmethod
    def _erase(frames: list[Tensor]) -> list[Tensor]:
        _, height, width = frames[0].shape
        erase_h = max(1, int(height * random.uniform(0.05, 0.18)))
        erase_w = max(1, int(width * random.uniform(0.05, 0.18)))
        y0 = random.randint(0, max(0, height - erase_h))
        x0 = random.randint(0, max(0, width - erase_w))
        out = []
        for frame in frames:
            erased = frame.clone()
            erased[:, y0 : y0 + erase_h, x0 : x0 + erase_w] = 0.0
            out.append(erased)
        return out
