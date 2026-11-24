from __future__ import annotations

import random
from typing import Any

import torch
from torch import Tensor


class VideoAugmentation:
    """Consistent clip-level augmentations for video windows."""

    def __init__(self, train: bool = True) -> None:
        self.train = train

    def __call__(self, frames: list[Tensor], targets: list[dict]) -> tuple[list[Tensor], list[dict]]:
        frames, targets, _ = self.apply(frames, targets)
        return frames, targets

    def apply(
        self,
        frames: list[Tensor],
        targets: list[dict],
        metadata: dict[str, list[Any]] | None = None,
    ) -> tuple[list[Tensor], list[dict], dict[str, list[Any]] | None]:
        if not self.train:
            return frames, targets, metadata

        # Never mutate cached or caller-owned target tensors in place.
        targets = [
            {
                **target,
                "boxes": target.get("boxes", torch.empty(0, 4)).clone(),
                "labels": target.get("labels", torch.empty(0, dtype=torch.long)).clone(),
            }
            for target in targets
        ]
        metadata = {key: list(values) for key, values in metadata.items()} if metadata is not None else None

        if random.random() < 0.5:
            frames = [torch.flip(frame, dims=(-1,)) for frame in frames]
            for target in targets:
                if target["boxes"].numel() > 0:
                    target["boxes"][:, 0] = 1.0 - target["boxes"][:, 0]

        if random.random() < 0.5:
            gamma = random.uniform(0.9, 1.1)
            beta = random.uniform(-0.05, 0.05)
            frames = [(frame * gamma + beta).clamp(0.0, 1.0) for frame in frames]

        if random.random() < 0.3:
            frames = list(reversed(frames))
            targets = list(reversed(targets))
            if metadata is not None:
                metadata = {key: list(reversed(values)) for key, values in metadata.items()}
        return frames, targets, metadata
