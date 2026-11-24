from __future__ import annotations

from typing import Any

import torch


class DRISHTICollator:
    """Collates clips while preserving variable-length per-frame targets."""

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        frame_targets = [item["frame_targets"] for item in batch]
        return {
            "frames": torch.stack([item["frames"] for item in batch], dim=0),
            "frame_targets": frame_targets,
            "targets": frame_targets,
            "image_ids": [item["image_ids"] for item in batch],
            "sequence": [item["sequence"] for item in batch],
            "dataset_url": batch[0].get("dataset_url"),
            "meta": [item.get("meta", {}) for item in batch],
        }
