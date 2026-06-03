"""Anti-UAV COCO-style data loading for the T-MoE detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F


MODELSCOPE_ANTI_UAV_URL = "https://modelscope.cn/datasets/ly261666/3rd_Anti-UAV"


@dataclass(frozen=True)
class AntiUAVDatasetPaths:
    image_root: str | None = None
    ann_file: str | None = None

    @property
    def is_complete(self) -> bool:
        return bool(self.image_root and self.ann_file)


def _resize_chw(image: Tensor, height: int, width: int) -> Tensor:
    if image.ndim != 3:
        raise ValueError("image tensor must have shape [channels, height, width]")
    return F.interpolate(
        image.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _empty_target() -> dict[str, Tensor]:
    return {
        "boxes": torch.zeros(0, 4, dtype=torch.float32),
        "labels": torch.zeros(0, dtype=torch.long),
    }


class ModelScopeAntiUAVCocoDataset(Dataset):
    """Temporal windows over a COCO-format Anti-UAV frame dataset.

    This implements the user's requested access pattern with
    ``torchvision.datasets.CocoDetection``. The ModelScope page is stored as
    metadata; actual training expects downloaded/extracted local COCO paths.
    """

    dataset_url = MODELSCOPE_ANTI_UAV_URL

    def __init__(
        self,
        root: str | Path,
        ann_file: str | Path,
        num_frames: int = 9,
        height: int = 448,
        width: int = 448,
    ) -> None:
        try:
            import torchvision.datasets as datasets
            import torchvision.transforms.functional as TF
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise ImportError(
                "Install torchvision to load COCO-format Anti-UAV data."
            ) from exc

        self.root = Path(root)
        self.ann_file = Path(ann_file)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self._to_tensor = TF.to_tensor
        self.base = datasets.CocoDetection(root=str(self.root), annFile=str(self.ann_file))
        self.order = sorted(
            range(len(self.base.ids)),
            key=lambda idx: self.base.coco.loadImgs(self.base.ids[idx])[0].get("file_name", ""),
        )

    def __len__(self) -> int:
        return len(self.order)

    def _window_indices(self, center_index: int) -> list[int]:
        radius = self.num_frames // 2
        last = len(self.order) - 1
        return [min(max(center_index + offset, 0), last) for offset in range(-radius, radius + 1)]

    @staticmethod
    def _annotations_to_target(annotations: list[dict[str, Any]], image_size: tuple[int, int]) -> dict[str, Tensor]:
        width, height = image_size
        boxes = []
        labels = []
        for ann in annotations:
            if ann.get("iscrowd", 0):
                continue
            x, y, box_w, box_h = [float(value) for value in ann.get("bbox", [0, 0, 0, 0])]
            if box_w <= 0 or box_h <= 0:
                continue
            cx = (x + box_w / 2.0) / max(width, 1)
            cy = (y + box_h / 2.0) / max(height, 1)
            boxes.append(
                [
                    min(max(cx, 0.0), 1.0),
                    min(max(cy, 0.0), 1.0),
                    min(max(box_w / max(width, 1), 0.0), 1.0),
                    min(max(box_h / max(height, 1), 0.0), 1.0),
                ]
            )
            labels.append(1)

        if not boxes:
            return _empty_target()
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _load_frame(self, ordered_index: int) -> tuple[Tensor, dict[str, Tensor], int]:
        dataset_index = self.order[ordered_index]
        image, annotations = self.base[dataset_index]
        image = image.convert("RGB")
        original_size = image.size
        tensor = _resize_chw(self._to_tensor(image), self.height, self.width)
        target = self._annotations_to_target(annotations, original_size)
        return tensor, target, int(self.base.ids[dataset_index])

    def __getitem__(self, index: int) -> dict[str, Any]:
        frames = []
        targets = []
        image_ids = []
        for ordered_index in self._window_indices(index):
            frame, target, image_id = self._load_frame(ordered_index)
            frames.append(frame)
            targets.append(target)
            image_ids.append(image_id)
        return {
            "frames": torch.stack(frames),
            "frame_targets": targets,
            "image_ids": image_ids,
            "dataset_url": self.dataset_url,
        }


class SyntheticAntiUAVDataset(Dataset):
    """Tiny moving-box dataset for smoke tests without external downloads."""

    def __init__(
        self,
        num_samples: int = 16,
        num_frames: int = 5,
        height: int = 64,
        width: int = 64,
    ) -> None:
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(10_000 + index)
        frames = 0.08 * torch.rand(
            self.num_frames,
            3,
            self.height,
            self.width,
            generator=generator,
        )
        box_size = max(3, min(self.height, self.width) // 10)
        start_x = int(torch.randint(0, max(1, self.width - box_size - self.num_frames), (1,), generator=generator))
        start_y = int(torch.randint(0, max(1, self.height - box_size), (1,), generator=generator))
        drift_y = int(torch.randint(-1, 2, (1,), generator=generator))
        targets = []
        for frame_idx in range(self.num_frames):
            x = min(max(start_x + frame_idx, 0), self.width - box_size)
            y = min(max(start_y + drift_y * frame_idx, 0), self.height - box_size)
            frames[frame_idx, :, y : y + box_size, x : x + box_size] = torch.tensor(
                [0.95, 0.92, 0.35]
            ).view(3, 1, 1)
            box = torch.tensor(
                [
                    (x + box_size / 2.0) / self.width,
                    (y + box_size / 2.0) / self.height,
                    box_size / self.width,
                    box_size / self.height,
                ],
                dtype=torch.float32,
            )
            targets.append({"boxes": box.view(1, 4), "labels": torch.ones(1, dtype=torch.long)})

        return {
            "frames": frames.clamp(0.0, 1.0),
            "frame_targets": targets,
            "image_ids": [index * self.num_frames + frame for frame in range(self.num_frames)],
            "dataset_url": MODELSCOPE_ANTI_UAV_URL,
        }


class AntiUAVDetectionCollator:
    """Batch Anti-UAV frames and assign boxes to patch targets."""

    def __init__(self, patch_grid_size: int) -> None:
        self.patch_grid_size = patch_grid_size

    def _targets_to_patch_tensors(
        self,
        frame_targets: list[dict[str, Tensor]],
    ) -> tuple[Tensor, Tensor, Tensor]:
        time = len(frame_targets)
        patches = self.patch_grid_size**2
        class_targets = torch.zeros(time, patches, dtype=torch.long)
        box_targets = torch.zeros(time, patches, 4, dtype=torch.float32)
        box_mask = torch.zeros(time, patches, dtype=torch.bool)
        for frame_idx, target in enumerate(frame_targets):
            boxes = target.get("boxes", torch.zeros(0, 4))
            for box in boxes:
                cx = float(box[0].clamp(0.0, 1.0))
                cy = float(box[1].clamp(0.0, 1.0))
                grid_x = min(self.patch_grid_size - 1, int(cx * self.patch_grid_size))
                grid_y = min(self.patch_grid_size - 1, int(cy * self.patch_grid_size))
                patch_idx = grid_y * self.patch_grid_size + grid_x
                class_targets[frame_idx, patch_idx] = 1
                box_targets[frame_idx, patch_idx] = box.clamp(0.0, 1.0)
                box_mask[frame_idx, patch_idx] = True
        return class_targets, box_targets, box_mask

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        class_targets = []
        box_targets = []
        box_masks = []
        for item in batch:
            labels, boxes, mask = self._targets_to_patch_tensors(item["frame_targets"])
            class_targets.append(labels)
            box_targets.append(boxes)
            box_masks.append(mask)

        return {
            "frames": torch.stack([item["frames"] for item in batch]),
            "class_targets": torch.stack(class_targets),
            "box_targets": torch.stack(box_targets),
            "box_mask": torch.stack(box_masks),
            "frame_targets": [item["frame_targets"] for item in batch],
            "image_ids": [item["image_ids"] for item in batch],
            "dataset_url": batch[0].get("dataset_url", MODELSCOPE_ANTI_UAV_URL),
        }
