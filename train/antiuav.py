"""Anti-UAV COCO-style data loading for the T-MoE detector."""

from __future__ import annotations

import json
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


@dataclass(frozen=True)
class AntiUAVVideoSequence:
    """One Anti-UAV-RGBT sequence folder with a modality video and labels."""

    name: str
    video_path: Path
    ann_path: Path
    boxes: tuple[Any, ...]
    exists: tuple[bool, ...]

    @property
    def num_frames(self) -> int:
        return max(len(self.boxes), len(self.exists))


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


def _bool_from_annotation(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "null"}
    return bool(value)


def _box_has_area(raw_box: Any) -> bool:
    try:
        values = [float(value) for value in raw_box[:4]]
    except (TypeError, ValueError):
        return False
    if len(values) < 4:
        return False
    return values[2] > 0 and values[3] > 0


def _candidate_boxes(raw_box: Any) -> list[Any]:
    if not isinstance(raw_box, (list, tuple)) or not raw_box:
        return []
    first = raw_box[0]
    if isinstance(first, (list, tuple)):
        return list(raw_box)
    return [raw_box]


def _read_antiuav_json(path: Path) -> tuple[tuple[Any, ...], tuple[bool, ...]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        raw_boxes = data
        raw_exists = None
    elif isinstance(data, dict):
        raw_boxes = (
            data.get("gt_rect")
            or data.get("gt_bbox")
            or data.get("bbox")
            or data.get("bboxes")
            or data.get("boxes")
        )
        raw_exists = (
            data.get("exist")
            or data.get("exists")
            or data.get("target_visible")
            or data.get("presence")
        )
    else:
        raise ValueError(f"Unsupported Anti-UAV annotation JSON at {path}")

    if not isinstance(raw_boxes, list):
        raise ValueError(
            f"Could not find frame boxes in {path}; expected a key like 'gt_rect'."
        )

    if raw_exists is not None and not isinstance(raw_exists, list):
        raise ValueError(f"Expected frame existence list in {path}")

    frame_count = max(len(raw_boxes), len(raw_exists or []))
    boxes = list(raw_boxes) + [[] for _ in range(frame_count - len(raw_boxes))]
    if raw_exists is None:
        exists = [_box_has_area(box) for box in boxes]
    else:
        exists = [_bool_from_annotation(value) for value in raw_exists]
        exists.extend(_box_has_area(boxes[idx]) for idx in range(len(exists), frame_count))
    return tuple(boxes), tuple(exists)


class AntiUAVRGBTVideoDataset(Dataset):
    """Temporal windows from the extracted Anti-UAV-RGBT video layout.

    Expected local structure for IR training:

    ``data_root/train/<sequence>/infrared.mp4``
    ``data_root/train/<sequence>/infrared.json``

    The loader also checks common ``label_new`` locations for validation/test
    annotations when they are not stored beside the videos.
    """

    dataset_url = "local Anti-UAV-RGBT extraction"

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        modality: str = "infrared",
        num_frames: int = 9,
        height: int = 448,
        width: int = 448,
        clip_stride: int = 4,
        frame_stride: int = 1,
        image_channels: int = 1,
        box_format: str = "xywh",
        label_dir_name: str = "label_new",
        sequence_ids: set[str] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.split_dir = self.data_root / split
        self.modality = modality
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.clip_stride = max(1, clip_stride)
        self.frame_stride = max(1, frame_stride)
        self.image_channels = image_channels
        self.box_format = box_format
        self.label_dir_name = label_dir_name

        if self.modality not in {"infrared", "visible"}:
            raise ValueError("modality must be 'infrared' or 'visible'")
        if self.image_channels not in {1, 3}:
            raise ValueError("image_channels must be 1 or 3")
        if self.box_format not in {"xywh", "xyxy"}:
            raise ValueError("box_format must be 'xywh' or 'xyxy'")
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Anti-UAV split directory not found: {self.split_dir}")

        self.sequences = self._discover_sequences(sequence_ids)
        if not self.sequences:
            raise FileNotFoundError(
                f"No {self.modality}.mp4 + annotations found under {self.split_dir}."
            )

        self.samples: list[tuple[int, int]] = []
        window_span = (self.num_frames - 1) * self.frame_stride + 1
        for seq_idx, sequence in enumerate(self.sequences):
            max_start = max(0, sequence.num_frames - window_span)
            starts = list(range(0, max_start + 1, self.clip_stride)) or [0]
            if starts[-1] != max_start:
                starts.append(max_start)
            self.samples.extend((seq_idx, start) for start in starts)

    def _annotation_candidates(self, sequence_dir: Path) -> list[Path]:
        label_root = self.data_root / self.label_dir_name
        name = sequence_dir.name
        return [
            sequence_dir / f"{self.modality}.json",
            label_root / self.split / name / f"{self.modality}.json",
            label_root / name / f"{self.modality}.json",
            label_root / self.split / f"{name}_{self.modality}.json",
            label_root / self.split / f"{self.modality}_{name}.json",
            label_root / self.split / f"{name}.json",
            label_root / f"{name}_{self.modality}.json",
            label_root / f"{self.modality}_{name}.json",
            label_root / f"{name}.json",
        ]

    def _find_annotation(self, sequence_dir: Path) -> Path | None:
        for candidate in self._annotation_candidates(sequence_dir):
            if candidate.exists():
                return candidate
        return None

    def _discover_sequences(self, sequence_ids: set[str] | None) -> list[AntiUAVVideoSequence]:
        sequences: list[AntiUAVVideoSequence] = []
        for sequence_dir in sorted(path for path in self.split_dir.iterdir() if path.is_dir()):
            if sequence_ids is not None and sequence_dir.name not in sequence_ids:
                continue
            video_path = sequence_dir / f"{self.modality}.mp4"
            ann_path = self._find_annotation(sequence_dir)
            if not video_path.exists() or ann_path is None:
                continue
            boxes, exists = _read_antiuav_json(ann_path)
            if not boxes:
                continue
            sequences.append(
                AntiUAVVideoSequence(
                    name=sequence_dir.name,
                    video_path=video_path,
                    ann_path=ann_path,
                    boxes=boxes,
                    exists=exists,
                )
            )
        return sequences

    def __len__(self) -> int:
        return len(self.samples)

    def _frame_indices(self, start: int, sequence: AntiUAVVideoSequence) -> list[int]:
        last = max(sequence.num_frames - 1, 0)
        return [
            min(start + frame_idx * self.frame_stride, last)
            for frame_idx in range(self.num_frames)
        ]

    def _read_frames(self, video_path: Path, frame_indices: list[int]) -> tuple[list[Tensor], tuple[int, int]]:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise ImportError(
                "Install opencv-python to read Anti-UAV-RGBT mp4 files in this loader."
            ) from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        frames: list[Tensor] = []
        previous_frame = None
        try:
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = cap.read()
                if not ok:
                    frame = previous_frame
                if frame is None:
                    frames.append(torch.zeros(self.image_channels, self.height, self.width))
                    continue
                previous_frame = frame
                source_height, source_width = frame.shape[:2]
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                tensor = torch.from_numpy(frame.copy()).float().div(255.0).unsqueeze(0)
                if self.image_channels == 3:
                    tensor = tensor.repeat(3, 1, 1)
                frames.append(_resize_chw(tensor, self.height, self.width))
        finally:
            cap.release()
        return frames, (source_width, source_height)

    def _target_from_box(
        self,
        raw_box: Any,
        exists: bool,
        image_size: tuple[int, int],
    ) -> dict[str, Tensor]:
        if not exists:
            return _empty_target()

        image_width, image_height = image_size
        boxes = []
        for candidate in _candidate_boxes(raw_box):
            try:
                values = [float(value) for value in candidate[:4]]
            except (TypeError, ValueError):
                continue
            if len(values) < 4:
                continue
            if self.box_format == "xyxy":
                x1, y1, x2, y2 = values
                x, y, box_w, box_h = x1, y1, x2 - x1, y2 - y1
            else:
                x, y, box_w, box_h = values
            if box_w <= 0 or box_h <= 0:
                continue
            cx = (x + box_w / 2.0) / max(image_width, 1)
            cy = (y + box_h / 2.0) / max(image_height, 1)
            boxes.append(
                [
                    min(max(cx, 0.0), 1.0),
                    min(max(cy, 0.0), 1.0),
                    min(max(box_w / max(image_width, 1), 0.0), 1.0),
                    min(max(box_h / max(image_height, 1), 0.0), 1.0),
                ]
            )

        if not boxes:
            return _empty_target()
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(boxes), dtype=torch.long),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        seq_idx, start = self.samples[index]
        sequence = self.sequences[seq_idx]
        frame_indices = self._frame_indices(start, sequence)
        frames, image_size = self._read_frames(sequence.video_path, frame_indices)
        targets = [
            self._target_from_box(
                sequence.boxes[frame_index],
                sequence.exists[frame_index] if frame_index < len(sequence.exists) else False,
                image_size,
            )
            for frame_index in frame_indices
        ]
        return {
            "frames": torch.stack(frames),
            "frame_targets": targets,
            "image_ids": [f"{sequence.name}:{frame_index}" for frame_index in frame_indices],
            "sequence": sequence.name,
            "dataset_url": self.dataset_url,
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
        image_channels: int = 3,
    ) -> None:
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.image_channels = image_channels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(10_000 + index)
        frames = 0.08 * torch.rand(
            self.num_frames,
            self.image_channels,
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
            if self.image_channels == 1:
                color = torch.tensor([0.95])
            else:
                color = torch.tensor([0.95, 0.92, 0.35])
            frames[frame_idx, :, y : y + box_size, x : x + box_size] = color.view(
                self.image_channels, 1, 1
            )
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
