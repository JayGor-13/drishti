from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from drishti_v2.data.utils import list_image_files, xywh_to_cxcywh, xyxy_to_cxcywh

MODELSCOPE_ANTI_UAV_URL = "https://modelscope.cn/datasets/ly261666/3rd_Anti-UAV"
BOX_KEYS = ("gt_rect", "gt_bbox", "bbox", "bboxes", "boxes")
EXIST_KEYS = ("exist", "exists", "target_visible", "presence")


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "null", "nan"}
    return bool(value)


def _candidate_boxes(value: Any) -> list[list[float]]:
    if value is None or value == []:
        return []
    if isinstance(value, (tuple, list)) and len(value) >= 4 and not isinstance(value[0], (tuple, list)):
        try:
            return [[float(value[0]), float(value[1]), float(value[2]), float(value[3])]]
        except (TypeError, ValueError):
            return []
    boxes = []
    if isinstance(value, (tuple, list)):
        for item in value:
            boxes.extend(_candidate_boxes(item))
    return boxes


def _box_exists(value: Any) -> bool:
    boxes = _candidate_boxes(value)
    return any(box[2] > 0 and box[3] > 0 for box in boxes)


def _read_antiuav_json(path: Path) -> tuple[tuple[Any, ...], tuple[bool, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        boxes = tuple(raw)
        exists = tuple(_box_exists(item) for item in boxes)
        return boxes, exists
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported Anti-UAV annotation format: {path}")

    box_values = None
    for key in BOX_KEYS:
        if key in raw:
            box_values = raw[key]
            break
    if box_values is None:
        box_values = []
    boxes = list(box_values)

    exists_values = None
    for key in EXIST_KEYS:
        if key in raw:
            exists_values = raw[key]
            break
    exists = [_truthy(item) for item in exists_values] if exists_values is not None else []

    length = max(len(boxes), len(exists))
    while len(boxes) < length:
        boxes.append([])
    while len(exists) < length:
        exists.append(_box_exists(boxes[len(exists)]))
    return tuple(boxes), tuple(exists)


def _resize_chw(image: Tensor, height: int, width: int) -> Tensor:
    return F.interpolate(image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)


def _pil_to_tensor(path: Path, image_channels: int) -> tuple[Tensor, tuple[int, int]]:
    mode = "L" if image_channels == 1 else "RGB"
    image = Image.open(path).convert(mode)
    original_size = (image.height, image.width)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if image_channels == 1:
        tensor = torch.from_numpy(array).unsqueeze(0)
    else:
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor.contiguous(), original_size


def _cv_frame_to_tensor(frame: np.ndarray, image_channels: int) -> Tensor:
    if frame.ndim == 2:
        tensor = torch.from_numpy(frame.copy()).float().div(255.0).unsqueeze(0)
        return tensor.repeat(3, 1, 1) if image_channels == 3 else tensor
    if image_channels == 1:
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return torch.from_numpy(gray.copy()).float().div(255.0).unsqueeze(0)
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).float().div(255.0).permute(2, 0, 1).contiguous()


def _empty_target() -> dict[str, Tensor]:
    return {"boxes": torch.zeros(0, 4, dtype=torch.float32), "labels": torch.zeros(0, dtype=torch.long)}


def _target_from_antiuav_box(
    raw_box: Any,
    exists: bool,
    image_size: tuple[int, int],
    box_format: str,
) -> dict[str, Tensor]:
    if not exists:
        return _empty_target()
    boxes = _candidate_boxes(raw_box)
    valid = [box for box in boxes if box[2] > 0 and box[3] > 0]
    if not valid:
        return _empty_target()
    raw = torch.tensor(valid, dtype=torch.float32)
    if box_format == "xywh":
        boxes_tensor = xywh_to_cxcywh(raw, image_size=image_size)
    elif box_format == "xyxy":
        boxes_tensor = xyxy_to_cxcywh(raw, image_size=image_size)
    else:
        raise ValueError(f"Unsupported box_format: {box_format}")
    keep = (boxes_tensor[:, 2] > 0) & (boxes_tensor[:, 3] > 0)
    boxes_tensor = boxes_tensor[keep]
    if boxes_tensor.numel() == 0:
        return _empty_target()
    return {"boxes": boxes_tensor, "labels": torch.ones(boxes_tensor.shape[0], dtype=torch.long)}


def _starts_for_sequence(num_frames_available: int, num_frames: int, frame_stride: int, clip_stride: int) -> list[int]:
    window_span = (num_frames - 1) * frame_stride + 1
    max_start = max(0, num_frames_available - window_span)
    starts = list(range(0, max_start + 1, clip_stride))
    if not starts:
        starts = [0]
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def _frame_indices(start: int, num_frames: int, frame_stride: int, sequence_len: int) -> list[int]:
    last = max(sequence_len - 1, 0)
    return [min(start + idx * frame_stride, last) for idx in range(num_frames)]


@dataclass(frozen=True)
class AntiUAVFrameSequence:
    name: str
    frame_paths: tuple[Path, ...]
    annotation_path: Path
    boxes: tuple[Any, ...]
    exists: tuple[bool, ...]

    @property
    def num_frames(self) -> int:
        return len(self.frame_paths)


@dataclass(frozen=True)
class AntiUAVVideoSequence:
    name: str
    video_path: Path
    annotation_path: Path
    boxes: tuple[Any, ...]
    exists: tuple[bool, ...]
    num_frames: int


class _AntiUAVBase(Dataset):
    dataset_url = MODELSCOPE_ANTI_UAV_URL

    def __init__(
        self,
        split: str,
        modality: str,
        num_frames: int,
        height: int,
        width: int,
        clip_stride: int,
        frame_stride: int,
        image_channels: int,
        box_format: str,
    ) -> None:
        self.split = split
        self.modality = modality
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.clip_stride = clip_stride
        self.frame_stride = frame_stride
        self.image_channels = image_channels
        self.box_format = box_format
        if image_channels not in {1, 3}:
            raise ValueError("image_channels must be 1 or 3")

    def _make_item(
        self,
        frames: list[Tensor],
        targets: list[dict[str, Tensor]],
        image_ids: list[str],
        sequence_name: str,
        frame_indices: list[int],
    ) -> dict[str, Any]:
        frame_targets = targets
        return {
            "frames": torch.stack(frames, dim=0),
            "frame_targets": frame_targets,
            "targets": frame_targets,
            "image_ids": image_ids,
            "sequence": sequence_name,
            "dataset_url": self.dataset_url,
            "meta": {"sequence": sequence_name, "frame_indices": frame_indices, "image_ids": image_ids},
        }


class AntiUAVExtractedFrameDataset(_AntiUAVBase):
    """Recommended Anti-UAV loader for pre-extracted frame directories."""

    def __init__(
        self,
        frames_root: str | Path,
        split: str = "train",
        modality: str = "visible",
        num_frames: int = 5,
        height: int = 448,
        width: int = 448,
        clip_stride: int = 4,
        frame_stride: int = 1,
        image_channels: int = 3,
        box_format: str = "xywh",
        sequence_ids: set[str] | None = None,
    ) -> None:
        super().__init__(split, modality, num_frames, height, width, clip_stride, frame_stride, image_channels, box_format)
        self.frames_root = Path(frames_root)
        self.sequences = self._discover_sequences(sequence_ids)
        self.samples = [
            (seq_idx, start)
            for seq_idx, sequence in enumerate(self.sequences)
            for start in _starts_for_sequence(sequence.num_frames, num_frames, frame_stride, clip_stride)
        ]
        if not self.samples:
            raise FileNotFoundError(f"No extracted Anti-UAV samples found under {self.frames_root / split}")

    def _discover_sequences(self, sequence_ids: set[str] | None) -> list[AntiUAVFrameSequence]:
        split_root = self.frames_root / self.split
        if not split_root.exists():
            raise FileNotFoundError(split_root)
        sequences = []
        for sequence_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            if sequence_ids and sequence_dir.name not in sequence_ids:
                continue
            frame_dir = sequence_dir / self.modality
            annotation_path = sequence_dir / f"{self.modality}.json"
            if not frame_dir.exists() or not annotation_path.exists():
                continue
            frame_paths = tuple(list_image_files(frame_dir))
            if not frame_paths:
                continue
            boxes, exists = _read_antiuav_json(annotation_path)
            sequences.append(AntiUAVFrameSequence(sequence_dir.name, frame_paths, annotation_path, boxes, exists))
        return sequences

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq_idx, start = self.samples[idx]
        sequence = self.sequences[seq_idx]
        indices = _frame_indices(start, self.num_frames, self.frame_stride, sequence.num_frames)
        frames = []
        targets = []
        image_ids = []
        for frame_idx in indices:
            frame, original_size = _pil_to_tensor(sequence.frame_paths[frame_idx], self.image_channels)
            frames.append(_resize_chw(frame, self.height, self.width))
            raw_box = sequence.boxes[frame_idx] if frame_idx < len(sequence.boxes) else []
            exists = sequence.exists[frame_idx] if frame_idx < len(sequence.exists) else _box_exists(raw_box)
            targets.append(_target_from_antiuav_box(raw_box, exists, original_size, self.box_format))
            image_ids.append(f"{self.split}/{sequence.name}/{self.modality}/{frame_idx:06d}")
        return self._make_item(frames, targets, image_ids, sequence.name, indices)


class AntiUAVRGBTVideoDataset(_AntiUAVBase):
    """Anti-UAV raw-video loader. Prefer extracted frames for training speed."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        modality: str = "visible",
        num_frames: int = 5,
        height: int = 448,
        width: int = 448,
        clip_stride: int = 4,
        frame_stride: int = 1,
        image_channels: int = 3,
        box_format: str = "xywh",
        label_dir_name: str = "label_new",
        sequence_ids: set[str] | None = None,
    ) -> None:
        super().__init__(split, modality, num_frames, height, width, clip_stride, frame_stride, image_channels, box_format)
        self.data_root = Path(data_root)
        self.label_dir_name = label_dir_name
        self.sequences = self._discover_sequences(sequence_ids)
        self.samples = [
            (seq_idx, start)
            for seq_idx, sequence in enumerate(self.sequences)
            for start in _starts_for_sequence(sequence.num_frames, num_frames, frame_stride, clip_stride)
        ]
        if not self.samples:
            raise FileNotFoundError(f"No raw Anti-UAV video samples found under {self.data_root / split}")

    def _annotation_candidates(self, sequence_dir: Path) -> list[Path]:
        name = sequence_dir.name
        label_root = self.data_root / self.label_dir_name
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
        return next((path for path in self._annotation_candidates(sequence_dir) if path.exists()), None)

    def _discover_sequences(self, sequence_ids: set[str] | None) -> list[AntiUAVVideoSequence]:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("AntiUAVRGBTVideoDataset requires opencv-python-headless") from exc

        split_root = self.data_root / self.split
        if not split_root.exists():
            raise FileNotFoundError(split_root)
        sequences = []
        for sequence_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            if sequence_ids and sequence_dir.name not in sequence_ids:
                continue
            video_path = sequence_dir / f"{self.modality}.mp4"
            annotation_path = self._find_annotation(sequence_dir)
            if not video_path.exists() or annotation_path is None:
                continue
            capture = cv2.VideoCapture(str(video_path))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            boxes, exists = _read_antiuav_json(annotation_path)
            num_frames = max(count, len(boxes), len(exists))
            if num_frames > 0:
                sequences.append(AntiUAVVideoSequence(sequence_dir.name, video_path, annotation_path, boxes, exists, num_frames))
        return sequences

    def __len__(self) -> int:
        return len(self.samples)

    def _read_frames(self, sequence: AntiUAVVideoSequence, indices: list[int]) -> tuple[list[Tensor], tuple[int, int]]:
        import cv2

        capture = cv2.VideoCapture(str(sequence.video_path))
        frames = []
        previous: Tensor | None = None
        original_size = (self.height, self.width)
        for frame_idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if ok:
                original_size = (int(frame.shape[0]), int(frame.shape[1]))
                tensor = _cv_frame_to_tensor(frame, self.image_channels)
                previous = tensor
            elif previous is not None:
                tensor = previous
            else:
                tensor = torch.zeros(self.image_channels, self.height, self.width)
            frames.append(_resize_chw(tensor, self.height, self.width))
        capture.release()
        return frames, original_size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq_idx, start = self.samples[idx]
        sequence = self.sequences[seq_idx]
        indices = _frame_indices(start, self.num_frames, self.frame_stride, sequence.num_frames)
        frames, original_size = self._read_frames(sequence, indices)
        targets = []
        image_ids = []
        for frame_idx in indices:
            raw_box = sequence.boxes[frame_idx] if frame_idx < len(sequence.boxes) else []
            exists = sequence.exists[frame_idx] if frame_idx < len(sequence.exists) else _box_exists(raw_box)
            targets.append(_target_from_antiuav_box(raw_box, exists, original_size, self.box_format))
            image_ids.append(f"{self.split}/{sequence.name}/{self.modality}/{frame_idx:06d}")
        return self._make_item(frames, targets, image_ids, sequence.name, indices)


class AntiUAVDataset(AntiUAVExtractedFrameDataset):
    """Backward-compatible alias for the recommended extracted-frame loader."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        num_frames: int = 5,
        frame_size: tuple[int, int] = (448, 448),
        clip_stride: int = 4,
        frame_stride: int = 1,
        modality: str = "visible",
        box_format: str = "xywh",
        augment: bool = False,
        sequence_filter: list[str] | None = None,
    ) -> None:
        del augment
        super().__init__(
            frames_root=data_root,
            split=split,
            modality=modality,
            num_frames=num_frames,
            height=frame_size[0],
            width=frame_size[1],
            clip_stride=clip_stride,
            frame_stride=frame_stride,
            image_channels=3,
            box_format=box_format,
            sequence_ids=set(sequence_filter) if sequence_filter else None,
        )


class SyntheticAntiUAVDataset(Dataset):
    """Deterministic synthetic Anti-UAV smoke dataset."""

    def __init__(
        self,
        num_samples: int = 16,
        num_frames: int = 5,
        height: int = 64,
        width: int = 64,
        image_channels: int = 3,
        length: int | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        if length is not None:
            num_samples = length
        if frame_size is not None:
            height, width = frame_size
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.image_channels = image_channels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(10_000 + idx)
        frames = 0.08 * torch.rand(self.num_frames, self.image_channels, self.height, self.width, generator=generator)
        rect_w = max(3, self.width // 18)
        rect_h = max(3, self.height // 18)
        max_x = max(1, self.width - rect_w - self.num_frames)
        max_y = max(1, self.height - rect_h - self.num_frames)
        start_x = int(torch.randint(0, max_x, (1,), generator=generator).item())
        start_y = int(torch.randint(0, max_y, (1,), generator=generator).item())
        drift_y = [-1, 0, 1][idx % 3]
        color = torch.tensor([0.95, 0.92, 0.35], dtype=torch.float32)[: self.image_channels].view(self.image_channels, 1, 1)
        if self.image_channels == 1:
            color = torch.tensor([0.95], dtype=torch.float32).view(1, 1, 1)
        frame_targets = []
        image_ids = []
        for t_idx in range(self.num_frames):
            x0 = min(max(start_x + t_idx, 0), self.width - rect_w)
            y0 = min(max(start_y + drift_y * t_idx, 0), self.height - rect_h)
            frames[t_idx, :, y0 : y0 + rect_h, x0 : x0 + rect_w] = color
            box = torch.tensor(
                [
                    (x0 + rect_w / 2.0) / self.width,
                    (y0 + rect_h / 2.0) / self.height,
                    rect_w / self.width,
                    rect_h / self.height,
                ],
                dtype=torch.float32,
            )
            frame_targets.append({"boxes": box.view(1, 4), "labels": torch.ones(1, dtype=torch.long)})
            image_ids.append(f"synthetic/{idx}/{t_idx:06d}")
        return {
            "frames": frames.clamp(0, 1),
            "frame_targets": frame_targets,
            "targets": frame_targets,
            "image_ids": image_ids,
            "sequence": f"synthetic_{idx}",
            "dataset_url": "synthetic://drishti-core-v2",
            "meta": {"sequence": f"synthetic_{idx}", "frame_indices": list(range(self.num_frames)), "image_ids": image_ids},
        }
