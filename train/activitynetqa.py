"""ActivityNetQA data loading for the unified T-MoE experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, random_split

try:
    from datasets import Dataset as HFDataset
    from datasets import load_dataset
except ImportError:  # pragma: no cover - exercised only when optional deps missing
    HFDataset = Any
    load_dataset = None


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_'-]+|[^\sA-Za-z0-9_]")
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True)
class ActivityNetQARecord:
    """Normalized ActivityNetQA row."""

    video_name: str
    question_id: str
    question: str
    answer: str
    question_type: str


def _video_lookup_keys(value: str) -> set[str]:
    """Return filename/stem variants that commonly identify one ActivityNet clip."""

    if not value:
        return set()
    name = Path(str(value)).name.strip()
    stem = Path(name).stem.strip()
    keys = {name.lower(), stem.lower()}
    if stem.startswith("v_"):
        keys.add(stem[2:].lower())
    else:
        keys.add(f"v_{stem}".lower())

    question_match = re.match(r"^v_(.+)_\d+$", stem)
    if question_match:
        video_stem = question_match.group(1)
        keys.update({video_stem.lower(), f"v_{video_stem}".lower()})
    return {key for key in keys if key}


class VideoFileIndex:
    """Fast lookup for videos extracted from one or more ActivityNetQA shards."""

    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root) if root else None
        self.paths_by_key: dict[str, Path] = {}
        if self.root is not None and self.root.exists():
            self.paths_by_key = self._build_index(self.root)

    def __len__(self) -> int:
        return len({path for path in self.paths_by_key.values()})

    @staticmethod
    def _build_index(root: Path) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            for key in _video_lookup_keys(path.name):
                index.setdefault(key, path)
        return index

    def find(self, record: ActivityNetQARecord) -> Path | None:
        candidates = [
            record.video_name,
            f"{record.video_name}.mp4" if record.video_name else "",
            f"v_{record.video_name}.mp4" if record.video_name else "",
            record.question_id,
            f"{record.question_id}.mp4" if record.question_id else "",
        ]
        for candidate in candidates:
            for key in _video_lookup_keys(candidate):
                path = self.paths_by_key.get(key)
                if path is not None:
                    return path
        return None


class SimpleQATokenizer:
    """Small word-level tokenizer fitted on ActivityNetQA text."""

    pad_token = "<pad>"
    unk_token = "<unk>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    video_token = "<video>"
    answer_token = "<answer>"

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.id_to_token = {idx: token for token, idx in vocab.items()}
        self.pad_token_id = vocab[self.pad_token]
        self.unk_token_id = vocab[self.unk_token]
        self.bos_token_id = vocab[self.bos_token]
        self.eos_token_id = vocab[self.eos_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def fit(
        cls,
        records: list[ActivityNetQARecord],
        max_vocab_size: int = 4096,
    ) -> "SimpleQATokenizer":
        special = [
            cls.pad_token,
            cls.unk_token,
            cls.bos_token,
            cls.eos_token,
            cls.video_token,
            cls.answer_token,
        ]
        counter: Counter[str] = Counter()
        for record in records:
            counter.update(cls.tokenize_text(record.question))
            counter.update(cls.tokenize_text(record.answer))

        vocab = {token: idx for idx, token in enumerate(special)}
        for token, _ in counter.most_common(max(0, max_vocab_size - len(vocab))):
            if token not in vocab:
                vocab[token] = len(vocab)
        return cls(vocab)

    @staticmethod
    def tokenize_text(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(str(text))]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self.vocab.get(token, self.unk_token_id) for token in self.tokenize_text(text)]
        if add_special_tokens:
            return [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        tokens = [
            self.id_to_token.get(int(idx), self.unk_token)
            for idx in ids
            if int(idx) not in {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        ]
        return " ".join(tokens)


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def load_activitynetqa_records(
    dataset_name: str = "lmms-lab/ActivityNetQA",
    split: str = "test",
    metadata_file: str | None = None,
    hf_token_env: str = "HF_TOKEN",
    limit_fraction: float | None = None,
    seed: int = 42,
) -> list[ActivityNetQARecord]:
    """Load and normalize ActivityNetQA rows from a local file or Hugging Face."""

    if load_dataset is None:
        raise ImportError(
            "Install optional experiment dependencies first: datasets, "
            "huggingface_hub, python-dotenv."
        )

    source_label = f"{dataset_name}:{split}"
    if metadata_file is not None:
        metadata_path = Path(metadata_file)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")
        loaders_by_suffix = {
            ".parquet": "parquet",
            ".csv": "csv",
            ".json": "json",
            ".jsonl": "json",
        }
        loader_name = loaders_by_suffix.get(metadata_path.suffix.lower())
        if loader_name is None:
            raise ValueError(
                "metadata_file must be .parquet, .csv, .json, or .jsonl; "
                f"got {metadata_path.suffix}"
            )
        dataset = load_dataset(loader_name, data_files=str(metadata_path), split="train")
        source_label = str(metadata_path)
    else:
        _load_dotenv_if_available()
        token = os.getenv(hf_token_env) or os.getenv("HUGGINGFACE_HUB_TOKEN")
        dataset = load_dataset(dataset_name, split=split, token=token)

    if limit_fraction is not None and 0.0 < limit_fraction < 1.0:
        count = max(1, math.ceil(len(dataset) * limit_fraction))
        dataset = dataset.shuffle(seed=seed).select(range(count))

    records: list[ActivityNetQARecord] = []
    for row in dataset:
        video_name = str(row.get("video_name") or row.get("video") or "")
        question_id = str(row.get("question_id") or row.get("id") or video_name)
        question = str(row.get("question") or "")
        answer = str(row.get("answer") or "")
        question_type = str(row.get("type") or "unknown")
        if question and answer:
            records.append(
                ActivityNetQARecord(
                    video_name=video_name,
                    question_id=question_id,
                    question=question,
                    answer=answer,
                    question_type=question_type,
                )
            )
    if not records:
        raise RuntimeError(f"No usable rows found in {source_label}")
    return records


class ActivityNetQADataset(Dataset):
    """ActivityNetQA text rows plus deterministic video tensors.

    The QA metadata can come from a local parquet file or Hugging Face. When
    local clips are available, this dataset decodes sampled video frames. Proxy
    tensors are kept only for smoke tests and debugging without video files.
    """

    def __init__(
        self,
        records: list[ActivityNetQARecord],
        tokenizer: SimpleQATokenizer,
        num_frames: int = 4,
        height: int = 32,
        width: int = 32,
        max_text_length: int = 64,
        video_root: str | None = None,
        allow_proxy_videos: bool = True,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.max_text_length = max_text_length
        self.video_root = Path(video_root) if video_root else None
        self.allow_proxy_videos = allow_proxy_videos
        self.video_index = VideoFileIndex(self.video_root)

    def __len__(self) -> int:
        return len(self.records)

    def _proxy_frames(self, record: ActivityNetQARecord) -> torch.Tensor:
        seed_bytes = f"{record.video_name}:{record.question_id}".encode("utf-8")
        seed = int(hashlib.sha256(seed_bytes).hexdigest()[:16], 16) % (2**31)
        generator = torch.Generator().manual_seed(seed)
        base = torch.rand(1, 3, self.height, self.width, generator=generator)
        frames = base.repeat(self.num_frames, 1, 1, 1)
        if self.num_frames > 1:
            jitter = 0.015 * torch.randn(
                self.num_frames,
                3,
                self.height,
                self.width,
                generator=generator,
            )
            frames = (frames + jitter).clamp(0.0, 1.0)
        return frames

    def _video_path(self, record: ActivityNetQARecord) -> Path | None:
        if self.video_root is None or not record.video_name:
            return None
        indexed = self.video_index.find(record)
        if indexed is not None:
            return indexed
        candidates = [
            self.video_root / record.video_name,
            self.video_root / f"{record.video_name}.mp4",
            self.video_root / f"v_{record.video_name}.mp4",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def has_video(self, record: ActivityNetQARecord) -> bool:
        return self._video_path(record) is not None

    def _resize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.numel() == 0:
            raise ValueError("decoded video contained no frames")
        if frames.shape[0] < self.num_frames:
            repeat_count = self.num_frames - frames.shape[0]
            frames = torch.cat([frames, frames[-1:].repeat(repeat_count, 1, 1, 1)], dim=0)
        indices = torch.linspace(0, frames.shape[0] - 1, self.num_frames).long()
        sampled = frames[indices].permute(0, 3, 1, 2).float() / 255.0
        return torch.nn.functional.interpolate(
            sampled,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )

    def _load_video_frames_torchvision(self, video_path: Path) -> torch.Tensor:
        try:
            from torchvision.io import read_video
        except ImportError:
            raise RuntimeError("torchvision is not installed")

        frames, _, _ = read_video(str(video_path), pts_unit="sec")
        return self._resize_frames(frames)

    def _load_video_frames_cv2(self, video_path: Path) -> torch.Tensor:
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise RuntimeError("opencv-python is not installed")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open {video_path}")
        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise RuntimeError("OpenCV reported zero frames")
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            sampled = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                sampled.append(frame[:, :, ::-1].copy())
            if not sampled:
                raise RuntimeError("OpenCV could not decode sampled frames")
            frames = torch.from_numpy(np.stack(sampled, axis=0))
            return self._resize_frames(frames)
        finally:
            capture.release()

    def _load_video_frames(self, record: ActivityNetQARecord) -> torch.Tensor:
        video_path = self._video_path(record)
        if video_path is None:
            if self.allow_proxy_videos:
                return self._proxy_frames(record)
            raise FileNotFoundError(f"No local video found for {record.video_name}")

        errors = []
        for loader in (self._load_video_frames_torchvision, self._load_video_frames_cv2):
            try:
                return loader(video_path)
            except Exception as exc:
                errors.append(f"{loader.__name__}: {exc}")

        if self.allow_proxy_videos:
            return self._proxy_frames(record)
        joined = "; ".join(errors)
        raise RuntimeError(f"Could not decode {video_path}. {joined}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        prompt_ids = [
            self.tokenizer.bos_token_id,
            self.tokenizer.vocab[self.tokenizer.video_token],
            *self.tokenizer.encode(record.question),
            self.tokenizer.vocab[self.tokenizer.answer_token],
        ]
        answer_ids = self.tokenizer.encode(record.answer) + [self.tokenizer.eos_token_id]
        input_ids = (prompt_ids + answer_ids)[: self.max_text_length]
        labels = [-100] * len(prompt_ids) + answer_ids
        labels = labels[: self.max_text_length]

        return {
            "frames": self._load_video_frames(record),
            "input_ids": input_ids,
            "labels": labels,
            "answer": record.answer,
            "question": record.question,
            "question_type": record.question_type,
            "video_name": record.video_name,
        }


class ActivityNetQACollator:
    """Pad ActivityNetQA batches."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        labels = []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "frames": torch.stack([item["frames"] for item in batch]),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "answers": [item["answer"] for item in batch],
            "questions": [item["question"] for item in batch],
            "question_types": [item["question_type"] for item in batch],
            "video_names": [item["video_name"] for item in batch],
        }


def split_records(
    records: list[ActivityNetQARecord],
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[list[ActivityNetQARecord], list[ActivityNetQARecord]]:
    if len(records) < 2:
        return records, records
    generator = torch.Generator().manual_seed(seed)
    train_len = max(1, int(len(records) * train_fraction))
    test_len = max(1, len(records) - train_len)
    if train_len + test_len > len(records):
        train_len = len(records) - 1
    train_subset, test_subset = random_split(records, [train_len, test_len], generator)
    return list(train_subset), list(test_subset)


def filter_records_with_available_videos(
    records: list[ActivityNetQARecord],
    video_root: str | Path,
) -> list[ActivityNetQARecord]:
    """Keep only QA rows whose video is present in a local video chunk directory."""

    index = VideoFileIndex(video_root)
    return [record for record in records if index.find(record) is not None]
