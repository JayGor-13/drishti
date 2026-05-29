"""Real video-language dataset and collation for Video VQA training."""

from __future__ import annotations

import json
import os
import torch
from torch.utils.data import Dataset
import numpy as np

# Avoid crash if decord is not installed in the local environment during dry-runs
try:
    import decord
    decord.bridge.set_bridge("torch")
except ImportError:
    decord = None


class VideoVQADataset(Dataset):
    """Dataset loading real MP4 videos and tokenizing question-answer annotations."""

    def __init__(
        self,
        json_path: str,
        video_dir: str,
        tokenizer: any,
        num_frames: int = 8,
        height: int = 224,
        width: int = 224,
    ) -> None:
        super().__init__()
        self.video_dir = video_dir
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.height = height
        self.width = width

        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def _load_video(self, video_name: str) -> torch.Tensor:
        """Decode and sample T frames from video file using decord."""
        video_path = os.path.join(self.video_dir, video_name)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if decord is None:
            # Fallback for dry runs / cpu tests
            return torch.zeros(self.num_frames, 3, self.height, self.width)

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        
        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices)  # [T, H, W, C]
        
        # Transform [T, H, W, C] to [T, C, H, W]
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        
        # Standard torchvision-style resize & normalization (CLIP-like)
        frames = torch.nn.functional.interpolate(
            frames, size=(self.height, self.width), mode="bilinear", align_corners=False
        )
        
        # Normalization values (CLIP default: mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])
        mean = torch.tensor([0.4814, 0.4578, 0.4082]).view(1, 3, 1, 1)
        std = torch.tensor([0.2686, 0.2613, 0.2757]).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        return frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list[int]]:
        item = self.data[idx]
        # Expected keys in Video-ChatGPT JSON format: 'video', 'QA' or 'question' / 'answer'
        # Adjust mapping if keys are slightly different
        video_name = item.get("video") or item.get("video_id")
        if not video_name.endswith(".mp4"):
            video_name += ".mp4"
            
        question = item.get("question") or item["QA"][0]["q"]
        answer_text = item.get("answer") or item["QA"][0]["a"]

        # Load video frames
        try:
            frames = self._load_video(video_name)
        except Exception as e:
            # Return dummy frames on error to avoid breaking training mid-epoch
            print(f"Error loading video {video_name}: {e}. Returning dummy zeros.")
            frames = torch.zeros(self.num_frames, 3, self.height, self.width)

        # Tokenize conversation using ChatML format:
        # User: <video>\n{Question}
        # Assistant: {Answer}
        prompt = f"<|im_start|>user\n<video>\n{question}<|im_end|>\n<|im_start|>assistant\n"
        answer = f"{answer_text}<|im_end|>"

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids  # Mask prompt tokens

        return {
            "frames": frames,
            "input_ids": input_ids,
            "labels": labels,
        }


class VideoQACollator:
    """Pad sequence lengths to the maximum sequence length in the batch."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        frames = torch.stack([item["frames"] for item in batch])
        
        # Pad input_ids and labels to max_len
        input_ids_list = [item["input_ids"] for item in batch]
        labels_list = [item["labels"] for item in batch]
        
        max_len = max(len(ids) for ids in input_ids_list)
        
        padded_ids = []
        padded_labels = []
        
        for ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)
            
        return {
            "frames": frames,
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
