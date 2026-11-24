import json

import numpy as np
import torch
from PIL import Image

from drishti_v2.data import AntiUAVExtractedFrameDataset, DRISHTICollator, SyntheticAntiUAVDataset


def test_extracted_dataset_contract_and_short_sequence_clamp(tmp_path):
    seq = tmp_path / "train" / "seq1"
    frame_dir = seq / "visible"
    frame_dir.mkdir(parents=True)
    for idx in range(3):
        image = Image.fromarray(np.full((20, 30, 3), idx * 40, dtype=np.uint8), mode="RGB")
        image.save(frame_dir / f"{idx:06d}.jpg")
    (seq / "visible.json").write_text(
        json.dumps({"gt_rect": [[1, 2, 4, 5], [2, 3, 4, 5], []], "exist": [1, 1, 0]}),
        encoding="utf-8",
    )

    dataset = AntiUAVExtractedFrameDataset(tmp_path, split="train", num_frames=5, height=16, width=16)
    item = dataset[0]
    assert item["frames"].shape == (5, 3, 16, 16)
    assert len(item["frame_targets"]) == 5
    assert item["meta"]["frame_indices"] == [0, 1, 2, 2, 2]
    assert item["frame_targets"][0]["boxes"].shape == (1, 4)
    assert item["frame_targets"][2]["boxes"].shape == (0, 4)
    assert item["targets"] is item["frame_targets"]
    assert len(item["image_ids"]) == 5


def test_synthetic_dataset_is_deterministic_and_collates():
    dataset = SyntheticAntiUAVDataset(num_samples=2, num_frames=5, height=32, width=32, image_channels=1)
    first = dataset[0]
    again = dataset[0]
    assert torch.equal(first["frames"], again["frames"])
    assert first["frames"].shape == (5, 1, 32, 32)
    batch = DRISHTICollator()([first, dataset[1]])
    assert batch["frames"].shape == (2, 5, 1, 32, 32)
    assert "frame_targets" in batch
    assert batch["targets"] is batch["frame_targets"]
