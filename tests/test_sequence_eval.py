import json

import torch
from PIL import Image

from drishti_v2.data.dataset import AntiUAVExtractedFrameDataset
from drishti_v2.experiments.run_sequence_eval import average_over_sequences, evaluate_sequence
from drishti_v2.models.config import DRISHTIConfig
from drishti_v2.models.pipeline import DRISHTIPipeline

SOURCES = ("motion", "edge", "grid", "guided", "pad")


def _write_sequence(root, split, name, frames=8, size=64):
    seq_dir = root / split / name
    (seq_dir / "infrared").mkdir(parents=True)
    rects = []
    exist = []
    for idx in range(frames):
        image = torch.zeros(size, size)
        x, y = 10 + 2 * idx, 20 + idx
        image[y : y + 4, x : x + 4] = 1.0
        Image.fromarray((image.numpy() * 255).astype("uint8")).save(seq_dir / "infrared" / f"{idx:06d}.png")
        visible = idx % 4 != 3
        rects.append([x, y, 4, 4] if visible else [])
        exist.append(1 if visible else 0)
    (seq_dir / "infrared.json").write_text(json.dumps({"gt_rect": rects, "exist": exist}), encoding="utf-8")


def test_sequence_eval_runs_end_to_end(tmp_path):
    _write_sequence(tmp_path, "val", "seq_a")
    _write_sequence(tmp_path, "val", "seq_b", frames=6)
    cfg = DRISHTIConfig(
        image_channels=1,
        image_height=64,
        image_width=64,
        crop_size=16,
        crop_scales=(1.0,),
        num_crops=4,
        temporal_window=3,
        ldmi_scales=(3,),
        motion_cnn_channels=(4, 4, 4),
        use_motion_gate=False,
        encoder_feature_dim=8,
        temporal_heads=2,
        temporal_layers=1,
        temporal_ffn_dim=16,
        num_experts=2,
        top_k=1,
        expert_ffn_dim=16,
        modality="infrared",
    )
    model = DRISHTIPipeline(cfg).eval()
    dataset = AntiUAVExtractedFrameDataset(
        frames_root=tmp_path,
        split="val",
        modality="infrared",
        num_frames=3,
        height=64,
        width=64,
        clip_stride=1,
        frame_stride=1,
        image_channels=1,
        box_format="xywh",
    )
    rows = []
    for sequence in dataset.sequences:
        for use_tracker in (True, False):
            metrics = evaluate_sequence(model, cfg, sequence, torch.device("cpu"), use_tracker=use_tracker)
            assert metrics["frames"] == sequence.num_frames
            assert metrics["visible_frames"] == sum(sequence.exists)
            assert 0.0 <= metrics["map50"] <= 1.0
            share = sum(metrics[f"top_box_source_{name}"] for name in SOURCES)
            assert share == 0.0 or abs(share - 1.0) < 1e-6
            rows.append({"sequence": sequence.name, **metrics})
    summary = average_over_sequences(rows)
    assert "antiuav_tracking_accuracy" in summary
    assert "gt_inside_some_crop" in summary
