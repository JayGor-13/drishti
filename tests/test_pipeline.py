import torch

from drishti_v2.models.config import DRISHTIConfig
from drishti_v2.models.pipeline import DRISHTIPipeline


def test_pipeline_end_to_end_shapes():
    cfg = DRISHTIConfig(image_height=64, image_width=64, crop_size=16, temporal_window=3, use_motion_gate=False)
    model = DRISHTIPipeline(cfg)
    out = model(torch.rand(2, 3, 3, 64, 64))
    assert out.heatmap.shape == (2, 1, 16, 16)
    assert out.objectness_logits.shape == (2, cfg.num_crops, 1)
    assert out.boxes.shape == (2, cfg.num_crops, 4)
    assert out.motion_gate_confidence.shape == (2,)
    assert len(out.all_heatmaps) == 3


def test_pipeline_dense_gate_fallback_shapes():
    cfg = DRISHTIConfig(
        image_height=64,
        image_width=64,
        crop_size=16,
        temporal_window=3,
        use_motion_gate=True,
        motion_gate_threshold=1.0,
    )
    model = DRISHTIPipeline(cfg)
    out = model(torch.rand(1, 3, 3, 64, 64))
    assert out.used_dense_mode is True
    assert out.objectness_logits.shape == (1, cfg.dense_num_crops, 1)
