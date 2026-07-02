import torch

from drishti_v2.models.config import DRISHTIConfig
from drishti_v2.models.pipeline import DRISHTIPipeline


def test_pipeline_end_to_end_shapes():
    cfg = DRISHTIConfig(image_height=64, image_width=64, crop_size=16, temporal_window=3)
    model = DRISHTIPipeline(cfg)
    out = model(torch.rand(2, 3, 3, 64, 64))
    assert out.heatmap.shape == (2, 1, 16, 16)
    assert out.objectness_logits.shape == (2, cfg.num_crops, 1)
    assert out.boxes.shape == (2, cfg.num_crops, 4)
