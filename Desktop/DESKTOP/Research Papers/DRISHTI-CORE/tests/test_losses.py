import torch

from drishti_v2.models.config import DRISHTIConfig
from drishti_v2.models.pipeline import DRISHTIPipeline
from drishti_v2.training.losses import DRISHTILoss


def test_loss_forward_on_synthetic_output():
    cfg = DRISHTIConfig(image_height=64, image_width=64, crop_size=16, temporal_window=3)
    model = DRISHTIPipeline(cfg)
    output = model(torch.rand(1, 3, 3, 64, 64))
    targets = [[{"boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]), "labels": torch.ones(1), "visible": True} for _ in range(3)]]
    losses = DRISHTILoss()(output, targets)
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])
