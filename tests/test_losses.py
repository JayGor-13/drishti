import torch
import pytest

from drishti_v2.models.config import DRISHTIConfig
from drishti_v2.models.pipeline import DRISHTIPipeline
from drishti_v2.training.losses import DRISHTILoss
from drishti_v2.training.stage_losses import StageLossFactory
from drishti_v2.training.stage_control import apply_training_stage


def test_loss_forward_on_synthetic_output():
    cfg = DRISHTIConfig(image_height=64, image_width=64, crop_size=16, temporal_window=3, use_motion_gate=False)
    model = DRISHTIPipeline(cfg)
    output = model(torch.rand(1, 3, 1, 64, 64))
    output.objectness_logits_seq[0].retain_grad()
    targets = [[{"boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]), "labels": torch.ones(1), "visible": True} for _ in range(3)]]
    losses = DRISHTILoss()(output, targets)
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert output.objectness_logits_seq[0].grad is not None


def test_stage_loss_factory_uses_config():
    cfg = DRISHTIConfig(image_height=64, image_width=64, crop_size=16, temporal_window=3, use_motion_gate=False)
    model = DRISHTIPipeline(cfg)
    output = model(torch.rand(1, 3, 1, 64, 64))
    targets = [[{"boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]), "labels": torch.ones(1)} for _ in range(3)]]
    losses = StageLossFactory.make_loss("stage1", config=cfg)(output, targets, all_heatmaps=output.all_heatmaps)
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])


@pytest.mark.parametrize("stage", ["stage2", "stage3", "stage4"])
def test_each_later_training_stage_runs_backward(stage):
    cfg = DRISHTIConfig(
        image_height=32,
        image_width=32,
        crop_size=8,
        crop_scales=(1.0,),
        num_crops=2,
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
    )
    model = DRISHTIPipeline(cfg)
    apply_training_stage(model, stage)
    output = model(torch.rand(1, 3, 1, 32, 32))
    targets = [[{"boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]), "labels": torch.ones(1)} for _ in range(3)]]
    losses = StageLossFactory.make_loss(stage, config=cfg)(output, targets)
    losses["loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
