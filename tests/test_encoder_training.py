from pathlib import Path

import pytest
import torch

from drishti_v2.models.config import DRISHTIConfig
from drishti_v2.models.crop_encoder import CropEncoder, ResNet18CropEncoder, build_crop_encoder
from drishti_v2.models.pipeline import DRISHTIPipeline
from drishti_v2.training import StageLossFactory, apply_training_stage

RESNET18_WEIGHTS = Path(__file__).resolve().parents[2] / "runs" / "pretrained" / "resnet18-f37072fd.pth"


def _small(**overrides):
    cfg = dict(
        image_channels=1, image_height=64, image_width=64, crop_size=16, crop_scales=(1.0, 2.0, 4.0),
        num_crops=2, temporal_window=3, ldmi_scales=(3,), motion_cnn_channels=(4, 4, 4), use_motion_gate=False,
        encoder_feature_dim=8, temporal_heads=2, temporal_layers=1, temporal_ffn_dim=16,
        num_experts=2, top_k=1, expert_ffn_dim=16,
    )
    cfg.update(overrides)
    return DRISHTIConfig(**cfg)


def _stage1_step(model, cfg):
    apply_training_stage(model, "stage1")
    out = model(torch.rand(2, 3, 1, 64, 64))
    targets = [[{"boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]), "labels": torch.ones(1)} for _ in range(3)] for _ in range(2)]
    StageLossFactory.make_loss("stage1", config=cfg)(out, targets, all_heatmaps=out.all_heatmaps)["loss"].backward()


def test_default_config_keeps_encoder_frozen_in_stage1():
    cfg = _small()
    model = DRISHTIPipeline(cfg)
    _stage1_step(model, cfg)
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.encoder.training is False


@pytest.mark.parametrize("backbone", ["simple", "resnet18"])
def test_encoder_trains_in_stage1_and_freezes_again_in_stage2(backbone):
    cfg = _small(train_encoder_in_stage1=True, encoder_backbone=backbone)
    model = DRISHTIPipeline(cfg)
    _stage1_step(model, cfg)
    assert model.encoder.training is True
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    model.train()
    apply_training_stage(model, "stage2")
    assert model.encoder.training is False
    assert not any(p.requires_grad for p in model.encoder.parameters())


def test_build_crop_encoder_picks_backbone():
    assert isinstance(build_crop_encoder(_small()), CropEncoder)
    assert isinstance(build_crop_encoder(_small(encoder_backbone="resnet18")), ResNet18CropEncoder)
    with pytest.raises(ValueError):
        build_crop_encoder(_small(encoder_backbone="vgg"))


@pytest.mark.skipif(not RESNET18_WEIGHTS.exists(), reason="ImageNet resnet18 weights not downloaded")
def test_resnet18_loads_imagenet_weights_exactly():
    enc = ResNet18CropEncoder(out_dim=256, in_channels=3, pretrained_path=str(RESNET18_WEIGHTS))
    ref = torch.load(RESNET18_WEIGHTS, map_location="cpu")
    assert enc.pretrained
    assert torch.equal(enc.conv1.weight, ref["conv1.weight"])
    assert torch.equal(enc.layer2[1].conv2.weight, ref["layer2.1.conv2.weight"])
    assert enc(torch.rand(4, 3, 64, 64)).shape == (4, 256)
    grey = ResNet18CropEncoder(out_dim=8, in_channels=1, pretrained_path=str(RESNET18_WEIGHTS))
    assert torch.allclose(grey.conv1.weight, ref["conv1.weight"].mean(dim=1, keepdim=True))
