import torch

from drishti_v2.models.config import DRISHTIConfig
from drishti_v2.models.crop_proposal import CropProposalEngine


def test_crop_proposal_budget_and_shapes():
    cfg = DRISHTIConfig(image_height=64, image_width=64, crop_size=16, use_motion_gate=False)
    engine = CropProposalEngine(cfg)
    frame = torch.rand(2, 3, 64, 64)
    heatmap = torch.rand(2, 1, 16, 16)
    out = engine(frame, heatmap, frame_index=4)
    assert out.centers.shape == (2, cfg.num_crops, 2)
    assert out.crops.shape == (2 * cfg.num_crops, 3, 16, 16)
    assert out.scores.shape == (2, cfg.num_crops)


def test_dense_crop_proposal_uses_grid_budget():
    cfg = DRISHTIConfig(image_height=64, image_width=64, crop_size=16, dense_grid_size=4)
    engine = CropProposalEngine(cfg)
    frame = torch.rand(2, 3, 64, 64)
    heatmap = torch.rand(2, 1, 16, 16)
    out = engine(frame, heatmap, frame_index=0, dense=True)
    assert out.centers.shape == (2, cfg.dense_num_crops, 2)
    assert out.crops.shape == (2 * cfg.dense_num_crops, 3, 16, 16)
