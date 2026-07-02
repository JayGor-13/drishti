import torch

from drishti_v2.models.temporal_fusion import CausalTemporalFusion


def test_temporal_fusion_shape_with_score_feature():
    model = CausalTemporalFusion(feature_dim=257, out_dim=256, nhead=4)
    out = model(torch.rand(2, 5, 8, 257))
    assert out.shape == (2, 8, 256)
