import torch

from drishti_v2.models.ldmi import LocalDifferentialMotion


def test_ldmi_shape_and_uniform_motion_suppression():
    ldmi = LocalDifferentialMotion(scales=(3,))
    base = torch.rand(2, 3, 16, 16)
    triplet = torch.cat([base, base + 0.1, base + 0.2], dim=1)
    out = ldmi(triplet)
    assert out.shape == (2, 15, 16, 16)
    assert out[:, :3].mean() < 0.05
