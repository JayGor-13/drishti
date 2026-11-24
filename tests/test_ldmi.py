import torch

from drishti_v2.models.ldmi import LocalDifferentialMotion


def test_ldmi_shape_and_uniform_motion_suppression():
    ldmi = LocalDifferentialMotion(image_channels=1, scales=(3,), use_sobel_edge=True)
    base = torch.rand(2, 1, 16, 16)
    triplet = torch.cat([base, base + 0.1, base + 0.2], dim=1).requires_grad_()
    out = ldmi(triplet)
    assert out.shape == (2, 10, 16, 16)
    assert out[:, :1].mean() < 0.05
    out[:, -1].mean().backward()
    assert triplet.grad is not None
