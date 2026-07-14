import torch

from drishti_v2.models.moe import SparseMoE


def test_sparse_moe_shape_and_balance_loss():
    moe = SparseMoE()
    out, diagnostics = moe(torch.rand(2, 8, 256))
    assert out.shape == (2, 8, 256)
    assert diagnostics.balance_loss.ndim == 0
    assert diagnostics.router_z_loss.ndim == 0
