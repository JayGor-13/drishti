import torch

from drishti_v2.models.moe import SparseMoE


def test_sparse_moe_shape_and_balance_loss():
    moe = SparseMoE()
    out, diag = moe(torch.rand(2, 8, 256))
    assert out.shape == (2, 8, 256)
    assert diag.balance_loss.ndim == 0

