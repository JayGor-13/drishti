import torch

from drishti_v2.models.motion_cnn import MotionCNN


def test_motion_cnn_output_shape():
    model = MotionCNN()
    heatmap = model(torch.rand(2, 15, 64, 64))
    assert heatmap.shape == (2, 1, 16, 16)
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0
