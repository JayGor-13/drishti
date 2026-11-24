import torch

from drishti_v2.models.detection_head import DetectionHead


def test_detection_head_shapes():
    head = DetectionHead()
    logits, boxes, offsets = head(torch.rand(2, 8, 256))
    assert logits.shape == (2, 8, 1)
    assert boxes.shape == (2, 8, 4)
    assert float(boxes.detach().min()) >= 0.0
    assert offsets.shape == (2, 8, 2)
    assert float(offsets.detach().abs().max()) <= 0.5
