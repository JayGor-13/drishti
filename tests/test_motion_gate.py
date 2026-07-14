import torch

from drishti_v2.models.motion_gate import MotionGate


def test_motion_gate_outputs_batch_confidence():
    gate = MotionGate()
    confidence = gate(torch.rand(3, 1, 16, 16))
    assert confidence.shape == (3,)
    assert float(confidence.min()) >= 0.0
    assert float(confidence.max()) <= 1.0
