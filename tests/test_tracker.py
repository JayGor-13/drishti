import torch

from drishti_v2.tracker import SimpleTracker


def test_tracker_birth_and_guidance():
    tracker = SimpleTracker(birth_threshold=0.3)
    tracker.update(torch.tensor([[0.5, 0.5, 0.1, 0.1]]), torch.tensor([[2.0]]))
    centers = tracker.get_guided_centers()
    assert centers is not None
    assert centers.shape == (1, 1, 2)
