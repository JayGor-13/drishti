import torch

from drishti_v2.models.motion_cnn import MotionCNN
from drishti_v2.training.focal_loss import heatmap_focal_loss
from drishti_v2.training.stage_losses import make_gt_heatmaps


def test_gt_heatmap_peak_is_exactly_one_for_fractional_centres():
    # Centres that do not fall on an integer heatmap pixel used to give peaks < 1.0,
    # so the focal loss saw zero positives and the heatmap collapsed to zero.
    targets = [{"boxes": torch.tensor([[0.3137, 0.6182, 0.02, 0.02]])}]
    heatmap = make_gt_heatmaps(targets, (112, 112), torch.device("cpu"))
    assert float(heatmap.max()) == 1.0
    assert int((heatmap >= 1.0).sum()) == 1

    single = MotionCNN.make_gt_heatmap(targets[0]["boxes"], (112, 112))
    assert float(single.max()) == 1.0


def test_focal_loss_pushes_prediction_up_at_the_peak():
    targets = [{"boxes": torch.tensor([[0.3137, 0.6182, 0.02, 0.02]])}]
    gt = make_gt_heatmaps(targets, (32, 32), torch.device("cpu"))
    pred = torch.full_like(gt, 0.01, requires_grad=True)
    loss = heatmap_focal_loss(pred, gt)
    loss.backward()
    peak = (gt >= 1.0).nonzero()[0]
    grad_at_peak = pred.grad[peak[0], peak[1], peak[2], peak[3]]
    # Gradient descent must increase the prediction at the positive pixel.
    assert float(grad_at_peak) < 0.0
    # And the loss is far from the near-zero value a positive-free loss collapses to.
    assert float(loss) > 1.0
