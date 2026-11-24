from __future__ import annotations

import random

import torch
from torch import nn

from drishti_v2.data.augmentations import VideoAugmentation
from drishti_v2.evaluation.metrics import compute_ap, detection_metrics
from drishti_v2.evaluation.visualize import draw_boxes_on_image, save_detection_figure
from drishti_v2.models.config import DRISHTIConfig
from drishti_v2.models.crop_encoder import CropEncoder
from drishti_v2.models.moe import SparseMoE
from drishti_v2.models.motion_cnn import MotionCNN
from drishti_v2.models.pipeline import DRISHTIPipeline
from drishti_v2.models.temporal_fusion import CausalTemporalFusion
from drishti_v2.tracker import SimpleTracker
from drishti_v2.training.ciou_loss import ciou_loss
from drishti_v2.training.motion_loss import motion_displacement_loss
from drishti_v2.training.stage_control import apply_training_stage
from drishti_v2.training.stage_losses import Stage4Loss, assign_crops


def test_default_config_matches_infrared_ldmi_and_multiscale_contract():
    config = DRISHTIConfig.from_yaml("configs/default.yaml")
    assert config.image_channels == 1
    assert config.modality == "infrared"
    assert config.motion_input_channels == 10
    assert config.encoder_in_channels == 3


def test_adaptive_heatmap_sigma_scales_with_box_size():
    tiny = torch.tensor([[0.5, 0.5, 0.01, 0.01]])
    large = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    tiny_map = MotionCNN.make_gt_heatmap(tiny, (32, 32))
    large_map = MotionCNN.make_gt_heatmap(large, (32, 32))
    assert int((large_map > 0.5).sum()) > int((tiny_map > 0.5).sum())


def test_assignment_is_unique_prediction_independent_and_unclamped():
    centers = torch.tensor([[[0.50, 0.50], [0.51, 0.50], [0.90, 0.90]]])
    logits = torch.zeros(1, 3, 1)
    targets = [{"boxes": torch.tensor([[0.50, 0.50, 0.20, 0.20], [0.52, 0.50, 0.20, 0.20]])}]
    first = assign_crops(centers, logits, torch.rand(1, 3, 4), targets, (0.1, 0.1), (16, 16))
    second = assign_crops(centers, logits, torch.rand(1, 3, 4), targets, (0.1, 0.1), (16, 16))
    labels, box_targets, offset_targets = first
    assert int(labels.sum()) == 2
    assert torch.equal(labels, second[0])
    assert torch.allclose(box_targets, second[1])
    assert float(box_targets[..., 2:].max()) > 1.0
    assert float(offset_targets.abs().max()) <= 0.5


def test_ciou_preserves_boxes_larger_than_relative_crop():
    box = torch.tensor([[0.5, 0.5, 2.0, 1.5]])
    assert float(ciou_loss(box, box)) < 1e-6


def test_motion_loss_uses_batch_time_order_and_masks_missing_targets():
    first = torch.full((2, 1, 5, 5), -10.0)
    second = torch.full((2, 1, 5, 5), -10.0)
    first[0, 0, 1, 1] = 10.0
    second[0, 0, 1, 2] = 10.0
    first[1, 0, 3, 3] = 10.0
    second[1, 0, 2, 3] = 10.0
    targets = [
        [
            {"boxes": torch.tensor([[0.25, 0.25, 0.1, 0.1]])},
            {"boxes": torch.empty(0, 4)},
        ],
        [
            {"boxes": torch.tensor([[0.75, 0.75, 0.1, 0.1]])},
            {"boxes": torch.tensor([[0.75, 0.50, 0.1, 0.1]])},
        ],
    ]
    loss = motion_displacement_loss([first, second], targets, temperature=0.1)
    assert float(loss) < 1e-6


def test_true_average_precision_uses_ranked_pr_curve():
    predictions = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "scores": torch.tensor([0.9])}]
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
    assert abs(compute_ap(predictions, targets, 0.5) - 1.0) < 1e-6
    assert detection_metrics(predictions, targets)["map75"] == 1.0


def test_sequence_augmentation_updates_boxes_order_and_metadata(monkeypatch):
    draws = iter([0.0, 1.0, 0.0])  # flip, no photometric jitter, reverse
    monkeypatch.setattr(random, "random", lambda: next(draws))
    frames = [torch.full((1, 2, 3), float(index)) for index in range(2)]
    targets = [
        {"boxes": torch.tensor([[0.2, 0.5, 0.1, 0.1]]), "labels": torch.ones(1, dtype=torch.long)},
        {"boxes": torch.tensor([[0.3, 0.5, 0.1, 0.1]]), "labels": torch.ones(1, dtype=torch.long)},
    ]
    original = targets[0]["boxes"].clone()
    output_frames, output_targets, metadata = VideoAugmentation().apply(
        frames,
        targets,
        {"image_ids": ["a", "b"], "frame_indices": [0, 1]},
    )
    assert float(output_frames[0][0, 0, 0]) == 1.0
    assert torch.allclose(output_targets[0]["boxes"][:, 0], torch.tensor([0.7]))
    assert metadata == {"image_ids": ["b", "a"], "frame_indices": [1, 0]}
    assert torch.equal(targets[0]["boxes"], original)


def test_temporal_padding_matches_equivalent_short_sequence_and_has_center_gradients():
    fusion = CausalTemporalFusion(
        feature_dim=3,
        out_dim=4,
        nhead=2,
        num_layers=1,
        ffn_dim=8,
        dropout=0.0,
        max_seq_len=5,
    ).eval()
    short = torch.randn(1, 2, 2, 3)
    short_centers = torch.rand(1, 2, 2, 2, requires_grad=True)
    sources = torch.tensor([[[0, 1], [0, 1]]])
    short_output = fusion(short, centers_seq=short_centers, source_labels_seq=sources)

    full = torch.cat([short, torch.randn(1, 3, 2, 3)], dim=1)
    full_centers = torch.cat([short_centers.detach(), torch.rand(1, 3, 2, 2)], dim=1)
    full_sources = torch.cat([sources, torch.full((1, 3, 2), 4, dtype=torch.long)], dim=1)
    mask = torch.tensor([[False, False, True, True, True]])
    full_output = fusion(full, centers_seq=full_centers, source_labels_seq=full_sources, padding_mask=mask)
    assert torch.allclose(short_output, full_output, atol=1e-6)
    short_output.sum().backward()
    assert short_centers.grad is not None


def test_sparse_moe_backward_is_safe_and_top1_router_gets_task_gradient():
    moe = SparseMoE(d_model=8, num_experts=3, top_k=1, ffn_dim=16, dropout=0.0)
    inputs = torch.randn(2, 4, 8, requires_grad=True)
    outputs, diagnostics = moe(inputs, source_labels=torch.zeros(2, 4, dtype=torch.long))
    (outputs.square().sum() + 0.01 * diagnostics.balance_loss).backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert moe.router.weight.grad is not None
    assert float(moe.router.weight.grad.abs().sum()) > 0.0


def test_box_conversion_uses_align_corners_span_and_subpixel_offset():
    config = DRISHTIConfig(
        image_height=10,
        image_width=10,
        crop_size=4,
        crop_scales=(1.0,),
        encoder_feature_dim=8,
        temporal_heads=2,
        num_experts=2,
        top_k=1,
    )
    model = DRISHTIPipeline(config)
    crop_boxes = torch.tensor([[[0.5, 0.5, 1.0, 1.0]]])
    centers = torch.tensor([[[0.5, 0.5]]])
    offsets = torch.tensor([[[0.5, 0.0]]])
    boxes = model._boxes_to_global(crop_boxes, centers, (10, 10), offsets, (5, 5))
    assert torch.allclose(boxes[..., 2], torch.tensor([[3.0 / 9.0]]))
    assert torch.allclose(boxes[..., 0], torch.tensor([[0.625]]))


def test_stream_triplets_pad_with_oldest_causal_frame():
    config = DRISHTIConfig(
        image_height=16,
        image_width=16,
        crop_size=8,
        crop_scales=(1.0,),
        num_crops=2,
        temporal_window=3,
        ldmi_scales=(3,),
        motion_cnn_channels=(4, 4, 4),
        encoder_feature_dim=8,
        temporal_heads=2,
        temporal_layers=1,
        temporal_ffn_dim=16,
        num_experts=2,
        top_k=1,
        expert_ffn_dim=16,
        use_motion_gate=False,
    )
    model = DRISHTIPipeline(config)
    captured = []
    handle = model.ldmi.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
    frame0 = torch.zeros(1, 1, 16, 16)
    frame1 = torch.ones(1, 1, 16, 16)
    model.forward_stream(frame0, 0)
    model.forward_stream(frame1, 1)
    handle.remove()
    assert torch.equal(captured[0], torch.cat([frame0, frame0, frame0], dim=1))
    assert torch.equal(captured[1], torch.cat([frame0, frame0, frame1], dim=1))


def test_crop_encoder_freeze_survives_parent_train_and_stage_freeze():
    encoder = CropEncoder(out_dim=8, in_channels=1)
    encoder.freeze()
    nn.Sequential(encoder).train()
    assert encoder.training is False

    config = DRISHTIConfig(
        image_height=16,
        image_width=16,
        crop_size=8,
        crop_scales=(1.0,),
        encoder_feature_dim=8,
        temporal_heads=2,
        temporal_layers=1,
        temporal_ffn_dim=16,
        num_experts=2,
        top_k=1,
        expert_ffn_dim=16,
    )
    model = DRISHTIPipeline(config)
    model.train()
    apply_training_stage(model, "stage2")
    assert model.encoder.training is False
    assert model.motion_cnn.training is False
    assert model.temporal.training is True


def test_tracker_uses_global_assignment_and_preserves_constant_velocity():
    tracker = SimpleTracker(dist_threshold=0.2, birth_threshold=0.3)
    tracker.update(
        torch.tensor([[0.0, 0.0, 0.1, 0.1], [0.2, 0.0, 0.1, 0.1]]),
        torch.tensor([[5.0], [5.0]]),
    )
    tracker.update(
        torch.tensor([[0.09, 0.0, 0.1, 0.1], [0.0, 0.12, 0.1, 0.1]]),
        torch.tensor([[5.0], [5.0]]),
    )
    assert len(tracker.tracks) == 2
    assert torch.allclose(tracker.tracks[0].center, torch.tensor([0.0, 0.12]))
    assert torch.allclose(tracker.tracks[1].center, torch.tensor([0.09, 0.0]))

    velocity_tracker = SimpleTracker(dist_threshold=0.2, birth_threshold=0.3)
    velocity_tracker.update(torch.tensor([[0.2, 0.5, 0.1, 0.1]]), torch.tensor([[5.0]]))
    velocity_tracker.update(torch.tensor([[0.3, 0.5, 0.1, 0.1]]), torch.tensor([[5.0]]))
    velocity_tracker.predict()
    velocity_tracker.update(torch.tensor([[0.4, 0.5, 0.1, 0.1]]), torch.tensor([[5.0]]))
    assert torch.allclose(velocity_tracker.tracks[0].velocity, torch.tensor([0.1, 0.0]), atol=1e-6)


def test_grayscale_visualization_accepts_single_channel(tmp_path):
    frame = torch.rand(1, 8, 8)
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    scores = torch.tensor([0.9])
    path = tmp_path / "gray.png"
    save_detection_figure(frame, boxes, scores, path)
    rendered = draw_boxes_on_image(frame, boxes, scores)
    assert path.exists()
    assert rendered.shape == (8, 8, 3)


class _FixedLoss(nn.Module):
    def __init__(self, values: dict[str, float], **weights: float) -> None:
        super().__init__()
        self.values = values
        for key, value in weights.items():
            setattr(self, key, value)

    def forward(self, *args, **kwargs):
        del args, kwargs
        return {key: torch.tensor(value) for key, value in self.values.items()}


def test_stage4_total_includes_weighted_balance_and_router_z_loss():
    loss = Stage4Loss()
    loss.stage1 = _FixedLoss(
        {"loss": 1.0, "heatmap": 0.0, "cls": 0.0, "bbox": 0.0, "offset": 0.0, "motion_disp": 0.0}
    )
    loss.stage2 = _FixedLoss({"temporal_consist": 2.0, "traj_smooth": 3.0}, w_tc=0.3, w_sm=0.1)
    loss.stage3 = _FixedLoss({"balance": 4.0, "z_loss": 5.0}, w_bal=0.2, w_zloss=0.05)
    result = loss(None, [])
    assert torch.allclose(result["loss"], torch.tensor(2.95))
