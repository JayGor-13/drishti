import torch

from models import (
    DRISHTIConfig,
    DRISHTIMoE,
    DRISHTIPipeline,
    FrozenCropEncoder,
    MotionCropProposal,
    TemporalFusion,
)
from train import (
    DRISHTICollator,
    SyntheticAntiUAVDataset,
    configure_drishti_training_stage,
    detector_loss,
    detector_stage_loss,
    moe_stage_loss,
    trainable_parameter_names,
)


def tiny_config() -> DRISHTIConfig:
    return DRISHTIConfig(
        crop_size=32,
        num_crops=4,
        feature_dim=32,
        temporal_input_dim=33,
        temporal_heads=4,
        temporal_layers=1,
        temporal_ffn_dim=64,
        moe_num_experts=4,
        moe_top_k=2,
        moe_ffn_dim=64,
    )


def synthetic_batch(num_samples: int = 2, num_frames: int = 5) -> dict:
    dataset = SyntheticAntiUAVDataset(
        num_samples=num_samples,
        num_frames=num_frames,
        height=64,
        width=64,
        image_channels=3,
    )
    return DRISHTICollator()([dataset[index] for index in range(num_samples)])


def test_motion_crop_proposal_contract():
    torch.manual_seed(0)
    proposer = MotionCropProposal(crop_size=32, num_crops=4)
    triplet = torch.randn(2, 9, 64, 64)

    output = proposer(triplet)

    assert output.crops.shape == (8, 3, 32, 32)
    assert output.motion_scores.shape == (8, 1)
    assert output.boxes.shape == (2, 4, 4)
    assert output.centers.shape == (2, 4, 2)
    assert output.heatmap.shape[:2] == (2, 1)
    assert torch.all((output.motion_scores >= 0.0) & (output.motion_scores <= 1.0))


def test_frozen_crop_encoder_has_no_trainable_backbone_by_default():
    encoder = FrozenCropEncoder(feature_dim=32, image_channels=3, frozen=True)
    crops = torch.randn(4, 3, 32, 32)

    features = encoder(crops)

    assert features.shape == (4, 32)
    assert not any(parameter.requires_grad for parameter in encoder.parameters())


def test_temporal_fusion_accepts_non_divisible_d_model():
    torch.manual_seed(1)
    fusion = TemporalFusion(tiny_config())
    features = torch.randn(2, 5, 4, 33)

    output = fusion(features)

    assert output.fused_features.shape == (2, 4, 32)
    assert output.temporal_tokens.shape == (2, 4, 5, 33)
    assert torch.isfinite(output.fused_features).all()


def test_sparse_moe_routes_top_two_experts():
    torch.manual_seed(2)
    moe = DRISHTIMoE(tiny_config())
    features = torch.randn(2, 4, 32)

    output = moe(features)

    assert output.hidden_states.shape == (2, 4, 32)
    assert output.router_probs.shape == (2, 4, 4)
    assert output.topk_indices.shape == (2, 4, 2)
    assert torch.allclose(output.router_probs.sum(dim=-1), torch.ones(2, 4), atol=1e-6)
    assert torch.isfinite(output.load_balance_loss)


def test_pipeline_forward_outputs_antiuav_predictions():
    torch.manual_seed(3)
    model = DRISHTIPipeline(tiny_config())
    frames = torch.randn(2, 5, 3, 64, 64)

    output = model(frames)

    assert output.heatmaps.shape[:3] == (2, 5, 1)
    assert output.proposal_boxes.shape == (2, 4, 4)
    assert output.temporal_features.shape == (2, 4, 32)
    assert output.moe_features.shape == (2, 4, 32)
    assert output.object_logits.shape == (2, 4, 1)
    assert output.boxes.shape == (2, 4, 4)
    assert output.predictions.shape == (2, 4, 6)
    assert output.router_topk.shape == (2, 4, 2)
    assert torch.all((output.predictions[..., :4] >= 0.0) & (output.predictions[..., :4] <= 1.0))
    assert torch.all((output.predictions[..., 4:] >= 0.0) & (output.predictions[..., 4:] <= 1.0))


def test_stage_specific_forward_paths_are_detector_only_and_temporal_only():
    torch.manual_seed(4)
    model = DRISHTIPipeline(tiny_config())
    frames = torch.randn(2, 5, 3, 64, 64)

    detector_output = model.forward_detector(frames)
    temporal_output = model.forward_temporal(frames)

    assert detector_output.crop_features.shape == (2, 4, 32)
    assert detector_output.object_logits.shape == (2, 4, 1)
    assert temporal_output.temporal_features.shape == (2, 4, 32)
    assert temporal_output.load_balance_loss.item() == 0.0


def test_stage_freezing_follows_drishti_core_order():
    model = DRISHTIPipeline(tiny_config())

    configure_drishti_training_stage(model, "detector")
    detector_names = trainable_parameter_names(model)
    assert any(name.startswith("motion_proposer") for name in detector_names)
    assert any(name.startswith("detection_head") for name in detector_names)
    assert not any(name.startswith("crop_encoder") for name in detector_names)
    assert not any(name.startswith("temporal_fusion") for name in detector_names)
    assert not any(name.startswith("moe") for name in detector_names)

    configure_drishti_training_stage(model, "temporal")
    temporal_names = trainable_parameter_names(model)
    assert temporal_names
    assert all(name.startswith("temporal_fusion") for name in temporal_names)

    configure_drishti_training_stage(model, "moe")
    moe_names = trainable_parameter_names(model)
    assert moe_names
    assert all(name.startswith("moe") for name in moe_names)

    configure_drishti_training_stage(model, "all")
    all_names = trainable_parameter_names(model)
    assert any(name.startswith("motion_proposer") for name in all_names)
    assert any(name.startswith("temporal_fusion") for name in all_names)
    assert any(name.startswith("moe") for name in all_names)
    assert not any(name.startswith("crop_encoder") for name in all_names)


def test_drishti_losses_are_finite_for_each_training_stage():
    torch.manual_seed(5)
    model = DRISHTIPipeline(tiny_config())
    batch = synthetic_batch()
    frames = batch["frames"]
    frame_targets = batch["frame_targets"]

    detector_parts = detector_stage_loss(model.forward_detector(frames), frame_targets)
    temporal_parts = detector_loss(model.forward_temporal(frames), frame_targets)
    moe_parts = moe_stage_loss(model(frames), frame_targets)

    assert torch.isfinite(detector_parts["det"])
    assert torch.isfinite(temporal_parts["det"])
    assert torch.isfinite(moe_parts["loss"])
    assert moe_parts["loss"].item() >= moe_parts["det"].item()


def test_drishti_collator_preserves_frame_targets():
    batch = synthetic_batch(num_samples=2, num_frames=5)

    assert batch["frames"].shape == (2, 5, 3, 64, 64)
    assert len(batch["frame_targets"]) == 2
    assert len(batch["frame_targets"][0]) == 5
    assert batch["frame_targets"][0][0]["boxes"].shape == (1, 4)


def test_one_optimization_step_updates_trainable_core():
    torch.manual_seed(6)
    model = DRISHTIPipeline(tiny_config())
    configure_drishti_training_stage(model, "all")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    batch = synthetic_batch()
    before = model.motion_proposer.motion_cnn[0].weight.detach().clone()

    output = model(batch["frames"])
    parts = moe_stage_loss(output, batch["frame_targets"])
    parts["loss"].backward()
    optimizer.step()

    after = model.motion_proposer.motion_cnn[0].weight.detach()
    assert not torch.allclose(before, after)
