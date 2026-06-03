import torch
import pytest

from models.cache import EventTokenCache
from models.moe_layer import MicroMoELayer
from models.router import ModalityAwareRouter
from models.tmoe_model import TMoEAntiDroneDetector, TMoEConfig
from train import AntiUAVDetectionCollator, SyntheticAntiUAVDataset
from train.loss import (
    cfcr_loss,
    detection_loss,
    expert_lora_similarity,
    load_balancing_loss,
    semantic_alignment,
)
from train.trainer import TMoETrainer, TrainingConfig


def tiny_config() -> TMoEConfig:
    return TMoEConfig(
        hidden_dim=32,
        ffn_dim=64,
        num_experts=4,
        top_k=2,
        num_layers=1,
        num_attention_heads=4,
        patch_grid_size=2,
        motion_dim=16,
        lora_rank=2,
        lora_alpha=4.0,
        cache_threshold=0.15,
        max_frames=8,
    )


def test_detector_output_shape_and_finiteness():
    torch.manual_seed(0)
    model = TMoEAntiDroneDetector(tiny_config())
    frames = torch.randn(2, 3, 3, 16, 16)

    output = model(frames, reset_cache=True)

    patches = model.config.patch_grid_size**2
    assert output.class_logits.shape == (2, 3, patches, 2)
    assert output.boxes.shape == (2, 3, patches, 4)
    assert torch.isfinite(output.class_logits).all()
    assert torch.all((output.boxes >= 0.0) & (output.boxes <= 1.0))


def test_cache_bypasses_static_second_frame():
    torch.manual_seed(1)
    hidden = 16
    layer = MicroMoELayer(
        hidden_dim=hidden,
        ffn_dim=32,
        num_experts=4,
        top_k=2,
        router_history_window=1,
    )
    cache = EventTokenCache(threshold=0.05)
    frame = torch.randn(1, 1, 4, hidden)
    tokens = frame.repeat(1, 2, 1, 1)
    motion_confidence = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])

    output = layer(tokens, motion_confidence=motion_confidence, cache=cache)

    assert output.stats.executed_tokens == 4
    assert output.stats.cached_tokens == 4
    assert torch.allclose(output.hidden_states[:, 0], output.hidden_states[:, 1], atol=1e-6)


def test_token_level_cache_mixed_motion():
    torch.manual_seed(5)
    hidden = 12
    layer = MicroMoELayer(
        hidden_dim=hidden,
        ffn_dim=24,
        num_experts=3,
        top_k=1,
        router_history_window=1,
    )
    cache = EventTokenCache(threshold=0.05)
    frame = torch.randn(1, 1, 4, hidden)
    tokens = frame.repeat(1, 2, 1, 1)
    tokens[:, 1, 2:] = tokens[:, 1, 2:] + 0.5
    motion_confidence = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]])
    expert_token_rows = 0

    def count_expert_rows(_module, inputs, _output):
        nonlocal expert_token_rows
        expert_token_rows += inputs[0].shape[0]

    handles = [expert.register_forward_hook(count_expert_rows) for expert in layer.experts]
    try:
        output = layer(tokens, motion_confidence=motion_confidence, cache=cache)
    finally:
        for handle in handles:
            handle.remove()

    ffn_branch = output.hidden_states - tokens
    assert output.stats.executed_tokens == 6
    assert output.stats.cached_tokens == 2
    assert expert_token_rows == 6
    assert torch.allclose(ffn_branch[:, 1, :2], ffn_branch[:, 0, :2], atol=1e-6)


def test_modality_router_uses_motion_embeddings():
    torch.manual_seed(2)
    router = ModalityAwareRouter(hidden_dim=12, num_experts=4, top_k=2)
    tokens = torch.randn(2, 3, 5, 12, requires_grad=True)
    still_motion = torch.zeros_like(tokens)
    moving_motion = torch.ones_like(tokens)

    still = router(tokens, still_motion)
    moving = router(tokens, moving_motion)
    loss = (moving.probs[..., 0] ** 2).mean()
    loss.backward()

    assert not torch.allclose(still.logits, moving.logits)
    assert tokens.grad is not None
    assert router.gate.weight.grad is not None


def test_expert_lora_similarity_can_detect_divergence():
    torch.manual_seed(3)
    layer = MicroMoELayer(
        hidden_dim=8,
        ffn_dim=8,
        num_experts=4,
        top_k=2,
        lora_rank=1,
        lora_alpha=2.0,
    )
    with torch.no_grad():
        for idx, expert in enumerate(layer.experts):
            for module in (expert.gate_proj, expert.up_proj, expert.down_proj):
                module.lora_b.zero_()
            expert.up_proj.lora_b[idx * 2 : idx * 2 + 2, 0] = 1.0

    similarity = expert_lora_similarity(layer.experts)

    assert similarity.item() < 0.2


def test_cfcr_loss_monotonicity():
    same = torch.tensor([[[[0.8, 0.2], [0.1, 0.9]], [[0.8, 0.2], [0.1, 0.9]]]])
    motion = torch.zeros(1, 2, 2)
    changed = torch.tensor([[[[0.8, 0.2], [0.1, 0.9]], [[0.2, 0.8], [0.9, 0.1]]]])

    static_loss = cfcr_loss(same, motion)
    changed_loss = cfcr_loss(changed, motion)

    assert static_loss.item() < 1e-7
    assert changed_loss.item() > static_loss.item()


def test_semantic_alignment_shape():
    features = torch.randn(2, 4, 5, 8)
    alignment = semantic_alignment(features)

    assert alignment.shape == (2, 3, 5, 5)
    assert torch.allclose(alignment.sum(dim=-1), torch.ones(2, 3, 5), atol=1e-5)


def test_collator_assigns_box_to_patch():
    collator = AntiUAVDetectionCollator(patch_grid_size=4)
    item = {
        "frames": torch.zeros(2, 3, 16, 16),
        "frame_targets": [
            {"boxes": torch.tensor([[0.6, 0.2, 0.1, 0.1]]), "labels": torch.ones(1, dtype=torch.long)},
            {"boxes": torch.tensor([[0.1, 0.9, 0.2, 0.2]]), "labels": torch.ones(1, dtype=torch.long)},
        ],
        "image_ids": [1, 2],
    }

    batch = collator([item])

    assert batch["frames"].shape == (1, 2, 3, 16, 16)
    assert batch["class_targets"].sum().item() == 2
    assert batch["box_mask"].sum().item() == 2
    assert batch["box_targets"].shape == (1, 2, 16, 4)


def test_detection_loss_is_finite():
    torch.manual_seed(4)
    logits = torch.randn(1, 2, 4, 2)
    boxes = torch.rand(1, 2, 4, 4)
    targets = torch.zeros(1, 2, 4, dtype=torch.long)
    targets[:, :, 1] = 1
    box_targets = torch.rand(1, 2, 4, 4)
    mask = targets == 1

    parts = detection_loss(logits, boxes, targets, box_targets, mask)

    assert torch.isfinite(parts["det"])
    assert parts["det"].item() > 0.0


def test_dense_routing_runs_all_experts_once_per_token():
    torch.manual_seed(6)
    hidden = 10
    layer = MicroMoELayer(
        hidden_dim=hidden,
        ffn_dim=20,
        num_experts=4,
        top_k=2,
        dense_routing=True,
    )
    tokens = torch.randn(1, 2, 3, hidden)
    expert_token_rows = 0

    def count_expert_rows(_module, inputs, _output):
        nonlocal expert_token_rows
        expert_token_rows += inputs[0].shape[0]

    handles = [expert.register_forward_hook(count_expert_rows) for expert in layer.experts]
    try:
        output = layer(tokens)
    finally:
        for handle in handles:
            handle.remove()

    assert output.stats.executed_tokens == 6
    assert expert_token_rows == 24


def test_trainer_step_optimizes_antiuav_batch():
    torch.manual_seed(7)
    dataset = SyntheticAntiUAVDataset(num_samples=2, num_frames=3, height=16, width=16)
    collator = AntiUAVDetectionCollator(patch_grid_size=2)
    batch = collator([dataset[0], dataset[1]])
    model = TMoEAntiDroneDetector(tiny_config())
    trainer = TMoETrainer(
        model,
        TrainingConfig(learning_rate=0.0, weight_decay=0.0, cfcr_warmup_steps=1),
    )

    metrics = trainer.train_step(batch)
    with torch.no_grad():
        output = model(batch["frames"], reset_cache=True)
        expected_aux = torch.stack(
            [load_balancing_loss(router.probs) for router in output.router_outputs]
        ).mean()

    assert metrics["aux"] == pytest.approx(expected_aux.item(), rel=1e-6)
    assert metrics["det"] > 0.0
