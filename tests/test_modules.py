import torch
import pytest

from models.cache import EventTokenCache
from models.moe_layer import MicroMoELayer
from models.router import TemporallyAwareRouter
from models.tmoe_model import TMoEConfig, TMoELLaVAMicro
from train import (
    ActivityNetQARecord,
    VideoFileIndex,
    filter_records_with_available_videos,
)
from train.loss import cfcr_loss, expert_lora_similarity, load_balancing_loss
from train.trainer import TMoETrainer, TrainingConfig
from experiment import parse_video_shards


def tiny_config() -> TMoEConfig:
    return TMoEConfig(
        vocab_size=97,
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
        max_text_length=16,
    )


def test_sequence_concatenation_shape():
    torch.manual_seed(0)
    model = TMoELLaVAMicro(tiny_config())
    frames = torch.randn(2, 3, 3, 16, 16)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))

    video_tokens, motion_embeddings, _ = model.encode_video(frames)
    sequence = model.build_multimodal_sequence(video_tokens, motion_embeddings, input_ids)

    patches = model.config.patch_grid_size**2
    assert sequence.shape == (2, 3 * patches * 2 + 5, model.config.hidden_dim)


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

    handles = [
        expert.register_forward_hook(count_expert_rows) for expert in layer.experts
    ]
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


def test_router_gradients_flow():
    torch.manual_seed(2)
    router = TemporallyAwareRouter(hidden_dim=12, num_experts=4, top_k=2)
    tokens = torch.randn(2, 3, 5, 12, requires_grad=True)

    result = router(tokens)
    loss = (result.probs[..., 0] ** 2).mean()
    loss.backward()

    assert tokens.grad is not None
    assert router.token_gate.weight.grad is not None
    assert router.context_gate.weight.grad is not None


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


def test_lm_head_output_shape_and_finiteness():
    torch.manual_seed(4)
    model = TMoELLaVAMicro(tiny_config())
    frames = torch.randn(1, 2, 3, 16, 16)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 6))

    output = model(frames, input_ids, reset_cache=True)

    assert output.next_token_logits.shape == (1, model.config.vocab_size)
    assert torch.isfinite(output.next_token_logits).all()


def test_trainer_averages_aux_loss_across_layers():
    torch.manual_seed(6)
    config = tiny_config()
    config.num_layers = 2
    model = TMoELLaVAMicro(config)
    with torch.no_grad():
        first_router = model.blocks[0].moe.router
        first_router.token_gate.weight.zero_()
        first_router.context_gate.weight.zero_()
        if first_router.motion_gate is not None:
            first_router.motion_gate.weight.zero_()

    trainer = TMoETrainer(
        model,
        TrainingConfig(learning_rate=0.0, weight_decay=0.0),
    )
    frames = torch.randn(1, 2, 3, 16, 16)
    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    labels = input_ids.clone()

    metrics = trainer.train_step(frames, input_ids, labels)
    with torch.no_grad():
        output = model(frames, input_ids, reset_cache=True)
        expected_aux = torch.stack(
            [load_balancing_loss(router.probs) for router in output.router_outputs]
        ).mean()

    assert metrics["aux"] == pytest.approx(expected_aux.item(), rel=1e-6)


def test_video_index_matches_activitynet_filename_variants(tmp_path):
    nested = tmp_path / "videos"
    nested.mkdir()
    (nested / "v_abc-123.mp4").write_bytes(b"not a real video")
    records = [
        ActivityNetQARecord("abc-123", "v_abc-123_2", "q", "a", "3"),
        ActivityNetQARecord("missing", "v_missing_1", "q", "a", "3"),
    ]

    index = VideoFileIndex(tmp_path)
    filtered = filter_records_with_available_videos(records, tmp_path)

    assert index.find(records[0]) == nested / "v_abc-123.mp4"
    assert filtered == [records[0]]


def test_parse_video_shards_accepts_ranges_commas_and_spaces():
    assert parse_video_shards(["1-3,5", "7 8"], all_shards=False) == (1, 2, 3, 5, 7, 8)
    assert parse_video_shards(None, all_shards=True) == tuple(range(1, 29))
