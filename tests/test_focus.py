import pytest
import torch
from torch.utils.data import DataLoader

from drishti_v2.data import DRISHTICollator, SyntheticAntiUAVDataset
from drishti_v2.models.config import DRISHTIConfig
from drishti_v2.models.pipeline import DRISHTIPipeline
from drishti_v2.training import DRISHTITrainer, StageLossFactory, apply_training_stage


def _cfg(mode, **kw):
    base = dict(image_channels=1, image_height=64, image_width=64, crop_size=16, crop_scales=(1.0, 2.0, 4.0), num_crops=4,
                temporal_window=3, ldmi_scales=(3,), motion_cnn_channels=(4, 4, 4), use_motion_gate=False,
                encoder_feature_dim=16, temporal_heads=2, temporal_layers=1, temporal_ffn_dim=16, num_experts=2, top_k=1,
                expert_ffn_dim=16, train_encoder_in_stage1=True, encoder_backbone="resnet18", focus_mode=mode,
                focus_max_steps=1 if mode == "fixed" else 3, focus_min_window_px=8.0, focus_max_window_px=32.0)
    base.update(kw)
    return DRISHTIConfig(**base)


def test_focus_off_by_default():
    assert DRISHTIPipeline(_cfg("off")).focus is None


@pytest.mark.parametrize("mode,max_windows", [("fixed", 2), ("adaptive", 6)])
def test_refine_changes_only_topk_boxes(mode, max_windows):
    model = DRISHTIPipeline(_cfg(mode)).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 1, 64, 64))
    assert out.boxes.shape == (2, 4, 4)
    assert torch.all(out.focus_windows <= max_windows) and torch.all(out.focus_windows >= 1)
    # boxes outside the top-2 by score are untouched
    with torch.no_grad():
        model.focus.config.focus_mode = "off"
        changed = (out.boxes - model._boxes_to_global(out.crop_boxes, out.proposal_centers, (64, 64), out.center_offsets, (16, 16))).abs().sum(-1) > 1e-6
    assert changed.sum(dim=1).le(2).all()


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_focus_loss_reaches_head_and_shared_encoder(mode):
    cfg = _cfg(mode)
    model = DRISHTIPipeline(cfg)
    apply_training_stage(model, "joint")
    targets = [[{"boxes": torch.tensor([[0.5, 0.45, 0.08, 0.08]])} for _ in range(3)] for _ in range(2)]
    opt = torch.optim.SGD(model.focus.parameters(), lr=0.1)
    for _ in range(2):  # the box head starts at zero, so the encoder gets gradient from the 2nd step on
        model.zero_grad()
        out = model(torch.rand(2, 3, 1, 64, 64))
        terms = model.focus.training_loss(out.frame_last, out.boxes, out.objectness_logits, targets)
        assert torch.isfinite(terms["focus"]) and terms["focus"] > 0
        (terms["focus"] + terms["focus_halt"]).backward()
        opt.step()
    assert model.focus.box_head.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    if mode == "adaptive":
        assert model.focus.halt_head.weight.grad.abs().sum() > 0


def test_trainer_logs_focus_loss():
    cfg = _cfg("adaptive")
    model = DRISHTIPipeline(cfg)
    ds = SyntheticAntiUAVDataset(num_samples=4, num_frames=3, height=64, width=64, image_channels=1)
    loader = DataLoader(ds, batch_size=2, collate_fn=DRISHTICollator())
    trainer = DRISHTITrainer(model, loader, None, StageLossFactory.make_loss("joint", config=cfg), output_dir="results/test_focus_trainer")
    history = trainer.fit(stage="joint", epochs=1, lr=1e-3)
    assert history[0]["train_focus"] > 0


def test_focus_learns_to_recentre_a_bright_target():
    torch.manual_seed(0)
    cfg = _cfg("fixed", encoder_backbone="simple")
    model = DRISHTIPipeline(cfg)
    apply_training_stage(model, "joint")
    params = list(model.focus.parameters()) + list(model.encoder.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)

    def batch():
        frames = torch.zeros(8, 1, 64, 64)
        targets = []
        for i in range(8):
            x, y = torch.randint(10, 50, (2,)).tolist()
            frames[i, 0, y - 2 : y + 2, x - 2 : x + 2] = 1.0
            targets.append([{"boxes": torch.tensor([[(x - 0.5) / 63, (y - 0.5) / 63, 4 / 63, 4 / 63]])}])
        return frames, targets

    losses = []
    for _ in range(80):
        frames, targets = batch()
        boxes = torch.rand(8, 4, 4) * 0.2 + 0.4
        loss = model.focus.training_loss(frames, boxes, torch.zeros(8, 4, 1), targets)["focus"]
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    assert sum(losses[-10:]) / 10 < 0.7 * sum(losses[:10]) / 10


def test_version3_layer_mix_and_fine_features_train():
    cfg = _cfg("off", head_layer_mix=True, head_fine_features=True)
    model = DRISHTIPipeline(cfg)
    assert model.layer_mix is not None and model.fine_proj is not None
    apply_training_stage(model, "joint")
    out = model(torch.rand(2, 3, 1, 64, 64))
    targets = [[{"boxes": torch.tensor([[0.5, 0.45, 0.08, 0.08]]), "labels": torch.ones(1)} for _ in range(3)] for _ in range(2)]
    loss = StageLossFactory.make_loss("joint", config=cfg)(out, targets, all_heatmaps=out.all_heatmaps,
                                                           logits_seq=out.objectness_logits_seq,
                                                           centers_seq=out.proposal_centers_seq, boxes_seq=out.boxes_seq)["loss"]
    loss.backward()
    assert model.layer_mix.logits.grad is not None and model.layer_mix.logits.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.fine_proj.parameters())
    model.eval()
    with torch.no_grad():
        model.reset_stream()
        s = model.forward_stream(torch.rand(1, 1, 64, 64), 0)
    assert s.boxes.shape == (1, 4, 4)
    assert torch.allclose(model.layer_mix.weights().sum(), torch.tensor(1.0))


def test_default_head_unchanged_without_version3():
    model = DRISHTIPipeline(_cfg("off"))
    assert model.layer_mix is None and model.fine_proj is None
