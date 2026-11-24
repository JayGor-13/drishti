import json

import torch
from PIL import Image

from drishti_v2.data.dataset import AntiUAVExtractedFrameDataset
from drishti_v2.evaluation.postprocess import smooth_sequence
from drishti_v2.experiments.run_sequence_eval import evaluate_sequence
from drishti_v2.models.config import DRISHTIConfig
from drishti_v2.models.pipeline import DRISHTIPipeline
from drishti_v2.tracker import TargetMemory
from drishti_v2.training import StageLossFactory, apply_training_stage


def _cfg(**kw):
    base = dict(image_channels=1, image_height=64, image_width=64, crop_size=16, crop_scales=(1.0, 2.0, 4.0), num_crops=4,
                temporal_window=3, ldmi_scales=(3,), motion_cnn_channels=(4, 4, 4), use_motion_gate=False,
                encoder_feature_dim=16, temporal_heads=2, temporal_layers=1, temporal_ffn_dim=16, num_experts=2, top_k=1,
                expert_ffn_dim=16, train_encoder_in_stage1=True, encoder_backbone="resnet18", modality="infrared")
    base.update(kw)
    return DRISHTIConfig(**base)


# ---------------------------------------------------------------- A: memory
def test_memory_keeps_looking_after_the_drone_is_lost_and_expires():
    mem = TargetMemory(threshold=0.5, max_age=3)
    assert mem.guided_centers() is None
    mem.update(torch.tensor([[0.40, 0.50, 0.02, 0.02]]), torch.tensor([0.9]))
    mem.update(torch.tensor([[0.42, 0.50, 0.02, 0.02]]), torch.tensor([0.9]))
    centers = mem.guided_centers()
    assert centers.shape == (1, 2, 2)
    for _ in range(2):  # drone lost: memory still predicts it moving right
        mem.update(torch.tensor([[0.9, 0.9, 0.02, 0.02]]), torch.tensor([0.1]))
    predicted = mem.guided_centers()[0, 0]
    assert predicted[0] > 0.42
    for _ in range(3):
        mem.update(torch.empty(0, 4), torch.empty(0))
    assert mem.guided_centers() is None


# ---------------------------------------------------------------- C: smoothing
def test_smoothing_holds_through_short_dropouts_and_reduces_jitter():
    box = torch.tensor([0.5, 0.5, 0.1, 0.1])
    preds = [{"boxes": box[None], "scores": torch.tensor([0.9])},
             {"boxes": (box + torch.tensor([0.02, 0, 0, 0]))[None], "scores": torch.tensor([0.9])},
             {"boxes": torch.empty(0, 4), "scores": torch.empty(0)},  # one-frame dropout
             {"boxes": box[None], "scores": torch.tensor([0.9])}]
    out = smooth_sequence(preds, threshold=0.3, hold=2, alpha=0.5)
    assert len(out) == 4
    assert out[2]["boxes"].shape == (1, 4) and float(out[2]["scores"]) >= 0.3  # held, so no flicker
    assert abs(float(out[1]["boxes"][0, 0]) - 0.51) < 1e-6  # half way between 0.50 and 0.52
    long_gap = smooth_sequence(preds[:1] + [{"boxes": torch.empty(0, 4), "scores": torch.empty(0)}] * 4, 0.3, hold=2)
    assert long_gap[3]["boxes"].numel() == 0  # gives up after the hold


# ---------------------------------------------------------------- D: recurrent heatmap
def test_recurrent_heatmap_adds_a_channel_and_trains_and_streams():
    cfg = _cfg(heatmap_recurrent=True)
    assert cfg.motion_input_channels == _cfg().motion_input_channels + 1
    model = DRISHTIPipeline(cfg)
    apply_training_stage(model, "joint")
    out = model(torch.rand(2, 3, 1, 64, 64))
    targets = [[{"boxes": torch.tensor([[0.5, 0.45, 0.08, 0.08]]), "labels": torch.ones(1)} for _ in range(3)] for _ in range(2)]
    loss = StageLossFactory.make_loss("joint", config=cfg)(out, targets, all_heatmaps=out.all_heatmaps,
                                                           logits_seq=out.objectness_logits_seq,
                                                           centers_seq=out.proposal_centers_seq, boxes_seq=out.boxes_seq)["loss"]
    loss.backward()
    first_conv = model.motion_cnn.net[0].weight
    assert first_conv.shape[1] == cfg.motion_input_channels and first_conv.grad[:, -1].abs().sum() > 0
    model.eval()
    model.reset_stream()
    with torch.no_grad():
        model.forward_stream(torch.rand(1, 1, 64, 64), 0)
        assert model._stream_prev_heatmap is not None
        model.forward_stream(torch.rand(1, 1, 64, 64), 1)
    model.reset_stream()
    assert model._stream_prev_heatmap is None


# ---------------------------------------------------------------- E: second focus step trains
def test_fixed_focus_with_two_steps_trains_both_steps():
    cfg = _cfg(focus_mode="fixed", focus_max_steps=2, focus_min_window_px=8.0, focus_max_window_px=32.0)
    model = DRISHTIPipeline(cfg).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 1, 64, 64))
    assert torch.all(out.focus_windows == 4)  # 2 candidates x 2 steps


# ---------------------------------------------------------------- A+B+C in the sequence evaluator
def _write_sequence(root, split, name, frames=10, size=64):
    seq_dir = root / split / name
    (seq_dir / "infrared").mkdir(parents=True)
    rects, exist = [], []
    for idx in range(frames):
        image = torch.zeros(size, size)
        x, y = 20 + idx, 30
        image[y : y + 4, x : x + 4] = 1.0
        Image.fromarray((image.numpy() * 255).astype("uint8")).save(seq_dir / "infrared" / f"{idx:06d}.png")
        rects.append([x, y, 4, 4]); exist.append(1)
    (seq_dir / "infrared.json").write_text(json.dumps({"gt_rect": rects, "exist": exist}), encoding="utf-8")


def test_evaluator_runs_with_memory_more_crops_and_smoothing(tmp_path):
    _write_sequence(tmp_path, "val", "seq_a")
    cfg = _cfg()
    cfg.num_crops = 6  # idea B: more crops than the 4 the model was built with
    model = DRISHTIPipeline(cfg).eval()
    ds = AntiUAVExtractedFrameDataset(frames_root=tmp_path, split="val", modality="infrared", num_frames=3, height=64,
                                      width=64, clip_stride=1, frame_stride=1, image_channels=1, box_format="xywh")
    m = evaluate_sequence(model, cfg, ds.sequences[0], torch.device("cpu"), use_tracker=True, threshold=0.0,
                          nms_iou=0.3, top1=True, memory=True, smooth=True)
    assert m["frames"] == 10 and 0.0 <= m["map50"] <= 1.0


def test_track1_mode_outputs_the_given_first_box(tmp_path):
    _write_sequence(tmp_path, "val", "seq_b")
    cfg = _cfg()
    model = DRISHTIPipeline(cfg).eval()
    ds = AntiUAVExtractedFrameDataset(frames_root=tmp_path, split="val", modality="infrared", num_frames=3, height=64,
                                      width=64, clip_stride=1, frame_stride=1, image_channels=1, box_format="xywh")
    raw: list = []
    m = evaluate_sequence(model, cfg, ds.sequences[0], torch.device("cpu"), use_tracker=True, threshold=0.3,
                          nms_iou=0.3, top1=True, raw_out=raw, init_first_box=True)
    gt0 = torch.tensor([(20 + 2) / 64, (30 + 2) / 64, 4 / 64, 4 / 64])  # frame 0 box, normalized cx, cy, w, h
    assert torch.allclose(torch.tensor(raw[0]["boxes"][0]), gt0, atol=1e-5) and raw[0]["scores"] == [1.0]
    assert m["frames"] == 10


def test_v5_external_detector_guides_crops_and_is_fused(tmp_path):
    _write_sequence(tmp_path, "val", "seq_c")
    cfg = _cfg(focus_mode="fixed", focus_max_steps=1, focus_min_window_px=8.0, focus_max_window_px=32.0)
    model = DRISHTIPipeline(cfg).eval()
    ds = AntiUAVExtractedFrameDataset(frames_root=tmp_path, split="val", modality="infrared", num_frames=3, height=64,
                                      width=64, clip_stride=1, frame_stride=1, image_channels=1, box_format="xywh")
    seq = ds.sequences[0]
    # a perfect "dense detector": the true box in every frame, plus a weak decoy
    frames = {str(i): {"boxes": [[(20 + i + 2) / 64, 32 / 64, 4 / 64, 4 / 64], [0.9, 0.9, 0.05, 0.05]], "scores": [0.95, 0.02]}
              for i in range(10)}
    external = {seq.name: frames}
    raw: list = []
    m = evaluate_sequence(model, cfg, seq, torch.device("cpu"), use_tracker=False, threshold=0.1, nms_iou=0.3, top1=True,
                          raw_out=raw, external=external, ext_topk=3, ext_min_score=0.05, fuse_ext=1.0, refine_ext=True)
    assert len(raw[0]["boxes"]) == cfg.num_crops + 1  # the decoy is below ext_min_score
    assert m["gt_inside_some_crop"] == 1.0  # detector boxes became guided crops: the drone is always covered
    assert m["recall"] > 0.5  # the fused detector box carries the output even with an untrained model
