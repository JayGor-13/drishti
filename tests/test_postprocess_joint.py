import torch

from drishti_v2.evaluation.postprocess import nms, postprocess
from drishti_v2.models.config import DRISHTIConfig
from drishti_v2.models.pipeline import DRISHTIPipeline
from drishti_v2.training import StageLossFactory, apply_training_stage


def test_nms_merges_duplicates_and_keeps_separate_boxes():
    boxes = torch.tensor([[0.50, 0.50, 0.10, 0.10], [0.505, 0.50, 0.10, 0.10], [0.20, 0.20, 0.05, 0.05]])
    scores = torch.tensor([0.6, 0.9, 0.4])
    kept_boxes, kept_scores = nms(boxes, scores, iou=0.5)
    assert kept_scores.tolist() == [0.8999999761581421, 0.4000000059604645]
    assert torch.allclose(kept_boxes[0], boxes[1])


def test_postprocess_top1_and_empty():
    boxes = torch.rand(8, 4) * 0.1 + 0.4
    scores = torch.linspace(0.1, 0.8, 8)
    b, s = postprocess(boxes, scores, top1=True)
    assert b.shape == (1, 4) and float(s) == float(scores.max())
    b, s = postprocess(torch.empty(0, 4), torch.empty(0), nms_iou=0.3, top1=True)
    assert b.numel() == 0 and s.numel() == 0


def _cfg(**kw):
    base = dict(image_channels=1, image_height=48, image_width=64, crop_size=16, crop_scales=(1.0, 2.0, 4.0), num_crops=2,
                temporal_window=3, ldmi_scales=(3,), motion_cnn_channels=(4, 4, 4), use_motion_gate=False,
                encoder_feature_dim=8, temporal_heads=2, temporal_layers=1, temporal_ffn_dim=16, num_experts=2, top_k=1,
                expert_ffn_dim=16, train_encoder_in_stage1=True, encoder_backbone="resnet18")
    base.update(kw)
    return DRISHTIConfig(**base)


def test_joint_stage_trains_every_module_on_non_square_frames():
    cfg = _cfg()
    model = DRISHTIPipeline(cfg)
    apply_training_stage(model, "joint")
    out = model(torch.rand(2, 3, 1, 48, 64))
    assert out.heatmap.shape[-2:] == (12, 16)
    targets = [[{"boxes": torch.tensor([[0.4, 0.6, 0.1, 0.1]]), "labels": torch.ones(1)} for _ in range(3)] for _ in range(2)]
    loss = StageLossFactory.make_loss("joint", config=cfg)(out, targets, all_heatmaps=out.all_heatmaps,
                                                           logits_seq=out.objectness_logits_seq,
                                                           centers_seq=out.proposal_centers_seq, boxes_seq=out.boxes_seq)["loss"]
    loss.backward()
    for name in ["motion_cnn", "encoder", "temporal", "head"]:
        module = getattr(model, name)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()), name
