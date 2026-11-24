import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_arbiter import Arbiter, metrics, pick, rule_scores  # noqa: E402

from drishti_v2.experiments.dump_candidates import FEATURES  # noqa: E402


def _seq(n=4, k=5):
    feats = torch.zeros(n, k, len(FEATURES))
    feats[..., FEATURES.index("is_ours")] = 1.0
    feats[:, -1, FEATURES.index("is_ours")] = 0.0
    feats[:, -1, FEATURES.index("is_ext")] = 1.0
    feats[..., FEATURES.index("score")] = torch.linspace(0.1, 0.9, k)
    boxes = torch.tensor([0.5, 0.5, 0.1, 0.1]).repeat(n, k, 1)
    boxes[:, 0] = torch.tensor([0.1, 0.1, 0.05, 0.05])  # the true box sits in slot 0 only
    gt = torch.tensor([0.1, 0.1, 0.05, 0.05]).repeat(n, 1)
    mask = torch.ones(n, k, dtype=torch.bool)
    mask[:, 3] = False
    iou = torch.zeros(n, k)
    iou[:, 0] = 1.0
    return {"feats": feats, "boxes": boxes, "mask": mask, "gt": gt, "visible": torch.ones(n, dtype=torch.bool), "iou": iou}


def test_arbiter_masks_padding_and_handles_empty_frames():
    for attn in (True, False):
        model = Arbiter(len(FEATURES), 16, attn=attn)
        mask = torch.ones(3, 5, dtype=torch.bool)
        mask[0, 2:] = False
        mask[1] = False  # a frame without candidates
        out = model(torch.randn(3, 5, len(FEATURES)), mask)
        assert out.shape == (3, 5)
        assert torch.isinf(out[~mask]).all() and torch.isfinite(out[mask]).all()


def test_rules_and_oracle():
    seq = _seq()
    names = list(FEATURES)
    # the highest raw score is the detector's (last slot), which is the wrong box
    assert metrics([seq], [rule_scores(seq, "max_score", names)], 0.05)["map50"] == 0.0
    assert metrics([seq], [rule_scores(seq, "oracle", names)], 0.05)["map50"] == 1.0
    ours = pick(seq, rule_scores(seq, "ours_only", names))
    assert all(torch.allclose(p["boxes"][0], torch.tensor([0.5, 0.5, 0.1, 0.1])) for p in ours)  # slot 3 is padding
