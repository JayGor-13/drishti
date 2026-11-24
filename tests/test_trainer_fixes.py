from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import drishti_v2.training.trainer as trainer_module
from drishti_v2.training.trainer import DRISHTITrainer


class _TinyModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))

    def forward(self, frames: torch.Tensor):
        prediction = self.weight * frames.mean()
        return SimpleNamespace(
            prediction=prediction,
            all_heatmaps=None,
            objectness_logits_seq=None,
            proposal_centers_seq=None,
            boxes_seq=None,
            moe_diagnostics=None,
        )


class _TinyLoss(nn.Module):
    def forward(self, output, targets):
        del targets
        return {"loss": output.prediction.square()}


def _loader():
    return [{"frames": torch.ones(1, 1), "targets": []}]


def test_best_checkpoint_saves_for_loss_above_one_and_resume_loads_model(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_module, "save_training_curves", lambda *args, **kwargs: None)
    model = _TinyModel(3.0)
    trainer = DRISHTITrainer(model, _loader(), None, _TinyLoss(), output_dir=tmp_path, device="cpu")
    trainer.fit(stage="all", epochs=1, lr=1e-3)

    checkpoint = tmp_path / "checkpoints" / "all_best.pt"
    assert checkpoint.exists()
    payload = torch.load(checkpoint, map_location="cpu")
    assert {"model", "optimizer", "scheduler", "epoch", "stage", "best_score"} <= payload.keys()
    assert payload["best_score"] < -1.0

    resumed_model = _TinyModel(-7.0)
    resumed = DRISHTITrainer(resumed_model, _loader(), None, _TinyLoss(), output_dir=tmp_path / "resume", device="cpu")
    history = resumed.fit(stage="all", epochs=1, lr=1e-3, resume_checkpoint=checkpoint)
    assert history == []
    assert torch.equal(resumed_model.state_dict()["weight"], payload["model"]["weight"])


def test_resume_rejects_partial_training_state(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_module, "save_training_curves", lambda *args, **kwargs: None)
    path = tmp_path / "partial.pt"
    torch.save({"model": _TinyModel(1.0).state_dict(), "epoch": 1}, path)
    trainer = DRISHTITrainer(_TinyModel(0.0), _loader(), None, _TinyLoss(), output_dir=tmp_path / "run")
    with pytest.raises(ValueError, match="missing keys"):
        trainer.fit(stage="all", epochs=2, lr=1e-3, resume_checkpoint=path)


def test_validation_runs_with_model_in_eval_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_module, "save_training_curves", lambda *args, **kwargs: None)
    observed_modes = []

    class _Evaluator:
        def __init__(self, model, loader, device):
            del loader, device
            self.model = model

        def evaluate(self, **kwargs):
            del kwargs
            observed_modes.append(self.model.training)
            return {"map50": 0.5}

    monkeypatch.setattr(trainer_module, "DRISHTIEvaluator", _Evaluator)
    trainer = DRISHTITrainer(_TinyModel(2.0), _loader(), [object()], _TinyLoss(), output_dir=tmp_path)
    trainer.fit(stage="all", epochs=1, lr=1e-3)
    assert observed_modes == [False]
