from __future__ import annotations

from torch import nn


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    module.train(trainable)
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def apply_training_stage(model: nn.Module, stage: str) -> None:
    """Apply staged freezing rules from the implementation plan."""

    stage = stage.lower()
    for parameter in model.parameters():
        parameter.requires_grad = False

    if stage in {"stage1", "detector"}:
        _set_trainable(model.motion_cnn, True)
        _set_trainable(model.head, True)
    elif stage in {"stage2", "temporal"}:
        _set_trainable(model.temporal, True)
    elif stage in {"stage3", "moe"}:
        _set_trainable(model.moe, True)
    elif stage in {"finetune", "e2e", "all"}:
        for parameter in model.parameters():
            parameter.requires_grad = True
        model.train()
    else:
        raise ValueError(f"Unknown training stage: {stage}")
