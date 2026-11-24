from __future__ import annotations

from torch import nn


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    module.train(trainable)
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def apply_training_stage(model: nn.Module, stage: str) -> None:
    """Apply staged freezing rules while protecting frozen normalization state."""

    stage = stage.lower()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    if stage in {"stage1", "detector"}:
        _set_trainable(model.motion_cnn, True)
        _set_trainable(model.head, True)
        config = getattr(model, "config", None)
        if getattr(config, "train_encoder_in_stage1", False) and hasattr(model.encoder, "unfreeze"):
            # Without this the crop encoder keeps its random initial weights through stages 1-3.
            model.encoder.unfreeze()
        if getattr(model, "motion_gate", None) is not None:
            _set_trainable(model.motion_gate, True)
        if getattr(model, "focus", None) is not None:
            _set_trainable(model.focus, True)
        for name in ("layer_mix", "fine_proj"):
            if getattr(model, name, None) is not None:
                _set_trainable(getattr(model, name), True)
    elif stage in {"stage2", "temporal"}:
        _set_trainable(model.temporal, True)
    elif stage in {"stage3", "moe"}:
        _set_trainable(model.moe, True)
    elif stage in {"stage4", "finetune", "e2e", "all", "joint"}:
        if hasattr(model, "encoder") and hasattr(model.encoder, "unfreeze"):
            model.encoder.unfreeze()
        for parameter in model.parameters():
            parameter.requires_grad = True
        model.train()
    else:
        raise ValueError(f"Unknown training stage: {stage}")
