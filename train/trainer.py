"""Small trainer wrapper for Anti-UAV smoke training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from models.tmoe_model import TMoEAntiDroneDetector
from .loss import (
    TMoELossWeights,
    cfcr_loss,
    detection_loss,
    load_balancing_loss,
    orthogonalization_loss,
    semantic_alignment,
)


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    cfcr_warmup_steps: int = 500
    loss_weights: TMoELossWeights = field(default_factory=TMoELossWeights)


class TMoETrainer:
    """Minimal Anti-UAV trainer that keeps all proposal losses wired."""

    def __init__(
        self,
        model: TMoEAntiDroneDetector,
        config: TrainingConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.global_step = 0

    def beta_cfcr(self) -> float:
        if self.config.cfcr_warmup_steps <= 0:
            return self.config.loss_weights.beta_cfcr
        scale = min(self.global_step / self.config.cfcr_warmup_steps, 1.0)
        return self.config.loss_weights.beta_cfcr * scale

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        device = next(self.model.parameters()).device
        frames = batch["frames"].to(device)
        class_targets = batch["class_targets"].to(device)
        box_targets = batch["box_targets"].to(device)
        box_mask = batch["box_mask"].to(device)

        output = self.model(frames, reset_cache=True)
        det_parts = detection_loss(
            output.class_logits,
            output.boxes,
            class_targets,
            box_targets,
            box_mask,
            weights=self.config.loss_weights,
        )
        alignment = semantic_alignment(output.semantic_tokens)
        aux = torch.stack([load_balancing_loss(router.probs) for router in output.router_outputs]).mean()
        cfcr = torch.stack(
            [cfcr_loss(router.probs, output.motion_confidence, alignment) for router in output.router_outputs]
        ).mean()
        ortho = torch.stack(
            [orthogonalization_loss(block.moe.experts) for block in self.model.blocks]
        ).mean()
        beta = self.beta_cfcr()
        weights = self.config.loss_weights
        loss = det_parts["det"] + weights.alpha_aux * aux + beta * cfcr
        loss = loss + weights.gamma_ortho * ortho

        loss.backward()
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        self.global_step += 1

        return {
            "loss": float(loss.detach().cpu()),
            "det": float(det_parts["det"].detach().cpu()),
            "cls": float(det_parts["cls"].detach().cpu()),
            "box_l1": float(det_parts["box_l1"].detach().cpu()),
            "giou": float(det_parts["giou"].detach().cpu()),
            "aux": float(aux.detach().cpu()),
            "cfcr": float(cfcr.detach().cpu()),
            "ortho": float(ortho.detach().cpu()),
            "beta_cfcr": beta,
        }
