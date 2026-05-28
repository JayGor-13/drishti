"""Small trainer wrapper for smoke training and future expansion."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from models.tmoe_model import TMoELLaVAMicro
from .loss import TMoELossWeights, total_tmoe_loss


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    loss_weights: TMoELossWeights = field(default_factory=TMoELossWeights)


class TMoETrainer:
    """Minimal trainer that keeps the research loss wired end-to-end."""

    def __init__(
        self,
        model: TMoELLaVAMicro,
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

    def train_step(
        self,
        frames: Tensor,
        input_ids: Tensor,
        labels: Tensor,
    ) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(frames, input_ids, reset_cache=True)
        first_router = output.router_outputs[0]
        first_block = self.model.blocks[0]
        losses = total_tmoe_loss(
            logits=output.logits,
            labels=labels,
            router_probs=first_router.probs,
            motion_confidence=output.motion_confidence,
            experts=first_block.moe.experts,
            weights=self.config.loss_weights,
        )
        losses["loss"].backward()
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        return {name: value.detach().item() for name, value in losses.items()}
