"""Small trainer wrapper for smoke training and future expansion."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from models.tmoe_model import TMoELLaVAMicro
from .loss import (
    TMoELossWeights,
    autoregressive_loss,
    cfcr_loss,
    load_balancing_loss,
    orthogonalization_loss,
)


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
        
        device = next(self.model.parameters()).device
        frames = frames.to(device)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        output = self.model(frames, input_ids, reset_cache=True)
        ar = autoregressive_loss(output.logits, labels)

        aux_losses = []
        cfcr_losses = []
        ortho_losses = []
        for block, router in zip(self.model.blocks, output.router_outputs):
            aux_losses.append(load_balancing_loss(router.probs))
            cfcr_losses.append(cfcr_loss(router.probs, output.motion_confidence))
            ortho_losses.append(orthogonalization_loss(block.moe.experts))
        if not aux_losses:
            raise RuntimeError("model must produce at least one router output")

        aux = torch.stack(aux_losses).mean()
        cfcr = torch.stack(cfcr_losses).mean()
        ortho = torch.stack(ortho_losses).mean()
        weights = self.config.loss_weights
        loss = ar + weights.alpha_aux * aux
        loss = loss + weights.beta_cfcr * cfcr + weights.gamma_ortho * ortho

        loss.backward()
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        return {
            "loss": loss.detach().item(),
            "ar": ar.detach().item(),
            "aux": aux.detach().item(),
            "cfcr": cfcr.detach().item(),
            "ortho": ortho.detach().item(),
        }
