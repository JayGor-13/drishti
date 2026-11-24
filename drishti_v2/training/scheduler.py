from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler


def make_scheduler(optimizer: Optimizer, epochs: int, eta_min: float = 1e-7) -> LRScheduler:
    return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=eta_min)
