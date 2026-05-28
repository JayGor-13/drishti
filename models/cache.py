"""Event-based temporal token cache."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EventTokenCache(nn.Module):
    """Cache previous-frame FFN outputs for low-motion visual tokens."""

    def __init__(self, threshold: float = 0.05, detach_writes: bool = True) -> None:
        super().__init__()
        self.threshold = threshold
        self.detach_writes = detach_writes
        self.register_buffer("values", torch.empty(0), persistent=False)
        self.register_buffer("valid", torch.empty(0, dtype=torch.bool), persistent=False)

    def reset(self) -> None:
        self.values = torch.empty(0, device=self.values.device)
        self.valid = torch.empty(0, dtype=torch.bool, device=self.valid.device)

    def ensure_shape(
        self,
        batch_size: int,
        slots: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        desired_values = (batch_size, slots, hidden_dim)
        desired_valid = (batch_size, slots)
        if (
            tuple(self.values.shape) != desired_values
            or self.values.device != device
            or self.values.dtype != dtype
        ):
            self.values = torch.zeros(desired_values, device=device, dtype=dtype)
        if tuple(self.valid.shape) != desired_valid or self.valid.device != device:
            self.valid = torch.zeros(desired_valid, device=device, dtype=torch.bool)

    def static_mask(self, confidence: Tensor) -> Tensor:
        return confidence < self.threshold

    def readable_mask(self, confidence: Tensor) -> Tensor:
        if self.valid.numel() == 0:
            return torch.zeros_like(confidence, dtype=torch.bool)
        return self.static_mask(confidence) & self.valid

    def read(self, mask: Tensor) -> Tensor:
        if self.values.numel() == 0:
            raise RuntimeError("cache shape has not been initialized")
        expanded = mask.unsqueeze(-1).expand_as(self.values)
        return torch.where(expanded, self.values, torch.zeros_like(self.values))

    def write(self, outputs: Tensor, mask: Tensor) -> None:
        """Update cache slots selected by mask.

        Args:
            outputs: Tensor shaped ``[batch, slots, hidden]``.
            mask: Boolean tensor shaped ``[batch, slots]``.
        """

        if self.values.numel() == 0:
            raise RuntimeError("cache shape has not been initialized")
        values = outputs.detach() if self.detach_writes else outputs
        expanded = mask.unsqueeze(-1).expand_as(self.values)
        self.values = torch.where(expanded, values, self.values)
        self.valid = self.valid | mask
