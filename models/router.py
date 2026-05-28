"""Temporally-conditioned router for homogeneous MoE experts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class RouterOutput:
    """Routing probabilities and top-k selections."""

    logits: Tensor
    probs: Tensor
    topk_indices: Tensor
    topk_scores: Tensor
    context: Tensor
    entropy: Tensor


class TemporallyAwareRouter(nn.Module):
    """Route tokens using current content, causal temporal context, and motion."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        history_window: int = 2,
        temperature: float = 1.0,
        use_motion_conditioning: bool = True,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.history_window = history_window
        self.temperature = temperature
        self.use_motion_conditioning = use_motion_conditioning

        kernel_size = history_window + 1
        self.temporal_context = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            groups=1,
            bias=False,
        )
        self.token_gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.context_gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.motion_gate = (
            nn.Linear(hidden_dim, num_experts, bias=False)
            if use_motion_conditioning
            else None
        )

    def causal_context(self, tokens: Tensor) -> Tensor:
        """Compute causal temporal context per spatial slot.

        Args:
            tokens: Tensor shaped ``[batch, time, slots, hidden]``.
        """

        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [batch, time, slots, hidden]")

        batch, time, slots, hidden = tokens.shape
        series = tokens.permute(0, 2, 3, 1).reshape(batch * slots, hidden, time)
        padded = F.pad(series, (self.history_window, 0))
        context = self.temporal_context(padded)
        return context.reshape(batch, slots, hidden, time).permute(0, 3, 1, 2)

    def forward(self, tokens: Tensor, motion_embeddings: Tensor | None = None) -> RouterOutput:
        """Return router probabilities and top-k experts.

        Args:
            tokens: Tensor shaped ``[batch, time, slots, hidden]``.
            motion_embeddings: Optional tensor with the same shape as tokens.
        """

        context = self.causal_context(tokens)
        logits = self.token_gate(tokens) + self.context_gate(context)
        if self.motion_gate is not None and motion_embeddings is not None:
            logits = logits + self.motion_gate(motion_embeddings)

        probs = torch.softmax(logits / self.temperature, dim=-1)
        topk_scores, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
        entropy = -(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log()).sum(dim=-1)
        return RouterOutput(
            logits=logits,
            probs=probs,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            context=context,
            entropy=entropy,
        )
