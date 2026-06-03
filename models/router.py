"""Modality-aware top-k router for motion-conditioned MoE detection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class RouterOutput:
    """Routing probabilities and top-k selections."""

    logits: Tensor
    probs: Tensor
    topk_indices: Tensor
    topk_scores: Tensor
    context: Tensor
    entropy: Tensor


class ModalityAwareRouter(nn.Module):
    """Route each patch from semantic and motion tokens.

    The architecture plan defines routing as ``softmax(W_r [x_i^t concat
    delta_i^t])``. Text tokens are not part of the AAAI workshop detector, but
    keeping ``motion_embeddings`` optional preserves the same modality-aware
    behavior: missing motion is represented by zeros.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
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
        self.temperature = temperature
        self.use_motion_conditioning = use_motion_conditioning
        router_input_dim = hidden_dim * 2 if use_motion_conditioning else hidden_dim
        self.gate = nn.Linear(router_input_dim, num_experts, bias=False)

    def forward(self, tokens: Tensor, motion_embeddings: Tensor | None = None) -> RouterOutput:
        """Return router probabilities and top-k experts.

        Args:
            tokens: Tensor shaped ``[batch, time, patches, hidden]``.
            motion_embeddings: Optional tensor with the same shape as tokens.
        """

        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [batch, time, patches, hidden]")
        if tokens.shape[-1] != self.hidden_dim:
            raise ValueError(f"expected hidden_dim={self.hidden_dim}, got {tokens.shape[-1]}")

        if self.use_motion_conditioning:
            if motion_embeddings is None:
                motion_embeddings = torch.zeros_like(tokens)
            if motion_embeddings.shape != tokens.shape:
                raise ValueError("motion_embeddings must match tokens when provided")
            router_input = torch.cat([tokens, motion_embeddings], dim=-1)
        else:
            router_input = tokens

        logits = self.gate(router_input)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        topk_scores, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
        entropy = -(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log()).sum(dim=-1)
        return RouterOutput(
            logits=logits,
            probs=probs,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            context=torch.zeros_like(tokens),
            entropy=entropy,
        )


class TemporallyAwareRouter(ModalityAwareRouter):
    """Backward-compatible alias for older imports.

    The current architecture uses direct semantic-motion concatenation rather
    than a causal temporal convolution. ``history_window`` is accepted so older
    constructors keep working while the behavior follows the Anti-UAV plan.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        history_window: int = 2,
        temperature: float = 1.0,
        use_motion_conditioning: bool = True,
    ) -> None:
        self.history_window = history_window
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            temperature=temperature,
            use_motion_conditioning=use_motion_conditioning,
        )
