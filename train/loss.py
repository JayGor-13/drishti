"""Loss functions for routing, temporal consistency, and expert diversity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.moe_layer import SwiGLUExpert


@dataclass
class TMoELossWeights:
    alpha_aux: float = 0.01
    beta_cfcr: float = 0.1
    gamma_ortho: float = 0.01


def autoregressive_loss(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Causal next-token cross entropy."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocab]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [batch, sequence]")
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def load_balancing_loss(router_probs: Tensor) -> Tensor:
    """Switch-style load balancing using soft probabilities and top-1 density."""

    if router_probs.ndim < 2:
        raise ValueError("router_probs must end with expert dimension")
    num_experts = router_probs.shape[-1]
    flat = router_probs.reshape(-1, num_experts)
    top1 = flat.argmax(dim=-1)
    density = F.one_hot(top1, num_classes=num_experts).float().mean(dim=0)
    density_proxy = flat.mean(dim=0)
    return num_experts * torch.sum(density * density_proxy)


def _kl_divergence(p: Tensor, q: Tensor) -> Tensor:
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)
    return (p * (p.log() - q.log())).sum(dim=-1)


def js_divergence(p: Tensor, q: Tensor) -> Tensor:
    midpoint = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, midpoint) + 0.5 * _kl_divergence(q, midpoint)


def cfcr_loss(
    router_probs: Tensor,
    motion_confidence: Tensor,
    alignment: Tensor | None = None,
) -> Tensor:
    """Cross-frame consistent routing loss with optional patch alignment.

    Args:
        router_probs: ``[batch, time, patches, experts]``.
        motion_confidence: ``[batch, time, patches]``.
        alignment: Optional ``[batch, time - 1, patches, patches]`` matrix.
            If omitted, identity alignment is used.
    """

    if router_probs.ndim != 4:
        raise ValueError("router_probs must have shape [batch, time, patches, experts]")
    if motion_confidence.shape != router_probs.shape[:3]:
        raise ValueError("motion_confidence must have shape [batch, time, patches]")
    batch, time, patches, _ = router_probs.shape
    if time < 2:
        return router_probs.new_zeros(())

    current = router_probs[:, :-1]
    nxt = router_probs[:, 1:]
    if alignment is None:
        eye = torch.eye(patches, device=router_probs.device, dtype=router_probs.dtype)
        alignment = eye.view(1, 1, patches, patches).expand(batch, time - 1, -1, -1)

    if alignment.shape != (batch, time - 1, patches, patches):
        raise ValueError("alignment must have shape [batch, time - 1, patches, patches]")

    pairwise = js_divergence(current.unsqueeze(3), nxt.unsqueeze(2))
    static_weight = (1.0 - motion_confidence[:, :-1]).clamp(0.0, 1.0).unsqueeze(-1)
    weighted = alignment * static_weight * pairwise
    normalizer = (alignment * static_weight).sum().clamp_min(1.0)
    return weighted.sum() / normalizer


def routing_entropy(router_probs: Tensor) -> Tensor:
    probs = router_probs.clamp_min(1e-8)
    return -(probs * probs.log()).sum(dim=-1).mean()


def expert_lora_similarity(experts: nn.ModuleList | list[SwiGLUExpert]) -> Tensor:
    """Mean absolute pairwise cosine similarity between expert adapter vectors."""

    vectors = [expert.lora_b_vector().float() for expert in experts]
    if len(vectors) < 2:
        return vectors[0].new_zeros(()) if vectors else torch.tensor(0.0)
    sims = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if vectors[i].norm() == 0 or vectors[j].norm() == 0:
                sims.append(vectors[i].new_zeros(()))
            else:
                sims.append(F.cosine_similarity(vectors[i], vectors[j], dim=0).abs())
    return torch.stack(sims).mean()


def orthogonalization_loss(experts: nn.ModuleList | list[SwiGLUExpert]) -> Tensor:
    return expert_lora_similarity(experts)


def total_tmoe_loss(
    logits: Tensor,
    labels: Tensor,
    router_probs: Tensor,
    motion_confidence: Tensor,
    experts: nn.ModuleList | list[SwiGLUExpert],
    weights: TMoELossWeights = TMoELossWeights(),
) -> dict[str, Tensor]:
    ar = autoregressive_loss(logits, labels)
    aux = load_balancing_loss(router_probs)
    cfcr = cfcr_loss(router_probs, motion_confidence)
    ortho = orthogonalization_loss(experts)
    total = ar + weights.alpha_aux * aux + weights.beta_cfcr * cfcr
    total = total + weights.gamma_ortho * ortho
    return {
        "loss": total,
        "ar": ar,
        "aux": aux,
        "cfcr": cfcr,
        "ortho": ortho,
    }
