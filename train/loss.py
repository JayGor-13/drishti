"""Loss functions for Anti-UAV detection, routing, and temporal consistency."""

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
    gamma_ortho: float = 0.0
    lambda_box: float = 5.0
    lambda_giou: float = 2.0


def focal_classification_loss(
    class_logits: Tensor,
    class_targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Multi-class focal loss for patch-wise drone/no-drone labels."""

    if class_logits.shape[:-1] != class_targets.shape:
        raise ValueError("class_targets must match class_logits without class dimension")
    num_classes = class_logits.shape[-1]
    flat_logits = class_logits.reshape(-1, num_classes)
    flat_targets = class_targets.reshape(-1).long()
    ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    pt = torch.exp(-ce)
    alpha_t = torch.where(flat_targets > 0, alpha, 1.0 - alpha).to(ce.dtype)
    return (alpha_t * (1.0 - pt).pow(gamma) * ce).mean()


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, width, height = boxes.unbind(dim=-1)
    half_w = width / 2.0
    half_h = height / 2.0
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)


def generalized_box_iou_aligned(pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
    """Aligned GIoU for normalized ``cx, cy, w, h`` boxes."""

    pred = cxcywh_to_xyxy(pred_boxes).clamp(0.0, 1.0)
    target = cxcywh_to_xyxy(target_boxes).clamp(0.0, 1.0)

    inter_top_left = torch.maximum(pred[:, :2], target[:, :2])
    inter_bottom_right = torch.minimum(pred[:, 2:], target[:, 2:])
    inter_wh = (inter_bottom_right - inter_top_left).clamp_min(0.0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    pred_wh = (pred[:, 2:] - pred[:, :2]).clamp_min(0.0)
    target_wh = (target[:, 2:] - target[:, :2]).clamp_min(0.0)
    pred_area = pred_wh[:, 0] * pred_wh[:, 1]
    target_area = target_wh[:, 0] * target_wh[:, 1]
    union = (pred_area + target_area - inter_area).clamp_min(1e-7)
    iou = inter_area / union

    enclosing_top_left = torch.minimum(pred[:, :2], target[:, :2])
    enclosing_bottom_right = torch.maximum(pred[:, 2:], target[:, 2:])
    enclosing_wh = (enclosing_bottom_right - enclosing_top_left).clamp_min(0.0)
    enclosing_area = (enclosing_wh[:, 0] * enclosing_wh[:, 1]).clamp_min(1e-7)
    return iou - (enclosing_area - union) / enclosing_area


def detection_loss(
    class_logits: Tensor,
    pred_boxes: Tensor,
    class_targets: Tensor,
    box_targets: Tensor,
    box_mask: Tensor,
    weights: TMoELossWeights = TMoELossWeights(),
) -> dict[str, Tensor]:
    """Patch detection loss from the architecture plan.

    ``class_targets`` is ``0`` for background and ``1`` for drone. Boxes are
    normalized ``cx, cy, w, h`` values assigned to the patch containing each
    drone center.
    """

    cls = focal_classification_loss(class_logits, class_targets)
    if pred_boxes.shape != box_targets.shape:
        raise ValueError("pred_boxes and box_targets must have the same shape")
    if box_mask.shape != pred_boxes.shape[:-1]:
        raise ValueError("box_mask must match pred_boxes without box dimension")

    if box_mask.any():
        pred_pos = pred_boxes[box_mask]
        target_pos = box_targets[box_mask]
        l1 = F.smooth_l1_loss(pred_pos, target_pos)
        giou = generalized_box_iou_aligned(pred_pos, target_pos)
        giou_loss = (1.0 - giou).mean()
    else:
        l1 = pred_boxes.new_zeros(())
        giou_loss = pred_boxes.new_zeros(())

    total = cls + weights.lambda_box * l1 + weights.lambda_giou * giou_loss
    return {
        "det": total,
        "cls": cls,
        "box_l1": l1,
        "giou": giou_loss,
    }


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


def semantic_alignment(semantic_tokens: Tensor, temperature: float = 0.1) -> Tensor:
    """Cosine-similarity patch alignment for adjacent frames."""

    if semantic_tokens.ndim != 4:
        raise ValueError("semantic_tokens must have shape [batch, time, patches, hidden]")
    batch, time, patches, _ = semantic_tokens.shape
    if time < 2:
        return semantic_tokens.new_empty(batch, 0, patches, patches)

    current = F.normalize(semantic_tokens[:, :-1], dim=-1)
    nxt = F.normalize(semantic_tokens[:, 1:], dim=-1)
    similarity = torch.einsum("btih,btjh->btij", current, nxt)
    return torch.softmax(similarity / temperature, dim=-1)


def cfcr_loss(
    router_probs: Tensor,
    motion_confidence: Tensor,
    alignment: Tensor | None = None,
) -> Tensor:
    """Cross-frame consistent routing loss with optional patch alignment."""

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


def total_antiuav_loss(
    class_logits: Tensor,
    pred_boxes: Tensor,
    class_targets: Tensor,
    box_targets: Tensor,
    box_mask: Tensor,
    router_probs: Tensor,
    motion_confidence: Tensor,
    experts: nn.ModuleList | list[SwiGLUExpert],
    semantic_tokens: Tensor | None = None,
    weights: TMoELossWeights = TMoELossWeights(),
    beta_cfcr: float | None = None,
) -> dict[str, Tensor]:
    det_parts = detection_loss(
        class_logits,
        pred_boxes,
        class_targets,
        box_targets,
        box_mask,
        weights=weights,
    )
    aux = load_balancing_loss(router_probs)
    alignment = semantic_alignment(semantic_tokens) if semantic_tokens is not None else None
    cfcr = cfcr_loss(router_probs, motion_confidence, alignment=alignment)
    ortho = orthogonalization_loss(experts)
    beta = weights.beta_cfcr if beta_cfcr is None else beta_cfcr
    total = det_parts["det"] + weights.alpha_aux * aux + beta * cfcr
    total = total + weights.gamma_ortho * ortho
    return {
        "loss": total,
        **det_parts,
        "aux": aux,
        "cfcr": cfcr,
        "ortho": ortho,
    }


total_tmoe_loss = total_antiuav_loss
