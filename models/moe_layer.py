"""Homogeneous top-k SwiGLU experts with event-cache bypass."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cache import EventTokenCache
from .router import RouterOutput, TemporallyAwareRouter


class LoRALinear(nn.Module):
    """Linear layer with an optional trainable low-rank adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 0,
        alpha: float = 1.0,
        bias: bool = False,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.base = nn.Linear(in_features, out_features, bias=bias)
        if freeze_base:
            self.base.weight.requires_grad_(False)
            if self.base.bias is not None:
                self.base.bias.requires_grad_(False)

        if rank > 0:
            self.lora_a = nn.Parameter(torch.empty(rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank if self.rank > 0 else 0.0

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        if self.rank > 0:
            adapter = F.linear(F.linear(x, self.lora_a), self.lora_b)
            y = y + adapter * self.scaling
        return y


class SwiGLUExpert(nn.Module):
    """LLaMA-style homogeneous FFN expert."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = LoRALinear(
            hidden_dim, ffn_dim, rank=lora_rank, alpha=lora_alpha, freeze_base=freeze_base
        )
        self.up_proj = LoRALinear(
            hidden_dim, ffn_dim, rank=lora_rank, alpha=lora_alpha, freeze_base=freeze_base
        )
        self.down_proj = LoRALinear(
            ffn_dim, hidden_dim, rank=lora_rank, alpha=lora_alpha, freeze_base=freeze_base
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def lora_b_vector(self) -> Tensor:
        matrices = [
            module.lora_b.flatten()
            for module in (self.gate_proj, self.up_proj, self.down_proj)
            if module.lora_b is not None
        ]
        if matrices:
            return torch.cat(matrices)
        return self.up_proj.base.weight.flatten()


@dataclass
class MoEForwardStats:
    executed_tokens: int
    cached_tokens: int
    total_tokens: int


@dataclass
class MoEForwardOutput:
    hidden_states: Tensor
    router: RouterOutput
    stats: MoEForwardStats


class MicroMoELayer(nn.Module):
    """Top-k homogeneous MoE layer with temporal cache bypass."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        router_history_window: int = 2,
        router_temperature: float = 1.0,
        use_motion_conditioning: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = False,
        dense_routing: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.dense_routing = dense_routing
        self.router = TemporallyAwareRouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            history_window=router_history_window,
            temperature=router_temperature,
            use_motion_conditioning=use_motion_conditioning,
        )
        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(
                    hidden_dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    freeze_base=freeze_base,
                )
                for _ in range(num_experts)
            ]
        )

    def _run_experts(self, flat_tokens: Tensor, flat_router: RouterOutput) -> Tensor:
        if self.dense_routing:
            return torch.stack([expert(flat_tokens) for expert in self.experts], dim=0).mean(dim=0)

        flat_topk_idx = flat_router.topk_indices.reshape(-1, self.top_k)
        flat_topk_scores = flat_router.topk_scores.reshape(-1, self.top_k)
        output = torch.zeros_like(flat_tokens)

        for expert_idx, expert in enumerate(self.experts):
            selected = flat_topk_idx == expert_idx
            if not selected.any():
                continue
            token_rows = selected.any(dim=-1)
            expert_input = flat_tokens[token_rows]
            expert_output = expert(expert_input)
            weights = (selected[token_rows].float() * flat_topk_scores[token_rows]).sum(
                dim=-1
            )
            output[token_rows] = output[token_rows] + expert_output * weights.unsqueeze(-1)

        return output

    def _slice_router(self, router: RouterOutput, frame_index: int) -> RouterOutput:
        return RouterOutput(
            logits=router.logits[:, frame_index : frame_index + 1],
            probs=router.probs[:, frame_index : frame_index + 1],
            topk_indices=router.topk_indices[:, frame_index : frame_index + 1],
            topk_scores=router.topk_scores[:, frame_index : frame_index + 1],
            context=router.context[:, frame_index : frame_index + 1],
            entropy=router.entropy[:, frame_index : frame_index + 1],
        )

    def forward(
        self,
        tokens: Tensor,
        motion_embeddings: Tensor | None = None,
        motion_confidence: Tensor | None = None,
        cache: EventTokenCache | None = None,
    ) -> MoEForwardOutput:
        """Apply the MoE branch and residual connection.

        Args:
            tokens: Tensor shaped ``[batch, time, slots, hidden]``.
            motion_embeddings: Optional tensor with the same shape as tokens.
            motion_confidence: Optional ``[batch, time, slots]`` confidence.
            cache: Optional event cache for visual tokens.
        """

        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [batch, time, slots, hidden]")
        batch, time, slots, hidden = tokens.shape
        if hidden != self.hidden_dim:
            raise ValueError(f"expected hidden_dim={self.hidden_dim}, got {hidden}")

        router = self.router(tokens, motion_embeddings=motion_embeddings)
        ffn_outputs = torch.zeros_like(tokens)
        executed_tokens = 0
        cached_tokens = 0

        if cache is None or motion_confidence is None:
            flat_tokens = tokens.reshape(batch * time * slots, hidden)
            ffn_outputs = self._run_experts(flat_tokens, router).view_as(tokens)
            executed_tokens = batch * time * slots
            return MoEForwardOutput(
                hidden_states=tokens + ffn_outputs,
                router=router,
                stats=MoEForwardStats(
                    executed_tokens=executed_tokens,
                    cached_tokens=cached_tokens,
                    total_tokens=batch * time * slots,
                ),
            )

        cache.ensure_shape(batch, slots, hidden, tokens.device, tokens.dtype)
        for frame_idx in range(time):
            frame_tokens = tokens[:, frame_idx]
            confidence = motion_confidence[:, frame_idx]
            cached_mask = cache.readable_mask(confidence)
            compute_mask = ~cached_mask

            if cached_mask.any():
                cached_branch = cache.read(cached_mask)
                ffn_outputs[:, frame_idx] = torch.where(
                    cached_mask.unsqueeze(-1), cached_branch, ffn_outputs[:, frame_idx]
                )
                cached_tokens += int(cached_mask.sum().item())

            if compute_mask.any():
                frame_router = self._slice_router(router, frame_idx)
                flat_compute_mask = compute_mask.reshape(-1)
                flat_frame_tokens = frame_tokens.reshape(batch * slots, hidden)
                active_tokens = flat_frame_tokens[flat_compute_mask]
                active_router = RouterOutput(
                    logits=frame_router.logits.reshape(batch * slots, -1)[
                        flat_compute_mask
                    ],
                    probs=frame_router.probs.reshape(batch * slots, -1)[
                        flat_compute_mask
                    ],
                    topk_indices=frame_router.topk_indices.reshape(batch * slots, -1)[
                        flat_compute_mask
                    ],
                    topk_scores=frame_router.topk_scores.reshape(batch * slots, -1)[
                        flat_compute_mask
                    ],
                    context=frame_router.context.reshape(batch * slots, -1)[
                        flat_compute_mask
                    ],
                    entropy=frame_router.entropy.reshape(batch * slots)[
                        flat_compute_mask
                    ],
                )
                active_branch = self._run_experts(active_tokens, active_router)
                flat_branch = torch.zeros(
                    batch * slots,
                    hidden,
                    device=tokens.device,
                    dtype=tokens.dtype,
                )
                flat_branch[flat_compute_mask] = active_branch
                flat_branch = flat_branch.reshape(batch, slots, hidden)
                ffn_outputs[:, frame_idx] = torch.where(
                    compute_mask.unsqueeze(-1),
                    flat_branch,
                    ffn_outputs[:, frame_idx],
                )
                cache.write(flat_branch, compute_mask)
                executed_tokens += int(compute_mask.sum().item())

        return MoEForwardOutput(
            hidden_states=tokens + ffn_outputs,
            router=router,
            stats=MoEForwardStats(
                executed_tokens=executed_tokens,
                cached_tokens=cached_tokens,
                total_tokens=batch * time * slots,
            ),
        )
