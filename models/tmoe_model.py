"""End-to-end Micro-MoE video-language scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .cache import EventTokenCache
from .moe_layer import MicroMoELayer, MoEForwardStats
from .motion_encoder import KinematicMotionEncoder
from .router import RouterOutput


@dataclass
class TMoEConfig:
    """Configuration for the CPU-runnable research scaffold."""

    vocab_size: int = 32000
    hidden_dim: int = 128
    ffn_dim: int = 256
    num_experts: int = 8
    top_k: int = 2
    num_layers: int = 2
    num_attention_heads: int = 4
    patch_grid_size: int = 4
    image_channels: int = 3
    motion_dim: int = 64
    router_history_window: int = 2
    router_temperature: float = 1.0
    cache_threshold: float = 0.05
    lora_rank: int = 0
    lora_alpha: float = 1.0
    max_text_length: int = 256


@dataclass
class TMoEModelOutput:
    logits: Tensor
    next_token_logits: Tensor
    video_tokens: Tensor
    motion_embeddings: Tensor
    motion_confidence: Tensor
    multimodal_sequence: Tensor
    router_outputs: list[RouterOutput] = field(default_factory=list)
    moe_stats: list[MoEForwardStats] = field(default_factory=list)


class PatchVisualEncoder(nn.Module):
    """Small visual patch encoder with CLIP-like output contract."""

    def __init__(
        self,
        hidden_dim: int,
        patch_grid_size: int,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.patch_grid_size = patch_grid_size
        stem_dim = max(1, hidden_dim // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(stem_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 5:
            raise ValueError(
                "frames must have shape [batch, time, channels, height, width]"
            )
        batch, time, channels, height, width = frames.shape
        flat = frames.reshape(batch * time, channels, height, width)
        features = self.stem(flat)
        pooled = nn.functional.adaptive_avg_pool2d(
            features, (self.patch_grid_size, self.patch_grid_size)
        )
        tokens = pooled.flatten(2).transpose(1, 2)
        return tokens.reshape(batch, time, self.patch_grid_size**2, -1)


class VideoMoEBlock(nn.Module):
    """Transformer-style video block followed by T-CHE MoE."""

    def __init__(self, config: TMoEConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_dim)
        self.attn = nn.MultiheadAttention(
            config.hidden_dim,
            num_heads=config.num_attention_heads,
            batch_first=True,
        )
        self.moe_norm = nn.LayerNorm(config.hidden_dim)
        self.moe = MicroMoELayer(
            hidden_dim=config.hidden_dim,
            ffn_dim=config.ffn_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            router_history_window=config.router_history_window,
            router_temperature=config.router_temperature,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
        )

    def forward(
        self,
        tokens: Tensor,
        motion_embeddings: Tensor,
        motion_confidence: Tensor,
        cache: EventTokenCache,
    ) -> tuple[Tensor, RouterOutput, MoEForwardStats]:
        batch, time, slots, hidden = tokens.shape
        sequence = tokens.reshape(batch, time * slots, hidden)
        attn_input = self.attn_norm(sequence)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = (sequence + attn_output).reshape(batch, time, slots, hidden)

        moe_input = self.moe_norm(tokens)
        moe_output = self.moe(
            moe_input,
            motion_embeddings=motion_embeddings,
            motion_confidence=motion_confidence,
            cache=cache,
        )
        tokens = tokens + (moe_output.hidden_states - moe_input)
        return tokens, moe_output.router, moe_output.stats


class TMoELLaVAMicro(nn.Module):
    """Minimal trainable skeleton for T-MoE-LLaVA 2.0."""

    def __init__(self, config: TMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.visual_encoder = PatchVisualEncoder(
            hidden_dim=config.hidden_dim,
            patch_grid_size=config.patch_grid_size,
            in_channels=config.image_channels,
        )
        self.motion_encoder = KinematicMotionEncoder(
            hidden_dim=config.hidden_dim,
            motion_dim=config.motion_dim,
            patch_grid_size=config.patch_grid_size,
            in_channels=config.image_channels,
        )
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, 1, config.patch_grid_size**2, config.hidden_dim)
        )
        self.temporal_pos = nn.Parameter(torch.zeros(1, 512, 1, config.hidden_dim))
        self.blocks = nn.ModuleList([VideoMoEBlock(config) for _ in range(config.num_layers)])
        self.caches = nn.ModuleList(
            [EventTokenCache(threshold=config.cache_threshold) for _ in range(config.num_layers)]
        )
        self.text_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.text_pos = nn.Parameter(torch.zeros(1, config.max_text_length, config.hidden_dim))
        self.fusion_norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def reset_caches(self) -> None:
        for cache in self.caches:
            cache.reset()

    def encode_video(self, frames: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        visual = self.visual_encoder(frames)
        motion = self.motion_encoder(frames)
        time = visual.shape[1]
        if time > self.temporal_pos.shape[1]:
            raise ValueError("number of frames exceeds temporal position capacity")
        tokens = visual + motion.embeddings + self.spatial_pos + self.temporal_pos[:, :time]
        return tokens, motion.embeddings, motion.confidence

    def build_multimodal_sequence(
        self,
        video_tokens: Tensor,
        motion_embeddings: Tensor,
        input_ids: Tensor,
    ) -> Tensor:
        """Concatenate visual, motion, and text tokens for inspection/training hooks."""

        text_tokens = self.text_embedding(input_ids)
        if text_tokens.shape[1] > self.config.max_text_length:
            raise ValueError("input_ids length exceeds config.max_text_length")
        text_tokens = text_tokens + self.text_pos[:, : text_tokens.shape[1]]
        batch = video_tokens.shape[0]
        visual_flat = video_tokens.reshape(batch, -1, self.config.hidden_dim)
        motion_flat = motion_embeddings.reshape(batch, -1, self.config.hidden_dim)
        return torch.cat([visual_flat, motion_flat, text_tokens], dim=1)

    def forward(
        self,
        frames: Tensor,
        input_ids: Tensor,
        reset_cache: bool = False,
    ) -> TMoEModelOutput:
        if reset_cache:
            self.reset_caches()

        video_tokens, motion_embeddings, motion_confidence = self.encode_video(frames)
        router_outputs: list[RouterOutput] = []
        moe_stats: list[MoEForwardStats] = []
        for block, cache in zip(self.blocks, self.caches):
            video_tokens, router, stats = block(
                video_tokens,
                motion_embeddings=motion_embeddings,
                motion_confidence=motion_confidence,
                cache=cache,
            )
            router_outputs.append(router)
            moe_stats.append(stats)

        text_tokens = self.text_embedding(input_ids)
        if text_tokens.shape[1] > self.config.max_text_length:
            raise ValueError("input_ids length exceeds config.max_text_length")
        text_tokens = text_tokens + self.text_pos[:, : text_tokens.shape[1]]

        video_context = video_tokens.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        fused_text = self.fusion_norm(text_tokens + video_context)
        logits = self.lm_head(fused_text)
        multimodal_sequence = self.build_multimodal_sequence(
            video_tokens, motion_embeddings, input_ids
        )
        return TMoEModelOutput(
            logits=logits,
            next_token_logits=logits[:, -1],
            video_tokens=video_tokens,
            motion_embeddings=motion_embeddings,
            motion_confidence=motion_confidence,
            multimodal_sequence=multimodal_sequence,
            router_outputs=router_outputs,
            moe_stats=moe_stats,
        )
