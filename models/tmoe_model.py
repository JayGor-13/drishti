"""T-MoE Anti-UAV detector scaffold."""

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
    """Configuration for the Anti-UAV detection model.

    The AAAI architecture targets ``hidden_dim=1024``, ``ffn_dim=4096``, a
    28x28 patch grid for 448x448 frames, 8 experts, top-2 routing, and a
    9-frame temporal window. Tests and smoke runs use smaller dimensions while
    preserving the same contracts.
    """

    hidden_dim: int = 128
    ffn_dim: int = 256
    num_experts: int = 8
    top_k: int = 2
    num_layers: int = 2
    num_attention_heads: int = 4
    patch_grid_size: int = 4
    image_channels: int = 3
    motion_dim: int = 64
    router_history_window: int = 0
    router_temperature: float = 1.0
    use_motion_conditioning: bool = True
    cache_threshold: float = 0.15
    lora_rank: int = 0
    lora_alpha: float = 1.0
    num_classes: int = 2
    dense_routing: bool = False
    use_temporal_cache: bool = True
    max_frames: int = 128


@dataclass
class TMoEDetectionOutput:
    class_logits: Tensor
    boxes: Tensor
    video_tokens: Tensor
    semantic_tokens: Tensor
    motion_embeddings: Tensor
    motion_confidence: Tensor
    router_outputs: list[RouterOutput] = field(default_factory=list)
    moe_stats: list[MoEForwardStats] = field(default_factory=list)


class LocateAnythingPatchEncoder(nn.Module):
    """Small LocateAnything-compatible patch encoder.

    LocateAnything-3B is the intended semantic backbone from the architecture
    plan. This local module keeps the same patch-token contract without pulling
    a 3B parameter dependency into tests or smoke runs.
    """

    def __init__(
        self,
        hidden_dim: int,
        patch_grid_size: int,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.patch_grid_size = patch_grid_size
        stem_dim = max(16, hidden_dim // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.SiLU(),
            nn.Conv2d(stem_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [batch, time, channels, height, width]")
        batch, time, channels, height, width = frames.shape
        flat = frames.reshape(batch * time, channels, height, width)
        features = self.stem(flat)
        pooled = nn.functional.adaptive_avg_pool2d(
            features, (self.patch_grid_size, self.patch_grid_size)
        )
        tokens = pooled.flatten(2).transpose(1, 2)
        return tokens.reshape(batch, time, self.patch_grid_size**2, -1)


class AntiUAVDetectionHead(nn.Module):
    """Patch-wise drone/no-drone classifier and normalized box regressor."""

    def __init__(self, hidden_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.box_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        features = self.norm(tokens)
        class_logits = self.classifier(features)
        boxes = torch.sigmoid(self.box_regressor(features))
        return class_logits, boxes


class VideoMoEBlock(nn.Module):
    """Self-attention followed by motion-conditioned MoE."""

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
            use_motion_conditioning=config.use_motion_conditioning,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            dense_routing=config.dense_routing,
        )

    def forward(
        self,
        tokens: Tensor,
        motion_embeddings: Tensor,
        motion_confidence: Tensor,
        cache: EventTokenCache | None,
    ) -> tuple[Tensor, RouterOutput, MoEForwardStats]:
        batch, time, patches, hidden = tokens.shape
        sequence = tokens.reshape(batch, time * patches, hidden)
        attn_input = self.attn_norm(sequence)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = (sequence + attn_output).reshape(batch, time, patches, hidden)

        moe_input = self.moe_norm(tokens)
        moe_output = self.moe(
            moe_input,
            motion_embeddings=motion_embeddings,
            motion_confidence=motion_confidence,
            cache=cache,
        )
        tokens = tokens + (moe_output.hidden_states - moe_input)
        return tokens, moe_output.router, moe_output.stats


class TMoEAntiDroneDetector(nn.Module):
    """Sparse, motion-conditioned detector for Anti-UAV RGB videos."""

    def __init__(self, config: TMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.semantic_encoder = LocateAnythingPatchEncoder(
            hidden_dim=config.hidden_dim,
            patch_grid_size=config.patch_grid_size,
            in_channels=config.image_channels,
        )
        self.motion_encoder = KinematicMotionEncoder(
            hidden_dim=config.hidden_dim,
            motion_dim=config.motion_dim,
            patch_grid_size=config.patch_grid_size,
            in_channels=config.image_channels,
            confidence_bias=-2.0,
        )
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, 1, config.patch_grid_size**2, config.hidden_dim)
        )
        self.temporal_pos = nn.Parameter(torch.zeros(1, config.max_frames, 1, config.hidden_dim))
        self.blocks = nn.ModuleList([VideoMoEBlock(config) for _ in range(config.num_layers)])
        self.caches = nn.ModuleList(
            [EventTokenCache(threshold=config.cache_threshold) for _ in range(config.num_layers)]
        )
        self.detection_head = AntiUAVDetectionHead(config.hidden_dim, config.num_classes)

    def reset_caches(self) -> None:
        for cache in self.caches:
            cache.reset()

    def set_dense_routing(self, dense_routing: bool) -> None:
        self.config.dense_routing = dense_routing
        for block in self.blocks:
            block.moe.dense_routing = dense_routing

    def add_expert_noise(self, std: float = 0.01) -> None:
        """Break expert symmetry after dense upcycling."""

        if std <= 0:
            return
        with torch.no_grad():
            for block in self.blocks:
                for expert in block.moe.experts:
                    for parameter in expert.parameters():
                        parameter.add_(torch.randn_like(parameter) * std)

    def encode_video(self, frames: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        semantic = self.semantic_encoder(frames)
        motion = self.motion_encoder(frames)
        time = semantic.shape[1]
        if time > self.temporal_pos.shape[1]:
            raise ValueError("number of frames exceeds temporal position capacity")
        tokens = semantic + motion.embeddings + self.spatial_pos + self.temporal_pos[:, :time]
        return tokens, semantic, motion.embeddings, motion.confidence

    def forward(
        self,
        frames: Tensor,
        reset_cache: bool = False,
        use_cache: bool | None = None,
    ) -> TMoEDetectionOutput:
        if reset_cache:
            self.reset_caches()

        use_cache = self.config.use_temporal_cache if use_cache is None else use_cache
        video_tokens, semantic_tokens, motion_embeddings, motion_confidence = self.encode_video(frames)
        router_outputs: list[RouterOutput] = []
        moe_stats: list[MoEForwardStats] = []
        for block, cache in zip(self.blocks, self.caches):
            video_tokens, router, stats = block(
                video_tokens,
                motion_embeddings=motion_embeddings,
                motion_confidence=motion_confidence,
                cache=cache if use_cache else None,
            )
            router_outputs.append(router)
            moe_stats.append(stats)

        class_logits, boxes = self.detection_head(video_tokens)
        return TMoEDetectionOutput(
            class_logits=class_logits,
            boxes=boxes,
            video_tokens=video_tokens,
            semantic_tokens=semantic_tokens,
            motion_embeddings=motion_embeddings,
            motion_confidence=motion_confidence,
            router_outputs=router_outputs,
            moe_stats=moe_stats,
        )
