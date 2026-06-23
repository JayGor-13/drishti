"""Stage 4: temporal fusion transformer."""

from __future__ import annotations

import math

from torch import Tensor, nn
import torch

from .config import DRISHTIConfig
from .types import TemporalFusionOutput


class FlexibleSelfAttention(nn.Module):
    """Self-attention that supports d_model=257 with nhead=4."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        self.num_heads = num_heads
        self.head_dim = math.ceil(embed_dim / num_heads)
        self.inner_dim = self.head_dim * num_heads
        self.qkv = nn.Linear(embed_dim, self.inner_dim * 3)
        self.out = nn.Linear(self.inner_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        batch, length, _ = tokens.shape
        qkv = self.qkv(tokens)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn, v).transpose(1, 2).reshape(batch, length, self.inner_dim)
        return self.out(context)


class TemporalFusionLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = FlexibleSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.dropout(self.attn(self.attn_norm(tokens)))
        tokens = tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))
        return tokens


class TemporalFusion(nn.Module):
    """Fuse five temporal crop-feature steps into eight 256-D crop features."""

    def __init__(self, config: DRISHTIConfig) -> None:
        super().__init__()
        self.input_dim = config.temporal_input_dim
        self.output_dim = config.feature_dim
        self.layers = nn.ModuleList(
            [
                TemporalFusionLayer(
                    embed_dim=config.temporal_input_dim,
                    num_heads=config.temporal_heads,
                    ffn_dim=config.temporal_ffn_dim,
                    dropout=config.temporal_dropout,
                )
                for _ in range(config.temporal_layers)
            ]
        )
        self.proj = nn.Linear(config.temporal_input_dim, config.feature_dim)

    def forward(self, features: Tensor) -> TemporalFusionOutput:
        if features.ndim != 4:
            raise ValueError("features must have shape [batch, time, crops, feature_dim+1]")
        batch, time, crops, dim = features.shape
        if dim != self.input_dim:
            raise ValueError(f"expected temporal input dim {self.input_dim}, got {dim}")
        tokens = features.permute(0, 2, 1, 3).reshape(batch * crops, time, dim)
        for layer in self.layers:
            tokens = layer(tokens)
        last_tokens = tokens[:, -1].reshape(batch, crops, dim)
        fused = self.proj(last_tokens)
        return TemporalFusionOutput(
            fused_features=fused,
            temporal_tokens=tokens.reshape(batch, crops, time, dim),
        )
