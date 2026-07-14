from __future__ import annotations

import torch
from torch import Tensor, nn


class CausalTemporalFusion(nn.Module):
    """Causal transformer over per-crop feature histories."""

    def __init__(
        self,
        feature_dim: int = 257,
        out_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 5,
    ) -> None:
        super().__init__()
        if out_dim % nhead != 0:
            raise ValueError("out_dim must be divisible by nhead")
        self.max_seq_len = max_seq_len
        self.input_proj = nn.Linear(feature_dim, out_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, out_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 4:
            raise ValueError(f"Expected [B, T, K, D], got {tuple(sequence.shape)}")
        batch, time, num_crops, dim = sequence.shape
        if time > self.max_seq_len:
            sequence = sequence[:, -self.max_seq_len :]
            time = self.max_seq_len
        x = sequence.permute(0, 2, 1, 3).reshape(batch * num_crops, time, dim)
        x = self.input_proj(x) + self.pos_embed[:, -time:]
        mask = torch.triu(torch.ones(time, time, device=x.device, dtype=torch.bool), diagonal=1)
        encoded = self.encoder(x, mask=mask)
        present = self.norm(encoded[:, -1])
        return present.reshape(batch, num_crops, -1)
