from __future__ import annotations

import torch
from torch import Tensor, nn


class CausalTemporalFusion(nn.Module):
    """Causal spatio-temporal transformer over per-crop feature histories."""

    def __init__(
        self,
        feature_dim: int = 257,
        out_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 5,
        num_sources: int = 5,
    ) -> None:
        super().__init__()
        if out_dim % nhead != 0:
            raise ValueError("out_dim must be divisible by nhead")
        self.max_seq_len = max_seq_len
        self.input_proj = nn.Linear(feature_dim, out_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, out_dim))
        self.center_proj = nn.Linear(2, out_dim)
        self.source_embed = nn.Embedding(num_sources, out_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        sequence: Tensor,
        centers_seq: Tensor | None = None,
        source_labels_seq: Tensor | None = None,
        padding_mask: Tensor | None = None,
        source_labels: Tensor | None = None,
    ) -> Tensor:
        if sequence.ndim != 4:
            raise ValueError(f"Expected [B, T, K, D], got {tuple(sequence.shape)}")
        batch, time, num_crops, dim = sequence.shape
        if time < 1:
            raise ValueError("Temporal sequence must contain at least one frame")

        # Backward-compatible shorthand for one source label per crop.
        if source_labels_seq is None and source_labels is not None:
            if source_labels.shape != (batch, num_crops):
                raise ValueError(f"Expected source_labels [B, K], got {tuple(source_labels.shape)}")
            source_labels_seq = source_labels.unsqueeze(1).expand(-1, time, -1)

        if time > self.max_seq_len:
            sequence = sequence[:, -self.max_seq_len :]
            if centers_seq is not None:
                centers_seq = centers_seq[:, -self.max_seq_len :]
            if source_labels_seq is not None:
                source_labels_seq = source_labels_seq[:, -self.max_seq_len :]
            if padding_mask is not None:
                padding_mask = padding_mask[:, -self.max_seq_len :]
            time = self.max_seq_len

        if centers_seq is not None and centers_seq.shape != (batch, time, num_crops, 2):
            raise ValueError(f"Expected centers_seq [B, T, K, 2], got {tuple(centers_seq.shape)}")
        if source_labels_seq is not None and source_labels_seq.shape != (batch, time, num_crops):
            raise ValueError(f"Expected source_labels_seq [B, T, K], got {tuple(source_labels_seq.shape)}")

        pad_mask = self._flatten_padding_mask(padding_mask, batch, time, num_crops, sequence.device)
        valid_time = time
        x = sequence.permute(0, 2, 1, 3).reshape(batch * num_crops, time, dim)

        if time < self.max_seq_len:
            pad_len = self.max_seq_len - time
            sequence_pad = sequence[:, -1:].expand(-1, pad_len, -1, -1)
            sequence = torch.cat([sequence, sequence_pad], dim=1)
            if centers_seq is not None:
                centers_seq = torch.cat([centers_seq, centers_seq[:, -1:].expand(-1, pad_len, -1, -1)], dim=1)
            if source_labels_seq is not None:
                source_labels_seq = torch.cat(
                    [source_labels_seq, source_labels_seq[:, -1:].expand(-1, pad_len, -1)], dim=1
                )
            generated_mask = torch.zeros(
                batch * num_crops,
                self.max_seq_len,
                dtype=torch.bool,
                device=sequence.device,
            )
            generated_mask[:, valid_time:] = True
            if pad_mask is not None:
                generated_mask[:, :valid_time] |= pad_mask
            pad_mask = generated_mask
            time = self.max_seq_len

        x = sequence.permute(0, 2, 1, 3).reshape(batch * num_crops, time, dim)
        x = self.input_proj(x) + self.pos_embed[:, :time]
        if centers_seq is not None:
            centers_flat = centers_seq.permute(0, 2, 1, 3).reshape(batch * num_crops, time, 2)
            x = x + self.center_proj(centers_flat)
        if source_labels_seq is not None:
            sources_flat = source_labels_seq.permute(0, 2, 1).reshape(batch * num_crops, time)
            sources_flat = sources_flat.clamp(0, self.source_embed.num_embeddings - 1)
            x = x + self.source_embed(sources_flat)

        if pad_mask is not None and (~pad_mask).sum(dim=1).eq(0).any():
            raise ValueError("Each temporal crop sequence must contain at least one unmasked frame")
        mask = torch.triu(torch.ones(time, time, device=x.device, dtype=torch.bool), diagonal=1)
        encoded = self.encoder(x, mask=mask, src_key_padding_mask=pad_mask)

        if pad_mask is None:
            present_indices = torch.full(
                (batch * num_crops,), valid_time - 1, dtype=torch.long, device=x.device
            )
        else:
            valid = ~pad_mask
            positions = torch.arange(time, device=x.device).view(1, time).expand_as(valid)
            present_indices = positions.masked_fill(~valid, -1).amax(dim=1).clamp_min(0)
        present = self.norm(encoded[torch.arange(encoded.shape[0], device=x.device), present_indices])
        return present.reshape(batch, num_crops, -1)

    @staticmethod
    def _flatten_padding_mask(
        padding_mask: Tensor | None,
        batch: int,
        time: int,
        num_crops: int,
        device: torch.device,
    ) -> Tensor | None:
        if padding_mask is None:
            return None
        padding_mask = padding_mask.to(device=device, dtype=torch.bool)
        if padding_mask.shape == (batch, time):
            return padding_mask.unsqueeze(1).expand(-1, num_crops, -1).reshape(batch * num_crops, time)
        if padding_mask.shape == (batch, time, num_crops):
            return padding_mask.permute(0, 2, 1).reshape(batch * num_crops, time)
        raise ValueError(
            f"Expected padding_mask [B, T] or [B, T, K], got {tuple(padding_mask.shape)}"
        )
