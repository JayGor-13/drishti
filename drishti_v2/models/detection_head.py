from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """Per-crop objectness, box regression, and sub-pixel offset head."""

    def __init__(self, feature_dim: int = 256, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.objectness_head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 1))
        self.box_stem = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.xy_head = nn.Linear(hidden_dim, 2)
        self.wh_head = nn.Linear(hidden_dim, 2)
        self.offset_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 2),
            nn.Tanh(),
        )

    def forward(self, features: Tensor, box_features: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """``box_features`` (optional) feeds the box and offset branches; objectness always uses ``features``."""

        box_in = features if box_features is None else box_features
        stem = self.box_stem(box_in)
        xy = torch.sigmoid(self.xy_head(stem))
        wh = F.softplus(self.wh_head(stem)) + 1e-4
        boxes = torch.cat([xy, wh], dim=-1)
        offsets = self.offset_head(box_in) * 0.5
        return self.objectness_head(features), boxes, offsets


class LayerMix(nn.Module):
    """Learned weighted mix of several layers' features (ELMo-style scalar mix).

    The head no longer has to read only the last module's output: it gets
    gamma * sum_i softmax(a)_i * LayerNorm_i(x_i), and the learned weights show which layers it uses.
    """

    def __init__(self, num_inputs: int, dim: int) -> None:
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_inputs)])
        self.logits = nn.Parameter(torch.zeros(num_inputs))
        self.gamma = nn.Parameter(torch.ones(1))

    def weights(self) -> Tensor:
        return torch.softmax(self.logits, dim=0)

    def forward(self, inputs: list[Tensor]) -> Tensor:
        w = self.weights()
        return self.gamma * sum(w[i] * norm(x) for i, (norm, x) in enumerate(zip(self.norms, inputs)))
