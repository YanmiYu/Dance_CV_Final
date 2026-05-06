"""Temporal wrapper around the old :class:`PoseGNNEncoder`."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets.common import NUM_JOINTS
from src.models.pose_gnn import PoseGNNEncoder


class PoseGNNTemporalEncoder(nn.Module):
    """Encode ``(B, T, 17, C)`` pose windows into L2-normalized embeddings."""

    def __init__(
        self,
        frame_embedding_dim: int = 128,
        out_embedding_dim: int = 128,
        dropout: float = 0.1,
        in_features: int = 3,
        pooling: str = "mean",
        frame_encoder: Optional[PoseGNNEncoder] = None,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "max"}:
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling
        self.in_features = int(in_features)
        self.out_embedding_dim = int(out_embedding_dim)
        self.frame_embedding_dim = int(frame_embedding_dim)
        self.frame_encoder = frame_encoder or PoseGNNEncoder(
            embedding_dim=self.frame_embedding_dim,
            dropout=dropout,
            in_features=self.in_features,
        )
        hidden = max(self.out_embedding_dim, self.frame_embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.frame_embedding_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.out_embedding_dim),
        )

    @property
    def embedding_dim(self) -> int:
        return self.out_embedding_dim

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 4 or x.shape[2] != NUM_JOINTS or x.shape[3] != self.in_features:
            raise ValueError(
                f"expected input shape (B, T, {NUM_JOINTS}, {self.in_features}), "
                f"got {tuple(x.shape)}"
            )
        B, T = x.shape[:2]
        x = x.float().reshape(B * T, NUM_JOINTS, self.in_features)
        z = self.frame_encoder(x).reshape(B, T, self.frame_embedding_dim)

        if mask is not None:
            if mask.shape != (B, T):
                raise ValueError(f"expected mask shape ({B}, {T}), got {tuple(mask.shape)}")
            if self.pooling == "max":
                z = z.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                z = z.max(dim=1).values
                z = torch.where(torch.isinf(z), torch.zeros_like(z), z)
            else:
                weights = mask.to(z.dtype).unsqueeze(-1)
                z = (z * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            z = z.max(dim=1).values if self.pooling == "max" else z.mean(dim=1)

        return F.normalize(self.projection(z), p=2, dim=-1)

    def frame_encoder_state_dict(self) -> dict:
        """Return old GNN frame encoder weights for legacy checkpoint loading."""
        return self.frame_encoder.state_dict()
