"""Minimal graph encoder for COCO-17 pose embeddings.

The model consumes one pose per sample as a graph:

  * nodes: 17 COCO joints
  * node features: normalized x, normalized y, confidence
  * edges: ``COCO_SKELETON`` plus self-loops

It intentionally uses only vanilla PyTorch, not PyTorch Geometric. The output
is an L2-normalized embedding suitable for triplet or contrastive losses.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets.common import COCO_SKELETON, NUM_JOINTS


def build_normalized_adjacency(
    num_joints: int = NUM_JOINTS,
    edges: Sequence[Tuple[int, int]] = COCO_SKELETON,
) -> torch.Tensor:
    """Build symmetric D^-1/2 A D^-1/2 adjacency with self-loops."""
    A = torch.eye(num_joints, dtype=torch.float32)
    for a, b in edges:
        if not (0 <= a < num_joints and 0 <= b < num_joints):
            raise ValueError(f"edge {(a, b)} is out of range for {num_joints} joints")
        A[a, b] = 1.0
        A[b, a] = 1.0

    degree = A.sum(dim=1).clamp_min(1.0)
    inv_sqrt = degree.pow(-0.5)
    return inv_sqrt[:, None] * A * inv_sqrt[None, :]


class SimpleGraphConv(nn.Module):
    """Graph convolution layer: aggregate with A_norm, then apply a Linear."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, adjacency_norm: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected x shape (B, V, C), got {tuple(x.shape)}")
        x = torch.einsum("ij,bjc->bic", adjacency_norm, x)
        return self.linear(x)


class PoseGNNEncoder(nn.Module):
    """COCO-17 pose graph encoder returning L2-normalized embeddings."""

    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        num_joints: int = NUM_JOINTS,
        in_features: int = 3,
    ) -> None:
        super().__init__()
        self.num_joints = int(num_joints)
        self.in_features = int(in_features)
        self.embedding_dim = int(embedding_dim)

        A_norm = build_normalized_adjacency(num_joints=self.num_joints)
        self.register_buffer("adjacency_norm", A_norm, persistent=False)

        self.conv1 = SimpleGraphConv(self.in_features, 64)
        self.conv2 = SimpleGraphConv(64, 128)
        self.conv3 = SimpleGraphConv(128, 128)
        self.dropout = nn.Dropout(float(dropout))
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.embedding_dim),
        )

    def _block(self, conv: SimpleGraphConv, x: torch.Tensor) -> torch.Tensor:
        x = conv(x, self.adjacency_norm)
        x = F.relu(x, inplace=True)
        return self.dropout(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode poses.

        Args:
            x: tensor of shape ``(B, 17, 3)``.

        Returns:
            tensor of shape ``(B, embedding_dim)`` with unit L2 norm.
        """
        if x.dim() != 3 or x.shape[1] != self.num_joints or x.shape[2] != self.in_features:
            raise ValueError(
                f"expected input shape (B, {self.num_joints}, {self.in_features}), "
                f"got {tuple(x.shape)}"
            )
        x = x.float()
        x = self._block(self.conv1, x)
        x = self._block(self.conv2, x)
        x = self._block(self.conv3, x)
        x = x.mean(dim=1)
        z = self.projection(x)
        return F.normalize(z, p=2, dim=-1)
