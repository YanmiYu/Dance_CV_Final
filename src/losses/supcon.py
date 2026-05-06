"""Supervised contrastive loss using cosine similarity."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be (B, D); got {tuple(embeddings.shape)}")
        if labels.dim() != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"labels must be (B,) matching embeddings rows; "
                f"got labels={tuple(labels.shape)}, emb={tuple(embeddings.shape)}"
            )
        B = embeddings.shape[0]
        if B < 2:
            return embeddings.sum() * 0.0

        z = F.normalize(embeddings, p=2, dim=-1)
        sim = z @ z.T / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        diag = torch.eye(B, dtype=torch.bool, device=z.device)
        labels = labels.view(-1)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~diag

        exp_sim = torch.exp(sim) * (~diag).to(sim.dtype)
        denom = exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_prob = sim - torch.log(denom)

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not torch.any(valid):
            return embeddings.sum() * 0.0

        pos_log_prob = (log_prob * pos_mask.to(log_prob.dtype)).sum(dim=1)
        mean_pos_log_prob = pos_log_prob[valid] / pos_count[valid].to(log_prob.dtype)
        return -mean_pos_log_prob.mean()


@torch.no_grad()
def positive_negative_cosine_means(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    z = F.normalize(embeddings, p=2, dim=-1)
    sim = z @ z.T
    B = z.shape[0]
    diag = torch.eye(B, dtype=torch.bool, device=z.device)
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~diag
    neg = (labels.unsqueeze(0) != labels.unsqueeze(1)) & ~diag
    mean_pos = float(sim[pos].mean()) if pos.any() else 0.0
    mean_neg = float(sim[neg].mean()) if neg.any() else 0.0
    return mean_pos, mean_neg
