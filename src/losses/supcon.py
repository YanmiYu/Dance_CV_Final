"""Supervised contrastive loss (Khosla et al., 2020).

We use the cosine-similarity formulation since the encoder outputs
L2-normalised embeddings:

    L_i = - 1/|P(i)| * sum_{p in P(i)} log( exp(sim(z_i, z_p) / T)
                                            / sum_{a != i} exp(sim(z_i, z_a) / T) )

where ``P(i)`` is the set of in-batch samples that share the anchor's
label. Anchors with no positives contribute zero loss (no NaN).
"""
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
            return embeddings.sum() * 0.0  # keeps autograd graph but yields 0

        z = F.normalize(embeddings, p=2, dim=-1)
        sim = z @ z.T / self.temperature                            # (B, B)

        # numerically stable log-softmax denom (subtract row-max).
        sim_max = sim.max(dim=1, keepdim=True).values.detach()
        sim = sim - sim_max

        diag_mask = torch.eye(B, dtype=torch.bool, device=z.device)
        labels = labels.view(-1)
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)        # (B, B)
        pos_mask = pos_mask & ~diag_mask                              # exclude self

        # log-prob over all non-self entries (denominator = sum over a != i).
        exp_sim = torch.exp(sim) * (~diag_mask).to(sim.dtype)
        denom = exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_prob = sim - torch.log(denom)                            # (B, B)

        pos_count = pos_mask.sum(dim=1)                              # (B,)
        valid = pos_count > 0
        if not torch.any(valid):
            return embeddings.sum() * 0.0

        pos_log_prob = (log_prob * pos_mask.to(log_prob.dtype)).sum(dim=1)
        mean_pos_log_prob = pos_log_prob[valid] / pos_count[valid].to(log_prob.dtype)
        return -mean_pos_log_prob.mean()


@torch.no_grad()
def positive_negative_cosine_means(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float]:
    """Return mean cosine similarity for positive vs negative pairs."""
    z = F.normalize(embeddings, p=2, dim=-1)
    sim = z @ z.T
    B = z.shape[0]
    diag = torch.eye(B, dtype=torch.bool, device=z.device)
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~diag
    neg = (labels.unsqueeze(0) != labels.unsqueeze(1)) & ~diag
    mean_pos = float(sim[pos].mean()) if pos.any() else 0.0
    mean_neg = float(sim[neg].mean()) if neg.any() else 0.0
    return mean_pos, mean_neg
