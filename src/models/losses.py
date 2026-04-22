"""Heatmap losses."""
from __future__ import annotations

import torch
import torch.nn as nn


class JointMSELoss(nn.Module):
    """Per-joint MSE on heatmaps with visibility masking.

    ``use_target_weight=True`` multiplies each joint's MSE by the supplied
    ``target_weight`` (e.g. 0 when the joint is not annotated).
    """

    def __init__(self, use_target_weight: bool = True) -> None:
        super().__init__()
        self.use_target_weight = use_target_weight
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(
        self,
        pred_heatmaps: torch.Tensor,      # (B, J, H, W)
        target_heatmaps: torch.Tensor,    # (B, J, H, W)
        target_weight: torch.Tensor,      # (B, J, 1)
    ) -> torch.Tensor:
        B, J, H, W = pred_heatmaps.shape
        pred = pred_heatmaps.reshape(B, J, -1)
        target = target_heatmaps.reshape(B, J, -1)
        if self.use_target_weight:
            w = target_weight.reshape(B, J, 1)
            pred = pred * w
            target = target * w
        return self.criterion(pred, target)
