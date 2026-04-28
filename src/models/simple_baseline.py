"""SimpleBaseline pose estimator (Xiao et al., ECCV 2018).

Architecture:
    ResNet-like backbone (trained from scratch, no pretrained weights)
    → DeconvHead (3x upsampling deconv layers)
    → 17 heatmaps (one per COCO joint)

Input:  (B, 3, 256, 192)  RGB image, person-cropped
Output: (B, 17, 64, 48)   heatmap per joint
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .heads import DeconvHead
from .layers import BasicBlock, init_weights


class _Backbone(nn.Module):
    """Small ResNet-like backbone built from scratch."""

    # (out_channels, num_blocks, stride)
    STAGES = [
        (64,  2, 1),
        (128, 2, 2),
        (256, 2, 2),
        (512, 2, 2),
    ]

    def __init__(self) -> None:
        super().__init__()
        # stem: 256x192 -> 64x48
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        layers = []
        in_c = 64
        for out_c, n_blocks, stride in self.STAGES:
            stage = [BasicBlock(in_c, out_c, stride=stride)]
            for _ in range(1, n_blocks):
                stage.append(BasicBlock(out_c, out_c))
            layers.append(nn.Sequential(*stage))
            in_c = out_c
        self.layers = nn.ModuleList(layers)
        self.out_channels = in_c  # 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        return x  # (B, 512, 8, 6) for 256x192 input


class SimpleBaseline(nn.Module):
    """Full pose model: backbone + deconv head."""

    def __init__(self, num_joints: int = 17) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.head = DeconvHead(
            in_channels=self.backbone.out_channels,
            num_joints=num_joints,
        )
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 256, 192) normalized RGB image
        Returns:
            heatmaps: (B, 17, 64, 48)
        """
        return self.head(self.backbone(x))

    @torch.no_grad()
    def predict_keypoints(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference and return (x, y) coords in heatmap space.

        Returns:
            keypoints: (B, 17, 2)  — (x, y) in [0, W_hm] x [0, H_hm]
        """
        heatmaps = self.forward(x)          # (B, 17, 64, 48)
        B, J, H, W = heatmaps.shape
        flat = heatmaps.view(B, J, -1)      # (B, 17, H*W)
        idx  = flat.argmax(dim=2)           # (B, 17)
        x_coord = (idx % W).float()
        y_coord = (idx // W).float()
        return torch.stack([x_coord, y_coord], dim=2)  # (B, 17, 2)
