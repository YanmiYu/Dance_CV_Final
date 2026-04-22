"""Simple Baseline pose network -- FROM SCRATCH.

``docs/project_decisions.md`` forbids loading any pretrained weights. The
backbone here is a small ResNet-like stack implemented by us.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .heads import DeconvHead
from .layers import BasicBlock, kaiming_init


class _SimpleResLikeBackbone(nn.Module):
    def __init__(self, stem_channels: int = 64, stages: List[Dict] | None = None) -> None:
        super().__init__()
        stages = stages or [
            {"channels": 64, "blocks": 2, "stride": 1},
            {"channels": 128, "blocks": 2, "stride": 2},
            {"channels": 256, "blocks": 2, "stride": 2},
            {"channels": 512, "blocks": 2, "stride": 2},
        ]
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layers = nn.ModuleList()
        in_c = stem_channels
        for s in stages:
            self.layers.append(self._make_stage(in_c, s["channels"], s["blocks"], s["stride"]))
            in_c = s["channels"]
        self.out_channels = in_c

    @staticmethod
    def _make_stage(in_c: int, out_c: int, n_blocks: int, stride: int) -> nn.Sequential:
        blocks = [BasicBlock(in_c, out_c, stride=stride)]
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock(out_c, out_c, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        return x


class SimpleBaselinePose(nn.Module):
    """Top-down pose model: backbone -> deconv head -> J heatmaps."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        assert not config.get("pretrained", False), (
            "Simple Baseline must NOT use pretrained weights. "
            "See docs/project_decisions.md."
        )
        bb = config.get("backbone", {})
        head = config.get("head", {})
        self.backbone = _SimpleResLikeBackbone(
            stem_channels=int(bb.get("stem_channels", 64)),
            stages=bb.get("stages"),
        )
        self.head = DeconvHead(
            in_channels=self.backbone.out_channels,
            num_joints=int(head.get("num_joints", 17)),
            num_deconv_layers=int(head.get("num_deconv_layers", 3)),
            deconv_channels=head.get("deconv_channels") or [256, 256, 256],
            deconv_kernels=head.get("deconv_kernels") or [4, 4, 4],
            final_kernel=int(head.get("final_kernel", 1)),
        )
        kaiming_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
