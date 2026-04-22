"""HRNet-W32 style backbone + pose head, implemented from scratch.

Follows the HRNet paper's high-to-low + low-to-high fusion pattern but
DOES NOT load any external pretrained weights. See
``docs/project_decisions.md`` section 1.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import FinalConvHead
from .layers import BasicBlock, Bottleneck, kaiming_init


class _HRBranch(nn.Module):
    """A single-resolution stream of ``num_blocks`` BasicBlocks."""

    def __init__(self, channels: int, num_blocks: int) -> None:
        super().__init__()
        blocks = [BasicBlock(channels, channels, stride=1) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _HRFuseModule(nn.Module):
    """Multi-resolution fusion: every branch gets every other branch's feature."""

    def __init__(self, channels: List[int]) -> None:
        super().__init__()
        self.num_branches = len(channels)
        self.channels = channels
        self.fuse_layers = nn.ModuleList()
        for i in range(self.num_branches):
            row = nn.ModuleList()
            for j in range(self.num_branches):
                if i == j:
                    row.append(nn.Identity())
                elif j > i:
                    # higher-res j -> upsample to match i
                    row.append(
                        nn.Sequential(
                            nn.Conv2d(channels[j], channels[i], kernel_size=1, bias=False),
                            nn.BatchNorm2d(channels[i]),
                        )
                    )
                else:
                    # lower-res j (i.e. higher resolution in HRNet's convention where
                    # index-0 is highest) -> downsample with conv strided by 2 * (i - j)
                    downs: list[nn.Module] = []
                    for k in range(i - j):
                        out_c = channels[i] if k == (i - j - 1) else channels[j]
                        downs += [
                            nn.Conv2d(channels[j] if k == 0 else channels[j], out_c, kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(out_c),
                        ]
                        if k < (i - j - 1):
                            downs.append(nn.ReLU(inplace=True))
                    row.append(nn.Sequential(*downs))
            self.fuse_layers.append(row)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        outs: list[torch.Tensor] = []
        for i in range(self.num_branches):
            acc = None
            for j in range(self.num_branches):
                y = self.fuse_layers[i][j](xs[j])
                if j > i:
                    # upsample to xs[i] spatial size
                    y = F.interpolate(y, size=xs[i].shape[-2:], mode="bilinear", align_corners=False)
                acc = y if acc is None else acc + y
            outs.append(self.relu(acc))
        return outs


class _HRStage(nn.Module):
    def __init__(self, num_blocks: List[int], channels: List[int], num_modules: int) -> None:
        super().__init__()
        assert len(num_blocks) == len(channels)
        self.num_modules = num_modules
        self.modules_list = nn.ModuleList()
        for _ in range(num_modules):
            branches = nn.ModuleList([_HRBranch(c, b) for c, b in zip(channels, num_blocks)])
            fuse = _HRFuseModule(channels)
            self.modules_list.append(nn.ModuleDict({"branches": branches, "fuse": fuse}))

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        for mod in self.modules_list:
            xs = [br(x) for br, x in zip(mod["branches"], xs)]
            xs = mod["fuse"](xs)
        return xs


class _HRTransition(nn.Module):
    """Create a NEW low-resolution branch from the previous stage's branches."""

    def __init__(self, prev_channels: List[int], new_channels: List[int]) -> None:
        super().__init__()
        assert len(new_channels) == len(prev_channels) + 1
        self.prev_n = len(prev_channels)
        self.transitions = nn.ModuleList()
        for i in range(len(new_channels)):
            if i < len(prev_channels):
                # keep branch; if channels differ, project with a 3x3
                if prev_channels[i] != new_channels[i]:
                    self.transitions.append(
                        nn.Sequential(
                            nn.Conv2d(prev_channels[i], new_channels[i], kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(new_channels[i]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.transitions.append(nn.Identity())
            else:
                # brand new lower-res branch built from the last previous branch, downsampled
                src = prev_channels[-1]
                self.transitions.append(
                    nn.Sequential(
                        nn.Conv2d(src, new_channels[i], kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(new_channels[i]),
                        nn.ReLU(inplace=True),
                    )
                )

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        outs: list[torch.Tensor] = []
        for i, tr in enumerate(self.transitions):
            if i < self.prev_n:
                outs.append(tr(xs[i]))
            else:
                outs.append(tr(xs[-1]))
        return outs


class HRNetPose(nn.Module):
    """HRNet-W32-style top-down pose network."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        assert not config.get("pretrained", False), (
            "HRNet must NOT load pretrained weights. See docs/project_decisions.md."
        )

        stem_c = int(config.get("stem", {}).get("out_channels", 64))
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_c, stem_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_c),
            nn.ReLU(inplace=True),
        )

        # Stage 1: bottlenecks
        s1 = config.get("stage1", {"num_blocks": 4, "channels": 64})
        self.stage1 = self._make_bottleneck_stage(
            in_c=stem_c,
            planes=int(s1.get("channels", 64)),
            num_blocks=int(s1.get("num_blocks", 4)),
        )
        s1_out = int(s1.get("channels", 64)) * Bottleneck.expansion

        # Stage 2
        s2 = config.get("stage2", {"num_branches": 2, "num_blocks": [4, 4], "channels": [32, 64], "num_modules": 1})
        self.trans_12 = _HRTransition(prev_channels=[s1_out], new_channels=list(s2["channels"]))
        self.stage2 = _HRStage(
            num_blocks=list(s2["num_blocks"]),
            channels=list(s2["channels"]),
            num_modules=int(s2.get("num_modules", 1)),
        )

        # Stage 3
        s3 = config.get("stage3", {"num_branches": 3, "num_blocks": [4, 4, 4], "channels": [32, 64, 128], "num_modules": 4})
        self.trans_23 = _HRTransition(prev_channels=list(s2["channels"]), new_channels=list(s3["channels"]))
        self.stage3 = _HRStage(
            num_blocks=list(s3["num_blocks"]),
            channels=list(s3["channels"]),
            num_modules=int(s3.get("num_modules", 4)),
        )

        # Stage 4
        s4 = config.get("stage4", {"num_branches": 4, "num_blocks": [4, 4, 4, 4], "channels": [32, 64, 128, 256], "num_modules": 3})
        self.trans_34 = _HRTransition(prev_channels=list(s3["channels"]), new_channels=list(s4["channels"]))
        self.stage4 = _HRStage(
            num_blocks=list(s4["num_blocks"]),
            channels=list(s4["channels"]),
            num_modules=int(s4.get("num_modules", 3)),
        )

        head = config.get("head", {"in_channels": 32, "num_joints": 17})
        self.head = FinalConvHead(
            in_channels=int(head.get("in_channels", s4["channels"][0])),
            num_joints=int(head.get("num_joints", 17)),
        )

        kaiming_init(self)

    @staticmethod
    def _make_bottleneck_stage(in_c: int, planes: int, num_blocks: int) -> nn.Sequential:
        blocks: list[nn.Module] = [Bottleneck(in_c, planes, stride=1)]
        for _ in range(1, num_blocks):
            blocks.append(Bottleneck(planes * Bottleneck.expansion, planes, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        xs = self.trans_12([x])
        xs = self.stage2(xs)
        xs = self.trans_23(xs)
        xs = self.stage3(xs)
        xs = self.trans_34(xs)
        xs = self.stage4(xs)
        # The highest-resolution branch (index 0) feeds the head.
        return self.head(xs[0])
