"""Pose heads.

Includes the deconv head used by Simple Baseline and the final 1x1 conv
head used by HRNet. Same input/output contract so downstream code is
indifferent to which model produced the heatmaps.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DeconvHead(nn.Module):
    """Three upsampling deconv blocks followed by a final 1x1 conv."""

    def __init__(
        self,
        in_channels: int,
        num_joints: int = 17,
        num_deconv_layers: int = 3,
        deconv_channels: List[int] | None = None,
        deconv_kernels: List[int] | None = None,
        final_kernel: int = 1,
    ) -> None:
        super().__init__()
        deconv_channels = deconv_channels or [256, 256, 256]
        deconv_kernels = deconv_kernels or [4, 4, 4]
        assert len(deconv_channels) == num_deconv_layers
        assert len(deconv_kernels) == num_deconv_layers

        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out, k in zip(deconv_channels, deconv_kernels):
            pad, out_pad = self._kernel_padding(k)
            layers += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=2, padding=pad, output_padding=out_pad, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.deconvs = nn.Sequential(*layers)
        self.final = nn.Conv2d(c_in, num_joints, kernel_size=final_kernel, padding=final_kernel // 2)

    @staticmethod
    def _kernel_padding(k: int):
        if k == 4:
            return 1, 0
        if k == 3:
            return 1, 1
        if k == 2:
            return 0, 0
        raise ValueError(f"unsupported deconv kernel size: {k}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.deconvs(x))


class FinalConvHead(nn.Module):
    """1x1 conv that maps HRNet's hi-res feature map to J heatmaps."""

    def __init__(self, in_channels: int, num_joints: int = 17) -> None:
        super().__init__()
        self.final = nn.Conv2d(in_channels, num_joints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(x)
