"""Deconvolutional head that upsamples backbone features into joint heatmaps."""
from __future__ import annotations

import torch
import torch.nn as nn


class DeconvHead(nn.Module):
    """
    3 deconv layers (each 2x upsample) followed by a 1x1 conv.
    Input:  (B, in_channels, H/32, W/32)
    Output: (B, num_joints, H/4, W/4)
    """

    def __init__(
        self,
        in_channels: int,
        num_joints: int = 17,
        deconv_channels: list[int] | None = None,
        deconv_kernels: list[int] | None = None,
    ) -> None:
        super().__init__()
        deconv_channels = deconv_channels or [256, 256, 256]
        deconv_kernels  = deconv_kernels  or [4, 4, 4]

        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out, k in zip(deconv_channels, deconv_kernels):
            padding, output_padding = self._padding(k)
            layers += [
                nn.ConvTranspose2d(
                    c_in, c_out, kernel_size=k, stride=2,
                    padding=padding, output_padding=output_padding, bias=False,
                ),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out

        self.deconvs = nn.Sequential(*layers)
        self.final   = nn.Conv2d(c_in, num_joints, kernel_size=1)

    @staticmethod
    def _padding(k: int) -> tuple[int, int]:
        if k == 4:
            return 1, 0
        if k == 3:
            return 1, 1
        if k == 2:
            return 0, 0
        raise ValueError(f"unsupported kernel size: {k}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.deconvs(x))
