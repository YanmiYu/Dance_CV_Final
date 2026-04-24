"""Single-person detector -- trained FROM SCRATCH.

This is a CenterNet-style anchor-free detector specialized for the single-
person assumption defined in ``docs/project_decisions.md`` section 2
(single-person, near-fixed camera, short clips). It is the only detector
this project ships, and it is trained from AIST++ 2D keypoint labels
(section 6) via the ``bbox_xyxy`` already present on every row.

No pretrained weights are loaded. The backbone is our own ResNet-like
stack (same primitives as :class:`SimpleBaselinePose`), and all weights
are Kaiming-initialized. See ``docs/project_decisions.md`` section 1.

Outputs (channel-wise, all at ``input_size / 4`` resolution):
    0 : person-center heatmap (unbounded logits; caller applies sigmoid).
    1 : bbox width, normalized to ``[0, 1]`` of the detector input width.
    2 : bbox height, normalized to ``[0, 1]`` of the detector input height.

Decoding a prediction is therefore::

    cy, cx = argmax(sigmoid(heatmap))
    bw, bh = size_w[cy, cx] * W_in, size_h[cy, cx] * H_in
    bbox = (cx*4 - bw/2, cy*4 - bh/2, cx*4 + bw/2, cy*4 + bh/2)
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .layers import BasicBlock, kaiming_init


class _DetectorBackbone(nn.Module):
    """Small ResNet-like backbone; stride-32 output (same as SimpleBaseline)."""

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


class _UpsampleHead(nn.Module):
    """Three deconv blocks -> stride-4 feature map -> center + size conv heads.

    Keeping the center head and the size head as separate final convs makes
    it trivial to apply different losses / weightings without slicing a
    packed tensor in the loss function.
    """

    def __init__(
        self,
        in_channels: int,
        deconv_channels: List[int] | None = None,
        deconv_kernels: List[int] | None = None,
    ) -> None:
        super().__init__()
        deconv_channels = deconv_channels or [256, 256, 256]
        deconv_kernels = deconv_kernels or [4, 4, 4]
        assert len(deconv_channels) == len(deconv_kernels)

        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out, k in zip(deconv_channels, deconv_kernels):
            pad, out_pad = self._kernel_padding(k)
            layers += [
                nn.ConvTranspose2d(
                    c_in, c_out, kernel_size=k, stride=2,
                    padding=pad, output_padding=out_pad, bias=False,
                ),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.deconvs = nn.Sequential(*layers)
        self.center = nn.Conv2d(c_in, 1, kernel_size=1)
        self.size = nn.Conv2d(c_in, 2, kernel_size=1)

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
        feat = self.deconvs(x)
        center = self.center(feat)
        size = self.size(feat)
        return torch.cat([center, size], dim=1)


class PersonDetector(nn.Module):
    """Heatmap-based single-person detector.

    ``config`` keys:
      - ``pretrained`` (must be false; enforced)
      - ``backbone``  : stem_channels, stages
      - ``head``      : deconv_channels, deconv_kernels
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        if config.get("pretrained", False):
            raise AssertionError(
                "PersonDetector must NOT use pretrained weights. "
                "See docs/project_decisions.md section 1."
            )
        bb = config.get("backbone", {})
        head = config.get("head", {})
        stages = bb.get("stages")
        self.backbone = _DetectorBackbone(
            stem_channels=int(bb.get("stem_channels", 64)),
            stages=stages,
        )
        deconv_kernels = head.get("deconv_kernels") or [4, 4, 4]
        self.head = _UpsampleHead(
            in_channels=self.backbone.out_channels,
            deconv_channels=head.get("deconv_channels") or [256, 256, 256],
            deconv_kernels=deconv_kernels,
        )
        # Stem (conv stride 2 + maxpool stride 2 = /4) * product(stage strides)
        # divided by 2^(number of deconv layers).
        backbone_stride = 4
        for s in (stages or [
            {"stride": 1}, {"stride": 2}, {"stride": 2}, {"stride": 2},
        ]):
            backbone_stride *= int(s.get("stride", 1))
        self._output_stride = max(1, backbone_stride // (2 ** len(deconv_kernels)))
        kaiming_init(self)

    @property
    def output_stride(self) -> int:
        """Input -> output downsampling factor (4 with default config)."""
        return self._output_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, 3, H/s, W/s). Channel 0 = center logits, 1-2 = size (w, h)."""
        return self.head(self.backbone(x))
