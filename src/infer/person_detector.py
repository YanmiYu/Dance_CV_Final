"""Runtime wrapper around a trained PersonDetector checkpoint.

Feeds a full BGR frame in, gets back an ``(x1, y1, x2, y2)`` bbox in
original-image coordinates. Used as a drop-in replacement for
:class:`src.infer.motion_crop.MotionCropper` whenever a detector
checkpoint is available. See ``docs/project_decisions.md`` sections 1
and 2 (no pretrained weights; single-person).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from src.models.person_detector import PersonDetector
from src.utils.config import load_yaml


def _check_internal_ckpt(path: str) -> None:
    normalized = os.path.normpath(path)
    assert "data/processed/" in normalized.replace("\\", "/"), (
        f"Detector checkpoint must live under data/processed/ (ours). Got: {path}. "
        f"See docs/project_decisions.md section 1."
    )


class PersonDetectorRuntime:
    """Load a trained :class:`PersonDetector` and run it on raw frames."""

    def __init__(
        self,
        model_config_path: str | Path,
        ckpt_path: str | Path,
        *,
        device: Optional[str] = None,
        input_size: Tuple[int, int] = (256, 256),
        score_threshold: float = 0.15,
    ) -> None:
        _check_internal_ckpt(str(ckpt_path))
        model_cfg = load_yaml(model_config_path)
        if model_cfg.get("pretrained", False):
            raise SystemExit(
                "pretrained=true is forbidden. See docs/project_decisions.md section 1."
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PersonDetector(model_cfg).to(self.device).eval()
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict):
            if "input_hw" in state:
                input_size = tuple(state["input_hw"])
            state = state.get("model", state)
        self.model.load_state_dict(state, strict=False)
        self.input_size = tuple(input_size)
        self.score_threshold = float(score_threshold)
        self.stride = int(self.model.output_stride)

    @torch.no_grad()
    def propose(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """Return ``(x1, y1, x2, y2)`` in original-image coords, or ``None``."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        src_h, src_w = frame_bgr.shape[:2]
        H, W = self.input_size
        resized = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        x = (
            torch.from_numpy(resized.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        preds = self.model(x)
        center = torch.sigmoid(preds[0, 0])
        flat = center.flatten()
        score, idx = flat.max(dim=0)
        if float(score.item()) < self.score_threshold:
            return None
        out_h, out_w = center.shape
        cy = int(idx.item()) // out_w
        cx = int(idx.item()) % out_w
        cx_in = (cx + 0.5) * self.stride
        cy_in = (cy + 0.5) * self.stride
        bw = float(preds[0, 1, cy, cx].item()) * W
        bh = float(preds[0, 2, cy, cx].item()) * H
        if bw <= 1 or bh <= 1:
            return None

        sx = src_w / float(W)
        sy = src_h / float(H)
        cx_src = cx_in * sx
        cy_src = cy_in * sy
        bw_src = bw * sx
        bh_src = bh * sy
        x1 = max(0.0, cx_src - bw_src / 2)
        y1 = max(0.0, cy_src - bh_src / 2)
        x2 = min(float(src_w), cx_src + bw_src / 2)
        y2 = min(float(src_h), cy_src + bh_src / 2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return (float(x1), float(y1), float(x2), float(y2))

    def reset(self) -> None:  # noqa: D401 -- parity with MotionCropper
        """No internal state to reset; exists so the class is a drop-in."""
        return
