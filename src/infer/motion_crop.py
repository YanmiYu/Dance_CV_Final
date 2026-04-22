"""Phase 9: motion-based bbox proposal per frame (NO pretrained detector).

We use MOG2 background subtraction + morphology + largest connected
component to estimate a single-person bbox in mostly-fixed-camera clips.
The EMA smoother in ``bbox_smoother.py`` keeps the bbox stable over time.

Fallbacks: if no connected component is large enough, we return ``None``
(the caller uses the previous smoothed bbox, or a center crop, or the
manual initialization box).
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class MotionCropper:
    def __init__(
        self,
        *,
        min_area_ratio: float = 0.01,
        history: int = 300,
        var_threshold: float = 16.0,
        morph_kernel: int = 5,
        pad_ratio: float = 0.25,
        center_crop_fallback: bool = True,
    ) -> None:
        self.min_area_ratio = float(min_area_ratio)
        self.pad_ratio = float(pad_ratio)
        self.center_crop_fallback = bool(center_crop_fallback)
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=int(history), varThreshold=float(var_threshold), detectShadows=False,
        )
        k = int(morph_kernel)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def reset(self) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    def propose(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """Return ``(x1, y1, x2, y2)`` or ``None`` if no confident proposal."""
        h, w = frame_bgr.shape[:2]
        fg = self._bg.apply(frame_bgr, learningRate=-1)
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kernel, iterations=2)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if num <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = int(np.argmax(areas)) + 1
        if areas[idx - 1] < self.min_area_ratio * (h * w):
            return self._center_fallback(h, w) if self.center_crop_fallback else None
        x, y, cw, ch, _ = stats[idx]
        # pad
        pad_w = int(round(cw * self.pad_ratio))
        pad_h = int(round(ch * self.pad_ratio))
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + cw + pad_w)
        y2 = min(h, y + ch + pad_h)
        return (float(x1), float(y1), float(x2), float(y2))

    @staticmethod
    def _center_fallback(h: int, w: int) -> Tuple[float, float, float, float]:
        # centered portrait-ish box covering the middle 60% horizontally and 95% vertically.
        bw = int(0.6 * w)
        bh = int(0.95 * h)
        x1 = (w - bw) // 2
        y1 = (h - bh) // 2
        return (float(x1), float(y1), float(x1 + bw), float(y1 + bh))
