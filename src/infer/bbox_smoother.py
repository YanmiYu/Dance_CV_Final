"""Exponential-moving-average smoother for per-frame bounding boxes."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class EMABBoxSmoother:
    """Smooth ``(x1, y1, x2, y2)`` bounding boxes across frames.

    ``alpha``: 0 = full smoothing (stuck to the initial box),
              1 = no smoothing (just pass through).
    """

    def __init__(self, alpha: float = 0.4, expand: float = 0.05) -> None:
        self.alpha = float(alpha)
        self.expand = float(expand)
        self._state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._state = None

    def update(self, bbox_xyxy: Optional[Tuple[float, float, float, float]]) -> Optional[np.ndarray]:
        if bbox_xyxy is None:
            return None if self._state is None else self._state.copy()
        b = np.asarray(bbox_xyxy, dtype=np.float32)
        if self.expand > 0:
            cx = (b[0] + b[2]) * 0.5
            cy = (b[1] + b[3]) * 0.5
            w = (b[2] - b[0]) * (1 + self.expand)
            h = (b[3] - b[1]) * (1 + self.expand)
            b = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
        if self._state is None:
            self._state = b
        else:
            self._state = self.alpha * b + (1 - self.alpha) * self._state
        return self._state.copy()
