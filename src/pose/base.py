"""Common interface for pose-extraction backends used by the integrate pipeline.

Each adapter (HRNet, SimpleBaseline, GNN) wraps an existing branch's inference
code and produces a uniform PoseRunResult so the downstream error / fusion
stages can stay backend-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PoseRunResult:
    name: str
    poses: Optional[np.ndarray]       # (T, 17, 3) keypoints in image coords; None for embedding-only backends
    embeddings: Optional[np.ndarray]  # (T, D) per-frame embeddings; None for keypoint-only backends
    bboxes: np.ndarray                # (T, 4) crop bboxes
    fps: float
    num_frames: int
    width: int
    height: int
    extra: dict = field(default_factory=dict)
