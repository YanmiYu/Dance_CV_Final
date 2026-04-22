"""COCO-17 joint metadata + shared dataset utilities.

This module is the SINGLE SOURCE OF TRUTH for:
  - joint names / indices
  - horizontal-flip joint remap
  - skeleton edges (for visualization)
  - body-part groupings

See ``docs/project_decisions.md`` section 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


# Index -> name.
COCO_17_JOINTS: Tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
NUM_JOINTS = 17

# Name -> index.
COCO_17_JOINT_INDEX: Dict[str, int] = {n: i for i, n in enumerate(COCO_17_JOINTS)}

# Horizontal flip map: when the image is mirrored left<->right, swap these
# left/right joint pairs. Central joints (nose) stay in place.
COCO_17_FLIP_PAIRS: Tuple[Tuple[int, int], ...] = (
    (1, 2),    # eyes
    (3, 4),    # ears
    (5, 6),    # shoulders
    (7, 8),    # elbows
    (9, 10),   # wrists
    (11, 12),  # hips
    (13, 14),  # knees
    (15, 16),  # ankles
)


def flip_keypoints(kps: np.ndarray, image_width: int) -> np.ndarray:
    """Horizontally flip (17, 3) keypoints and swap left/right pairs.

    ``kps[..., 0]`` is x in pixels, ``kps[..., 1]`` is y, ``kps[..., 2]`` is
    visibility/confidence. Returned array is a copy.
    """
    out = kps.copy()
    out[..., 0] = (image_width - 1) - out[..., 0]
    for a, b in COCO_17_FLIP_PAIRS:
        out[..., [a, b], :] = out[..., [b, a], :]
    return out


# Skeleton edges (for viz). Order is arbitrary but stable.
COCO_SKELETON: Tuple[Tuple[int, int], ...] = (
    (5, 7), (7, 9),          # left arm
    (6, 8), (8, 10),         # right arm
    (5, 6),                  # shoulders
    (11, 13), (13, 15),      # left leg
    (12, 14), (14, 16),      # right leg
    (11, 12),                # hips
    (5, 11), (6, 12),        # torso sides
    (0, 1), (0, 2), (1, 3), (2, 4),  # face
    (0, 5), (0, 6),          # neck-ish
)

# Per-joint OKS sigmas (COCO convention; used only by metrics when asked).
COCO_OKS_SIGMAS: Tuple[float, ...] = (
    0.026, 0.025, 0.025, 0.035, 0.035,
    0.079, 0.079, 0.072, 0.072, 0.062,
    0.062, 0.107, 0.107, 0.087, 0.087,
    0.089, 0.089,
)


# Body-part groupings (must be kept in sync with configs/data/compare.yaml).
BODY_PART_GROUPS: Dict[str, Tuple[int, ...]] = {
    "head":      (0, 1, 2, 3, 4),
    "left_arm":  (5, 7, 9),
    "right_arm": (6, 8, 10),
    "torso":     (5, 6, 11, 12),
    "left_leg":  (11, 13, 15),
    "right_leg": (12, 14, 16),
}


@dataclass
class AnnotationRecord:
    """Internal unified annotation schema (see Phase 5.2 of the plan)."""
    image_path: str
    image_id: str
    dataset_name: str
    bbox_xyxy: List[float]
    keypoints_xyv: List[List[float]]   # shape (17, 3): x, y, visibility
    center: List[float]
    scale: List[float]
    meta: Dict

    def validate(self) -> None:
        if len(self.keypoints_xyv) != NUM_JOINTS:
            raise ValueError(
                f"Expected {NUM_JOINTS} joints (COCO-17), got {len(self.keypoints_xyv)}. "
                f"See docs/project_decisions.md section 3."
            )
        for kp in self.keypoints_xyv:
            if len(kp) != 3:
                raise ValueError(f"Each keypoint must be [x, y, v], got {kp}")
        if len(self.bbox_xyxy) != 4:
            raise ValueError(f"bbox_xyxy must have 4 numbers, got {self.bbox_xyxy}")


def bbox_to_center_scale(bbox_xyxy: Iterable[float], aspect_ratio: float, pixel_std: float = 200.0) -> Tuple[List[float], List[float]]:
    """Convert xyxy bbox to (center, scale) used by top-down crops.

    ``scale`` is expressed as (w, h) divided by ``pixel_std`` so that a scale
    of 1.0 corresponds to ~200px bounding box side (following the convention
    used by Simple Baselines and HRNet reference code).
    """
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    if w / h > aspect_ratio:
        h = w / aspect_ratio
    else:
        w = h * aspect_ratio
    # mild padding
    w *= 1.25
    h *= 1.25
    return [float(cx), float(cy)], [float(w / pixel_std), float(h / pixel_std)]
