"""Lightweight visualization helpers for pose overlays.

We draw with OpenCV only to keep headless environments simple.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.datasets.common import COCO_SKELETON


_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
    (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
    (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
    (132, 56, 255), (82, 0, 133),
]


def draw_pose(
    image: np.ndarray,
    keypoints: np.ndarray,
    *,
    min_conf: float = 0.2,
    radius: int = 4,
    thickness: int = 2,
    skeleton: Sequence[Tuple[int, int]] = COCO_SKELETON,
) -> np.ndarray:
    """Draw a single-person 17-joint pose onto a BGR image.

    ``keypoints``: ``(17, 3)`` with ``(x, y, confidence)`` or ``(x, y, visibility)``.
    Returns a new image; input is not modified.
    """
    img = image.copy()
    if keypoints is None:
        return img
    kps = np.asarray(keypoints, dtype=np.float32)
    assert kps.shape == (17, 3), f"keypoints must be (17,3), got {kps.shape}"

    # Skeleton edges.
    for (a, b) in skeleton:
        ca, cb = kps[a, 2], kps[b, 2]
        if ca >= min_conf and cb >= min_conf:
            pa = (int(round(kps[a, 0])), int(round(kps[a, 1])))
            pb = (int(round(kps[b, 0])), int(round(kps[b, 1])))
            cv2.line(img, pa, pb, (220, 220, 220), thickness, lineType=cv2.LINE_AA)

    # Joints.
    for i in range(17):
        if kps[i, 2] >= min_conf:
            p = (int(round(kps[i, 0])), int(round(kps[i, 1])))
            cv2.circle(img, p, radius, _COLORS[i % len(_COLORS)], -1, lineType=cv2.LINE_AA)

    return img


def draw_bbox(image: np.ndarray, bbox_xyxy: Iterable[float], color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    img = image.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def side_by_side(a: np.ndarray, b: np.ndarray, label_a: Optional[str] = None, label_b: Optional[str] = None) -> np.ndarray:
    """Resize two images to the same height and concatenate horizontally."""
    h = min(a.shape[0], b.shape[0])
    ar = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
    br = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    if label_a:
        cv2.putText(ar, label_a, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if label_b:
        cv2.putText(br, label_b, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate([ar, br], axis=1)
