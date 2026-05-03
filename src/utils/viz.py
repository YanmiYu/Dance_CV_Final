"""Lightweight visualization helpers for pose overlays.

We draw with OpenCV only to keep headless environments simple.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.datasets.common import COCO_SKELETON


_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
    (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
    (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
    (132, 56, 255), (82, 0, 133),
]


# Body-part bones for severity-colored rendering. The union of these tuples
# covers the same edges as COCO_SKELETON; partitioning by body part lets us
# color each region (left_arm, torso, ...) independently.
PART_BONES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "head":      ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)),
    "left_arm":  ((5, 7), (7, 9)),
    "right_arm": ((6, 8), (8, 10)),
    "torso":     ((5, 6), (5, 11), (6, 12), (11, 12)),
    "left_leg":  ((11, 13), (13, 15)),
    "right_leg": ((12, 14), (14, 16)),
}


SEVERITY_BGR: Dict[str, Tuple[int, int, int]] = {
    "green":   (80, 220, 80),    # aligned
    "yellow":  (40, 220, 240),   # mildly off
    "red":     (60, 60, 240),    # significantly off
    "neutral": (220, 220, 220),
}


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


def pose_to_canvas(
    pose: np.ndarray,
    canvas_shape: Tuple[int, ...],
    *,
    scale_frac: float = 0.30,
    y_anchor_frac: float = 0.62,
) -> np.ndarray:
    """Map raw 2D keypoints into canvas pixel coords.

    The pose is hip-centered and torso-scaled so that the skeleton sits
    consistently inside the panel regardless of the source video resolution
    or the person's pixel size. ``canvas_shape`` is ``(H, W)`` (extra dims
    are ignored) and the returned array has shape ``(17, 2)`` with int32
    pixel coordinates.
    """
    H, W = int(canvas_shape[0]), int(canvas_shape[1])
    pts = np.asarray(pose, dtype=np.float32)[:, :2].copy()
    hip = (pts[11] + pts[12]) * 0.5
    shoulder = (pts[5] + pts[6]) * 0.5
    torso_len = float(np.linalg.norm(shoulder - hip))
    if not np.isfinite(torso_len) or torso_len < 1e-3:
        # Fall back to the full pose extent so degenerate poses still render.
        spread = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        torso_len = max(spread * 0.25, 1.0)
    scale = (H * scale_frac) / torso_len
    out = (pts - hip) * scale
    out[:, 0] += W * 0.5
    out[:, 1] += H * float(y_anchor_frac)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.int32)


def draw_skeleton_blank(
    canvas_shape: Tuple[int, int, int],
    pose: np.ndarray,
    *,
    severity_per_part: Optional[Dict[str, str]] = None,
    base_color: Tuple[int, int, int] = (220, 220, 220),
    joint_color: Optional[Tuple[int, int, int]] = None,
    min_conf: float = 0.2,
    line_thickness: int = 3,
    joint_radius: int = 4,
) -> np.ndarray:
    """Render the COCO-17 skeleton onto a black canvas.

    When ``severity_per_part`` is provided, each body-part region is drawn
    in its severity color (green / yellow / red); otherwise the whole
    skeleton uses ``base_color``. Joints are always drawn in
    ``joint_color`` (defaults to ``base_color``).
    """
    H, W = int(canvas_shape[0]), int(canvas_shape[1])
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    if pose is None:
        return canvas
    pts = pose_to_canvas(pose, (H, W))
    confs = np.asarray(pose, dtype=np.float32)[:, 2]
    for part, bones in PART_BONES.items():
        if severity_per_part is None:
            color = base_color
        else:
            color = SEVERITY_BGR.get(severity_per_part.get(part, "green"), base_color)
        for (a, b) in bones:
            if confs[a] >= min_conf and confs[b] >= min_conf:
                cv2.line(
                    canvas,
                    (int(pts[a, 0]), int(pts[a, 1])),
                    (int(pts[b, 0]), int(pts[b, 1])),
                    color,
                    line_thickness,
                    lineType=cv2.LINE_AA,
                )
    jc = joint_color if joint_color is not None else base_color
    for j in range(17):
        if confs[j] >= min_conf:
            cv2.circle(
                canvas,
                (int(pts[j, 0]), int(pts[j, 1])),
                joint_radius,
                jc,
                -1,
                lineType=cv2.LINE_AA,
            )
    return canvas


def draw_skeleton_overlay(
    image: np.ndarray,
    pose: np.ndarray,
    *,
    severity_per_part: Optional[Dict[str, str]] = None,
    base_color: Tuple[int, int, int] = (220, 220, 220),
    joint_color: Optional[Tuple[int, int, int]] = None,
    min_conf: float = 0.2,
    line_thickness: int = 3,
    joint_radius: int = 4,
) -> np.ndarray:
    """Draw a skeleton over an existing image.

    Unlike :func:`draw_skeleton_blank`, this assumes ``pose[..., :2]`` are
    already in the same pixel space as ``image`` (the AIST 2D keypoints
    are in source-video pixel coordinates, so they line up after resizing
    the frame and the pose by the same factor).

    When ``severity_per_part`` is provided, each body-part region is drawn
    in its severity color; otherwise the whole skeleton uses ``base_color``.
    Returns a new image; the input is not modified.
    """
    img = image.copy()
    if pose is None:
        return img
    arr = np.asarray(pose, dtype=np.float32)
    pts = arr[:, :2]
    confs = arr[:, 2]
    for part, bones in PART_BONES.items():
        if severity_per_part is None:
            color = base_color
        else:
            color = SEVERITY_BGR.get(severity_per_part.get(part, "green"), base_color)
        for (a, b) in bones:
            if confs[a] >= min_conf and confs[b] >= min_conf:
                cv2.line(
                    img,
                    (int(pts[a, 0]), int(pts[a, 1])),
                    (int(pts[b, 0]), int(pts[b, 1])),
                    color,
                    line_thickness,
                    lineType=cv2.LINE_AA,
                )
    jc = joint_color if joint_color is not None else base_color
    for j in range(17):
        if confs[j] >= min_conf:
            cv2.circle(
                img,
                (int(pts[j, 0]), int(pts[j, 1])),
                joint_radius,
                jc,
                -1,
                lineType=cv2.LINE_AA,
            )
    return img


def draw_severity_legend(
    canvas: np.ndarray,
    *,
    origin: Tuple[int, int] = (10, 10),
    box: int = 14,
    gap: int = 6,
) -> np.ndarray:
    """Draw a small green/yellow/red key in the top-left corner."""
    img = canvas
    x, y = origin
    items = [("aligned", "green"), ("mildly off", "yellow"), ("off", "red")]
    for label, sev in items:
        color = SEVERITY_BGR[sev]
        cv2.rectangle(img, (x, y), (x + box, y + box), color, -1)
        cv2.putText(
            img,
            label,
            (x + box + gap, y + box - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y += box + gap
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
