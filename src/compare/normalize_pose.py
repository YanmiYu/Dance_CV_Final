"""Phase 11: normalize poses for comparison.

Rules (see docs/project_decisions.md + configs/data/compare.yaml):
  * translate so the root joint (hip_center) is at the origin
  * scale by torso length (or shoulder width / bbox diag, configurable)
  * do NOT rotate away camera viewpoint (dance orientation matters)
  * mask low-confidence joints
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.datasets.common import COCO_17_JOINT_INDEX, NUM_JOINTS


@dataclass
class NormalizeConfig:
    root_joint: str = "hip_center"        # "hip_center" | "pelvis_midpoint" (alias)
    scale_by: str = "torso"               # "torso" | "shoulder_width" | "bbox_diag"
    min_visibility: float = 0.2
    orient_torso: bool = False            # keep viewpoint for dance; only set True if asked


def _hip_center(xy: np.ndarray) -> np.ndarray:
    lh = COCO_17_JOINT_INDEX["left_hip"]
    rh = COCO_17_JOINT_INDEX["right_hip"]
    return 0.5 * (xy[..., lh, :] + xy[..., rh, :])


def _shoulder_center(xy: np.ndarray) -> np.ndarray:
    ls = COCO_17_JOINT_INDEX["left_shoulder"]
    rs = COCO_17_JOINT_INDEX["right_shoulder"]
    return 0.5 * (xy[..., ls, :] + xy[..., rs, :])


def _torso_length(xy: np.ndarray) -> np.ndarray:
    """Return per-frame torso length; shape (T,)."""
    sc = _shoulder_center(xy)
    hc = _hip_center(xy)
    return np.linalg.norm(sc - hc, axis=-1)


def _shoulder_width(xy: np.ndarray) -> np.ndarray:
    ls = COCO_17_JOINT_INDEX["left_shoulder"]
    rs = COCO_17_JOINT_INDEX["right_shoulder"]
    return np.linalg.norm(xy[..., ls, :] - xy[..., rs, :], axis=-1)


def _bbox_diag(xy: np.ndarray) -> np.ndarray:
    mn = xy.min(axis=-2)
    mx = xy.max(axis=-2)
    return np.linalg.norm(mx - mn, axis=-1)


def _orient_torso(xy: np.ndarray) -> np.ndarray:
    """Rotate so the shoulder axis is horizontal (mild orientation fix)."""
    ls = COCO_17_JOINT_INDEX["left_shoulder"]
    rs = COCO_17_JOINT_INDEX["right_shoulder"]
    diff = xy[..., rs, :] - xy[..., ls, :]         # (T, 2)
    theta = np.arctan2(diff[..., 1], diff[..., 0]) # (T,)
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.zeros((*xy.shape[:-2], 2, 2), dtype=np.float32)
    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c
    return np.einsum("...ij,...kj->...ki", R, xy)


def normalize_sequence(poses: np.ndarray, cfg: NormalizeConfig | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize a pose sequence.

    Args:
        poses: ``(T, 17, 3)`` -- (x, y, conf).

    Returns:
        normalized_poses: ``(T, 17, 3)`` with x,y normalized and conf unchanged.
        mask:  ``(T, 17)`` boolean, True where the joint is reliable for scoring.
    """
    cfg = cfg or NormalizeConfig()
    assert poses.ndim == 3 and poses.shape[1] == NUM_JOINTS and poses.shape[2] == 3
    xy = poses[..., :2].astype(np.float32).copy()
    conf = poses[..., 2].astype(np.float32)

    root = _hip_center(xy)                                 # (T, 2)
    xy = xy - root[..., None, :]

    if cfg.scale_by == "torso":
        s = _torso_length(xy + root[..., None, :])         # back-out for stability
    elif cfg.scale_by == "shoulder_width":
        s = _shoulder_width(xy + root[..., None, :])
    elif cfg.scale_by == "bbox_diag":
        s = _bbox_diag(xy + root[..., None, :])
    else:
        raise ValueError(f"unknown scale_by {cfg.scale_by!r}")
    # avoid div-by-zero; use the sequence median as fallback
    med = float(np.median(s[s > 1e-3])) if np.any(s > 1e-3) else 1.0
    s = np.where(s > 1e-3, s, med)
    xy = xy / s[..., None, None]

    if cfg.orient_torso:
        xy = _orient_torso(xy)

    mask = conf >= cfg.min_visibility

    out = np.concatenate([xy, conf[..., None]], axis=-1).astype(np.float32)
    return out, mask.astype(bool)
