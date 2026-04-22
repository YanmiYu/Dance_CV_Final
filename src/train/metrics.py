"""Pose metrics (all computed from scratch).

Implements:
  * PCK (Percentage of Correct Keypoints) with configurable threshold
  * OKS-based AP-style scalar (simple mean OKS)
  * Mean joint error in pixels
  * Scale-normalized joint error
  * Missing-joint rate
  * Temporal jitter score on videos
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from src.datasets.common import COCO_OKS_SIGMAS, NUM_JOINTS


def mean_joint_error_px(pred_xy: np.ndarray, gt_xy: np.ndarray, vis: np.ndarray) -> float:
    """Mean L2 distance on visible joints."""
    mask = vis > 0
    if mask.sum() == 0:
        return float("nan")
    diff = pred_xy - gt_xy
    dist = np.linalg.norm(diff, axis=-1)
    return float(dist[mask].mean())


def scale_normalized_mje(pred_xy: np.ndarray, gt_xy: np.ndarray, vis: np.ndarray, scale_px: float) -> float:
    if scale_px <= 0:
        return float("nan")
    return mean_joint_error_px(pred_xy, gt_xy, vis) / float(scale_px)


def pck(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    vis: np.ndarray,
    scale_px: float,
    threshold: float = 0.05,
) -> float:
    """Percentage of joints with L2 error <= ``threshold * scale_px``."""
    if scale_px <= 0:
        return float("nan")
    mask = vis > 0
    if mask.sum() == 0:
        return float("nan")
    d = np.linalg.norm(pred_xy - gt_xy, axis=-1) / scale_px
    return float((d[mask] <= threshold).mean())


def oks(pred_xy: np.ndarray, gt_xy: np.ndarray, vis: np.ndarray, area: float) -> float:
    """Single-person OKS using COCO sigmas."""
    sigmas = np.asarray(COCO_OKS_SIGMAS, dtype=np.float32)
    d2 = np.sum((pred_xy - gt_xy) ** 2, axis=-1)
    vars_ = (sigmas * 2.0) ** 2
    e = d2 / (vars_ * max(area, 1.0) * 2.0)
    mask = vis > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.exp(-e[mask]).mean())


def missing_rate(vis_pred: np.ndarray, min_confidence: float = 0.1) -> float:
    """Fraction of joints flagged as missing (confidence < ``min_confidence``)."""
    return float((vis_pred < min_confidence).mean())


def temporal_jitter(poses_tjv: np.ndarray, fps: float = 30.0) -> float:
    """Mean per-joint acceleration magnitude (px / frame^2), a cheap smoothness proxy.

    ``poses_tjv``: ``(T, J, 2|3)``.
    """
    xy = poses_tjv[..., :2]
    if xy.shape[0] < 3:
        return float("nan")
    accel = xy[2:] - 2 * xy[1:-1] + xy[:-2]
    mag = np.linalg.norm(accel, axis=-1)
    return float(mag.mean())


def summarize_epoch(metrics_per_item: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_per_item:
        return {}
    keys = metrics_per_item[0].keys()
    out: Dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in metrics_per_item if m.get(k) is not None and not np.isnan(m[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def per_joint_error(pred_xy: np.ndarray, gt_xy: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Mean error per joint, over the items/frames axis. Shape: (J,)."""
    assert pred_xy.shape[-2] == NUM_JOINTS
    d = np.linalg.norm(pred_xy - gt_xy, axis=-1)
    mask = vis > 0
    out = np.zeros(NUM_JOINTS, dtype=np.float32)
    for j in range(NUM_JOINTS):
        m = mask[..., j]
        out[j] = float(d[..., j][m].mean()) if m.any() else float("nan")
    return out
