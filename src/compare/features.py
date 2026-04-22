"""Phase 12: feature extraction from a normalized pose sequence.

Features (all frame-aligned, same T):
  * normalized joint coords (2J-dim per frame)
  * first-difference velocities
  * second-difference accelerations (optional)
  * selected limb angles (elbows, knees, shoulders, hips)
  * torso orientation proxy (angle of shoulder axis)
  * motion energy per frame (sum |v|)

All are confidence-aware -- values where the mask is False are zero-weighted
in downstream scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.datasets.common import BODY_PART_GROUPS, COCO_17_JOINT_INDEX, NUM_JOINTS


@dataclass
class FeatureConfig:
    use_normalized_coords: bool = True
    use_velocities: bool = True
    use_accelerations: bool = True
    use_limb_angles: bool = True
    use_motion_energy: bool = True
    smoothing_window: int = 5


# (joint_a, center, joint_b) -- angle at "center" joint.
_LIMB_ANGLES: Dict[str, Tuple[str, str, str]] = {
    "left_elbow":    ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow":   ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder":("right_hip", "right_shoulder", "right_elbow"),
    "left_hip":      ("left_shoulder", "left_hip", "left_knee"),
    "right_hip":     ("right_shoulder", "right_hip", "right_knee"),
    "left_knee":     ("left_hip", "left_knee", "left_ankle"),
    "right_knee":    ("right_hip", "right_knee", "right_ankle"),
}


def _angle_at(center: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v1 = a - center
    v2 = b - center
    cos = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8)
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)


def _moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.shape[0] < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        for d in range(x.shape[2]):
            out[:, j, d] = np.convolve(x[:, j, d], kernel, mode="same")
    return out


def extract_features(
    normalized_poses: np.ndarray,   # (T, 17, 3) from normalize_pose.normalize_sequence
    mask: np.ndarray,               # (T, 17) visibility mask
    cfg: FeatureConfig | None = None,
) -> Dict[str, np.ndarray]:
    cfg = cfg or FeatureConfig()
    T = normalized_poses.shape[0]
    assert normalized_poses.shape == (T, NUM_JOINTS, 3)
    xy = normalized_poses[..., :2].astype(np.float32)
    xy_s = _moving_avg(xy, cfg.smoothing_window)

    out: Dict[str, np.ndarray] = {}
    if cfg.use_normalized_coords:
        out["coords"] = xy_s.copy()                     # (T, 17, 2)

    if cfg.use_velocities:
        v = np.zeros_like(xy_s)
        v[1:] = xy_s[1:] - xy_s[:-1]
        out["velocities"] = v                           # (T, 17, 2)

    if cfg.use_accelerations:
        a = np.zeros_like(xy_s)
        a[1:-1] = xy_s[2:] - 2 * xy_s[1:-1] + xy_s[:-2]
        out["accelerations"] = a                        # (T, 17, 2)

    if cfg.use_limb_angles:
        angle_keys: List[str] = list(_LIMB_ANGLES.keys())
        angles = np.zeros((T, len(angle_keys)), dtype=np.float32)
        for i, k in enumerate(angle_keys):
            a, c, b = _LIMB_ANGLES[k]
            ia, ic, ib = COCO_17_JOINT_INDEX[a], COCO_17_JOINT_INDEX[c], COCO_17_JOINT_INDEX[b]
            angles[:, i] = _angle_at(xy_s[:, ic], xy_s[:, ia], xy_s[:, ib])
        out["limb_angles"] = angles                     # (T, 8)
        out["limb_angle_keys"] = np.asarray(angle_keys, dtype=object)

    if cfg.use_motion_energy:
        v = np.zeros_like(xy_s)
        v[1:] = xy_s[1:] - xy_s[:-1]
        mag = np.linalg.norm(v, axis=-1)                # (T, 17)
        out["motion_energy"] = mag.mean(axis=-1)        # (T,)

    out["mask"] = mask.astype(bool)                     # (T, 17)
    return out


def joint_indices_for_group(group: str) -> Tuple[int, ...]:
    return tuple(BODY_PART_GROUPS[group])


def framewise_distance_vector(feats: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten per-frame features into a single vector used by DTW distance.

    Keeps coords + velocities by default -- acceleration is too noisy for DTW.
    """
    parts: List[np.ndarray] = []
    if "coords" in feats:
        parts.append(feats["coords"].reshape(feats["coords"].shape[0], -1))
    if "velocities" in feats:
        parts.append(feats["velocities"].reshape(feats["velocities"].shape[0], -1))
    if "limb_angles" in feats:
        parts.append(feats["limb_angles"])
    return np.concatenate(parts, axis=-1).astype(np.float32)
