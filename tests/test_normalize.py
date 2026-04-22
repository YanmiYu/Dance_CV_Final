"""Pose normalization invariance to translation + scale."""
from __future__ import annotations

import numpy as np

from src.compare.normalize_pose import NormalizeConfig, normalize_sequence


def _toy_pose(T: int = 30) -> np.ndarray:
    """A simple standing pose that drifts slightly each frame."""
    base = np.array(
        [
            [0, -50], [-4, -53], [4, -53], [-6, -52], [6, -52],  # head
            [-20, -30], [20, -30], [-28, -10], [28, -10], [-30, 10], [30, 10],  # arms
            [-12, 20], [12, 20], [-14, 60], [14, 60], [-16, 100], [16, 100],     # legs
        ],
        dtype=np.float32,
    )
    out = np.zeros((T, 17, 3), dtype=np.float32)
    for t in range(T):
        out[t, :, :2] = base + np.array([t * 0.2, t * 0.1], dtype=np.float32)
        out[t, :, 2] = 1.0
    return out


def test_normalize_is_invariant_to_translation():
    pose = _toy_pose()
    n1, m1 = normalize_sequence(pose, NormalizeConfig())
    shifted = pose.copy()
    shifted[..., 0] += 500
    shifted[..., 1] -= 300
    n2, m2 = normalize_sequence(shifted, NormalizeConfig())
    assert np.allclose(n1[..., :2], n2[..., :2], atol=1e-5)
    assert np.array_equal(m1, m2)


def test_normalize_is_invariant_to_uniform_scale():
    pose = _toy_pose()
    n1, _ = normalize_sequence(pose, NormalizeConfig(scale_by="torso"))
    scaled = pose.copy()
    scaled[..., :2] *= 3.0
    n2, _ = normalize_sequence(scaled, NormalizeConfig(scale_by="torso"))
    assert np.allclose(n1[..., :2], n2[..., :2], atol=1e-4)
