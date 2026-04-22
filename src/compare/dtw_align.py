"""Phase 13: DTW alignment with a Sakoe-Chiba band.

Implementation notes:
  * we operate on per-frame feature vectors (see ``features.framewise_distance_vector``)
  * distance is weighted L2; ``feature_weights`` lets us up-weight arms over legs
  * monotonic, non-decreasing path, symmetric-2 step pattern
  * Sakoe-Chiba band constrains |i - j * n/m| <= band_width
  * optional warp penalty per horizontal/vertical step
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DTWConfig:
    band_ratio: float = 0.15
    warp_penalty: float = 0.05
    feature_weights: Optional[np.ndarray] = None  # (D,) per-dim weights


@dataclass
class DTWResult:
    cost: float
    path: np.ndarray         # (L, 2) pairs (i_a, i_b)
    aligned_a_idx: np.ndarray  # (L,)
    aligned_b_idx: np.ndarray  # (L,)
    timing_skew_sec: float   # mean signed delay of B relative to A, seconds


def _pairwise_distance(A: np.ndarray, B: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    """Weighted squared-L2 distance between every pair."""
    if weights is None:
        diff = A[:, None, :] - B[None, :, :]
        return (diff ** 2).sum(axis=-1)
    w = weights.astype(np.float32)
    diff = A[:, None, :] - B[None, :, :]
    return ((diff ** 2) * w).sum(axis=-1)


def dtw_align(A: np.ndarray, B: np.ndarray, cfg: DTWConfig | None = None, fps: float = 30.0) -> DTWResult:
    """Run DTW between feature sequences A and B.

    ``A``: ``(Ta, D)``; ``B``: ``(Tb, D)``.
    """
    cfg = cfg or DTWConfig()
    Ta, Da = A.shape
    Tb, Db = B.shape
    assert Da == Db, f"feature dims mismatch: {Da} vs {Db}"

    cost = _pairwise_distance(A, B, cfg.feature_weights)

    band_w = max(1, int(cfg.band_ratio * max(Ta, Tb)))
    INF = float("inf")
    D = np.full((Ta + 1, Tb + 1), INF, dtype=np.float32)
    D[0, 0] = 0.0
    for i in range(1, Ta + 1):
        j_ref = int(round((i / Ta) * Tb))
        j_lo = max(1, j_ref - band_w)
        j_hi = min(Tb, j_ref + band_w)
        for j in range(j_lo, j_hi + 1):
            c = cost[i - 1, j - 1]
            match = D[i - 1, j - 1]
            insert = D[i - 1, j] + cfg.warp_penalty
            delete = D[i, j - 1] + cfg.warp_penalty
            best = min(match, insert, delete)
            D[i, j] = c + best

    # Backtrack.
    i, j = Ta, Tb
    path: list[tuple[int, int]] = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            choices = [
                (D[i - 1, j - 1], (i - 1, j - 1)),
                (D[i - 1, j], (i - 1, j)),
                (D[i, j - 1], (i, j - 1)),
            ]
            _, (i, j) = min(choices, key=lambda kv: kv[0])
        if i >= 1 and j >= 1:
            path.append((i - 1, j - 1))
    path.reverse()
    path_np = np.asarray(path, dtype=np.int32)

    timing_skew = float(np.mean((path_np[:, 1] - path_np[:, 0]))) / max(fps, 1e-6)

    return DTWResult(
        cost=float(D[Ta, Tb] / max(len(path_np), 1)),
        path=path_np,
        aligned_a_idx=path_np[:, 0],
        aligned_b_idx=path_np[:, 1],
        timing_skew_sec=timing_skew,
    )


def build_feature_weights(feat_dim: int, num_joints: int, upper_body_boost: float = 1.5) -> np.ndarray:
    """Build a crude (D,) weight vector that up-weights upper-body coord dims.

    ``coords`` occupy the first ``2*J`` dims, ``velocities`` the next ``2*J``,
    and ``limb_angles`` the remaining dims.
    """
    from src.datasets.common import BODY_PART_GROUPS

    w = np.ones(feat_dim, dtype=np.float32)
    upper = set(BODY_PART_GROUPS["left_arm"]) | set(BODY_PART_GROUPS["right_arm"]) | set(BODY_PART_GROUPS["head"]) | set(BODY_PART_GROUPS["torso"])
    # coords block
    for j in range(num_joints):
        if j in upper:
            w[2 * j] *= upper_body_boost
            w[2 * j + 1] *= upper_body_boost
    # velocities block
    base = 2 * num_joints
    for j in range(num_joints):
        if j in upper:
            w[base + 2 * j] *= upper_body_boost
            w[base + 2 * j + 1] *= upper_body_boost
    return w
