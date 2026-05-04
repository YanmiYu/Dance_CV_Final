"""
alignment.py — DTW-based temporal alignment of two keypoint sequences.
Owner: Member 2

Uses fastdtw (O(N) time and space) when installed so it stays fast on
15–30 fps × 15 s clips. A small exact-DTW fallback keeps tests and lightweight
environments working. Cost between two frames = mean Euclidean distance across
all 17 joints (x, y only).

The warping path is also used downstream in scoring.py to map aligned frame
indices back to real timestamps.
"""

from __future__ import annotations

import numpy as np

try:
    from fastdtw import fastdtw as _fastdtw
except ImportError:  # pragma: no cover - exercised only without optional dep
    _fastdtw = None

KeypointSequence = np.ndarray  # shape (T, 17, 3)


def _frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _exact_dtw_path(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    """Small exact DTW fallback used when fastdtw is not installed."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    back = np.zeros((n, m, 2), dtype=np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = (
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1],
            )
            step = int(np.argmin(choices))
            if step == 0:
                pi, pj = i - 2, j - 1
            elif step == 1:
                pi, pj = i - 1, j - 2
            else:
                pi, pj = i - 2, j - 2
            cost[i, j] = _frame_distance(a[i - 1], b[j - 1]) + choices[step]
            back[i - 1, j - 1] = (max(pi, 0), max(pj, 0))

    path: list[tuple[int, int]] = []
    i, j = n - 1, m - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        i, j = (int(v) for v in back[i, j])
    path.reverse()
    return path


def dtw_align(
    bench: KeypointSequence,
    user: KeypointSequence,
) -> tuple[KeypointSequence, KeypointSequence, list[tuple[int, int]]]:
    """Align two normalized keypoint sequences using Dynamic Time Warping.

    Parameters
    ----------
    bench : np.ndarray, shape (T_b, 17, 3)
        Normalized benchmark sequence.
    user : np.ndarray, shape (T_u, 17, 3)
        Normalized learner sequence.

    Returns
    -------
    bench_aligned : np.ndarray, shape (T', 17, 3)
    user_aligned  : np.ndarray, shape (T', 17, 3)
        Both sequences reindexed to the same length T' via the warping path.
    path : list of (i_bench, i_user) tuples
        The DTW warping path.
    """
    # fastdtw expects 1-D or 2-D feature vectors; flatten each frame to (34,)
    bench_flat = bench[:, :, :2].reshape(len(bench), -1)
    user_flat = user[:, :, :2].reshape(len(user), -1)

    if _fastdtw is not None:
        _, path = _fastdtw(bench_flat, user_flat, dist=_frame_distance)
    else:
        path = _exact_dtw_path(bench_flat, user_flat)

    bench_idx = [p[0] for p in path]
    user_idx = [p[1] for p in path]

    bench_aligned = bench[bench_idx]
    user_aligned = user[user_idx]

    return bench_aligned, user_aligned, path


def warping_path_to_timestamps(
    path: list[tuple[int, int]],
    fps: float,
) -> np.ndarray:
    """Convert the user side of the warping path to timestamps in seconds.

    Returns shape (T',) with the timestamp in seconds for each aligned frame.
    """
    user_indices = np.array([p[1] for p in path])
    return user_indices / fps
