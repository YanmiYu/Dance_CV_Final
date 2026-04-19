"""
alignment.py — DTW-based temporal alignment of two keypoint sequences.
Owner: Member 2

Uses fastdtw (O(N) time and space) so it stays fast on 15–30 fps × 15 s clips.
Cost between two frames = mean Euclidean distance across all 17 joints (x, y only).

The warping path is also used downstream in scoring.py to map aligned frame
indices back to real timestamps.
"""

from __future__ import annotations

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

KeypointSequence = np.ndarray  # shape (T, 17, 3)


def _frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Euclidean distance across all joints between two frames.

    Parameters
    ----------
    a, b : np.ndarray, shape (17, 3)
        Normalized keypoint frames (x, y, confidence).
    """
    return float(np.mean(np.linalg.norm(a[:, :2] - b[:, :2], axis=1)))


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

    _, path = fastdtw(bench_flat, user_flat, dist=euclidean)

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
