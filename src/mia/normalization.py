"""
normalization.py — Body-size and camera-distance normalization.
Owner: Member 2

After normalization, two sequences are comparable regardless of:
  - How close/far the dancer stands from the camera
  - Differences in height between the benchmark dancer and the learner

Steps:
  1. Center: translate all joints so the hip midpoint sits at (0, 0)
  2. Scale: divide all coordinates by torso length (neck midpoint → hip midpoint)
"""

from __future__ import annotations

import numpy as np

# COCO joint indices used for normalization anchors
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP, _R_HIP = 11, 12

KeypointSequence = np.ndarray  # shape (T, 17, 3)


def normalize(seq: KeypointSequence) -> KeypointSequence:
    """Center and scale a keypoint sequence frame by frame.

    Parameters
    ----------
    seq : np.ndarray, shape (T, 17, 3)
        Raw or smoothed keypoint sequence. axis-2 = [x, y, confidence].

    Returns
    -------
    np.ndarray, shape (T, 17, 3)
        Normalized sequence. x and y are in torso-length units centered
        on the hip midpoint. Confidence channel is unchanged.
    """
    seq = seq.copy()
    T = seq.shape[0]

    for t in range(T):
        hip_mid = (seq[t, _L_HIP, :2] + seq[t, _R_HIP, :2]) / 2.0
        neck_mid = (seq[t, _L_SHOULDER, :2] + seq[t, _R_SHOULDER, :2]) / 2.0
        torso_len = np.linalg.norm(neck_mid - hip_mid)

        if torso_len < 1e-6:
            # Dancer not detected this frame; skip normalization
            continue

        seq[t, :, :2] = (seq[t, :, :2] - hip_mid) / torso_len

    return seq


def torso_length(seq: KeypointSequence) -> np.ndarray:
    """Return per-frame torso length (in pixels) before normalization.

    Useful for sanity-checking that normalization is working correctly.
    Returns shape (T,).
    """
    hip_mid = (seq[:, _L_HIP, :2] + seq[:, _R_HIP, :2]) / 2.0
    neck_mid = (seq[:, _L_SHOULDER, :2] + seq[:, _R_SHOULDER, :2]) / 2.0
    return np.linalg.norm(neck_mid - hip_mid, axis=1)
