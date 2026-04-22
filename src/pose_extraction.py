"""
pose_extraction.py — PoseEstimator wrapper, smoothing, and interpolation.
Owner: Member 1

Uses the MediaPipe Solutions API (mediapipe==0.10.9, CPU-only, no model file needed).

Output: COCO 17-joint layout
  0 nose       1 L_eye    2 R_eye    3 L_ear    4 R_ear
  5 L_shoulder 6 R_shoulder 7 L_elbow 8 R_elbow
  9 L_wrist   10 R_wrist  11 L_hip  12 R_hip
 13 L_knee    14 R_knee   15 L_ankle 16 R_ankle
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

# MediaPipe BlazePose 33-landmark → COCO 17-joint mapping
_MP_TO_COCO: list[int] = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

KeypointSequence = np.ndarray  # shape (T, 17, 3) — [x_px, y_px, confidence]


class PoseEstimator:
    """Wraps MediaPipe Pose and exposes a unified extract API.

    Parameters
    ----------
    backend : {"mediapipe"}
        Only "mediapipe" supported locally. Kept as a parameter so MMPose
        can be swapped in on Oscar without changing calling code.
    """

    def __init__(self, backend: str = "mediapipe") -> None:
        self.backend = backend
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if self.backend == "mediapipe":
            import mediapipe as mp
            self._model = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Extract keypoints from a single BGR frame.

        Returns
        -------
        np.ndarray, shape (17, 3)
            Each row is [x_px, y_px, visibility].
            All zeros when no person is detected.
        """
        self._load_model()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._model.process(rgb)

        kp = np.zeros((17, 3), dtype=np.float32)
        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            for coco_idx, mp_idx in enumerate(_MP_TO_COCO):
                kp[coco_idx] = [
                    lm[mp_idx].x * w,
                    lm[mp_idx].y * h,
                    lm[mp_idx].visibility,
                ]
        return kp

    def extract_sequence(self, frames: list[np.ndarray]) -> KeypointSequence:
        """Extract keypoints from a list of frames.

        Returns
        -------
        KeypointSequence : np.ndarray, shape (T, 17, 3)
        """
        return np.stack([self.extract(f) for f in frames], axis=0)

    def close(self) -> None:
        if self._model is not None:
            self._model.close()
            self._model = None


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def smooth(
    seq: KeypointSequence,
    window: int = 7,
    polyorder: int = 2,
) -> KeypointSequence:
    """Apply Savitzky-Golay filter along the time axis to reduce jitter.

    Only the x and y channels are smoothed; visibility is left unchanged.
    """
    seq = seq.copy()
    T = seq.shape[0]
    win = min(window, T if T % 2 == 1 else T - 1)
    if win < polyorder + 1:
        return seq
    seq[:, :, 0] = savgol_filter(seq[:, :, 0], win, polyorder, axis=0)
    seq[:, :, 1] = savgol_filter(seq[:, :, 1], win, polyorder, axis=0)
    return seq


def interpolate_missing(
    seq: KeypointSequence,
    confidence_threshold: float = 0.3,
) -> KeypointSequence:
    """Fill frames where a joint's visibility is below threshold via linear interpolation."""
    seq = seq.copy()
    T, J, _ = seq.shape
    for j in range(J):
        low = seq[:, j, 2] < confidence_threshold
        if not low.any():
            continue
        good_idx = np.where(~low)[0]
        if len(good_idx) < 2:
            continue
        for ch in range(2):
            seq[:, j, ch] = np.interp(
                np.arange(T), good_idx, seq[good_idx, j, ch]
            )
    return seq


def save_keypoints(seq: KeypointSequence, path: str | Path) -> None:
    """Save a keypoint sequence to a .npy file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), seq)


def load_keypoints(path: str | Path) -> KeypointSequence:
    """Load a keypoint sequence from a .npy file. Returns shape (T, 17, 3)."""
    return np.load(str(path))
