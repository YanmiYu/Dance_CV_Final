"""
video_io.py — Frame loading, standardization, and export utilities.
Owner: Member 4
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator


def load_video(path: str | Path) -> tuple[list[np.ndarray], float]:
    """Load all frames from a video file.

    Returns
    -------
    frames : list of np.ndarray
        BGR frames, each shape (H, W, 3).
    fps : float
        Original frames-per-second of the video.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def standardize(
    frames: list[np.ndarray],
    src_fps: float,
    target_fps: float = 30.0,
    target_resolution: tuple[int, int] = (720, 1280),
) -> tuple[list[np.ndarray], float]:
    """Resample to target_fps and resize to target_resolution (H, W).

    Returns the resampled frame list and the new fps.
    """
    h, w = target_resolution

    # Temporal resampling via index selection
    n_src = len(frames)
    duration_s = n_src / src_fps
    n_dst = max(1, int(duration_s * target_fps))
    indices = np.linspace(0, n_src - 1, n_dst).astype(int)
    resampled = [cv2.resize(frames[i], (w, h)) for i in indices]
    return resampled, target_fps


def export_side_by_side(
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
    out_path: str | Path,
    fps: float = 30.0,
    label_a: str = "Benchmark",
    label_b: str = "Learner",
) -> None:
    """Write a side-by-side MP4 from two equal-length frame lists."""
    if not frames_a or not frames_b:
        raise ValueError("Frame lists must not be empty.")
    h, w = frames_a[0].shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w * 2, h))

    n = min(len(frames_a), len(frames_b))
    for i in range(n):
        fa = frames_a[i].copy()
        fb = frames_b[i].copy()
        _put_label(fa, label_a)
        _put_label(fb, label_b)
        combined = np.concatenate([fa, fb], axis=1)
        writer.write(combined)
    writer.release()


def frame_generator(path: str | Path) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (frame, timestamp_ms) one frame at a time without loading all into memory."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        yield frame, ts
    cap.release()


def _put_label(frame: np.ndarray, label: str) -> None:
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
