"""Video IO helpers: ffprobe wrapper + OpenCV readers/writers.

We intentionally prefer ``ffprobe`` for metadata (exact, fast) and fall back
to OpenCV if ffprobe is missing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: str
    ok: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    num_frames: int = 0
    duration_sec: float = 0.0
    error: Optional[str] = None


def ffprobe_meta(path: str | Path) -> VideoMeta:
    path = Path(path)
    if not path.exists():
        return VideoMeta(path=str(path), ok=False, error="file not found")
    if shutil.which("ffprobe"):
        try:
            out = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate,nb_frames,duration",
                    "-of",
                    "json",
                    str(path),
                ],
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            info = json.loads(out).get("streams", [{}])[0]
            num, den = (info.get("avg_frame_rate", "0/1") + "/1").split("/")[:2]
            fps = float(num) / float(den) if float(den) > 0 else 0.0
            nb = int(info.get("nb_frames") or 0)
            dur = float(info.get("duration") or 0.0)
            if nb == 0 and fps > 0 and dur > 0:
                nb = int(round(fps * dur))
            return VideoMeta(
                path=str(path),
                ok=True,
                width=int(info.get("width") or 0),
                height=int(info.get("height") or 0),
                fps=fps,
                num_frames=nb,
                duration_sec=dur,
            )
        except Exception as e:  # pragma: no cover
            return VideoMeta(path=str(path), ok=False, error=f"ffprobe: {e}")

    # Fallback: OpenCV.
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return VideoMeta(path=str(path), ok=False, error="cv2 cannot open")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = nb / fps if fps > 0 else 0.0
        return VideoMeta(path=str(path), ok=True, width=width, height=height, fps=fps, num_frames=nb, duration_sec=dur)
    finally:
        cap.release()


def probe_playable(path: str | Path) -> bool:
    """Decode first, middle and last frames via OpenCV; ok only if all three decode."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            return False
        indices = sorted({0, max(0, n // 2), max(0, n - 1)})
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, _ = cap.read()
            if not ok:
                return False
        return True
    finally:
        cap.release()


def iter_frames(path: str | Path, stride: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(frame_index, bgr_frame)`` pairs."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if stride <= 1 or idx % stride == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def write_video(
    path: str | Path,
    frames: Iterator[np.ndarray],
    fps: float,
    size: Tuple[int, int],
    fourcc: str = "mp4v",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
