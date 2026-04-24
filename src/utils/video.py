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


def _pad_to_even(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Ensure the frame has the requested (even) dimensions; pad with replicated border."""
    h, w = frame.shape[:2]
    if w == target_w and h == target_h:
        return frame
    pad_right = max(0, target_w - w)
    pad_bottom = max(0, target_h - h)
    if pad_right == 0 and pad_bottom == 0:
        # Frame is larger than target; crop from top-left.
        return frame[:target_h, :target_w]
    return cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)


def _write_video_ffmpeg(
    path: Path,
    frames: Iterator[np.ndarray],
    fps: float,
    size: Tuple[int, int],
) -> bool:
    """Encode H.264/yuv420p MP4 via imageio-ffmpeg's bundled ffmpeg. Returns True on success."""
    try:
        import imageio_ffmpeg  # type: ignore
    except ImportError:
        return False

    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return False

    w, h = size
    w2 = w + (w % 2)
    h2 = h + (h % 2)
    cmd = [
        exe,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w2}x{h2}",
        "-pix_fmt", "bgr24",
        "-r", f"{float(fps)}",
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        "-preset", "medium",
        "-movflags", "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdin is not None
        for frame in frames:
            framed = _pad_to_even(frame, w2, h2)
            if not framed.flags["C_CONTIGUOUS"]:
                framed = np.ascontiguousarray(framed)
            proc.stdin.write(framed.tobytes())
        proc.stdin.close()
        ret = proc.wait()
    except BrokenPipeError:
        proc.wait()
        return False
    except Exception:
        proc.kill()
        proc.wait()
        return False
    return ret == 0 and path.exists() and path.stat().st_size > 0


def _write_video_opencv(
    path: Path,
    frames: Iterator[np.ndarray],
    fps: float,
    size: Tuple[int, int],
    fourcc: str,
) -> bool:
    """Encode via cv2.VideoWriter with the requested FourCC. Returns True on success."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
    if not writer.isOpened():
        writer.release()
        return False
    wrote_any = False
    try:
        for frame in frames:
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
            wrote_any = True
    finally:
        writer.release()
    return wrote_any and path.exists() and path.stat().st_size > 0


def write_video(
    path: str | Path,
    frames: Iterator[np.ndarray],
    fps: float,
    size: Tuple[int, int],
    fourcc: str = "mp4v",
) -> None:
    """Write a video that plays in macOS Preview/QuickTime and browsers.

    Encoder preference (first that works wins):
      1. H.264/yuv420p MP4 via imageio-ffmpeg's bundled ffmpeg.
      2. cv2.VideoWriter with FourCC ``avc1`` (H.264), when OpenCV's build supports it.
      3. cv2.VideoWriter with the supplied ``fourcc`` (default ``mp4v``). This path
         may produce files that some players (macOS Preview, Safari) cannot open.
    """
    import warnings

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frames_list = list(frames)
    if not frames_list:
        return

    if _write_video_ffmpeg(path, iter(frames_list), fps, size):
        return

    if _write_video_opencv(path, iter(frames_list), fps, size, "avc1"):
        return

    warnings.warn(
        "Falling back to cv2 FourCC '%s' for %s; install 'imageio-ffmpeg' for "
        "broadly-playable H.264 output." % (fourcc, path),
        RuntimeWarning,
        stacklevel=2,
    )
    if not _write_video_opencv(path, iter(frames_list), fps, size, fourcc):
        raise RuntimeError(f"Failed to write video to {path} with any available encoder")
