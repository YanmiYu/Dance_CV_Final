"""Extract frames from videos at a given stride for downstream labeling / inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.utils.video import iter_frames


def extract(video_path: str | Path, out_dir: str | Path, stride: int = 1, prefix: str | None = None) -> int:
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix or video_path.stem
    n = 0
    for idx, frame in iter_frames(video_path, stride=stride):
        out = out_dir / f"{stem}_{idx:06d}.jpg"
        cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        n += 1
    return n


def _main() -> None:
    p = argparse.ArgumentParser(description="Extract frames from a video.")
    p.add_argument("--video", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()
    n = extract(args.video, args.out_dir, stride=args.stride)
    print(f"Extracted {n} frames -> {args.out_dir}")


if __name__ == "__main__":
    _main()
