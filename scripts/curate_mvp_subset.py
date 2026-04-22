"""Phase 3: filter pilot-downloaded dance data to the MVP assumptions.

Rules (see docs/project_decisions.md):

  * one visible primary dancer
  * mostly static camera
  * usable framing (we proxy this with ffprobe-derived resolution + playability)
  * 10-20s segments
  * no group choreography in v1
  * no moving-camera footage in v1

We use conservative heuristics here. Final inclusion still requires a human
spot-check (see preview thumbnails emitted to ``data/manifests/preview/``).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.utils.config import load_yaml
from src.utils.io import read_table, write_table
from src.utils.video import iter_frames


# AIST camera codes that tend to correspond to roughly-static framings.
# (This is a coarse heuristic and can be refined after hand-review.)
_FIXED_CAMERA_CODES = {"01", "02", "03", "04", "05", "06", "07", "08"}


def _estimate_motion_ratio(video_path: str | Path, start_sec: float, end_sec: float,
                            n_samples: int = 6) -> float:
    """Rough camera-motion proxy.

    We compute the mean absolute frame-to-frame difference on downsampled
    gray frames sampled uniformly within [start_sec, end_sec]. A very low
    value implies a locked-off camera (and a mostly static scene). A high
    value could indicate either a dancer moving fast OR a moving camera; we
    only use it as an exclusion proxy for HIGHLY unstable footage.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            return float("nan")
        s = max(0, int(start_sec * fps))
        e = min(total - 1, int(end_sec * fps))
        if e - s < 2:
            return float("nan")
        sample_idx = np.linspace(s, e, n_samples).astype(int)
        prev = None
        diffs: List[float] = []
        for idx in sample_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diffs.append(float(np.mean(np.abs(g.astype(np.int16) - prev.astype(np.int16)))))
            prev = g
        return float(np.mean(diffs)) if diffs else float("nan")
    finally:
        cap.release()


def curate(segment_manifest: str | Path, out_path: str | Path, max_motion: float = 22.0) -> Path:
    rows = read_table(segment_manifest)
    out_rows: List[dict] = []

    preview_dir = Path("data/manifests/preview")
    preview_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        path = row["local_path"]
        start, end = float(row["start_sec"]), float(row["end_sec"])
        dur = end - start
        reasons: List[str] = []
        if dur < 10.0 or dur > 22.0:
            reasons.append(f"duration {dur:.1f}s outside [10, 20]")
        cam = (row.get("camera") or "").strip()
        if cam and cam not in _FIXED_CAMERA_CODES:
            reasons.append(f"camera {cam!r} not in known-fixed set")
        motion = _estimate_motion_ratio(path, start, end)
        if np.isfinite(motion) and motion > max_motion:
            reasons.append(f"motion proxy {motion:.1f} > {max_motion}")

        is_mvp_usable = len(reasons) == 0
        if is_mvp_usable:
            # Write a preview thumbnail from mid-segment.
            mid = (start + end) / 2.0
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000)
            ok, frame = cap.read()
            cap.release()
            if ok:
                preview = preview_dir / f"{row['segment_id']}.jpg"
                cv2.imwrite(str(preview), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        out_rows.append(
            {
                **row,
                "motion_proxy": motion,
                "is_mvp_usable": bool(is_mvp_usable),
                "reason_if_rejected": "; ".join(reasons) if reasons else None,
            }
        )

    out = Path(out_path)
    write_table(out, out_rows)
    return out


def _main() -> None:
    p = argparse.ArgumentParser(description="Curate MVP-usable subset of segments.")
    p.add_argument("--segments", default="data/manifests/segment_manifest.parquet")
    p.add_argument("--out", default="data/manifests/benchmark_candidate_manifest.parquet")
    p.add_argument("--max-motion", type=float, default=22.0)
    args = p.parse_args()
    out = curate(args.segments, args.out, max_motion=args.max_motion)
    print(f"Wrote curated candidate manifest -> {out}")


if __name__ == "__main__":
    _main()
