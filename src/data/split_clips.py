"""Split long videos into 10-20s candidate clip segments.

This does not re-encode by default; it just records (start_sec, end_sec)
offsets into the parent video inside a manifest.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.utils.config import load_yaml
from src.utils.io import read_table, write_table
from src.utils.video import ffprobe_meta


def split_into_segments(
    cfg_path: str | Path,
    out_path: str | Path,
    segment_len: float = 15.0,
    min_len: float = 10.0,
    max_len: float = 20.0,
) -> Path:
    cfg = load_yaml(cfg_path)
    rows = read_table(Path(cfg["manifest_out"]))

    segments: List[dict] = []
    for row in rows:
        local = row.get("local_path")
        if not local or not row.get("playable"):
            continue
        meta = ffprobe_meta(local)
        if not meta.ok or meta.duration_sec <= 0:
            continue
        dur = meta.duration_sec
        start = 0.0
        seg_idx = 0
        while start < dur:
            end = min(dur, start + segment_len)
            if end - start >= min_len:
                segments.append(
                    {
                        "parent_clip_id": row["clip_id"],
                        "segment_id": f"{row['clip_id']}_seg{seg_idx:02d}",
                        "local_path": local,
                        "start_sec": float(start),
                        "end_sec": float(end),
                        "duration_sec": float(end - start),
                        "genre": row.get("genre"),
                        "camera": row.get("camera"),
                        "dancer": row.get("dancer"),
                    }
                )
                seg_idx += 1
            start += segment_len
            if start + min_len > dur:
                break
        # Ensure even short-ish videos that fit in [min_len, max_len] end up as one segment.
        if seg_idx == 0 and min_len <= dur <= max_len + 1.0:
            segments.append(
                {
                    "parent_clip_id": row["clip_id"],
                    "segment_id": f"{row['clip_id']}_seg00",
                    "local_path": local,
                    "start_sec": 0.0,
                    "end_sec": float(dur),
                    "duration_sec": float(dur),
                    "genre": row.get("genre"),
                    "camera": row.get("camera"),
                    "dancer": row.get("dancer"),
                }
            )

    out = Path(out_path)
    write_table(out, segments)
    return out


def _main() -> None:
    p = argparse.ArgumentParser(description="Split downloaded videos into 10-20s candidate segments.")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="data/manifests/segment_manifest.parquet")
    p.add_argument("--segment-len", type=float, default=15.0)
    args = p.parse_args()
    out = split_into_segments(args.config, args.out, segment_len=args.segment_len)
    print(f"Wrote segment manifest -> {out}")


if __name__ == "__main__":
    _main()
