"""Phase 2.5: ffprobe every downloaded video and mark corrupt files."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.utils.config import load_yaml
from src.utils.io import read_table, write_table
from src.utils.video import ffprobe_meta, probe_playable


def probe_all(cfg_path: str | Path) -> Path:
    cfg = load_yaml(cfg_path)
    manifest_path = Path(cfg["manifest_out"])
    rows: List[dict] = read_table(manifest_path)

    for row in rows:
        row.setdefault("width", None)
        row.setdefault("height", None)
        row.setdefault("fps", None)
        row.setdefault("num_frames", None)
        row.setdefault("duration_sec", None)
        row.setdefault("playable", None)
        row.setdefault("probe_error", None)
        local = row.get("local_path")
        if not local:
            continue
        p = Path(local)
        if not p.exists():
            row["playable"] = False
            row["probe_error"] = "missing file"
            continue
        meta = ffprobe_meta(p)
        row["width"] = meta.width
        row["height"] = meta.height
        row["fps"] = meta.fps
        row["num_frames"] = meta.num_frames
        row["duration_sec"] = meta.duration_sec
        if not meta.ok:
            row["playable"] = False
            row["probe_error"] = meta.error
            continue
        row["playable"] = bool(probe_playable(p))
        row["probe_error"] = None if row["playable"] else "first/middle/last decode failed"

    write_table(manifest_path, rows)
    return manifest_path


def _main() -> None:
    p = argparse.ArgumentParser(description="ffprobe + decode-triad validation for downloaded videos.")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    out = probe_all(args.config)
    print(f"Probed videos. Manifest updated -> {out}")


if __name__ == "__main__":
    _main()
