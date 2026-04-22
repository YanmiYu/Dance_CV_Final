"""Manifest builders.

There are two kinds of manifests:

1. AIST raw-url manifest (Phase 2) -- built by ``src.data.load_csv_urls``.
2. Pair manifest for our own benchmark/imitation clips (Phase 4) -- built here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.data.annotate_helpers import parse_pair_filename
from src.utils.io import write_table


def build_pair_manifest(clips_dir: str | Path, out_path: str | Path) -> Path:
    """Scan a directory of paired clips and produce ``pair_manifest``.

    See ``docs/recording_protocol.md`` for naming + column rules.
    """
    clips_dir = Path(clips_dir)
    rows: List[dict] = []
    for p in sorted(clips_dir.rglob("*.mp4")):
        meta = parse_pair_filename(p.name)
        song_id = meta["song_id"] or ""
        phrase_id = meta["phrase_id"] or ""
        role = meta["role"] or ""
        performer_id = "benchmark" if role == "benchmark" else role
        rows.append(
            {
                "pair_id": f"{song_id}_{phrase_id}" if song_id and phrase_id else p.stem,
                "song_id": song_id,
                "phrase_id": phrase_id,
                "role": role,
                "performer_id": performer_id,
                "camera_id": meta.get("camera_id"),
                "take": meta.get("take"),
                "path": str(p),
                "start_sec": None,
                "end_sec": None,
                "parse_ok": bool(meta.get("parse_ok")),
            }
        )
    out = Path(out_path)
    write_table(out, rows)
    return out


def _main() -> None:
    p = argparse.ArgumentParser(description="Build pair manifest for benchmark/imitation clips.")
    p.add_argument("--clips-dir", required=True)
    p.add_argument("--out", default="data/manifests/pair_manifest.parquet")
    args = p.parse_args()
    out = build_pair_manifest(args.clips_dir, args.out)
    print(f"Wrote pair manifest -> {out}")


if __name__ == "__main__":
    _main()
