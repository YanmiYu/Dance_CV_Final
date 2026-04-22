"""Phase 2.1: read the (headerless, one-column) URL CSV, clean it up.

Usage::

    python -m src.data.load_csv_urls --config configs/data/aist_urls.yaml
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

from src.data.annotate_helpers import parse_aist_filename
from src.utils.config import load_yaml
from src.utils.io import write_table


def _read_raw_urls(csv_path: str | Path) -> List[str]:
    """Read urls from a possibly-headerless one-column CSV."""
    urls: List[str] = []
    with open(csv_path, "r") as f:
        for raw in f:
            s = raw.strip().strip(",")  # strip stray commas if any
            if not s:
                continue
            # Skip plausible header line.
            if s.lower() in {"url", "urls", "video_url"}:
                continue
            urls.append(s)
    return urls


def _dedupe_keep_order(xs: Iterable[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_raw_url_manifest(cfg_path: str | Path) -> Path:
    cfg = load_yaml(cfg_path)
    csv_path = Path(cfg["csv_path"])
    regex = re.compile(cfg["filename_regex"])
    out_path = Path(cfg["manifest_out"])

    urls = _read_raw_urls(csv_path)
    urls = [u for u in urls if u.lower().endswith(".mp4")]
    urls = _dedupe_keep_order(urls)

    rows: List[dict] = []
    for i, url in enumerate(urls):
        filename = url.rsplit("/", 1)[-1]
        meta = parse_aist_filename(filename, regex=regex)
        rows.append(
            {
                "clip_id": f"aist_{i:06d}",
                "url": url,
                "filename": filename,
                "genre": meta.get("genre"),
                "situation": meta.get("situation"),
                "camera": meta.get("camera"),
                "dancer": meta.get("dancer"),
                "music": meta.get("music"),
                "choreography": meta.get("chore"),
                "parse_ok": meta.get("parse_ok", False),
            }
        )

    write_table(out_path, rows)
    return out_path


def _main() -> None:
    p = argparse.ArgumentParser(description="Build data/manifests/raw_url_manifest.parquet")
    p.add_argument("--config", required=True, help="configs/data/aist_urls.yaml")
    args = p.parse_args()
    out = build_raw_url_manifest(args.config)
    print(f"Wrote raw url manifest -> {out}")


if __name__ == "__main__":
    _main()
