"""Phase 2.4: pilot-subset downloader.

Downloads a small number of videos first, updates the manifest with local
paths, and preserves resumability via ``.part`` files.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import List

import requests
from tqdm import tqdm

from src.utils.config import load_yaml
from src.utils.io import read_table, write_table


def _download_one(url: str, out_path: Path, timeout: float, max_retries: int) -> tuple[bool, str | None]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    last_err: str | None = None
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
            tmp.replace(out_path)
            return True, None
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    return False, last_err


def _pick_pilot(rows: List[dict], n: int, rng_seed: int, prefer_cameras: List[str]) -> List[int]:
    """Prefer rows whose parsed ``camera`` is in ``prefer_cameras``; random-sample the rest."""
    rng = random.Random(rng_seed)
    preferred = [i for i, r in enumerate(rows) if r.get("camera") in prefer_cameras]
    rest = [i for i, r in enumerate(rows) if r.get("camera") not in prefer_cameras]
    rng.shuffle(preferred)
    rng.shuffle(rest)
    out = preferred[:n]
    if len(out) < n:
        out += rest[: n - len(out)]
    return out


def download_pilot(cfg_path: str | Path) -> Path:
    cfg = load_yaml(cfg_path)
    manifest_path = Path(cfg["manifest_out"])
    rows = read_table(manifest_path)

    d = cfg.get("download", {})
    n = int(d.get("pilot_n", 50))
    rng_seed = int(d.get("rng_seed", 123))
    out_dir = Path(d.get("out_dir", "data/raw_videos"))
    timeout = float(d.get("timeout_sec", 60))
    max_retries = int(d.get("max_retries", 3))
    prefer_cameras = list(
        {c for item in d.get("prefer", []) for c in item.get("cameras", [])}
    ) or []

    indices = _pick_pilot(rows, n, rng_seed, prefer_cameras)
    for i, row in enumerate(rows):
        row.setdefault("local_path", None)
        row.setdefault("download_status", None)
        row.setdefault("download_error", None)

    for idx in tqdm(indices, desc="pilot download"):
        row = rows[idx]
        filename = row["filename"]
        out_path = out_dir / filename
        if out_path.exists():
            row["local_path"] = str(out_path)
            row["download_status"] = "already_present"
            continue
        ok, err = _download_one(row["url"], out_path, timeout=timeout, max_retries=max_retries)
        row["local_path"] = str(out_path) if ok else None
        row["download_status"] = "ok" if ok else "failed"
        row["download_error"] = err

    write_table(manifest_path, rows)
    return manifest_path


def _main() -> None:
    p = argparse.ArgumentParser(description="Download pilot subset of videos.")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    out = download_pilot(args.config)
    print(f"Pilot download finished. Manifest updated -> {out}")


if __name__ == "__main__":
    _main()
