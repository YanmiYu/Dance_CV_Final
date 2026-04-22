"""Phase 2.3: verify downloadability of a small sample via HEAD / partial GET.

We intentionally only check a sample to avoid hammering the server.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import requests

from src.utils.config import load_yaml
from src.utils.io import read_table, write_table


def _head_ok(url: str, timeout: float = 30.0) -> tuple[int, bool]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        return r.status_code, r.status_code < 400
    except requests.RequestException:
        pass
    try:
        r = requests.get(url, stream=True, timeout=timeout, headers={"Range": "bytes=0-1023"})
        ok = r.status_code < 400
        r.close()
        return r.status_code, ok
    except requests.RequestException:
        return 0, False


def validate(cfg_path: str | Path) -> Path:
    cfg = load_yaml(cfg_path)
    in_path = Path(cfg["manifest_out"])
    rows = read_table(in_path)

    n_sample = int(cfg.get("download", {}).get("head_check_sample", 20))
    rng = random.Random(cfg.get("download", {}).get("rng_seed", 123))
    sample_ix = set(rng.sample(range(len(rows)), min(n_sample, len(rows))))

    out_rows: List[dict] = []
    for i, row in enumerate(rows):
        if i in sample_ix:
            code, ok = _head_ok(row["url"])
            row = {**row, "http_status": code, "head_ok": bool(ok)}
        else:
            row = {**row, "http_status": None, "head_ok": None}
        out_rows.append(row)

    write_table(in_path, out_rows)
    return in_path


def _main() -> None:
    p = argparse.ArgumentParser(description="Validate a sample of URLs by HTTP HEAD.")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    out = validate(args.config)
    print(f"Updated manifest with HEAD checks -> {out}")


if __name__ == "__main__":
    _main()
