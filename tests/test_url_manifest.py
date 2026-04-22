"""CSV ingestion (Phase 2.1) with tolerance to headerless + mixed inputs."""
from __future__ import annotations

import json
from pathlib import Path

from src.data.load_csv_urls import build_raw_url_manifest


def test_build_raw_url_manifest_dedup_and_invalid(tmp_path: Path) -> None:
    csv = tmp_path / "urls.csv"
    csv.write_text(
        "https://example.com/videos/gBR_sBM_c01_d04_mBR0_ch01.mp4\n"
        "https://example.com/videos/gBR_sBM_c01_d04_mBR0_ch01.mp4\n"    # dupe
        "\n"
        "https://example.com/videos/funky_filename.mp4\n"               # parse_ok=false
        "https://example.com/text.txt\n"                                # dropped (no mp4)
    )
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"csv_path: {csv}\n"
        f"manifest_out: {tmp_path}/manifest.jsonl\n"
        "filename_regex: '^g(?P<genre>[A-Z0-9]+)_s(?P<situation>[A-Z0-9]+)_c(?P<camera>\\d+)_d(?P<dancer>\\d+)_m(?P<music>[A-Z0-9]+)_ch(?P<chore>\\d+)\\.mp4$'\n"
    )
    out = build_raw_url_manifest(cfg)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    parse_oks = [r["parse_ok"] for r in rows]
    assert parse_oks.count(True) == 1
    assert parse_oks.count(False) == 1
    good = next(r for r in rows if r["parse_ok"])
    assert good["genre"] == "BR"
    assert good["camera"] == "01"
