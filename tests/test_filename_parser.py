"""CSV/filename parsing robustness (Phase 2)."""
from __future__ import annotations

from src.data.annotate_helpers import parse_aist_filename, parse_pair_filename


def test_aist_canonical_filename_parses_all_fields():
    meta = parse_aist_filename("gBR_sBM_c01_d04_mBR0_ch01.mp4")
    assert meta["parse_ok"] is True
    assert meta == {
        "genre": "BR", "situation": "BM", "camera": "01",
        "dancer": "04", "music": "BR0", "chore": "01", "parse_ok": True,
    }


def test_aist_malformed_filename_returns_nulls_not_exception():
    meta = parse_aist_filename("weird_file_name.mp4")
    assert meta["parse_ok"] is False
    assert meta["genre"] is None
    assert meta["camera"] is None


def test_pair_filename_parses_role_and_take():
    meta = parse_pair_filename("blackpink_01_benchmark_take1_camA.mp4")
    assert meta["parse_ok"] is True
    assert meta["song_id"] == "blackpink"
    assert meta["phrase_id"] == "01"
    assert meta["role"] == "benchmark"
    assert meta["take"] == "take1"
    assert meta["camera_id"] == "camA"


def test_pair_filename_handles_multi_token_song_id():
    meta = parse_pair_filename("doja_paint_the_town_02_userA_take3_camB.mp4")
    assert meta["parse_ok"] is True
    assert meta["song_id"] == "doja_paint_the_town"
    assert meta["role"] == "userA"
