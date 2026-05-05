"""Tests for the Basic Dance all-genre index builder."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.datasets.basic_dance_index import (
    build_index_csv,
    iter_basic_dance_records,
    load_index_csv,
    parse_basic_dance_filename,
)


def _write_fake_pkl(path: Path, shape: tuple[int, ...]) -> None:
    arr = np.zeros(shape, dtype=np.float32)
    arr[..., 2] = 0.9
    with path.open("wb") as f:
        pickle.dump({"keypoints2d": arr}, f)


def test_parse_basic_dance_filename_handles_multiple_genres() -> None:
    for filename, genre, music in [
        ("gBR_sBM_c01_d04_mBR0_ch04.pkl", "BR", "BR0"),
        ("gPO_sBM_cAll_d05_mPO1_ch03.pkl", "PO", "PO1"),
        ("gLO_sBM_c09_d07_mLO2_ch10.pkl", "LO", "LO2"),
        ("gKR_sBM_cAll_d01_mKR0_ch01.pkl", "KR", "KR0"),
    ]:
        parsed = parse_basic_dance_filename(filename)
        assert parsed is not None
        assert parsed["genre"] == genre
        assert parsed["music"] == music
        assert parsed["situation"] == "BM"


def test_call_expansion_can_produce_c01_row(tmp_path: Path) -> None:
    pkl_dir = tmp_path / "pkls"
    pkl_dir.mkdir()
    _write_fake_pkl(pkl_dir / "gHO_sBM_cAll_d04_mHO0_ch01.pkl", (9, 64, 17, 3))

    rows = list(iter_basic_dance_records(pkl_dir, situation="sBM", use_cameras=["c01"]))

    assert len(rows) == 1
    assert rows[0].camera == "c01"
    assert rows[0].camera_index == 0
    assert rows[0].genre_label == "gHO"
    assert rows[0].dance_label == "gHO_mHO0_ch01"


def test_index_builder_filters_sbm_and_c01_only_all_genres(tmp_path: Path) -> None:
    pkl_dir = tmp_path / "pkls"
    pkl_dir.mkdir()
    _write_fake_pkl(pkl_dir / "gBR_sBM_cAll_d04_mBR0_ch01.pkl", (9, 64, 17, 3))
    _write_fake_pkl(pkl_dir / "gPO_sBM_cAll_d05_mPO0_ch01.pkl", (9, 64, 17, 3))
    _write_fake_pkl(pkl_dir / "gHO_sFM_cAll_d05_mHO0_ch01.pkl", (9, 64, 17, 3))
    _write_fake_pkl(pkl_dir / "gKR_sBM_c02_d06_mKR0_ch01.pkl", (64, 17, 3))

    out_csv = tmp_path / "index.csv"
    n = build_index_csv(
        pkl_dir,
        out_csv,
        situation="sBM",
        use_cameras=["c01"],
        genres="all",
    )
    rows = load_index_csv(out_csv)

    assert n == 2
    assert len(rows) == 2
    assert {row["genre_label"] for row in rows} == {"gBR", "gPO"}
    assert {row["camera"] for row in rows} == {"c01"}
    assert {row["situation"] for row in rows} == {"sBM"}


def test_index_builder_can_whitelist_legacy_gbr(tmp_path: Path) -> None:
    pkl_dir = tmp_path / "pkls"
    pkl_dir.mkdir()
    _write_fake_pkl(pkl_dir / "gBR_sBM_cAll_d04_mBR0_ch01.pkl", (9, 64, 17, 3))
    _write_fake_pkl(pkl_dir / "gPO_sBM_cAll_d05_mPO0_ch01.pkl", (9, 64, 17, 3))

    out_csv = tmp_path / "index.csv"
    build_index_csv(pkl_dir, out_csv, situation="sBM", use_cameras=["c01"], genres=["gBR"])
    rows = load_index_csv(out_csv)

    assert len(rows) == 1
    assert rows[0]["genre_label"] == "gBR"
