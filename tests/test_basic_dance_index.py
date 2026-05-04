"""Tests for the Basic Dance index parser + builder."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from src.datasets.basic_dance_index import (
    build_index_csv,
    iter_basic_dance_records,
    load_index_csv,
    parse_basic_dance_filename,
)


def test_parse_basic_dance_filename_per_camera():
    out = parse_basic_dance_filename("gBR_sBM_c01_d04_mBR0_ch04.pkl")
    assert out is not None
    assert out["genre"] == "BR"
    assert out["situation"] == "BM"
    assert out["camera"] == "01"
    assert out["dancer"] == "04"
    assert out["music"] == "BR0"
    assert out["chore"] == "04"


def test_parse_basic_dance_filename_consolidated():
    out = parse_basic_dance_filename("gHO_sBM_cAll_d06_mHO5_ch10.pkl")
    assert out is not None
    assert out["camera"] == "All"
    assert out["genre"] == "HO"
    assert out["chore"] == "10"


def test_parse_basic_dance_filename_rejects_garbage():
    assert parse_basic_dance_filename("not_a_real_file.pkl") is None
    assert parse_basic_dance_filename("gBR_sBM_c01_d04_mBR0_ch04.mp4") is None


def _write_fake_pkl(path: Path, shape: tuple) -> None:
    arr = np.zeros(shape, dtype=np.float32)
    arr[..., 2] = 0.9  # confidence
    payload = {"keypoints2d": arr, "det_scores": np.zeros(shape[:-2]), "timestamps": np.arange(shape[-3])}
    with path.open("wb") as f:
        pickle.dump(payload, f)


def test_index_builder_expands_consolidated_pkl(tmp_path: Path):
    pkl_dir = tmp_path / "pkls"
    pkl_dir.mkdir()
    # cAll file with 9 cameras and 64 frames
    _write_fake_pkl(pkl_dir / "gBR_sBM_cAll_d04_mBR0_ch01.pkl", (9, 64, 17, 3))
    # single-camera file
    _write_fake_pkl(pkl_dir / "gBR_sBM_c01_d04_mBR0_ch02.pkl", (64, 17, 3))
    # non-sBM should be filtered out
    _write_fake_pkl(pkl_dir / "gBR_sFM_cAll_d04_mBR0_ch03.pkl", (9, 64, 17, 3))

    out_csv = tmp_path / "index.csv"
    n = build_index_csv(pkl_dir, out_csv, situation="sBM")
    assert n == 9 + 1
    rows = load_index_csv(out_csv)
    assert len(rows) == 10
    cams = sorted({r["camera"] for r in rows})
    assert cams == [f"c{i:02d}" for i in range(1, 10)]
    # dance_label uses gXX_mXX_chXX
    assert {r["dance_label"] for r in rows} == {
        "gBR_mBR0_ch01", "gBR_mBR0_ch02"
    }
    # genre_label is just the genre prefix
    assert all(r["genre_label"] == "gBR" for r in rows)


def test_index_builder_filters_use_cameras(tmp_path: Path):
    pkl_dir = tmp_path / "pkls"
    pkl_dir.mkdir()
    _write_fake_pkl(pkl_dir / "gBR_sBM_cAll_d04_mBR0_ch01.pkl", (9, 64, 17, 3))
    out_csv = tmp_path / "index.csv"
    n = build_index_csv(pkl_dir, out_csv, situation="sBM",
                        use_cameras=["c01", "c02"])
    assert n == 2
    rows = load_index_csv(out_csv)
    assert sorted(r["camera"] for r in rows) == ["c01", "c02"]
