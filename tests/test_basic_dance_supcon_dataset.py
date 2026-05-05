"""Tests for SupCon pose-window dataset and balanced sampler."""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.datasets.balanced_batch_sampler import DanceLabelBalancedBatchSampler
from src.datasets.basic_dance_index import build_index_csv, load_index_csv
from src.datasets.basic_dance_supcon_dataset import (
    BasicDanceSupConDataset,
    filter_labels_with_min_rows,
    split_index_rows,
    supcon_collate,
)


def _make_pose(num_frames: int, num_cams: int = 1) -> np.ndarray:
    if num_cams == 1:
        kp = np.zeros((num_frames, 17, 3), dtype=np.float32)
    else:
        kp = np.zeros((num_cams, num_frames, 17, 3), dtype=np.float32)
    base = np.array(
        [
            [0.0, -2.0], [-0.1, -2.1], [0.1, -2.1], [-0.2, -2.0], [0.2, -2.0],
            [-1.0, -1.0], [1.0, -1.0], [-1.4, 0.0], [1.4, 0.0],
            [-1.6, 1.0], [1.6, 1.0], [-0.5, 1.5], [0.5, 1.5],
            [-0.6, 3.0], [0.6, 3.0], [-0.6, 4.5], [0.6, 4.5],
        ],
        dtype=np.float32,
    ) * 100.0 + 200.0
    kp[..., :2] = base
    kp[..., 2] = 0.95
    return kp


def _write(path: Path, kp: np.ndarray) -> None:
    with path.open("wb") as f:
        pickle.dump({"keypoints2d": kp}, f)


def _build_index(tmp_path: Path) -> Path:
    pkl_dir = tmp_path / "pkls"
    pkl_dir.mkdir()
    _write(pkl_dir / "gBR_sBM_cAll_d04_mBR0_ch01.pkl", _make_pose(80, 9))
    _write(pkl_dir / "gBR_sBM_cAll_d05_mBR0_ch01.pkl", _make_pose(80, 9))
    _write(pkl_dir / "gPO_sBM_cAll_d06_mPO0_ch01.pkl", _make_pose(80, 9))
    _write(pkl_dir / "gPO_sBM_cAll_d07_mPO0_ch01.pkl", _make_pose(80, 9))
    out_csv = tmp_path / "index.csv"
    build_index_csv(pkl_dir, out_csv, situation="sBM", use_cameras=["c01"], genres="all")
    return out_csv


def test_dataset_window_shape_and_metadata(tmp_path: Path) -> None:
    rows = load_index_csv(_build_index(tmp_path))
    ds = BasicDanceSupConDataset(rows, window_size=32, window_stride=16, random_start=False)
    item = ds[0]

    assert item["pose_window"].shape == (32, 17, 3)
    assert item["mask"].shape == (32, 17)
    assert item["meta"]["dancer"].startswith("d")
    assert item["meta"]["music_id"].startswith("m")
    assert item["meta"]["choreography_id"].startswith("ch")
    assert torch.isfinite(item["pose_window"]).all()


def test_dataset_collate(tmp_path: Path) -> None:
    rows = load_index_csv(_build_index(tmp_path))
    ds = BasicDanceSupConDataset(rows, window_size=16, random_start=False)
    batch = supcon_collate([ds[i] for i in range(4)])

    assert batch["pose_window"].shape == (4, 16, 17, 3)
    assert batch["mask"].shape == (4, 16, 17)
    assert batch["dance_label_id"].shape == (4,)
    assert batch["genre_label_id"].shape == (4,)
    assert len(batch["meta"]) == 4


def test_split_dance_label_no_leakage(tmp_path: Path) -> None:
    rows = load_index_csv(_build_index(tmp_path))
    train, val = split_index_rows(rows, split_mode="dance_label", val_ratio=0.5, seed=1)

    assert train and val
    assert {r["dance_label"] for r in train}.isdisjoint({r["dance_label"] for r in val})


def test_filter_labels_with_min_rows_drops_singletons() -> None:
    rows = [
        {"dance_label": "a"},
        {"dance_label": "a"},
        {"dance_label": "b"},
    ]
    filtered = filter_labels_with_min_rows(rows, min_rows_per_label=2)
    assert [r["dance_label"] for r in filtered] == ["a", "a"]


def test_balanced_batch_sampler(tmp_path: Path) -> None:
    rows = load_index_csv(_build_index(tmp_path))
    ds = BasicDanceSupConDataset(rows, window_size=16, random_start=True, seed=0)
    sampler = DanceLabelBalancedBatchSampler(
        ds.labels(),
        n_labels_per_batch=2,
        n_samples_per_label=2,
        num_batches=4,
        seed=0,
    )

    assert len(sampler) == 4
    for batch in sampler:
        counts = Counter(ds.labels()[i] for i in batch)
        assert len(counts) == 2
        assert set(counts.values()) == {2}
