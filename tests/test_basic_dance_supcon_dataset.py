"""Tests for the SupCon temporal-window dataset and balanced sampler."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.datasets.balanced_batch_sampler import DanceLabelBalancedBatchSampler
from src.datasets.basic_dance_index import build_index_csv, load_index_csv
from src.datasets.basic_dance_supcon_dataset import (
    BasicDanceSupConDataset,
    split_index_rows,
    supcon_collate,
)


def _make_pose(num_frames: int, num_cams: int = 1, *, scale: float = 100.0) -> np.ndarray:
    """Synthetic but anatomically-plausible pose array with non-zero torso."""
    if num_cams == 1:
        kp = np.zeros((num_frames, 17, 3), dtype=np.float32)
    else:
        kp = np.zeros((num_cams, num_frames, 17, 3), dtype=np.float32)
    # set joints to a rough COCO layout so normalize_sequence has a torso.
    # joints: 5 left_shoulder, 6 right_shoulder, 11 left_hip, 12 right_hip
    base = np.array([
        [0.0, -2.0],  # 0 nose
        [-0.1, -2.1], [0.1, -2.1],
        [-0.2, -2.0], [0.2, -2.0],
        [-1.0, -1.0], [1.0, -1.0],
        [-1.4, 0.0], [1.4, 0.0],
        [-1.6, 1.0], [1.6, 1.0],
        [-0.5, 1.5], [0.5, 1.5],
        [-0.6, 3.0], [0.6, 3.0],
        [-0.6, 4.5], [0.6, 4.5],
    ], dtype=np.float32) * scale + scale * 2
    if num_cams == 1:
        kp[..., :2] = base[None]
        kp[..., 2] = 0.95
    else:
        kp[..., :2] = base[None, None]
        kp[..., 2] = 0.95
    return kp


def _write(path: Path, kp: np.ndarray) -> None:
    payload = {"keypoints2d": kp,
               "det_scores": np.full(kp.shape[:-2], 0.95, dtype=np.float32),
               "timestamps": np.arange(kp.shape[-3], dtype=np.float32)}
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _build_index(tmp_path: Path) -> Path:
    pkl_dir = tmp_path / "pkls"
    pkl_dir.mkdir()
    # Two choreographies, each with a 9-camera consolidated PKL.
    _write(pkl_dir / "gBR_sBM_cAll_d04_mBR0_ch01.pkl", _make_pose(80, 9))
    _write(pkl_dir / "gBR_sBM_cAll_d04_mBR0_ch02.pkl", _make_pose(80, 9))
    _write(pkl_dir / "gHO_sBM_cAll_d05_mHO0_ch01.pkl", _make_pose(80, 9))
    out_csv = tmp_path / "index.csv"
    build_index_csv(pkl_dir, out_csv, situation="sBM")
    return out_csv


def test_dataset_window_shape_and_labels(tmp_path: Path):
    csv_path = _build_index(tmp_path)
    rows = load_index_csv(csv_path)
    ds = BasicDanceSupConDataset(rows, window_size=32, window_stride=16,
                                  random_start=False, normalize=True, seed=0)
    assert len(ds) > 0
    item = ds[0]
    assert item["pose_window"].shape == (32, 17, 3)
    assert item["mask"].shape == (32, 17)
    assert item["dance_label_id"].dim() == 0
    assert item["genre_label_id"].dim() == 0
    # values must be finite after normalisation.
    assert torch.isfinite(item["pose_window"]).all()


def test_dataset_collate(tmp_path: Path):
    csv_path = _build_index(tmp_path)
    rows = load_index_csv(csv_path)
    ds = BasicDanceSupConDataset(rows, window_size=16, random_start=False, seed=0)
    batch = supcon_collate([ds[i] for i in range(4)])
    assert batch["pose_window"].shape == (4, 16, 17, 3)
    assert batch["mask"].shape == (4, 16, 17)
    assert batch["dance_label_id"].shape == (4,)
    assert batch["genre_label_id"].shape == (4,)
    assert len(batch["meta"]) == 4


def test_split_dance_label_no_leakage(tmp_path: Path):
    csv_path = _build_index(tmp_path)
    rows = load_index_csv(csv_path)
    train, val = split_index_rows(rows, split_mode="dance_label",
                                   val_ratio=0.5, seed=1)
    assert len(train) > 0 and len(val) > 0
    assert set(r["dance_label"] for r in train).isdisjoint(
        set(r["dance_label"] for r in val))


def test_split_camera_holds_out_cameras(tmp_path: Path):
    csv_path = _build_index(tmp_path)
    rows = load_index_csv(csv_path)
    train, val = split_index_rows(rows, split_mode="camera",
                                   val_cameras=["c08", "c09"])
    assert {r["camera"] for r in val} == {"c08", "c09"}
    assert "c08" not in {r["camera"] for r in train}
    assert "c09" not in {r["camera"] for r in train}


def test_balanced_batch_sampler(tmp_path: Path):
    csv_path = _build_index(tmp_path)
    rows = load_index_csv(csv_path)
    ds = BasicDanceSupConDataset(rows, window_size=16, random_start=True, seed=0)
    sampler = DanceLabelBalancedBatchSampler(
        ds.labels(),
        n_labels_per_batch=2,
        n_samples_per_label=3,
        num_batches=4,
        seed=0,
    )
    assert len(sampler) == 4
    assert sampler.batch_size == 6
    seen = []
    for batch in sampler:
        assert len(batch) == 6
        labs = [ds.labels()[i] for i in batch]
        # Each batch has exactly 2 distinct labels with 3 samples each.
        from collections import Counter
        c = Counter(labs)
        assert len(c) == 2
        for v in c.values():
            assert v == 3
        seen.append(tuple(sorted(c.keys())))
    assert len(seen) == 4
