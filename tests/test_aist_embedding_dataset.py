"""Tests for AIST++ embedding triplet sampling."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.datasets.aist_embedding_dataset import AISTEmbeddingDataset


def _write_keypoints(path: Path, frames: int = 80, offset: float = 0.0) -> None:
    rng = np.random.default_rng(0)
    kps = rng.uniform(0, 1000, size=(frames, 17, 3)).astype(np.float32)
    kps[..., 0] += offset
    kps[..., 1] += offset
    kps[..., 2] = rng.uniform(0.25, 1.0, size=(frames, 17)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"keypoints2d": kps}, f)


def test_dataset_length_and_shapes(tmp_path: Path) -> None:
    _write_keypoints(tmp_path / "gBR_sBM_c01_d04_mBR0_ch01.pkl", frames=80)
    _write_keypoints(tmp_path / "gBR_sBM_c01_d04_mBR0_ch02.pkl", frames=70)

    ds = AISTEmbeddingDataset(tmp_path, positive_range=5, seed=0, verbose=False)
    sample = ds[40]

    assert len(ds) == 150
    assert set(sample) == {"anchor", "positive", "negative"}
    assert sample["anchor"].shape == (17, 3)
    assert sample["positive"].shape == (17, 3)
    assert sample["negative"].shape == (17, 3)
    assert torch.is_tensor(sample["anchor"])
    assert float(sample["anchor"][..., :2].min()) >= 0.0
    assert float(sample["anchor"][..., :2].max()) <= 1.0


def test_positive_frame_is_near_anchor(tmp_path: Path) -> None:
    _write_keypoints(tmp_path / "gBR_sBM_c01_d04_mBR0_ch01.pkl", frames=80)

    ds = AISTEmbeddingDataset(tmp_path, positive_range=5, seed=1, verbose=False)
    anchor, positive, _ = ds.sample_triplet_indices(40)

    assert anchor[0] == positive[0]
    assert positive[1] != anchor[1]
    assert abs(positive[1] - anchor[1]) <= 5


def test_far_frame_negative_is_far_from_anchor(tmp_path: Path) -> None:
    _write_keypoints(tmp_path / "gBR_sBM_c01_d04_mBR0_ch01.pkl", frames=80)

    ds = AISTEmbeddingDataset(tmp_path, positive_range=5, seed=2, verbose=False)
    anchor, _, negative = ds.sample_triplet_indices(40)

    assert anchor[0] == negative[0]
    assert abs(negative[1] - anchor[1]) > 30


def test_different_video_negative_prefers_different_choreography(tmp_path: Path) -> None:
    _write_keypoints(tmp_path / "gBR_sBM_c01_d04_mBR0_ch01.pkl", frames=80)
    _write_keypoints(tmp_path / "gBR_sBM_c01_d04_mBR0_ch02.pkl", frames=80)

    ds = AISTEmbeddingDataset(
        tmp_path,
        positive_range=5,
        negative_strategy="different_video",
        seed=3,
        verbose=False,
    )
    anchor, _, negative = ds.sample_triplet_indices(40)

    assert anchor[0] != negative[0]
    assert anchor[0].endswith("ch01")
    assert negative[0].endswith("ch02")
