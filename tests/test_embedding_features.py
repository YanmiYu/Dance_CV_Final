"""Tests for GNN embedding feature extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_load_and_encode_pose_gnn_checkpoint(tmp_path: Path) -> None:
    from src.compare.embedding_features import encode_pose_sequence, load_pose_gnn_encoder
    from src.compare.dtw_align import DTWConfig, dtw_align
    from src.models.pose_gnn import PoseGNNEncoder

    ckpt = tmp_path / "pose_gnn.pt"
    model = PoseGNNEncoder(embedding_dim=128, dropout=0.0)
    torch.save({"model": model.state_dict(), "embedding_dim": 128}, ckpt)

    loaded, device = load_pose_gnn_encoder(ckpt, device="cpu")
    poses_a = np.random.default_rng(0).random((12, 17, 3), dtype=np.float32)
    poses_b = np.random.default_rng(1).random((10, 17, 3), dtype=np.float32)

    emb_a = encode_pose_sequence(loaded, poses_a, device=device, batch_size=5)
    emb_b = encode_pose_sequence(loaded, poses_b, device=device, batch_size=5)

    assert emb_a.shape == (12, 128)
    assert emb_b.shape == (10, 128)
    np.testing.assert_allclose(np.linalg.norm(emb_a, axis=1), 1.0, atol=1e-5)

    dtw = dtw_align(emb_a, emb_b, DTWConfig(band_ratio=0.5, feature_weights=None), fps=30.0)
    assert dtw.path.shape[1] == 2
    assert dtw.cost >= 0.0
