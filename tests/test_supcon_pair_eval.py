"""Tests for SupCon pair-similarity grouping metrics."""
from __future__ import annotations

import numpy as np

from src.train.train_pose_gnn_supcon import evaluate_pair_similarity


def test_pair_similarity_eval_groups_expected_pair_types() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    metas = [
        {"dance_label": "gBR_mBR0_ch01", "genre_label": "gBR", "dancer": "d04"},
        {"dance_label": "gBR_mBR0_ch01", "genre_label": "gBR", "dancer": "d05"},
        {"dance_label": "gBR_mBR0_ch02", "genre_label": "gBR", "dancer": "d04"},
        {"dance_label": "gPO_mPO0_ch01", "genre_label": "gPO", "dancer": "d04"},
        {"dance_label": "gPO_mPO0_ch01", "genre_label": "gPO", "dancer": "d05"},
        {"dance_label": "gKR_mKR0_ch01", "genre_label": "gKR", "dancer": "d04"},
    ]

    out = evaluate_pair_similarity(embeddings, metas, max_pairs_per_type=100, seed=0)
    pairs = out["pair_types"]

    assert pairs["same_dance_label_same_or_diff_dancer"]["num_pairs"] == 2
    assert pairs["same_choreography_different_dancer"]["num_pairs"] == 2
    assert pairs["same_genre_different_choreography"]["num_pairs"] == 2
    assert pairs["different_genre"]["num_pairs"] > 0
    assert (
        out["gaps"]["same_choreography_different_dancer_mean_minus_same_genre_different_choreography_mean"]
        is not None
    )
