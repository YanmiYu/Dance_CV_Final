"""Scoring stability on toy examples."""
from __future__ import annotations

import numpy as np

from src.compare.dtw_align import DTWConfig, dtw_align
from src.compare.features import FeatureConfig, extract_features, framewise_distance_vector
from src.compare.normalize_pose import NormalizeConfig, normalize_sequence
from src.compare.score import ScoreConfig, compare_features


def _toy_sequence(T: int, rng: np.random.Generator, jitter: float = 0.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, T, dtype=np.float32)
    pose = np.zeros((T, 17, 3), dtype=np.float32)
    for j in range(17):
        pose[:, j, 0] = 300 + 10 * j + 20 * np.sin(t + j * 0.1)
        pose[:, j, 1] = 200 + 5 * j + 20 * np.cos(t + j * 0.1)
        pose[:, j, 2] = 1.0
    pose[..., :2] += rng.normal(scale=jitter, size=pose[..., :2].shape).astype(np.float32)
    return pose


def test_identical_sequences_score_perfect():
    rng = np.random.default_rng(0)
    pose = _toy_sequence(80, rng, jitter=0.0)
    norm, mask = normalize_sequence(pose, NormalizeConfig())
    feats = extract_features(norm, mask, FeatureConfig())
    A = framewise_distance_vector(feats)
    dtw = dtw_align(A, A, DTWConfig(band_ratio=0.3), fps=30.0)
    res = compare_features(feats, feats, dtw, ScoreConfig(), fps=30.0)
    assert res.overall_score > 95.0
    for part, s in res.per_body_part_score.items():
        assert s > 90.0, f"{part}: {s}"


def test_perturbed_sequence_scores_lower():
    rng = np.random.default_rng(0)
    base = _toy_sequence(80, rng, jitter=0.0)
    noisy = _toy_sequence(80, rng, jitter=8.0)
    for p in (base, noisy):
        n, m = normalize_sequence(p, NormalizeConfig())
        p_feats = extract_features(n, m, FeatureConfig())
        p_dist = framewise_distance_vector(p_feats)
    base_norm, base_mask = normalize_sequence(base, NormalizeConfig())
    noisy_norm, noisy_mask = normalize_sequence(noisy, NormalizeConfig())
    bf = extract_features(base_norm, base_mask, FeatureConfig())
    nf = extract_features(noisy_norm, noisy_mask, FeatureConfig())
    A = framewise_distance_vector(bf)
    B = framewise_distance_vector(nf)

    dtw_same = dtw_align(A, A, DTWConfig(band_ratio=0.3), fps=30.0)
    dtw_diff = dtw_align(A, B, DTWConfig(band_ratio=0.3), fps=30.0)
    r_same = compare_features(bf, bf, dtw_same, ScoreConfig(), fps=30.0)
    r_diff = compare_features(bf, nf, dtw_diff, ScoreConfig(), fps=30.0)
    assert r_diff.overall_score < r_same.overall_score
