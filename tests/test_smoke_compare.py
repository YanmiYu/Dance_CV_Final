"""Smoke test B: a toy benchmark/user pair runs through normalize -> DTW ->
score -> feedback end to end.

Does NOT require a trained model: we synthesize poses arrays directly.
"""
from __future__ import annotations

import numpy as np

from src.compare.dtw_align import DTWConfig, dtw_align
from src.compare.feedback import generate_feedback
from src.compare.features import FeatureConfig, extract_features, framewise_distance_vector
from src.compare.normalize_pose import NormalizeConfig, normalize_sequence
from src.compare.score import ScoreConfig, compare_features


def _make_seq(T: int, seed: int, jitter: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, T, dtype=np.float32)
    pose = np.zeros((T, 17, 3), dtype=np.float32)
    for j in range(17):
        pose[:, j, 0] = 320 + 12 * j + 20 * np.sin(t + j * 0.15)
        pose[:, j, 1] = 240 + 8 * j + 20 * np.cos(t + j * 0.15)
        pose[:, j, 2] = 1.0
    if jitter > 0:
        pose[..., :2] += rng.normal(scale=jitter, size=pose[..., :2].shape).astype(np.float32)
    return pose


def test_end_to_end_compare_produces_report_and_feedback():
    bench = _make_seq(120, seed=0)
    user = _make_seq(120, seed=1, jitter=5.0)

    bn, bm = normalize_sequence(bench, NormalizeConfig())
    un, um = normalize_sequence(user, NormalizeConfig())
    bf = extract_features(bn, bm, FeatureConfig())
    uf = extract_features(un, um, FeatureConfig())
    A = framewise_distance_vector(bf)
    B = framewise_distance_vector(uf)
    dtw = dtw_align(A, B, DTWConfig(band_ratio=0.3), fps=30.0)
    result = compare_features(bf, uf, dtw, ScoreConfig(), fps=30.0)

    assert 0.0 <= result.overall_score <= 100.0
    assert set(result.per_body_part_score) == {"head", "left_arm", "right_arm", "torso", "left_leg", "right_leg"}
    assert len(result.worst_windows) > 0
    fb = generate_feedback(result)
    assert len(fb) >= 3
    assert any("benchmark" in line.lower() for line in fb)
