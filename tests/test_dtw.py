"""DTW monotonicity + timing skew on toy signals."""
from __future__ import annotations

import numpy as np

from src.compare.dtw_align import DTWConfig, dtw_align


def test_path_is_monotonic_and_covers_endpoints():
    rng = np.random.default_rng(0)
    T, D = 60, 5
    A = rng.normal(size=(T, D)).astype(np.float32)
    B = rng.normal(size=(T, D)).astype(np.float32)
    res = dtw_align(A, B, DTWConfig(band_ratio=0.2, warp_penalty=0.0), fps=30.0)
    p = res.path
    # monotonic
    assert np.all(np.diff(p[:, 0]) >= 0)
    assert np.all(np.diff(p[:, 1]) >= 0)
    # start and end
    assert tuple(p[0]) == (0, 0)
    assert tuple(p[-1]) == (T - 1, T - 1)


def test_same_sequence_at_different_speeds_aligns_with_low_cost():
    T = 120
    t = np.linspace(0, 4 * np.pi, T).astype(np.float32)
    A = np.stack([np.sin(t), np.cos(t)], axis=-1)

    # B is the same motion played 1.5x faster with padding.
    idx = np.clip(np.arange(T) * 1.5, 0, T - 1).astype(int)
    B = A[idx].copy()

    res = dtw_align(A, B, DTWConfig(band_ratio=0.4), fps=30.0)
    assert res.cost < 0.05


def test_timing_skew_direction():
    T = 100
    t = np.linspace(0, 2 * np.pi, T).astype(np.float32)
    A = np.sin(t).reshape(-1, 1).astype(np.float32)
    # B is A shifted right by 10 samples (i.e. B is behind).
    B = np.zeros_like(A)
    B[10:] = A[:-10]
    res = dtw_align(A, B, DTWConfig(band_ratio=0.3), fps=30.0)
    assert res.timing_skew_sec > 0
