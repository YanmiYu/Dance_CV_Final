"""Phase 10: temporal cleanup after per-frame pose inference.

Provides:
  * confidence-aware interpolation for short gaps
  * Savitzky-Golay smoothing (default) or One-Euro filter
  * outlier rejection based on impossible jumps
  * left/right swap fix for obvious flip mistakes
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from src.datasets.common import COCO_17_FLIP_PAIRS, NUM_JOINTS


@dataclass
class SmoothConfig:
    min_confidence: float = 0.2
    max_gap_frames: int = 8
    savgol_window: int = 9
    savgol_poly: int = 3
    outlier_jump_px: float = 80.0
    fix_flip_swaps: bool = True


def _confidence_interpolate(xy: np.ndarray, conf: np.ndarray, cfg: SmoothConfig) -> np.ndarray:
    """Linearly interpolate low-confidence positions over short gaps."""
    T, J, _ = xy.shape
    out = xy.copy()
    for j in range(J):
        ok = conf[:, j] >= cfg.min_confidence
        if ok.sum() < 2:
            continue
        # find runs of "bad" between good anchors
        i = 0
        while i < T:
            if ok[i]:
                i += 1
                continue
            start = i
            while i < T and not ok[i]:
                i += 1
            end = i
            gap_len = end - start
            if gap_len > cfg.max_gap_frames or start == 0 or end >= T:
                continue
            a, b = start - 1, end
            for k in range(start, end):
                t = (k - a) / float(b - a)
                out[k, j] = (1 - t) * xy[a, j] + t * xy[b, j]
    return out


def _savgol_smooth(xy: np.ndarray, cfg: SmoothConfig) -> np.ndarray:
    T = xy.shape[0]
    win = min(cfg.savgol_window, T if T % 2 == 1 else T - 1)
    if win < 3 or win <= cfg.savgol_poly:
        return xy
    if win % 2 == 0:
        win -= 1
    smoothed = xy.copy()
    smoothed[..., 0] = savgol_filter(xy[..., 0], window_length=win, polyorder=cfg.savgol_poly, axis=0)
    smoothed[..., 1] = savgol_filter(xy[..., 1], window_length=win, polyorder=cfg.savgol_poly, axis=0)
    return smoothed


def _reject_jumps(xy: np.ndarray, cfg: SmoothConfig) -> np.ndarray:
    T, J, _ = xy.shape
    out = xy.copy()
    for t in range(1, T):
        step = np.linalg.norm(out[t] - out[t - 1], axis=-1)
        bad = step > cfg.outlier_jump_px
        out[t][bad] = out[t - 1][bad]
    return out


def _fix_left_right_swaps(xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """If two paired joints (e.g. left/right wrist) swap between adjacent frames
    with significantly less total error after un-swapping, un-swap.
    """
    T = xy.shape[0]
    out = xy.copy()
    for t in range(1, T):
        for a, b in COCO_17_FLIP_PAIRS:
            prev_a, prev_b = out[t - 1, a], out[t - 1, b]
            cur_a, cur_b = out[t, a], out[t, b]
            same = np.linalg.norm(cur_a - prev_a) + np.linalg.norm(cur_b - prev_b)
            swap = np.linalg.norm(cur_a - prev_b) + np.linalg.norm(cur_b - prev_a)
            if swap + 1e-3 < 0.6 * same and conf[t, a] > 0 and conf[t, b] > 0:
                out[t, a], out[t, b] = cur_b.copy(), cur_a.copy()
    return out


def smooth_sequence(poses: np.ndarray, cfg: Optional[SmoothConfig] = None) -> np.ndarray:
    """Input ``poses``: ``(T, 17, 3)``. Returns same shape with smoothed xy + original conf."""
    cfg = cfg or SmoothConfig()
    assert poses.ndim == 3 and poses.shape[1] == NUM_JOINTS and poses.shape[2] == 3
    xy = poses[..., :2].astype(np.float32).copy()
    conf = poses[..., 2].astype(np.float32).copy()

    if cfg.fix_flip_swaps:
        xy = _fix_left_right_swaps(xy, conf)
    xy = _confidence_interpolate(xy, conf, cfg)
    xy = _reject_jumps(xy, cfg)
    xy = _savgol_smooth(xy, cfg)

    out = np.concatenate([xy, conf[..., None]], axis=-1).astype(np.float32)
    return out


def _main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Temporally smooth a (T,17,3) poses.npy file.")
    p.add_argument("--in-poses", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    poses = np.load(args.in_poses)
    smoothed = smooth_sequence(poses)
    np.save(args.out, smoothed)
    print(f"wrote smoothed poses -> {args.out}  shape={smoothed.shape}")


if __name__ == "__main__":
    _main()
