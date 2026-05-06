"""Keypoint-stream error / similarity computation.

Given two raw (T, 17, 3) keypoint sequences from a single pose model,
this module produces everything the fusion layer needs:

  - DTW-aligned (T', 17, 3) pairs, plus the warping path and per-frame
    timestamps (Mia's fastdtw + torso-length normalization, which is
    what the LSTM was trained against).
  - per-part error signals (T', 6): mean joint error in torso-length units.
  - LSTM probabilities (T', 6) when a checkpoint is supplied; otherwise
    falls back to thresholding the per-part error signal.
  - per-frame cosine similarity between aligned, normalized keypoints
    flattened to (T', 34) — the "Direct Similarity" branch of the
    diagram's error-detection block.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.mia.alignment import dtw_align, warping_path_to_timestamps
from src.mia.dataset import PART_ORDER, build_diff_features
from src.mia.normalization import normalize
from src.mia.scoring import (
    compute_joint_errors,
    per_part_error_over_time,
)


@dataclass
class KeypointStream:
    name: str                          # pose-model name
    bench_aligned: np.ndarray          # (T', 17, 3) torso-normalized
    user_aligned: np.ndarray           # (T', 17, 3) torso-normalized
    path: list[tuple[int, int]]        # warping path
    timestamps: np.ndarray             # (T',) seconds, derived from user-side warp
    part_signal: np.ndarray            # (T', 6) per-part error in torso units (column order = PART_ORDER)
    part_probs: Optional[np.ndarray]   # (T', 6) LSTM P(off, t) — None if no LSTM ckpt
    cosine_sim: np.ndarray             # (T',) per-frame cosine on flattened normalized coords
    fps: float


def _per_part_signal(joint_errors: np.ndarray) -> np.ndarray:
    """Stack per_part_error_over_time into a (T', 6) array in PART_ORDER columns."""
    parts = per_part_error_over_time(joint_errors)
    return np.stack([parts[p] for p in PART_ORDER], axis=1).astype(np.float32)


def _keypoint_cosine(bench_al: np.ndarray, user_al: np.ndarray) -> np.ndarray:
    a = bench_al[:, :, :2].reshape(len(bench_al), -1).astype(np.float32)
    b = user_al[:, :, :2].reshape(len(user_al), -1).astype(np.float32)
    a /= np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b /= np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    return np.clip((a * b).sum(axis=1), -1.0, 1.0).astype(np.float32)


def _maybe_lstm_probs(
    bench_al: np.ndarray,
    user_al: np.ndarray,
    ckpt_path: Optional[str],
    device: str = "cpu",
    calibration: Optional[dict] = None,
) -> Optional[np.ndarray]:
    if not ckpt_path or not Path(ckpt_path).exists():
        return None
    import torch
    from src.mia.model import load_checkpoint
    model, _ = load_checkpoint(ckpt_path, device=device)
    feats = build_diff_features(bench_al, user_al)              # (T', 24)
    x = torch.from_numpy(feats).float().unsqueeze(0).to(device) # (1, T', 24)
    probs = model.predict_proba(x).squeeze(0).cpu().numpy().astype(np.float32)
    return calibrate_lstm_probabilities(probs, calibration)


def calibrate_lstm_probabilities(
    probs: np.ndarray,
    calibration: Optional[dict] = None,
) -> np.ndarray:
    """Calibrate Mia LSTM probabilities before they enter fusion.

    The imported LSTM can be over-confident on HRNet/SimpleBaseline keypoints.
    Calibration keeps the signal continuous while reducing that over-flagging.
    Supported methods:
      - ``logit``: sigmoid(logit(p) * probability_scale + probability_bias)
      - ``affine``: clip(p * probability_scale + probability_bias, 0, 1)
    """
    out = np.asarray(probs, dtype=np.float32)
    if calibration is None:
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    scale = float(calibration.get("probability_scale", 1.0))
    bias = float(calibration.get("probability_bias", 0.0))
    method = str(calibration.get("calibration_method", "logit")).lower()
    if method == "none":
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    if method == "affine":
        calibrated = (out * scale) + bias
    elif method == "logit":
        clipped = np.clip(out, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped))
        calibrated = 1.0 / (1.0 + np.exp(-((logits * scale) + bias)))
    else:
        raise ValueError(f"unknown LSTM calibration method: {method!r}")
    return np.clip(calibrated, 0.0, 1.0).astype(np.float32, copy=False)


def build(
    name: str,
    bench_kp: np.ndarray,
    user_kp: np.ndarray,
    fps: float,
    *,
    lstm_ckpt: Optional[str] = None,
    lstm_calibration: Optional[dict] = None,
    device: str = "cpu",
) -> KeypointStream:
    """Build a complete KeypointStream from raw (T, 17, 3) sequences."""
    bench_n = normalize(bench_kp.astype(np.float32))
    user_n = normalize(user_kp.astype(np.float32))
    bench_al, user_al, path = dtw_align(bench_n, user_n)
    timestamps = warping_path_to_timestamps(path, fps)

    joint_errs = compute_joint_errors(bench_al, user_al)
    part_signal = _per_part_signal(joint_errs)
    cosine_sim = _keypoint_cosine(bench_al, user_al)
    part_probs = _maybe_lstm_probs(
        bench_al,
        user_al,
        lstm_ckpt,
        device=device,
        calibration=lstm_calibration,
    )

    return KeypointStream(
        name=name,
        bench_aligned=bench_al,
        user_aligned=user_al,
        path=path,
        timestamps=timestamps,
        part_signal=part_signal,
        part_probs=part_probs,
        cosine_sim=cosine_sim,
        fps=float(fps),
    )
