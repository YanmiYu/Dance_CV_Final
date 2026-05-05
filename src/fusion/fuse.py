"""Fuse per-pose-model error / similarity streams into a single output.

Pipeline:
  1. Pick the canonical time axis (longest keypoint stream's timestamps).
  2. Resample each model's per-frame curves onto that axis with np.interp.
  3. Average per-part "off" signals across keypoint streams.
     Use LSTM probabilities when available, else use a thresholded geometric
     part-error signal scaled into [0, 1].
  4. Average per-frame cosine similarity across all streams (keypoint cosine
     + GNN embedding cosine).
  5. Map cosine similarity from [-1, 1] to [0, 1], then blend its inverse
     with P_off into the final off signal.
  6. Threshold final_off to find_off_moments-style intervals.
  7. Compute overall_score from final_off (mean across part * time).
  8. Generate feedback through Mia's feedback module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.error.embedding_stream import EmbeddingStream
from src.error.keypoint_stream import KeypointStream
from src.mia.dataset import PART_ORDER
from src.mia.feedback import format_report, generate_feedback
from src.mia.scoring import Interval, find_off_moments


@dataclass
class FusionResult:
    overall_score: float
    intervals: list[Interval]
    feedback: list[str]
    markdown_report: str
    time_axis: np.ndarray                        # (T',) seconds
    final_off: np.ndarray                        # (T', 6)
    sim_avg: np.ndarray                          # (T',)
    part_probs_avg: np.ndarray                   # (T', 6)
    per_model_part_signal: dict[str, np.ndarray] # name -> (T', 6) on canonical axis
    per_model_part_probs: dict[str, np.ndarray]  # name -> (T', 6) on canonical axis (LSTM only)
    per_model_cosine: dict[str, np.ndarray]      # name -> (T',) on canonical axis
    fps: float
    extra: dict = field(default_factory=dict)


def _dedupe_time_axis(src_t: np.ndarray, src_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by time and average duplicate timestamps before interpolation."""
    src_t = np.asarray(src_t, dtype=np.float32)
    src_y = np.asarray(src_y, dtype=np.float32)
    order = np.argsort(src_t, kind="stable")
    t = src_t[order]
    y = src_y[order]
    unique_t, inverse = np.unique(t, return_inverse=True)
    if len(unique_t) == len(t):
        return t, y

    if y.ndim == 1:
        out = np.zeros((len(unique_t),), dtype=np.float32)
    else:
        out = np.zeros((len(unique_t), y.shape[1]), dtype=np.float32)
    counts = np.bincount(inverse).astype(np.float32)
    for i, row in enumerate(y):
        out[inverse[i]] += row
    if out.ndim == 1:
        out /= counts
    else:
        out /= counts[:, None]
    return unique_t, out


def _interp_curve(src_t: np.ndarray, src_y: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    src_t, src_y = _dedupe_time_axis(src_t, src_y)
    if len(src_t) == 0:
        shape = (len(dst_t),) if src_y.ndim == 1 else (len(dst_t), src_y.shape[1])
        return np.zeros(shape, dtype=np.float32)
    if len(src_t) == 1:
        if src_y.ndim == 1:
            return np.full((len(dst_t),), float(src_y[0]), dtype=np.float32)
        return np.repeat(src_y[:1], len(dst_t), axis=0).astype(np.float32)
    if src_y.ndim == 1:
        return np.interp(dst_t, src_t, src_y).astype(np.float32)
    out = np.zeros((len(dst_t), src_y.shape[1]), dtype=np.float32)
    for c in range(src_y.shape[1]):
        out[:, c] = np.interp(dst_t, src_t, src_y[:, c])
    return out


def _signal_to_off_prob(signal: np.ndarray, threshold: float) -> np.ndarray:
    """Map a torso-length error signal into [0, 1] via a soft step around threshold.

    Below threshold -> 0; at threshold -> 0.5; well above -> 1.
    """
    return (1.0 / (1.0 + np.exp(-8.0 * (signal - threshold)))).astype(np.float32)


def _normalize_body_part_weights(weights: Optional[dict[str, float]]) -> dict[str, float]:
    if not weights:
        return {part: 1.0 / len(PART_ORDER) for part in PART_ORDER}
    normalized_input = {str(k).upper(): float(v) for k, v in weights.items()}
    total = sum(max(0.0, normalized_input.get(part, 0.0)) for part in PART_ORDER)
    if total <= 0.0:
        return {part: 1.0 / len(PART_ORDER) for part in PART_ORDER}
    return {
        part: max(0.0, normalized_input.get(part, 0.0)) / total
        for part in PART_ORDER
    }


def fuse(
    keypoint_streams: list[KeypointStream],
    embedding_streams: list[EmbeddingStream],
    *,
    geom_threshold: float = 0.25,
    off_threshold: float = 0.5,
    similarity_weight: float = 0.4,
    min_duration_s: float = 0.5,
    body_part_weights: Optional[dict[str, float]] = None,
) -> FusionResult:
    if not keypoint_streams:
        raise ValueError("fuse() requires at least one keypoint stream")

    # 1. Canonical axis = the longest keypoint stream's timestamps.
    canonical = max(keypoint_streams, key=lambda s: len(s.timestamps))
    time_axis = canonical.timestamps.astype(np.float32)

    per_model_part_signal: dict[str, np.ndarray] = {}
    per_model_part_probs: dict[str, np.ndarray] = {}
    per_model_cosine: dict[str, np.ndarray] = {}

    off_components: list[np.ndarray] = []
    sim_components: list[np.ndarray] = []

    # 2. Keypoint streams: resample, contribute to off + similarity averages.
    for s in keypoint_streams:
        sig = _interp_curve(s.timestamps, s.part_signal, time_axis)            # (T', 6)
        per_model_part_signal[s.name] = sig
        if s.part_probs is not None:
            probs = _interp_curve(s.timestamps, s.part_probs, time_axis)        # (T', 6)
            per_model_part_probs[s.name] = probs
            off_components.append(probs)
        else:
            off_components.append(_signal_to_off_prob(sig, geom_threshold))
        cos = _interp_curve(s.timestamps, s.cosine_sim, time_axis)              # (T',)
        per_model_cosine[s.name] = cos
        sim_components.append(cos)

    # 3. Embedding streams contribute only to similarity.
    for s in embedding_streams:
        cos = _interp_curve(s.timestamps, s.cosine_sim, time_axis)
        per_model_cosine[s.name] = cos
        sim_components.append(cos)

    part_probs_avg = np.mean(np.stack(off_components, axis=0), axis=0)          # (T', 6)
    sim_avg = np.mean(np.stack(sim_components, axis=0), axis=0).clip(-1.0, 1.0) # (T',)
    sim_score = ((sim_avg + 1.0) * 0.5).clip(0.0, 1.0)
    sim_off = (1.0 - sim_score).clip(0.0, 1.0)                                  # higher = more off

    # 4. Blend: per-part off plus broadcast similarity-off.
    final_off = ((1.0 - similarity_weight) * part_probs_avg
                 + similarity_weight * sim_off[:, None]).astype(np.float32)

    # 5. Intervals & overall score.
    part_signal_dict = {p: final_off[:, i] for i, p in enumerate(PART_ORDER)}
    intervals = find_off_moments(
        part_signal_dict,
        threshold=off_threshold,
        fps=canonical.fps,
        min_duration_s=min_duration_s,
        timestamps=time_axis,
    )

    body_part_weights = _normalize_body_part_weights(body_part_weights)
    weighted_off = sum(
        body_part_weights.get(p, 0.0) * float(part_signal_dict[p].mean())
        for p in PART_ORDER
    )
    overall_score = float(max(0.0, min(100.0, (1.0 - weighted_off) * 100.0)))

    feedback_lines = generate_feedback(intervals)
    markdown = format_report(overall_score, intervals, feedback_lines)

    return FusionResult(
        overall_score=overall_score,
        intervals=intervals,
        feedback=feedback_lines,
        markdown_report=markdown,
        time_axis=time_axis,
        final_off=final_off,
        sim_avg=sim_avg,
        part_probs_avg=part_probs_avg,
        per_model_part_signal=per_model_part_signal,
        per_model_part_probs=per_model_part_probs,
        per_model_cosine=per_model_cosine,
        fps=canonical.fps,
        extra={
            "canonical_stream": canonical.name,
            "geom_threshold": geom_threshold,
            "off_threshold": off_threshold,
            "similarity_weight": similarity_weight,
            "min_duration_s": min_duration_s,
        },
    )
