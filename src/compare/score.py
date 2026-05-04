"""Phase 14: interpretable deviation scores.

Given aligned benchmark/imitation feature sequences plus a DTW path, we
compute:

  * per-aligned-frame joint error
  * per-body-part scores (weighted by configs/data/compare.yaml)
  * per-time-window scores
  * top-k worst time windows
  * top-k most off body parts

We keep all intermediate values so the overall score can be explained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.compare.dtw_align import DTWResult
from src.datasets.common import BODY_PART_GROUPS, NUM_JOINTS


@dataclass
class ScoreConfig:
    body_part_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "head": 0.8, "left_arm": 1.3, "right_arm": 1.3,
            "torso": 1.1, "left_leg": 0.7, "right_leg": 0.7,
        }
    )
    score_weights: Dict[str, float] = field(
        default_factory=lambda: {"pose_geometry": 0.6, "limb_angle": 0.3, "timing": 0.1}
    )
    seconds_per_window: float = 1.0
    top_k_worst_windows: int = 3
    top_k_worst_parts: int = 2
    normalize_reference: str = "median"   # "median" | "fixed"
    fixed_reference_scale: float = 1.0    # used when normalize_reference == "fixed"


@dataclass
class ScoreResult:
    overall_score: float                   # 0..100
    pose_geometry_score: float             # 0..100
    limb_angle_score: float                # 0..100
    timing_score: float                    # 0..100
    per_body_part_score: Dict[str, float]  # 0..100
    per_body_part_error_mean: Dict[str, float]
    per_window_score: np.ndarray           # (W,)
    per_window_time_sec: np.ndarray        # (W, 2) -> (start, end) in seconds (benchmark time)
    worst_windows: List[Tuple[float, float, float]]  # (start, end, score)
    worst_parts: List[Tuple[str, float]]
    timing_skew_sec: float


def _err_to_score_0_100(mean_err: float, ref_err: float) -> float:
    """Convert a mean error into a 0-100 score.

    Interpretation: when err == 0, score is 100. When err == ref_err, score
    is 50. Decay is exponential so that worse errors keep decreasing.
    """
    if ref_err <= 0:
        return 100.0 if mean_err <= 0 else 0.0
    return float(100.0 * np.exp(-np.log(2.0) * mean_err / ref_err))


def _warp_into(
    feats: Dict[str, np.ndarray],
    indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in feats.items():
        if not isinstance(v, np.ndarray) or v.dtype == object:
            continue
        if v.shape[0] != 0 and v.shape[0] == feats.get("mask", v).shape[0]:
            out[k] = v[indices]
    return out


def _combined_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    return mask_a & mask_b


def compare_features(
    feats_a: Dict[str, np.ndarray],
    feats_b: Dict[str, np.ndarray],
    dtw: DTWResult,
    cfg: ScoreConfig | None = None,
    fps: float = 30.0,
) -> ScoreResult:
    cfg = cfg or ScoreConfig()

    # Warp both sequences along the DTW path.
    a_idx = dtw.aligned_a_idx
    b_idx = dtw.aligned_b_idx
    coords_a = feats_a["coords"][a_idx]         # (L, 17, 2)
    coords_b = feats_b["coords"][b_idx]         # (L, 17, 2)
    angles_a = feats_a.get("limb_angles", np.zeros((len(a_idx), 0), dtype=np.float32))[a_idx] if "limb_angles" in feats_a else None
    angles_b = feats_b.get("limb_angles", np.zeros((len(b_idx), 0), dtype=np.float32))[b_idx] if "limb_angles" in feats_b else None
    mask_a = feats_a["mask"][a_idx]             # (L, 17)
    mask_b = feats_b["mask"][b_idx]
    mask = mask_a & mask_b

    # Per-aligned-frame joint errors (L, 17).
    joint_err = np.linalg.norm(coords_a - coords_b, axis=-1)
    joint_err[~mask] = np.nan

    # Reference error for 0..100 mapping.
    if cfg.normalize_reference == "median":
        vals = joint_err[~np.isnan(joint_err)]
        ref_err = float(np.median(vals)) if vals.size else 1.0
        ref_err = max(ref_err, 1e-3)
    else:
        ref_err = cfg.fixed_reference_scale

    # Per body-part mean error and 0..100 score.
    per_part_err: Dict[str, float] = {}
    per_part_score: Dict[str, float] = {}
    for part, idxs in BODY_PART_GROUPS.items():
        sub = joint_err[:, list(idxs)]
        vals = sub[~np.isnan(sub)]
        mean = float(np.mean(vals)) if vals.size else 0.0
        per_part_err[part] = mean
        per_part_score[part] = _err_to_score_0_100(mean, ref_err)

    # Geometry score = weighted mean of per-part scores.
    weights = np.array([cfg.body_part_weights.get(p, 1.0) for p in BODY_PART_GROUPS], dtype=np.float32)
    part_scores_arr = np.array([per_part_score[p] for p in BODY_PART_GROUPS], dtype=np.float32)
    pose_geometry_score = float((part_scores_arr * weights).sum() / max(float(weights.sum()), 1e-6))

    # Limb-angle score.
    if angles_a is not None and angles_b is not None and angles_a.size and angles_b.size:
        ang_err = np.abs(angles_a - angles_b)
        ang_err_mean = float(np.degrees(ang_err).mean()) if ang_err.size else 0.0
        limb_angle_score = _err_to_score_0_100(ang_err_mean, 20.0)  # 20 degrees -> 50
    else:
        ang_err_mean = 0.0
        limb_angle_score = 100.0

    # Timing score from DTW skew: perfect when skew is zero.
    timing_score = _err_to_score_0_100(abs(dtw.timing_skew_sec), 0.5)  # 0.5 s -> 50

    # Overall.
    sw = cfg.score_weights
    denom = max(float(sum(sw.values())), 1e-6)
    overall = (
        sw.get("pose_geometry", 0) * pose_geometry_score
        + sw.get("limb_angle", 0) * limb_angle_score
        + sw.get("timing", 0) * timing_score
    ) / denom

    # Per-time-window score on the aligned path, in benchmark-time seconds.
    window = int(round(cfg.seconds_per_window * fps))
    window = max(window, 4)
    # we define a window over benchmark frames (feats_a timeline)
    Ta = feats_a["coords"].shape[0]
    num_windows = max(1, (Ta + window - 1) // window)
    per_window_score = np.zeros(num_windows, dtype=np.float32)
    per_window_time = np.zeros((num_windows, 2), dtype=np.float32)
    for w in range(num_windows):
        f_lo, f_hi = w * window, min((w + 1) * window, Ta)
        # Select aligned pairs whose A-index is inside [f_lo, f_hi)
        sel = (a_idx >= f_lo) & (a_idx < f_hi)
        if not sel.any():
            per_window_score[w] = np.nan
            per_window_time[w] = (f_lo / fps, f_hi / fps)
            continue
        sub = joint_err[sel]
        vals = sub[~np.isnan(sub)]
        mean = float(np.mean(vals)) if vals.size else 0.0
        per_window_score[w] = _err_to_score_0_100(mean, ref_err)
        per_window_time[w] = (f_lo / fps, f_hi / fps)

    # Top-k worst windows (lowest scores; treat NaN as best so it isn't "worst").
    score_arr = np.where(np.isnan(per_window_score), 101.0, per_window_score)
    order = np.argsort(score_arr)
    worst_windows: List[Tuple[float, float, float]] = []
    for w in order[: cfg.top_k_worst_windows]:
        if np.isnan(per_window_score[w]):
            continue
        worst_windows.append((float(per_window_time[w, 0]), float(per_window_time[w, 1]), float(per_window_score[w])))

    # Top-k worst body parts.
    worst_parts = sorted(per_part_score.items(), key=lambda kv: kv[1])[: cfg.top_k_worst_parts]

    return ScoreResult(
        overall_score=float(overall),
        pose_geometry_score=float(pose_geometry_score),
        limb_angle_score=float(limb_angle_score),
        timing_score=float(timing_score),
        per_body_part_score=per_part_score,
        per_body_part_error_mean=per_part_err,
        per_window_score=per_window_score,
        per_window_time_sec=per_window_time,
        worst_windows=worst_windows,
        worst_parts=worst_parts,
        timing_skew_sec=float(dtw.timing_skew_sec),
    )


def score_result_to_dict(result: ScoreResult) -> Dict:
    return {
        "overall_score": result.overall_score,
        "pose_geometry_score": result.pose_geometry_score,
        "limb_angle_score": result.limb_angle_score,
        "timing_score": result.timing_score,
        "per_body_part_score": result.per_body_part_score,
        "per_body_part_error_mean": result.per_body_part_error_mean,
        "per_window_score": result.per_window_score.tolist(),
        "per_window_time_sec": result.per_window_time_sec.tolist(),
        "worst_windows": [
            {"start_sec": s, "end_sec": e, "score": sc} for (s, e, sc) in result.worst_windows
        ],
        "worst_parts": [{"part": p, "score": sc} for (p, sc) in result.worst_parts],
        "timing_skew_sec": result.timing_skew_sec,
    }
