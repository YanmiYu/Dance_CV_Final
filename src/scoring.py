"""
scoring.py — Per-joint error computation, find_off_moments, and overall score.
Owner: Member 2

Pipeline:
  aligned sequences
    → compute_joint_errors()     shape (T', 17)
    → per_part_error_over_time() dict[part → (T',)]
    → find_off_moments()         List[Interval]
    → overall_score()            float 0–100
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KeypointSequence = np.ndarray  # shape (T, 17, 3)

# COCO body-part groupings
BODY_PARTS: dict[str, list[int]] = {
    "LEFT_ARM":   [5, 7, 9],       # L shoulder, L elbow, L wrist
    "RIGHT_ARM":  [6, 8, 10],      # R shoulder, R elbow, R wrist
    "LEFT_LEG":   [11, 13, 15],    # L hip, L knee, L ankle
    "RIGHT_LEG":  [12, 14, 16],    # R hip, R knee, R ankle
    "TORSO":      [5, 6, 11, 12],  # shoulders + hips
    "HEAD":       [0],             # nose
}

# Default body-part weights for the overall score (sum to 1.0)
DEFAULT_WEIGHTS: dict[str, float] = {
    "LEFT_ARM":  1 / 6,
    "RIGHT_ARM": 1 / 6,
    "LEFT_LEG":  1 / 6,
    "RIGHT_LEG": 1 / 6,
    "TORSO":     1 / 6,
    "HEAD":      1 / 6,
}

# Deviation thresholds for color coding (normalized units)
THRESHOLD_GOOD     = 0.15
THRESHOLD_MODERATE = 0.35


@dataclass
class Interval:
    """A contiguous time window where a body part deviates beyond the threshold."""
    start_s:    float
    end_s:      float
    part:       str
    mean_error: float

    def __str__(self) -> str:
        return (
            f"At {self.start_s:.1f} s – {self.end_s:.1f} s "
            f"your {self.part.replace('_', ' ')} is off "
            f"(mean error {self.mean_error:.2f})."
        )


@dataclass
class AnalysisResult:
    """Full output of the scoring pipeline."""
    overall_score:   float
    intervals:       list[Interval]
    part_errors:     dict[str, np.ndarray]   # part → (T',) error time series
    warping_path:    list[tuple[int, int]]
    feedback_lines:  list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_joint_errors(
    bench_aligned: KeypointSequence,
    user_aligned: KeypointSequence,
) -> np.ndarray:
    """Per-joint Euclidean error at each aligned frame.

    Parameters
    ----------
    bench_aligned, user_aligned : np.ndarray, shape (T', 17, 3)
        DTW-aligned normalized sequences.

    Returns
    -------
    np.ndarray, shape (T', 17)
        errors[t, j] = Euclidean distance between benchmark and user at joint j, frame t.
    """
    diff = bench_aligned[:, :, :2] - user_aligned[:, :, :2]  # (T', 17, 2)
    return np.linalg.norm(diff, axis=2)                        # (T', 17)


def per_part_error_over_time(
    joint_errors: np.ndarray,
) -> dict[str, np.ndarray]:
    """Average joint errors within each body part, per frame.

    Parameters
    ----------
    joint_errors : np.ndarray, shape (T', 17)

    Returns
    -------
    dict mapping part name → np.ndarray of shape (T',)
    """
    return {
        part: joint_errors[:, joints].mean(axis=1)
        for part, joints in BODY_PARTS.items()
    }


def find_off_moments(
    part_errors: dict[str, np.ndarray],
    threshold: float = 0.25,
    fps: float = 30.0,
    min_duration_s: float = 0.5,
    timestamps: np.ndarray | None = None,
) -> list[Interval]:
    """Detect contiguous time windows where a body part exceeds the error threshold.

    Parameters
    ----------
    part_errors : dict[str, np.ndarray]
        Output of per_part_error_over_time().
    threshold : float
        Normalized-unit error above which a frame is considered "off".
    fps : float
        Frames per second of the aligned sequence (used to convert frame index → seconds).
    min_duration_s : float
        Intervals shorter than this are discarded (single-frame noise filter).
    timestamps : np.ndarray or None
        If provided (shape (T',)), use these timestamps instead of frame_idx / fps.
        Useful when the warping path produces non-uniform timestamps.

    Returns
    -------
    list[Interval]
        Sorted by start_s.
    """
    intervals: list[Interval] = []
    min_frames = max(1, int(min_duration_s * fps))

    for part, errors in part_errors.items():
        above = errors > threshold
        in_run = False
        run_start = 0

        for t, flag in enumerate(above):
            if flag and not in_run:
                in_run = True
                run_start = t
            elif not flag and in_run:
                run_len = t - run_start
                if run_len >= min_frames:
                    intervals.append(_make_interval(
                        part, run_start, t - 1, errors, fps, timestamps))
                in_run = False

        if in_run:
            t = len(above) - 1
            run_len = t - run_start + 1
            if run_len >= min_frames:
                intervals.append(_make_interval(
                    part, run_start, t, errors, fps, timestamps))

    intervals.sort(key=lambda iv: iv.start_s)
    return intervals


def overall_score(
    part_errors: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
    scale_factor: float = 100.0,
) -> float:
    """Compute a single 0–100 similarity score.

    score = 100 × (1 − weighted_mean_error / max_expected_error)
    Clamped to [0, 100].
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    max_expected = 1.0  # 1 torso-length is treated as maximally wrong
    weighted_err = sum(
        weights.get(part, 0.0) * float(errors.mean())
        for part, errors in part_errors.items()
    )
    return float(max(0.0, min(100.0, (1.0 - weighted_err / max_expected) * scale_factor)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_interval(
    part: str,
    t_start: int,
    t_end: int,
    errors: np.ndarray,
    fps: float,
    timestamps: np.ndarray | None,
) -> Interval:
    if timestamps is not None:
        start_s = float(timestamps[t_start])
        end_s = float(timestamps[t_end])
    else:
        start_s = t_start / fps
        end_s = t_end / fps
    mean_err = float(errors[t_start: t_end + 1].mean())
    return Interval(start_s=start_s, end_s=end_s, part=part, mean_error=mean_err)
