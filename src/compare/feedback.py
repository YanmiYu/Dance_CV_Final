"""Phase 15: rule-based textual feedback.

Every sentence is traceable to quantitative values in ``ScoreResult``. We
intentionally do NOT use an LLM here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.compare.score import ScoreResult


PART_DISPLAY = {
    "head": "head/face",
    "left_arm": "left arm",
    "right_arm": "right arm",
    "torso": "torso",
    "left_leg": "left leg",
    "right_leg": "right leg",
}


@dataclass(frozen=True)
class FeedbackInterval:
    start_sec: float
    end_sec: float
    body_part: str
    level: str   # "yellow" | "red"
    message: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _severity(score: float) -> str:
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "mostly correct"
    if score >= 60:
        return "off in places"
    if score >= 40:
        return "noticeably off"
    return "quite off"


def generate_feedback(result: ScoreResult) -> List[str]:
    msgs: List[str] = []

    overall_tag = _severity(result.overall_score)
    msgs.append(
        f"Overall similarity vs the benchmark: {result.overall_score:.1f}/100 -- {overall_tag}."
    )
    msgs.append(
        f"Pose geometry: {result.pose_geometry_score:.1f}/100. "
        f"Limb angles: {result.limb_angle_score:.1f}/100. "
        f"Timing: {result.timing_score:.1f}/100."
    )

    if abs(result.timing_skew_sec) > 0.1:
        direction = "ahead of" if result.timing_skew_sec < 0 else "behind"
        msgs.append(
            f"On average, your dance is {direction} the benchmark by {abs(result.timing_skew_sec):.2f}s."
        )
    else:
        msgs.append("Your timing closely tracks the benchmark (skew < 0.1s).")

    if result.worst_parts:
        p, sc = result.worst_parts[0]
        display = PART_DISPLAY.get(p, p)
        msgs.append(
            f"The body region that drifts the most from the benchmark is your {display} ({sc:.1f}/100)."
        )
        if len(result.worst_parts) > 1:
            p2, sc2 = result.worst_parts[1]
            msgs.append(
                f"Next is your {PART_DISPLAY.get(p2, p2)} ({sc2:.1f}/100)."
            )

    if result.worst_windows:
        lines = []
        for start, end, sc in result.worst_windows:
            lines.append(f"  - {start:.1f}s -> {end:.1f}s (score {sc:.1f})")
        msgs.append("Weakest time windows (benchmark time):\n" + "\n".join(lines))

    # Per-part quick summary.
    ordered = sorted(result.per_body_part_score.items(), key=lambda kv: -kv[1])
    summary = ", ".join(f"{PART_DISPLAY.get(p, p)}={sc:.0f}" for p, sc in ordered)
    msgs.append(f"Per-region scores: {summary}.")

    return msgs


def _median_smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    """Moving median of a 1D array. ``window`` must be odd; <=1 is a no-op."""
    if window <= 1 or x.size <= window:
        return x.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(x.astype(np.float32, copy=False), (pad, pad), mode="edge")
    out = np.empty_like(x, dtype=np.float32)
    for i in range(x.size):
        out[i] = np.median(padded[i : i + window])
    return out


def _severity_from_thresholds(value: float, low_t: float, high_t: float) -> str:
    if not np.isfinite(value):
        return "green"
    if value < low_t:
        return "green"
    if value < high_t:
        return "yellow"
    return "red"


def severity_thresholds(
    per_part_err_seq: Dict[str, np.ndarray],
    *,
    quantile_low: float = 0.33,
    quantile_high: float = 0.66,
) -> Tuple[float, float]:
    """Return ``(low, high)`` thresholds from the pooled error distribution."""
    if not per_part_err_seq:
        return 0.0, 1.0
    pooled = np.concatenate(
        [np.asarray(v, dtype=np.float32).ravel() for v in per_part_err_seq.values()]
    )
    finite = pooled[np.isfinite(pooled)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.quantile(finite, quantile_low))
    high = float(np.quantile(finite, quantile_high))
    if high <= low:
        high = low + 1e-6
    return low, high


def severity_sequence_per_part(
    per_part_err_seq: Dict[str, np.ndarray],
    *,
    smoothing_window: int = 5,
    quantile_low: float = 0.33,
    quantile_high: float = 0.66,
) -> Tuple[Dict[str, List[str]], Tuple[float, float]]:
    """Smooth and threshold per-part error sequences into severity strings.

    Returns ``(severities, (low_t, high_t))`` where ``severities[part]`` is a
    list of ``"green" | "yellow" | "red"`` of the same length as the input.
    """
    if not per_part_err_seq:
        return {}, (0.0, 1.0)
    low_t, high_t = severity_thresholds(
        per_part_err_seq, quantile_low=quantile_low, quantile_high=quantile_high
    )
    out: Dict[str, List[str]] = {}
    for part, errs in per_part_err_seq.items():
        smoothed = _median_smooth_1d(np.asarray(errs, dtype=np.float32), smoothing_window)
        out[part] = [_severity_from_thresholds(float(v), low_t, high_t) for v in smoothed]
    return out, (low_t, high_t)


def _format_interval_message(part: str, start: float, end: float, level: str) -> str:
    pretty = PART_DISPLAY.get(part, part).split("/")[0]
    intensity = "significantly" if level == "red" else "moderately"
    return f"At {start:.1f}s-{end:.1f}s your {pretty} is {intensity} off"


def intervals_from_severity(
    severities: Dict[str, List[str]],
    timestamps_sec: np.ndarray,
    *,
    min_duration_sec: float = 0.3,
) -> List[FeedbackInterval]:
    """Walk per-part severity strings and emit contiguous yellow/red runs.

    Each green tag flushes the current run; runs that last at least
    ``min_duration_sec`` are kept. The level of a run is the worst severity
    seen during it (red beats yellow).
    """
    timestamps_sec = np.asarray(timestamps_sec, dtype=np.float32)
    out: List[FeedbackInterval] = []
    for part, sev_list in severities.items():
        cur = None
        for sev, t in zip(sev_list, timestamps_sec):
            if sev in ("yellow", "red"):
                if cur is None:
                    cur = {"start": float(t), "end": float(t), "level": sev}
                else:
                    cur["end"] = float(t)
                    if sev == "red":
                        cur["level"] = "red"
            else:
                if cur is not None and (cur["end"] - cur["start"]) >= min_duration_sec:
                    out.append(
                        FeedbackInterval(
                            start_sec=cur["start"],
                            end_sec=cur["end"],
                            body_part=part,
                            level=cur["level"],
                            message=_format_interval_message(
                                part, cur["start"], cur["end"], cur["level"]
                            ),
                        )
                    )
                cur = None
        if cur is not None and (cur["end"] - cur["start"]) >= min_duration_sec:
            out.append(
                FeedbackInterval(
                    start_sec=cur["start"],
                    end_sec=cur["end"],
                    body_part=part,
                    level=cur["level"],
                    message=_format_interval_message(
                        part, cur["start"], cur["end"], cur["level"]
                    ),
                )
            )
    out.sort(key=lambda fi: (fi.start_sec, fi.body_part))
    return out


def extract_feedback_intervals(
    per_part_err_seq: Dict[str, np.ndarray],
    timestamps_sec: np.ndarray,
    *,
    smoothing_window: int = 5,
    min_duration_sec: float = 0.3,
    quantile_low: float = 0.33,
    quantile_high: float = 0.66,
) -> List[FeedbackInterval]:
    """Extract contiguous yellow/red runs as user-facing intervals.

    The error sequences are median-smoothed first to suppress lone-frame
    flicker, then thresholded into severity bands; only intervals lasting at
    least ``min_duration_sec`` are surfaced.
    """
    if not per_part_err_seq:
        return []
    severities, _ = severity_sequence_per_part(
        per_part_err_seq,
        smoothing_window=smoothing_window,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )
    return intervals_from_severity(
        severities, timestamps_sec, min_duration_sec=min_duration_sec
    )
