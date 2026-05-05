"""Rule-based coaching report utilities for integrated fusion outputs."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.mia.dataset import PART_ORDER
from src.mia.scoring import Interval


PART_LABELS = {
    "LEFT_ARM": "left arm",
    "RIGHT_ARM": "right arm",
    "LEFT_LEG": "left leg",
    "RIGHT_LEG": "right leg",
    "TORSO": "torso",
    "HEAD": "head",
}

PART_CUES = {
    "LEFT_ARM": "Slow the arm pathway down, then match the final hand shape on the beat.",
    "RIGHT_ARM": "Mark the elbow and wrist path separately before returning to full speed.",
    "LEFT_LEG": "Rehearse the weight shift and landing shape with a smaller step first.",
    "RIGHT_LEG": "Check the knee and ankle line at the end of the phrase, then add speed.",
    "TORSO": "Keep the chest and hips organized before layering the arm details back in.",
    "HEAD": "Use the head as a timing marker and keep the gaze change crisp.",
}


def _clamp_score(value: float) -> float:
    return float(max(0.0, min(100.0, value)))


def _score_from_off(value: float) -> float:
    return _clamp_score((1.0 - float(value)) * 100.0)


def _score_from_cosine(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    sim = ((arr.clip(-1.0, 1.0) + 1.0) * 0.5).mean()
    return _clamp_score(float(sim) * 100.0)


def _quality_label(score: float) -> str:
    if score >= 90.0:
        return "excellent"
    if score >= 80.0:
        return "strong"
    if score >= 70.0:
        return "solid"
    if score >= 60.0:
        return "developing"
    if score >= 45.0:
        return "needs attention"
    return "needs focused practice"


def _format_seconds(value: float) -> str:
    return f"{float(value):.1f}s"


def _window_step(time_axis: np.ndarray, fps: float) -> float:
    if len(time_axis) > 1:
        diffs = np.diff(np.asarray(time_axis, dtype=np.float32))
        diffs = diffs[diffs > 1e-6]
        if diffs.size:
            return float(np.median(diffs))
    return 1.0 / max(float(fps), 1e-6)


def _interval_union_duration(intervals: list[Interval]) -> float:
    spans = sorted((float(iv.start_s), float(iv.end_s)) for iv in intervals if iv.end_s >= iv.start_s)
    if not spans:
        return 0.0
    total = 0.0
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        total += cur_end - cur_start
        cur_start, cur_end = start, end
    total += cur_end - cur_start
    return float(max(0.0, total))


def _part_scores(final_off: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    arr = np.asarray(final_off, dtype=np.float32)
    if arr.size == 0:
        return ({part: 0.0 for part in PART_ORDER}, {part: 1.0 for part in PART_ORDER})
    means = arr.mean(axis=0)
    off_mean = {part: float(means[i]) for i, part in enumerate(PART_ORDER)}
    scores = {part: _score_from_off(off_mean[part]) for part in PART_ORDER}
    return scores, off_mean


def _timeline_windows(
    *,
    time_axis: np.ndarray,
    final_off: np.ndarray,
    sim_avg: np.ndarray,
    part_probs_avg: np.ndarray,
    fps: float,
    seconds_per_window: float = 1.0,
) -> list[dict[str, Any]]:
    time = np.asarray(time_axis, dtype=np.float32)
    off = np.asarray(final_off, dtype=np.float32)
    sim = np.asarray(sim_avg, dtype=np.float32)
    probs = np.asarray(part_probs_avg, dtype=np.float32)
    if time.size == 0 or off.size == 0:
        return []

    start = float(time.min())
    end = float(time.max() + _window_step(time, fps))
    windows: list[dict[str, Any]] = []
    cursor = start
    while cursor < end:
        next_cursor = min(end, cursor + seconds_per_window)
        if next_cursor >= end:
            mask = (time >= cursor) & (time <= next_cursor)
        else:
            mask = (time >= cursor) & (time < next_cursor)
        if mask.any():
            off_window = off[mask]
            prob_window = probs[mask] if probs.size else off_window
            part_off = off_window.mean(axis=0)
            weakest_idx = int(np.argmax(part_off))
            confidence = _score_from_off(float(off_window.mean()))
            similarity = _score_from_cosine(sim[mask]) if sim.size else 0.0
            windows.append(
                {
                    "start_s": float(cursor),
                    "end_s": float(next_cursor),
                    "score": confidence,
                    "confidence": confidence,
                    "similarity": similarity,
                    "off_pose_mean": float(off_window.mean()),
                    "weakest_part": PART_ORDER[weakest_idx],
                    "weakest_part_score": _score_from_off(float(part_off[weakest_idx])),
                    "mean_part_probability": float(prob_window.mean()),
                }
            )
        cursor = next_cursor
    return windows


def _interval_feedback(intervals: list[Interval]) -> list[str]:
    if not intervals:
        return [
            "No sustained off-pose moments were detected. Keep the same control while raising speed or phrase length."
        ]

    lines: list[str] = []
    for iv in intervals:
        label = PART_LABELS.get(iv.part, iv.part.replace("_", " ").lower())
        cue = PART_CUES.get(iv.part, "Rehearse this phrase slowly, then return to tempo.")
        lines.append(
            f"{_format_seconds(iv.start_s)} to {_format_seconds(iv.end_s)}: "
            f"your {label} drifted most (off-pose {iv.mean_error:.2f}). {cue}"
        )
    return lines


def build_coaching_sections(
    *,
    overall_score: float,
    intervals: list[Interval],
    time_axis: np.ndarray,
    final_off: np.ndarray,
    sim_avg: np.ndarray,
    part_probs_avg: np.ndarray,
    per_model_cosine: dict[str, np.ndarray],
    fps: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create deterministic report fields from fused time-series outputs."""
    part_score, part_off_mean = _part_scores(final_off)
    ranked = sorted(part_score.items(), key=lambda item: item[1], reverse=True)
    weakest = sorted(part_score.items(), key=lambda item: item[1])
    movement_similarity = _score_from_cosine(sim_avg)
    pose_control = float(np.mean(list(part_score.values()))) if part_score else 0.0
    confidence = _score_from_off(float(np.asarray(final_off, dtype=np.float32).mean()))
    total_off_s = _interval_union_duration(intervals)
    windows = _timeline_windows(
        time_axis=time_axis,
        final_off=final_off,
        sim_avg=sim_avg,
        part_probs_avg=part_probs_avg,
        fps=fps,
    )
    model_similarity = {
        name: _score_from_cosine(values)
        for name, values in sorted(per_model_cosine.items())
    }

    strongest_part, strongest_score = ranked[0]
    weakest_part, weakest_score = weakest[0]
    summary = (
        f"Overall match is {_quality_label(overall_score)} at {overall_score:.1f}/100. "
        f"Your strongest area is the {PART_LABELS[strongest_part]} "
        f"({strongest_score:.1f}/100), and the first practice priority is the "
        f"{PART_LABELS[weakest_part]} ({weakest_score:.1f}/100)."
    )
    if not intervals:
        summary += " No sustained off-pose windows were detected, so the next goal is consistency at full speed."

    strengths = [
        f"{PART_LABELS[part].title()} control was {_quality_label(score)} ({score:.1f}/100)."
        for part, score in ranked[:2]
    ]
    if movement_similarity >= 85.0:
        strengths.append(f"Overall movement similarity stayed high ({movement_similarity:.1f}/100).")
    if confidence >= 85.0:
        strengths.append(f"Frame-by-frame confidence stayed steady ({confidence:.1f}/100).")
    strengths = strengths[:4]

    grouped: dict[str, list[Interval]] = {}
    for iv in intervals:
        grouped.setdefault(iv.part, []).append(iv)
    priorities: list[dict[str, Any]] = []
    for part, score in weakest[:3]:
        part_intervals = grouped.get(part, [])
        total_part_s = sum(max(0.0, iv.end_s - iv.start_s) for iv in part_intervals)
        window_text = "no sustained flagged window"
        if part_intervals:
            longest = max(part_intervals, key=lambda iv: iv.end_s - iv.start_s)
            window_text = f"{_format_seconds(longest.start_s)} to {_format_seconds(longest.end_s)}"
        priorities.append(
            {
                "part": part,
                "label": PART_LABELS[part],
                "score": score,
                "off_pose_mean": part_off_mean[part],
                "flagged_time_s": float(total_part_s),
                "main_window": window_text,
                "cue": PART_CUES[part],
            }
        )
    if not intervals:
        priorities = [
            {
                "part": weakest_part,
                "label": PART_LABELS[weakest_part],
                "score": weakest_score,
                "off_pose_mean": part_off_mean[weakest_part],
                "flagged_time_s": 0.0,
                "main_window": "no sustained flagged window",
                "cue": PART_CUES[weakest_part],
            }
        ]

    practice_plan = [
        f"Replay the weakest section for {PART_LABELS[weakest_part]} at half speed for 3 clean reps.",
        "Add the benchmark video back in and mark only the first and last shape of each count.",
        "Return to full tempo and record one pass, checking whether the same timestamp still appears.",
    ]
    if not intervals:
        practice_plan = [
            "Run one full-tempo pass to keep the clean baseline.",
            f"Add difficulty by sharpening the {PART_LABELS[weakest_part]} pathway without changing timing.",
            "Record a second pass and compare consistency across the whole phrase.",
        ]

    status_notes = [
        f"Canonical stream: {(extra or {}).get('canonical_stream') or 'unknown'}.",
        f"Fusion similarity weight: {float((extra or {}).get('similarity_weight', 0.0)):.2f}.",
        f"Off-pose threshold: {float((extra or {}).get('off_threshold', 0.0)):.2f}.",
    ]

    return {
        "score_breakdown": {
            "overall": _clamp_score(overall_score),
            "movement_similarity": movement_similarity,
            "pose_control": _clamp_score(pose_control),
            "confidence": confidence,
            "total_off_pose_time_s": total_off_s,
            "interval_count": len(intervals),
        },
        "per_body_part_score": part_score,
        "per_body_part_off_mean": part_off_mean,
        "model_similarity_score": model_similarity,
        "timeline_windows": windows,
        "coaching_report": {
            "headline": f"{overall_score:.1f}/100 - {_quality_label(overall_score).title()} match",
            "summary": summary,
            "strengths": strengths,
            "improvement_priorities": priorities,
            "practice_plan": practice_plan,
            "timestamped_feedback": _interval_feedback(intervals),
            "status_notes": status_notes,
        },
    }


def format_coaching_markdown(
    *,
    overall_score: float,
    report_sections: dict[str, Any],
    intervals: list[Interval],
    models_enabled: list[str] | None = None,
    lstm_status: dict[str, Any] | None = None,
) -> str:
    """Render the rich report as compact Markdown for saved artifacts."""
    coaching = report_sections["coaching_report"]
    breakdown = report_sections["score_breakdown"]
    part_scores = report_sections["per_body_part_score"]
    lines = [
        "# Dance Practice Report",
        "",
        f"## Overall Score: {overall_score:.1f} / 100",
        "",
        coaching["summary"],
        "",
        "## Scorecard",
        "",
        f"- Movement similarity: {breakdown['movement_similarity']:.1f}/100",
        f"- Pose control: {breakdown['pose_control']:.1f}/100",
        f"- Confidence: {breakdown['confidence']:.1f}/100",
        f"- Flagged off-pose time: {breakdown['total_off_pose_time_s']:.1f}s across {breakdown['interval_count']} intervals",
        "",
        "## What Went Well",
        "",
    ]
    lines.extend(f"- {item}" for item in coaching["strengths"])
    lines.extend(["", "## What To Improve First", ""])
    for item in coaching["improvement_priorities"]:
        lines.append(
            f"- {item['label'].title()}: {item['score']:.1f}/100, "
            f"main window {item['main_window']}. {item['cue']}"
        )
    lines.extend(["", "## Timestamped Feedback", ""])
    lines.extend(f"- {item}" for item in coaching["timestamped_feedback"])
    lines.extend(["", "## Practice Plan", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(coaching["practice_plan"], start=1))
    lines.extend(["", "## Body Part Scores", ""])
    for part in PART_ORDER:
        lines.append(f"- {PART_LABELS[part].title()}: {part_scores[part]:.1f}/100")
    lines.extend(["", "## Model Notes", ""])
    if models_enabled:
        lines.append(f"- Models enabled: {', '.join(models_enabled)}")
    if lstm_status:
        used = "yes" if lstm_status.get("used") else "no"
        streams = ", ".join(lstm_status.get("streams") or []) or "none"
        lines.append(f"- LSTM used: {used} ({streams})")
    lines.extend(f"- {item}" for item in coaching["status_notes"])
    lines.append("")
    return "\n".join(lines)
