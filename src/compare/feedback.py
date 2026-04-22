"""Phase 15: rule-based textual feedback.

Every sentence is traceable to quantitative values in ``ScoreResult``. We
intentionally do NOT use an LLM here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.compare.score import ScoreResult


PART_DISPLAY = {
    "head": "head/face",
    "left_arm": "left arm",
    "right_arm": "right arm",
    "torso": "torso",
    "left_leg": "left leg",
    "right_leg": "right leg",
}


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
