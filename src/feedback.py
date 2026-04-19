"""
feedback.py — Convert scored intervals into plain-English feedback sentences.
Owner: Member 4

Rule-based: maps (body_part, error_level, duration) to a human-readable sentence.
"""

from __future__ import annotations

from scoring import Interval, THRESHOLD_GOOD, THRESHOLD_MODERATE


def generate_feedback(intervals: list[Interval]) -> list[str]:
    """Convert a list of Interval objects into plain-English feedback lines.

    Parameters
    ----------
    intervals : list[Interval]
        Output of find_off_moments(), sorted by start_s.

    Returns
    -------
    list[str]
        One sentence per interval, e.g.:
        "At 3.2 s – 5.8 s your RIGHT ARM is significantly off."
    """
    if not intervals:
        return ["Great job! No significant deviations detected."]

    lines: list[str] = []
    for iv in intervals:
        part_label = iv.part.replace("_", " ").title()
        severity = _severity_label(iv.mean_error)
        duration = iv.end_s - iv.start_s
        line = (
            f"At {iv.start_s:.1f} s \u2013 {iv.end_s:.1f} s "
            f"your {part_label} is {severity} "
            f"({duration:.1f} s window)."
        )
        lines.append(line)

    return lines


def format_report(
    overall_score: float,
    intervals: list[Interval],
    feedback_lines: list[str],
) -> str:
    """Format the full analysis result as a Markdown string for display.

    Parameters
    ----------
    overall_score : float
        0–100 similarity score.
    intervals : list[Interval]
    feedback_lines : list[str]
        Output of generate_feedback().

    Returns
    -------
    str — Markdown-formatted report.
    """
    header = f"## Overall Similarity Score: {overall_score:.0f} / 100\n"

    if not intervals:
        body = "\n**No significant deviations detected.** Great performance!\n"
    else:
        body = "\n### Timestamped Feedback\n\n"
        for line in feedback_lines:
            body += f"- {line}\n"

    return header + body


def _severity_label(mean_error: float) -> str:
    if mean_error < THRESHOLD_GOOD:
        return "slightly off"
    if mean_error < THRESHOLD_MODERATE:
        return "moderately off"
    return "significantly off"
