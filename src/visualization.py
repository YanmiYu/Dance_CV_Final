"""
visualization.py — Skeleton overlay, comparison video renderer, and timeline chart.
Owner: Member 4

Color coding by error level (normalized units):
  Good     < 0.15  → green  #2ecc71
  Moderate 0.15–0.35 → yellow #f39c12
  Off      > 0.35  → red    #e74c3c
"""

from __future__ import annotations

import numpy as np
import cv2
import plotly.graph_objects as go

from scoring import BODY_PARTS, THRESHOLD_GOOD, THRESHOLD_MODERATE, Interval

KeypointSequence = np.ndarray  # shape (T, 17, 3)

# BGR colors for OpenCV drawing
_COLOR_GOOD     = (113, 204, 46)   # green
_COLOR_MODERATE = (18, 156, 243)   # yellow (BGR)
_COLOR_OFF      = (60, 76, 231)    # red (BGR)

# COCO skeleton edges: pairs of joint indices
SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]


def _error_to_color(error: float) -> tuple[int, int, int]:
    if error < THRESHOLD_GOOD:
        return _COLOR_GOOD
    if error < THRESHOLD_MODERATE:
        return _COLOR_MODERATE
    return _COLOR_OFF


def overlay_skeleton(
    frame: np.ndarray,
    kp: np.ndarray,
    joint_errors: np.ndarray | None = None,
) -> np.ndarray:
    """Draw a skeleton on a copy of frame, optionally color-coded by error.

    Parameters
    ----------
    frame : np.ndarray, BGR image.
    kp : np.ndarray, shape (17, 3) — [x_px, y_px, confidence] in pixel coords.
    joint_errors : np.ndarray or None, shape (17,)
        Per-joint error values for color coding. If None, all joints are drawn green.

    Returns
    -------
    np.ndarray — annotated frame.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Draw edges
    for (j1, j2) in SKELETON_EDGES:
        if kp[j1, 2] < 0.3 or kp[j2, 2] < 0.3:
            continue
        p1 = (int(kp[j1, 0]), int(kp[j1, 1]))
        p2 = (int(kp[j2, 0]), int(kp[j2, 1]))
        color = _COLOR_GOOD
        if joint_errors is not None:
            color = _error_to_color(max(joint_errors[j1], joint_errors[j2]))
        cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)

    # Draw joints
    for j in range(17):
        if kp[j, 2] < 0.3:
            continue
        px, py = int(kp[j, 0]), int(kp[j, 1])
        color = _COLOR_GOOD
        if joint_errors is not None:
            color = _error_to_color(joint_errors[j])
        cv2.circle(out, (px, py), 5, color, -1, cv2.LINE_AA)

    return out


def render_comparison_video(
    bench_frames: list[np.ndarray],
    user_frames: list[np.ndarray],
    bench_kp: KeypointSequence,
    user_kp: KeypointSequence,
    joint_errors: np.ndarray,
    intervals: list[Interval],
    out_path: str,
    fps: float = 30.0,
    timestamps: np.ndarray | None = None,
) -> None:
    """Write a side-by-side MP4 with color-coded skeleton overlays.

    During "off" intervals a red border is drawn on the user's panel.

    Parameters
    ----------
    bench_frames, user_frames : frame lists (aligned length T').
    bench_kp, user_kp : KeypointSequence shape (T', 17, 3) — pixel coords.
    joint_errors : np.ndarray shape (T', 17).
    intervals : list[Interval] from find_off_moments().
    out_path : str — output .mp4 path.
    fps : float
    timestamps : np.ndarray shape (T',) — real time in seconds for each aligned
        frame (from warping_path_to_timestamps). If supplied, the red border
        and timestamp label use DTW-accurate time; otherwise falls back to t/fps.
    """
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    T = min(len(bench_frames), len(user_frames), len(bench_kp), len(user_kp))
    h, w = bench_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w * 2, h))

    # Build sorted interval list of (start_s, end_s) for O(1) lookup per frame
    off_spans = [(iv.start_s, iv.end_s) for iv in intervals]

    def _is_off(t: int) -> bool:
        t_s = float(timestamps[t]) if timestamps is not None else t / fps
        return any(s <= t_s <= e for s, e in off_spans)

    for t in range(T):
        t_s = float(timestamps[t]) if timestamps is not None else t / fps
        errors_t = joint_errors[t]

        bf = overlay_skeleton(bench_frames[t], bench_kp[t])
        uf = overlay_skeleton(user_frames[t], user_kp[t], errors_t)

        if _is_off(t):
            cv2.rectangle(uf, (0, 0), (w - 1, h - 1), (0, 0, 220), 6)

        cv2.putText(bf, f"{t_s:.1f}s", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1, cv2.LINE_AA)

        combined = np.concatenate([bf, uf], axis=1)
        writer.write(combined)

    writer.release()


def plot_error_timeline(
    part_errors: dict[str, np.ndarray],
    intervals: list[Interval],
    fps: float = 30.0,
    timestamps: np.ndarray | None = None,
) -> go.Figure:
    """Plotly line chart: time (s) vs. per-part error, with shaded "off" regions.

    Parameters
    ----------
    timestamps : np.ndarray or None
        Real per-frame timestamps from warping_path_to_timestamps(). When
        supplied the x-axis uses DTW-accurate time; otherwise falls back to
        uniform t/fps spacing.

    Returns a Plotly Figure for st.plotly_chart().
    """
    fig = go.Figure()

    colors = {
        "LEFT_ARM":   "#3498db",
        "RIGHT_ARM":  "#e74c3c",
        "LEFT_LEG":   "#2ecc71",
        "RIGHT_LEG":  "#f39c12",
        "TORSO":      "#9b59b6",
        "HEAD":       "#1abc9c",
    }

    for part, errors in part_errors.items():
        T = len(errors)
        times = timestamps if timestamps is not None else np.arange(T) / fps
        fig.add_trace(go.Scatter(
            x=times, y=errors,
            mode="lines",
            name=part.replace("_", " ").title(),
            line=dict(color=colors.get(part, "#888888"), width=1.5),
        ))

    # Shaded "off" regions
    for iv in intervals:
        fig.add_vrect(
            x0=iv.start_s, x1=iv.end_s,
            fillcolor="red", opacity=0.12,
            layer="below", line_width=0,
        )

    # Threshold lines
    fig.add_hline(y=THRESHOLD_GOOD, line_dash="dot",
                  line_color="green", annotation_text="good")
    fig.add_hline(y=THRESHOLD_MODERATE, line_dash="dot",
                  line_color="orange", annotation_text="moderate")

    fig.update_layout(
        title="Per-Body-Part Error Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Normalized Error",
        legend_title="Body Part",
        template="plotly_white",
    )
    return fig


def plot_part_scores(part_scores: dict[str, float]) -> go.Figure:
    """Plotly bar chart of per-part scores (0–100)."""
    parts = [p.replace("_", " ").title() for p in part_scores]
    scores = list(part_scores.values())

    bar_colors = [
        "#2ecc71" if s >= 70 else "#f39c12" if s >= 40 else "#e74c3c"
        for s in scores
    ]

    fig = go.Figure(go.Bar(
        x=parts, y=scores,
        marker_color=bar_colors,
        text=[f"{s:.0f}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        title="Per-Body-Part Similarity Score",
        yaxis=dict(range=[0, 110], title="Score (0–100)"),
        xaxis_title="Body Part",
        template="plotly_white",
    )
    return fig
