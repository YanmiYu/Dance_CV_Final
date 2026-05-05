"""Streamlit dashboard for dance practice comparison reports.

Run with:

    streamlit run src/app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app.report_data import discover_report_runs, load_dashboard_data


st.set_page_config(page_title="Dance Practice Review", layout="wide", initial_sidebar_state="expanded")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --paper: #fbf7f0;
            --ink: #1f2933;
            --muted: #697386;
            --line: #e6ded2;
            --teal: #0f766e;
            --coral: #d95f4f;
            --amber: #b7791f;
            --card: #ffffff;
        }
        .stApp {
            background: var(--paper);
            color: var(--ink);
        }
        div[data-testid="stHeader"] {
            background: rgba(251, 247, 240, 0.86);
        }
        section[data-testid="stSidebar"] {
            background: #f2eadf;
            border-right: 1px solid var(--line);
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        .dashboard-title {
            font-size: 2.2rem;
            font-weight: 760;
            margin-bottom: 0.15rem;
        }
        .dashboard-subtitle {
            color: var(--muted);
            font-size: 1rem;
            margin-bottom: 1.1rem;
        }
        .metric-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            min-height: 112px;
            box-shadow: 0 12px 26px rgba(31, 41, 51, 0.06);
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.75rem;
            line-height: 1.1;
            font-weight: 760;
            margin-top: 0.35rem;
        }
        .metric-note {
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: 0.3rem;
        }
        .coach-panel {
            background: #fffdf9;
            border: 1px solid var(--line);
            border-left: 4px solid var(--teal);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            min-height: 148px;
        }
        .coach-panel.focus {
            border-left-color: var(--coral);
        }
        .coach-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 760;
            margin-bottom: 0.45rem;
        }
        .coach-line {
            color: var(--ink);
            margin: 0.28rem 0;
            line-height: 1.38;
        }
        .small-muted {
            color: var(--muted);
            font-size: 0.9rem;
        }
        div[data-testid="stMetric"] {
            background: #fffdf9;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _score_level(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Strong"
    if score >= 70:
        return "Solid"
    if score >= 60:
        return "Developing"
    return "Needs practice"


def _metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _base_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fffdf9",
        font={"family": "Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif", "color": "#1f2933"},
        margin={"l": 40, "r": 24, "t": 42, "b": 36},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def _gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0},
                "bar": {"color": "#0f766e"},
                "bgcolor": "#fffdf9",
                "borderwidth": 1,
                "bordercolor": "#e6ded2",
                "steps": [
                    {"range": [0, 60], "color": "#f5d7cf"},
                    {"range": [60, 80], "color": "#f5e6bd"},
                    {"range": [80, 100], "color": "#cde8df"},
                ],
            },
        )
    )
    return _base_layout(fig, height=285)


def _body_part_chart(rows: list[dict]) -> go.Figure | None:
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("score", ascending=True)
    fig = px.bar(
        df,
        x="score",
        y="label",
        orientation="h",
        color="score",
        color_continuous_scale=[(0, "#d95f4f"), (0.6, "#d8a23a"), (1, "#0f766e")],
        range_color=[0, 100],
        text=df["score"].map(lambda value: f"{value:.1f}"),
        labels={"score": "Score", "label": ""},
        title="Body-Part Scores",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[0, 105], gridcolor="#eee6da")
    fig.update_yaxes(categoryorder="array", categoryarray=df["label"].tolist())
    return _base_layout(fig, height=360)


def _timeline_chart(data: dict) -> go.Figure | None:
    curves = data["curves"]
    time_axis = curves.get("time_axis") or []
    confidence = curves.get("confidence") or []
    similarity = curves.get("similarity") or []
    if not time_axis or (not confidence and not similarity):
        return None
    fig = go.Figure()
    if similarity:
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=similarity,
                mode="lines",
                name="Similarity",
                line={"color": "#2563eb", "width": 2.4},
            )
        )
    if confidence:
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=confidence,
                mode="lines",
                name="Confidence",
                line={"color": "#0f766e", "width": 2.4},
            )
        )
    for item in data["interval_rows"]:
        if item.get("end_s", 0) > item.get("start_s", 0):
            fig.add_vrect(
                x0=item["start_s"],
                x1=item["end_s"],
                fillcolor="#d95f4f",
                opacity=0.14,
                line_width=0,
            )
    fig.update_yaxes(range=[0, 105], title="Score", gridcolor="#eee6da")
    fig.update_xaxes(title="Benchmark time (s)", gridcolor="#f3ece2")
    fig.update_layout(title="Similarity and Confidence Over Time")
    return _base_layout(fig, height=380)


def _heatmap(data: dict) -> go.Figure | None:
    curves = data["curves"]
    off = np.asarray(curves.get("final_off") or [], dtype=np.float32)
    time_axis = curves.get("time_axis") or []
    labels = curves.get("part_labels") or []
    if off.ndim != 2 or off.size == 0 or not time_axis:
        return None
    fig = go.Figure(
        data=go.Heatmap(
            x=time_axis,
            y=labels,
            z=(off.T * 100.0),
            colorscale=[[0, "#e7f3ee"], [0.55, "#f2d08a"], [1, "#d95f4f"]],
            zmin=0,
            zmax=100,
            colorbar={"title": "Off-pose %"},
        )
    )
    fig.update_layout(title="Body-Part Off-Pose Heatmap")
    fig.update_xaxes(title="Benchmark time (s)")
    fig.update_yaxes(title="")
    return _base_layout(fig, height=390)


def _interval_chart(rows: list[dict]) -> go.Figure | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["mid_s"] = (df["start_s"] + df["end_s"]) / 2.0
    df["duration_s"] = (df["end_s"] - df["start_s"]).clip(lower=0.05)
    fig = px.scatter(
        df,
        x="mid_s",
        y="label",
        size="duration_s",
        color="score",
        color_continuous_scale=[(0, "#d95f4f"), (0.6, "#d8a23a"), (1, "#0f766e")],
        range_color=[0, 100],
        hover_data={"start_s": ":.2f", "end_s": ":.2f", "mean_error": ":.2f", "score": ":.1f"},
        labels={"mid_s": "Benchmark time (s)", "label": "", "score": "Score"},
        title="Flagged Time Windows",
    )
    fig.update_xaxes(gridcolor="#eee6da")
    fig.update_yaxes(categoryorder="total ascending")
    return _base_layout(fig, height=330)


def _model_chart(rows: list[dict]) -> go.Figure | None:
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("score", ascending=True)
    fig = px.bar(
        df,
        x="score",
        y="model",
        orientation="h",
        color="score",
        color_continuous_scale=[(0, "#d95f4f"), (0.6, "#d8a23a"), (1, "#0f766e")],
        range_color=[0, 100],
        text=df["score"].map(lambda value: f"{value:.1f}"),
        labels={"score": "Similarity Score", "model": ""},
        title="Per-Model Similarity",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[0, 105], gridcolor="#eee6da")
    return _base_layout(fig, height=300)


def _coach_panel(title: str, lines: list[str], *, focus: bool = False) -> None:
    css_class = "coach-panel focus" if focus else "coach-panel"
    body = "".join(f'<p class="coach-line">{line}</p>' for line in lines[:4])
    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="coach-title">{title}</div>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


_inject_css()

runs = discover_report_runs()
st.sidebar.title("Report")
if not runs:
    st.warning(
        "No reports found under data/reports or results. Run the integrated pipeline first, then reopen this app."
    )
    st.stop()

labels = [run.label for run in runs]
choice = st.sidebar.selectbox("Run", labels, index=0)
run = next(item for item in runs if item.label == choice)
data = load_dashboard_data(run.path)

st.sidebar.caption(f"Source: {run.path}")
st.sidebar.caption(f"Format: {data['report_type']}")
if data["models_enabled"]:
    st.sidebar.write("Models")
    st.sidebar.write(", ".join(data["models_enabled"]))
st.sidebar.write("LSTM")
st.sidebar.write("Used" if data["lstm_used"] else "Not used")

st.markdown('<div class="dashboard-title">Dance Practice Review</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="dashboard-subtitle">{data["coaching_report"].get("headline", "Performance summary")} - {choice}</div>',
    unsafe_allow_html=True,
)

breakdown = data["score_breakdown"]
body_rows = data["body_part_rows"]
strongest = max(body_rows, key=lambda row: row["score"]) if body_rows else None
weakest = min(body_rows, key=lambda row: row["score"]) if body_rows else None
off_time = float(breakdown.get("total_off_pose_time_s", 0.0))

top_cols = st.columns(4)
with top_cols[0]:
    _metric_card("Overall", f"{data['overall_score']:.1f}", _score_level(data["overall_score"]))
with top_cols[1]:
    _metric_card("Strongest Area", strongest["label"] if strongest else "N/A", f"{strongest['score']:.1f}/100" if strongest else "")
with top_cols[2]:
    _metric_card("Focus Next", weakest["label"] if weakest else "N/A", f"{weakest['score']:.1f}/100" if weakest else "")
with top_cols[3]:
    _metric_card("Flagged Time", f"{off_time:.1f}s", f"{int(breakdown.get('interval_count', 0))} intervals")

overview, timeline, body_parts, coach, diagnostics, video = st.tabs(
    ["Overview", "Timeline", "Body Parts", "Coach Report", "Model Diagnostics", "Video"]
)

with overview:
    left, right = st.columns([0.95, 1.35])
    with left:
        st.plotly_chart(_gauge(data["overall_score"]), width="stretch", key="overview_gauge")
    with right:
        _coach_panel("What went well", data["coaching_report"].get("strengths") or [])
        st.write("")
        priority_lines = []
        for item in data["coaching_report"].get("improvement_priorities") or []:
            priority_lines.append(
                f"{item.get('label', 'Priority')}: {float(item.get('score', 0.0)):.1f}/100. {item.get('cue', '')}"
            )
        _coach_panel("Focus next", priority_lines, focus=True)

    chart_cols = st.columns([1, 1])
    with chart_cols[0]:
        fig = _body_part_chart(body_rows)
        if fig is not None:
            st.plotly_chart(fig, width="stretch", key="overview_body_part_scores")
        else:
            st.info("No body-part scores are available for this run.")
    with chart_cols[1]:
        fig = _timeline_chart(data)
        if fig is not None:
            st.plotly_chart(fig, width="stretch", key="overview_timeline")
        else:
            curve_png = data["artifacts"].get("curve_png")
            if curve_png:
                st.image(curve_png, caption="Pipeline curve artifact")
            else:
                st.info("No timeline curves are available for this run.")

with timeline:
    fig = _timeline_chart(data)
    if fig is not None:
        st.plotly_chart(fig, width="stretch", key="timeline_similarity_confidence")
    heat = _heatmap(data)
    if heat is not None:
        st.plotly_chart(heat, width="stretch", key="timeline_off_pose_heatmap")
    interval_fig = _interval_chart(data["interval_rows"])
    if interval_fig is not None:
        st.plotly_chart(interval_fig, width="stretch", key="timeline_flagged_windows")
    else:
        st.success("No sustained off-pose intervals were detected.")
    if data["timeline_windows"]:
        st.dataframe(pd.DataFrame(data["timeline_windows"]), width="stretch", hide_index=True)

with body_parts:
    fig = _body_part_chart(body_rows)
    if fig is not None:
        st.plotly_chart(fig, width="stretch", key="body_parts_scores")
        st.dataframe(pd.DataFrame(body_rows), width="stretch", hide_index=True)
    else:
        st.info("No body-part breakdown is available.")
    if data["interval_rows"]:
        st.subheader("Timestamped Trouble Spots")
        st.dataframe(pd.DataFrame(data["interval_rows"]), width="stretch", hide_index=True)

with coach:
    coaching = data["coaching_report"]
    st.subheader("Summary")
    st.write(coaching.get("summary", "No coaching summary is available."))
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("What Went Well")
        for item in coaching.get("strengths") or []:
            st.write(f"- {item}")
    with col_b:
        st.subheader("What To Improve First")
        for item in coaching.get("improvement_priorities") or []:
            st.write(
                f"- {item.get('label', 'Priority')}: {float(item.get('score', 0.0)):.1f}/100. "
                f"{item.get('cue', '')}"
            )
    st.subheader("Timestamped Feedback")
    for item in coaching.get("timestamped_feedback") or []:
        st.write(f"- {item}")
    st.subheader("Practice Plan")
    for idx, item in enumerate(coaching.get("practice_plan") or [], start=1):
        st.write(f"{idx}. {item}")

with diagnostics:
    diag_cols = st.columns(3)
    with diag_cols[0]:
        st.metric("Movement similarity", f"{float(breakdown.get('movement_similarity', 0.0)):.1f}")
    with diag_cols[1]:
        st.metric("Pose control", f"{float(breakdown.get('pose_control', 0.0)):.1f}")
    with diag_cols[2]:
        st.metric("Confidence", f"{float(breakdown.get('confidence', 0.0)):.1f}")
    model_fig = _model_chart(data["model_rows"])
    if model_fig is not None:
        st.plotly_chart(model_fig, width="stretch", key="diagnostics_model_scores")
    if data["lstm"]:
        st.subheader("LSTM Status")
        st.json(data["lstm"])
    fusion = data["raw_report"].get("fusion_params") or {}
    if fusion:
        st.subheader("Fusion Parameters")
        st.json(fusion)
    with st.expander("Raw report.json"):
        st.json(json.loads(json.dumps(data["raw_report"], default=str)))

with video:
    aligned = data["artifacts"].get("aligned_video")
    if aligned:
        st.video(aligned)
    else:
        st.info("No aligned side-by-side video was rendered for this run.")
    image_cols = st.columns(2)
    with image_cols[0]:
        if data["artifacts"].get("curve_png"):
            st.image(data["artifacts"]["curve_png"], caption="Saved timeline curve")
    with image_cols[1]:
        if data["artifacts"].get("summary_png"):
            st.image(data["artifacts"]["summary_png"], caption="Saved body-part summary")
