"""Streamlit demo for the integrated dance-CV pipeline.

Run with::

    streamlit run src/app/streamlit_app.py

Reads precomputed artifacts from ``data/reports/<run>/`` produced by either:
  * ``python run.py --benchmark ... --learner ... --out data/reports/<run>``
    (current pipeline -- top-level keys), or
  * ``python -m src.compare.render_report ...`` (legacy -- nested ``scores`` block).
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Dance Practice CV Tool", layout="wide")
st.title("Dance Practice -- CV Comparison")

st.caption(
    "This app reads reports produced by `python run.py --benchmark ... --learner ... "
    "--out data/reports/<name>` (or the legacy `render_report` CLI)."
)

reports_dir = Path("data/reports")
runs = sorted([p for p in reports_dir.glob("*") if p.is_dir() and (p / "report.json").exists()])
if not runs:
    st.warning(
        "No reports found under `data/reports/`. "
        "Run `python run.py --benchmark ... --learner ... --out data/reports/<name>` first."
    )
    st.stop()

choice = st.sidebar.selectbox(
    "Report", options=[r.name for r in runs], index=len(runs) - 1
)
run_dir = reports_dir / choice
report = json.loads((run_dir / "report.json").read_text())


def _is_legacy(r: dict) -> bool:
    return isinstance(r.get("scores"), dict) and "overall_score" in r["scores"]


# -----------------------------------------------------------------------------
# Top metrics.
# -----------------------------------------------------------------------------
col_top = st.columns(2)
with col_top[0]:
    if _is_legacy(report):
        s = report["scores"]
        st.metric("Overall score", f"{s['overall_score']:.1f} / 100")
        if "pose_geometry_score" in s:
            st.metric("Pose geometry", f"{s['pose_geometry_score']:.1f}")
        if "limb_angle_score" in s:
            st.metric("Limb angles", f"{s['limb_angle_score']:.1f}")
        if "timing_score" in s:
            st.metric("Timing", f"{s['timing_score']:.1f}")
    else:
        st.metric("Overall score", f"{float(report['overall_score']):.1f} / 100")
        st.metric("FPS", f"{float(report.get('fps', 0.0)):.1f}")
        st.metric("LSTM used", "yes" if report.get("lstm_used") else "no")
        models = report.get("models_enabled", [])
        if models:
            st.metric("Models enabled", ", ".join(models))

with col_top[1]:
    summary_png = run_dir / "summary.png"
    curve_png = run_dir / report.get("curve_png", "report_curves.png")
    if summary_png.exists():
        st.image(str(summary_png), caption="Per-body-part similarity")
    elif curve_png.exists():
        st.image(str(curve_png), caption="Per-stream off-pose curves")

# -----------------------------------------------------------------------------
# Feedback.
# -----------------------------------------------------------------------------
st.subheader("Feedback")
feedback = report.get("feedback", [])
if feedback:
    for line in feedback:
        st.write(line)
else:
    st.write("No feedback lines were produced.")

# -----------------------------------------------------------------------------
# Per-body-part scores / intervals.
# -----------------------------------------------------------------------------
if _is_legacy(report):
    per_part = report["scores"].get("per_body_part_score", {})
    if per_part:
        st.subheader("Per-body-part scores")
        st.table({k: [v] for k, v in per_part.items()})

    worst = report["scores"].get("worst_windows", [])
    if worst:
        st.subheader("Worst time windows")
        st.table(
            {
                "start_sec": [w["start_sec"] for w in worst],
                "end_sec": [w["end_sec"] for w in worst],
                "score": [w["score"] for w in worst],
            }
        )
else:
    intervals = report.get("intervals", [])
    if intervals:
        st.subheader("Off-pose intervals")
        st.table(
            {
                "start_s": [round(iv["start_s"], 2) for iv in intervals],
                "end_s": [round(iv["end_s"], 2) for iv in intervals],
                "part": [iv["part"] for iv in intervals],
                "mean_error": [round(iv["mean_error"], 3) for iv in intervals],
            }
        )

        # Mean error per body part (lower = better).
        agg: dict[str, list[float]] = {}
        for iv in intervals:
            agg.setdefault(iv["part"], []).append(float(iv["mean_error"]))
        st.subheader("Mean error per body part (lower = better)")
        st.table({k: [round(sum(v) / len(v), 3)] for k, v in agg.items()})
    else:
        st.write("No off-pose intervals were detected.")

    fusion = report.get("fusion_params") or {}
    if fusion:
        with st.expander("Fusion parameters"):
            st.json(fusion)

# -----------------------------------------------------------------------------
# Aligned video.
# -----------------------------------------------------------------------------
aligned = run_dir / "aligned_side.mp4"
if aligned.exists():
    st.subheader("Aligned side-by-side")
    st.video(str(aligned))
else:
    st.info("No `aligned_side.mp4` was rendered for this run.")

st.caption(f"Source report: {run_dir / 'report.json'}")
