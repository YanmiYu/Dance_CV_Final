"""Phase 16: Streamlit demo. Built LAST, after the CLI pipeline works.

Run with::

    streamlit run src/app/streamlit_app.py

The app reads precomputed artifacts from ``data/reports/<run>/`` produced by
``src.compare.render_report`` so it never recomputes on each user click.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Dance Practice CV Tool", layout="wide")
st.title("Dance Practice -- CV Comparison")

st.caption(
    "This app assumes you have already run the CLI end-to-end: "
    "`python -m src.compare.render_report --benchmark ... --user ... --ckpt ...`."
)

# -----------------------------------------------------------------------------
# Mode 1: browse saved reports.
# -----------------------------------------------------------------------------
reports_dir = Path("data/reports")
runs = sorted([p for p in reports_dir.glob("*") if p.is_dir() and (p / "report.json").exists()])
if not runs:
    st.warning(
        "No reports found under `data/reports/`. "
        "Run `python -m src.compare.render_report --help` first."
    )
    st.stop()

choice = st.sidebar.selectbox(
    "Report", options=[r.name for r in runs] if runs else ["(none)"], index=max(0, len(runs) - 1)
)
if not runs or choice == "(none)":
    st.stop()
run_dir = reports_dir / choice
report = json.loads((run_dir / "report.json").read_text())

col_top = st.columns(2)
with col_top[0]:
    st.metric("Overall score", f"{report['scores']['overall_score']:.1f} / 100")
    st.metric("Pose geometry", f"{report['scores']['pose_geometry_score']:.1f}")
    st.metric("Limb angles", f"{report['scores']['limb_angle_score']:.1f}")
    st.metric("Timing", f"{report['scores']['timing_score']:.1f}")
with col_top[1]:
    summary_png = run_dir / "summary.png"
    if summary_png.exists():
        st.image(str(summary_png), caption="Per-body-part similarity")

st.subheader("Feedback")
for line in report.get("feedback", []):
    st.write(line)

st.subheader("Per-body-part scores")
st.table({k: [v] for k, v in report["scores"]["per_body_part_score"].items()})

st.subheader("Worst time windows")
worst = report["scores"].get("worst_windows", [])
if worst:
    st.table(
        {
            "start_sec": [w["start_sec"] for w in worst],
            "end_sec": [w["end_sec"] for w in worst],
            "score": [w["score"] for w in worst],
        }
    )
else:
    st.write("No worst-window entries produced.")

aligned = run_dir / "aligned_side.mp4"
if aligned.exists():
    st.subheader("Aligned side-by-side")
    st.video(str(aligned))
else:
    st.info("No `aligned_side.mp4` was rendered for this run (pass without `--no-video`).")

st.caption(f"Source report: {run_dir/'report.json'}")
