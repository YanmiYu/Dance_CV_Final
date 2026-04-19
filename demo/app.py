"""
app.py — Streamlit UI orchestrating the full analysis pipeline.
Owner: Member 4

Run with:
    streamlit run demo/app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

# Make src/ importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_io import load_video, standardize, export_side_by_side
from pose_extraction import PoseEstimator, smooth, interpolate_missing
from normalization import normalize
from alignment import dtw_align, warping_path_to_timestamps
from scoring import (
    compute_joint_errors,
    per_part_error_over_time,
    find_off_moments,
    overall_score,
    AnalysisResult,
)
from visualization import (
    render_comparison_video,
    plot_error_timeline,
    plot_part_scores,
)
from feedback import generate_feedback, format_report

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Dance Practice Tool",
    page_icon="🕺",
    layout="wide",
)

st.title("Dance Choreography Practice Tool")
st.caption("Compare your performance against a benchmark and get timestamped feedback.")

# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    backend = st.selectbox("Pose estimator", ["mediapipe"], index=0)
    fps_target = st.slider("Target FPS (lower = faster)", 10, 30, 15)
    threshold = st.slider(
        "Deviation threshold (normalized units)",
        min_value=0.10, max_value=0.50, value=0.25, step=0.05,
    )
    min_duration = st.slider(
        "Minimum 'off' interval duration (s)",
        min_value=0.3, max_value=2.0, value=0.5, step=0.1,
    )

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

col_a, col_b = st.columns(2)
with col_a:
    bench_file = st.file_uploader("Upload Benchmark Video (10–15 s)", type=["mp4", "mov", "avi"])
with col_b:
    user_file = st.file_uploader("Upload Your Video (10–15 s)", type=["mp4", "mov", "avi"])

run_button = st.button("Run Analysis", type="primary", disabled=not (bench_file and user_file))

if not run_button:
    st.stop()

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

with st.spinner("Loading videos..."):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_bench:
        f_bench.write(bench_file.read())
        bench_path = f_bench.name
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_user:
        f_user.write(user_file.read())
        user_path = f_user.name

    bench_frames_raw, bench_src_fps = load_video(bench_path)
    user_frames_raw, user_src_fps = load_video(user_path)
    bench_frames, fps = standardize(bench_frames_raw, bench_src_fps, fps_target)
    user_frames, _   = standardize(user_frames_raw,  user_src_fps,  fps_target)

with st.spinner("Extracting pose keypoints..."):
    estimator = PoseEstimator(backend=backend)
    bench_kp_raw = estimator.extract_sequence(bench_frames)
    user_kp_raw  = estimator.extract_sequence(user_frames)
    estimator.close()

    bench_kp = interpolate_missing(smooth(bench_kp_raw))
    user_kp  = interpolate_missing(smooth(user_kp_raw))

with st.spinner("Normalizing and aligning sequences..."):
    bench_norm = normalize(bench_kp)
    user_norm  = normalize(user_kp)
    bench_aligned, user_aligned, path = dtw_align(bench_norm, user_norm)
    timestamps = warping_path_to_timestamps(path, fps)

with st.spinner("Computing deviation scores..."):
    joint_errors   = compute_joint_errors(bench_aligned, user_aligned)
    part_errors    = per_part_error_over_time(joint_errors)
    intervals      = find_off_moments(part_errors, threshold, fps, min_duration, timestamps)
    score          = overall_score(part_errors)
    feedback_lines = generate_feedback(intervals)

    # Map aligned keypoints back to pixel space for rendering
    # (use raw kp reindexed by the warping path)
    bench_idx  = [p[0] for p in path]
    user_idx   = [p[1] for p in path]
    bench_kp_px = bench_kp[bench_idx]
    user_kp_px  = user_kp[user_idx]
    bench_frames_al = [bench_frames[min(i, len(bench_frames)-1)] for i in bench_idx]
    user_frames_al  = [user_frames[min(i, len(user_frames)-1)]  for i in user_idx]

with st.spinner("Rendering comparison video..."):
    out_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    render_comparison_video(
        bench_frames_al, user_frames_al,
        bench_kp_px, user_kp_px,
        joint_errors, intervals,
        out_video, fps,
    )

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

st.divider()
st.subheader(f"Overall Similarity Score: {score:.0f} / 100")

st.video(out_video)

st.divider()
st.subheader("Error Timeline")
st.plotly_chart(plot_error_timeline(part_errors, intervals, fps), use_container_width=True)

# Per-part scores (0–100)
part_score_values = {
    part: max(0.0, min(100.0, (1.0 - float(errs.mean())) * 100))
    for part, errs in part_errors.items()
}
st.plotly_chart(plot_part_scores(part_score_values), use_container_width=True)

st.divider()
st.subheader("Timestamped Feedback")
if not intervals:
    st.success("Great job! No significant deviations detected.")
else:
    for line in feedback_lines:
        st.warning(line)

st.divider()
st.markdown(format_report(score, intervals, feedback_lines))
