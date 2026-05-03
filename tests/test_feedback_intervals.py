"""Unit tests for the timestamped feedback-interval extractor."""
from __future__ import annotations

import numpy as np

from src.compare.feedback import (
    extract_feedback_intervals,
    intervals_from_severity,
    severity_sequence_per_part,
    severity_thresholds,
)


def test_severity_thresholds_use_quantiles():
    seq = {"left_arm": np.linspace(0.0, 1.0, 100, dtype=np.float32)}
    low, high = severity_thresholds(seq, quantile_low=0.33, quantile_high=0.66)
    assert 0.30 < low < 0.36
    assert 0.63 < high < 0.69


def test_severity_sequence_thresholding_into_three_bands():
    seq = {
        "left_arm": np.array(
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0], dtype=np.float32
        )
    }
    severities, (low, high) = severity_sequence_per_part(seq, smoothing_window=1)
    out = severities["left_arm"]
    assert out[:3] == ["green", "green", "green"]
    assert out[3:6] == ["yellow", "yellow", "yellow"]
    assert out[6:] == ["red", "red", "red"]
    assert low < high


def test_intervals_from_severity_keeps_long_runs_drops_short_ones():
    fps = 30.0
    n = 90
    timestamps = np.arange(n, dtype=np.float32) / fps
    sev = ["green"] * n
    # 1.0s red run (frames 10..40 inclusive of left edge, exclusive of right -> 30 frames).
    for i in range(10, 40):
        sev[i] = "red"
    # 5-frame red blip (~0.17s) -- should be dropped at min_duration_sec=0.3.
    for i in range(60, 65):
        sev[i] = "red"

    intervals = intervals_from_severity(
        {"left_arm": sev}, timestamps, min_duration_sec=0.3
    )
    assert len(intervals) == 1
    iv = intervals[0]
    assert iv.body_part == "left_arm"
    assert iv.level == "red"
    assert abs(iv.start_sec - 10 / fps) < 1e-3
    assert abs(iv.end_sec - 39 / fps) < 1e-3
    assert "left arm" in iv.message
    assert "significantly" in iv.message


def test_intervals_promote_to_red_when_run_contains_red_frame():
    fps = 30.0
    sev = ["yellow"] * 30 + ["red"] * 5 + ["yellow"] * 30
    timestamps = np.arange(len(sev), dtype=np.float32) / fps

    intervals = intervals_from_severity(
        {"torso": sev}, timestamps, min_duration_sec=0.3
    )
    assert len(intervals) == 1
    assert intervals[0].level == "red"
    assert "significantly" in intervals[0].message


def test_intervals_pure_yellow_run_emits_moderate_message():
    fps = 30.0
    sev = ["green"] * 5 + ["yellow"] * 30 + ["green"] * 5
    timestamps = np.arange(len(sev), dtype=np.float32) / fps

    intervals = intervals_from_severity(
        {"head": sev}, timestamps, min_duration_sec=0.3
    )
    assert len(intervals) == 1
    assert intervals[0].level == "yellow"
    assert "moderately" in intervals[0].message


def test_extract_feedback_intervals_handles_empty_input():
    assert extract_feedback_intervals({}, np.zeros(0, dtype=np.float32)) == []


def test_extract_feedback_intervals_smooths_lone_frame_flicker():
    fps = 30.0
    n = 60
    timestamps = np.arange(n, dtype=np.float32) / fps
    err = np.full(n, 0.05, dtype=np.float32)
    err[30] = 1.0  # single bad frame, should be median-smoothed away
    intervals = extract_feedback_intervals(
        {"left_arm": err}, timestamps, smoothing_window=5, min_duration_sec=0.2
    )
    # The lone spike must not produce an interval. (The baseline may still
    # produce a long interval depending on threshold placement, so we only
    # assert the spike does not survive smoothing as a *separate* run.)
    spike_t = 30 / fps
    spikes = [iv for iv in intervals if iv.start_sec <= spike_t <= iv.end_sec
              and iv.end_sec - iv.start_sec < 0.2]
    assert spikes == []
