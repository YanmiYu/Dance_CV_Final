from __future__ import annotations

from pathlib import Path

import numpy as np

from src.error.embedding_stream import EmbeddingStream
from src.error.keypoint_stream import KeypointStream
from src.fusion.fuse import fuse
from src.pipeline.run_pipeline import _write_curve_plot


def test_fuse_combines_keypoint_lstm_and_embedding_streams(tmp_path: Path) -> None:
    timestamps = np.array([0.0, 0.1, 0.1, 0.2, 0.3], dtype=np.float32)
    part_signal = np.array(
        [
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            [0.40, 0.05, 0.05, 0.05, 0.05, 0.05],
            [0.60, 0.05, 0.05, 0.05, 0.05, 0.05],
            [0.10, 0.10, 0.30, 0.30, 0.10, 0.10],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        ],
        dtype=np.float32,
    )
    part_probs = np.clip(part_signal * 1.5, 0.0, 1.0)
    keypoint_stream = KeypointStream(
        name="hrnet",
        bench_aligned=np.zeros((5, 17, 3), dtype=np.float32),
        user_aligned=np.zeros((5, 17, 3), dtype=np.float32),
        path=[(i, i) for i in range(5)],
        timestamps=timestamps,
        part_signal=part_signal,
        part_probs=part_probs,
        cosine_sim=np.array([0.95, 0.5, 0.4, 0.85, 0.95], dtype=np.float32),
        fps=10.0,
    )
    embedding_stream = EmbeddingStream(
        name="gnn",
        cosine_sim=np.array([0.9, 0.3, 0.8, 0.95, 0.95], dtype=np.float32),
        timestamps=timestamps,
    )

    result = fuse(
        [keypoint_stream],
        [embedding_stream],
        off_threshold=0.35,
        min_duration_s=0.0,
        body_part_weights={
            "LEFT_ARM": 1.3,
            "RIGHT_ARM": 1.3,
            "LEFT_LEG": 0.7,
            "RIGHT_LEG": 0.7,
            "TORSO": 1.1,
            "HEAD": 0.8,
        },
    )

    assert 0.0 <= result.overall_score <= 100.0
    assert result.final_off.shape == (5, 6)
    assert result.sim_avg.shape == (5,)
    assert "hrnet" in result.per_model_part_probs
    assert "gnn" in result.per_model_cosine
    assert result.intervals
    assert result.score_breakdown["overall"] == result.overall_score
    assert set(result.per_body_part_score) == {
        "LEFT_ARM",
        "RIGHT_ARM",
        "LEFT_LEG",
        "RIGHT_LEG",
        "TORSO",
        "HEAD",
    }
    assert all(0.0 <= score <= 100.0 for score in result.per_body_part_score.values())
    assert min(result.per_body_part_score, key=result.per_body_part_score.get) == "LEFT_ARM"
    assert result.model_similarity_score["gnn"] <= 100.0
    assert result.timeline_windows
    assert result.coaching_report["improvement_priorities"]
    assert "What Went Well" in result.markdown_report
    assert "Practice Plan" in result.markdown_report

    curve_path = tmp_path / "report_curves.png"
    _write_curve_plot(curve_path, result)
    assert curve_path.exists()
    assert curve_path.stat().st_size > 0


def test_fuse_rich_report_handles_clean_runs() -> None:
    timestamps = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    part_signal = np.zeros((12, 6), dtype=np.float32)
    stream = KeypointStream(
        name="hrnet",
        bench_aligned=np.zeros((12, 17, 3), dtype=np.float32),
        user_aligned=np.zeros((12, 17, 3), dtype=np.float32),
        path=[(i, i) for i in range(12)],
        timestamps=timestamps,
        part_signal=part_signal,
        part_probs=np.zeros((12, 6), dtype=np.float32),
        cosine_sim=np.ones((12,), dtype=np.float32),
        fps=12.0,
    )

    result = fuse([stream], [], off_threshold=0.5)

    assert not result.intervals
    assert result.score_breakdown["interval_count"] == 0
    assert result.score_breakdown["total_off_pose_time_s"] == 0.0
    assert result.overall_score > 95.0
    assert result.coaching_report["strengths"]
    assert result.coaching_report["practice_plan"]
    assert "No sustained off-pose moments" in result.coaching_report["timestamped_feedback"][0]
