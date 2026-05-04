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

    curve_path = tmp_path / "report_curves.png"
    _write_curve_plot(curve_path, result)
    assert curve_path.exists()
    assert curve_path.stat().st_size > 0
