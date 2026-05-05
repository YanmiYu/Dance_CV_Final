"""Unit tests for the GNN embedding similarity score and combined report scoring."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
torch = pytest.importorskip("torch")


def _make_seq(T: int, phase: float = 0.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, T, dtype=np.float32)
    pose = np.zeros((T, 17, 3), dtype=np.float32)
    for j in range(17):
        pose[:, j, 0] = 320 + 10 * j + 20 * np.sin(t + phase + j * 0.1)
        pose[:, j, 1] = 240 + 8 * j + 20 * np.cos(t + phase + j * 0.1)
        pose[:, j, 2] = 1.0
    return pose


def _write_cached_pose(run_dir: Path, name: str, poses: np.ndarray) -> None:
    pred = run_dir / name
    pred.mkdir(parents=True)
    np.save(pred / "poses.npy", poses)
    (pred / "meta.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "num_frames": int(poses.shape[0]),
                "width": 640,
                "height": 480,
                "crop_mode": "detector_union",
            }
        )
    )


def _write_compare_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "normalization:",
                "  scale_by: torso",
                "  min_visibility: 0.2",
                "  canonical_orient: false",
                "features:",
                "  smoothing_window: 3",
                "dtw:",
                "  band_ratio: 0.5",
                "  warp_penalty: 0.05",
                "body_part_weights: {}",
                "score_weights: {}",
                "windowing:",
                "  seconds_per_window: 1.0",
                "  top_k_worst_windows: 2",
                "  top_k_worst_parts: 2",
            ]
        )
    )


def test_compute_embedding_similarity_identical_sequences():
    from src.compare.embedding_features import compute_embedding_similarity

    T, D = 30, 128
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((T, D)).astype(np.float32)
    emb = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    a_idx = np.arange(T, dtype=np.int64)

    stats = compute_embedding_similarity(emb, emb, a_idx, a_idx)
    assert stats["score"] == pytest.approx(100.0, abs=1e-4)
    assert stats["mean_cosine_similarity"] == pytest.approx(1.0, abs=1e-5)
    assert stats["mean_distance"] == pytest.approx(0.0, abs=1e-5)
    assert stats["num_aligned_steps"] == T


def test_compute_embedding_similarity_orthogonal_zero():
    from src.compare.embedding_features import compute_embedding_similarity

    T, D = 10, 4
    a = np.zeros((T, D), dtype=np.float32)
    a[:, 0] = 1.0
    b = np.zeros((T, D), dtype=np.float32)
    b[:, 1] = 1.0  # orthogonal to a
    idx = np.arange(T, dtype=np.int64)

    stats = compute_embedding_similarity(a, b, idx, idx)
    assert stats["score"] == pytest.approx(0.0, abs=1e-5)
    assert stats["mean_cosine_similarity"] == pytest.approx(0.0, abs=1e-5)
    assert stats["mean_distance"] == pytest.approx(np.sqrt(2.0), abs=1e-5)


def test_compute_embedding_similarity_negative_clamps_score_to_zero():
    from src.compare.embedding_features import compute_embedding_similarity

    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[-1.0, 0.0]], dtype=np.float32)
    idx = np.zeros(1, dtype=np.int64)

    stats = compute_embedding_similarity(a, b, idx, idx)
    assert stats["score"] == 0.0
    assert stats["mean_cosine_similarity"] == pytest.approx(-1.0, abs=1e-5)


def test_compute_embedding_similarity_handles_empty_input():
    from src.compare.embedding_features import compute_embedding_similarity

    empty = np.zeros((0, 128), dtype=np.float32)
    idx = np.zeros((0,), dtype=np.int64)
    stats = compute_embedding_similarity(empty, empty, idx, idx)
    assert stats["score"] == 0.0
    assert stats["num_aligned_steps"] == 0


def test_pairwise_cosine_similarity_returns_frame_matrix():
    from src.compare.embedding_features import pairwise_cosine_similarity

    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    sim = pairwise_cosine_similarity(a, b)

    assert sim.shape == (2, 2)
    assert sim[0, 0] == pytest.approx(1.0)
    assert sim[1, 0] == pytest.approx(0.0)
    assert sim[0, 1] == pytest.approx(1.0 / np.sqrt(2.0))


def _prepare_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "report"
    _write_cached_pose(run_dir, "benchmark_pose", _make_seq(40, phase=0.0))
    _write_cached_pose(run_dir, "user_pose", _make_seq(38, phase=0.1))
    cfg = tmp_path / "compare.yaml"
    _write_compare_config(cfg)
    return run_dir, cfg


def test_raw_features_report_has_no_embedding_fields(tmp_path: Path) -> None:
    from src.compare.render_report import run

    run_dir, cfg = _prepare_run(tmp_path)
    out = run(
        "benchmark.mp4", "user.mp4",
        "unused_model.yaml", "unused_ckpt.pt",
        str(cfg), str(run_dir),
        render_video=False,
    )
    report = json.loads((out / "report.json").read_text())

    assert "embedding" not in report
    assert set(report) == {
        "benchmark_video",
        "user_video",
        "fps_used_for_timing",
        "dtw",
        "scores",
        "feedback",
    }


def test_gnn_embedding_report_combines_geometry_and_embedding_scores(tmp_path: Path) -> None:
    from src.compare.render_report import run
    from src.models.pose_gnn import PoseGNNEncoder

    run_dir, cfg = _prepare_run(tmp_path)
    ckpt = tmp_path / "pose_gnn.pt"
    model = PoseGNNEncoder(embedding_dim=128, dropout=0.0)
    torch.save({"model": model.state_dict(), "embedding_dim": 128}, ckpt)

    out = run(
        "benchmark.mp4", "user.mp4",
        "unused_model.yaml", "unused_ckpt.pt",
        str(cfg), str(run_dir),
        render_video=False,
        alignment_method="gnn_embedding",
        gnn_checkpoint=str(ckpt),
        gnn_device="cpu",
        embedding_score_weight=0.4,
    )
    report = json.loads((out / "report.json").read_text())

    embedding = report["embedding"]
    assert embedding["alignment_method"] == "gnn_embedding"
    assert embedding["embedding_similarity_score"] is not None
    assert embedding["heatmap"] == "embedding_similarity_heatmap.png"
    assert embedding["over_time_plot"] == "embedding_similarity_over_time.png"
    assert (out / embedding["heatmap"]).exists()
    assert (out / embedding["over_time_plot"]).exists()
    assert embedding["heatmap_stats"]["shape"] == [40, 38]
    assert -1.0 <= embedding["heatmap_stats"]["min"] <= 1.0
    assert -1.0 <= embedding["heatmap_stats"]["max"] <= 1.0
    assert 0.0 <= embedding["embedding_similarity_score"] <= 100.0
    assert embedding["combined_score"] is not None
    assert embedding["embedding_score_weight"] == pytest.approx(0.4)

    raw_overall = report["scores"]["overall_score"]
    emb = embedding["embedding_similarity_score"]
    expected_combined = 0.6 * raw_overall + 0.4 * emb
    assert embedding["combined_score"] == pytest.approx(expected_combined, abs=1e-4)

    assert "scores" in report
    assert "per_body_part_score" in report["scores"]


def test_embedding_score_weight_is_clamped_to_unit_interval(tmp_path: Path) -> None:
    from src.compare.render_report import run
    from src.models.pose_gnn import PoseGNNEncoder

    run_dir, cfg = _prepare_run(tmp_path)
    ckpt = tmp_path / "pose_gnn.pt"
    model = PoseGNNEncoder(embedding_dim=128, dropout=0.0)
    torch.save({"model": model.state_dict(), "embedding_dim": 128}, ckpt)

    out = run(
        "benchmark.mp4", "user.mp4",
        "unused_model.yaml", "unused_ckpt.pt",
        str(cfg), str(run_dir),
        render_video=False,
        alignment_method="gnn_embedding",
        gnn_checkpoint=str(ckpt),
        gnn_device="cpu",
        embedding_score_weight=2.5,  # out-of-range, must clamp to 1.0
    )
    report = json.loads((out / "report.json").read_text())
    assert report["embedding"]["embedding_score_weight"] == pytest.approx(1.0)
    # With weight=1.0 the combined score is just the embedding score.
    assert report["embedding"]["combined_score"] == pytest.approx(
        report["embedding"]["embedding_similarity_score"]
    )
