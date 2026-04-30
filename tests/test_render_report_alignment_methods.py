"""Render-report alignment method smoke tests with cached synthetic poses."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")
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
        json.dumps({"fps": 30.0, "num_frames": int(poses.shape[0]), "width": 640, "height": 480})
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


def _prepare_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "report"
    _write_cached_pose(run_dir, "benchmark_pose", _make_seq(40, phase=0.0))
    _write_cached_pose(run_dir, "user_pose", _make_seq(38, phase=0.1))
    cfg = tmp_path / "compare.yaml"
    _write_compare_config(cfg)
    return run_dir, cfg


def test_raw_features_is_render_report_default() -> None:
    from src.compare.render_report import run

    assert inspect.signature(run).parameters["alignment_method"].default == "raw_features"


def test_render_report_raw_features_smoke(tmp_path: Path) -> None:
    from src.compare.render_report import run

    run_dir, cfg = _prepare_run(tmp_path)
    out = run(
        "benchmark.mp4",
        "user.mp4",
        "unused_model.yaml",
        "unused_ckpt.pt",
        str(cfg),
        str(run_dir),
        render_video=False,
    )
    report = json.loads((out / "report.json").read_text())
    assert report["alignment"]["method"] == "raw_features"
    assert report["alignment"]["feature_shape_benchmark"][0] == 40
    assert "scores" in report


def test_render_report_gnn_embedding_smoke(tmp_path: Path) -> None:
    from src.compare.render_report import run
    from src.models.pose_gnn import PoseGNNEncoder

    run_dir, cfg = _prepare_run(tmp_path)
    ckpt = tmp_path / "pose_gnn.pt"
    model = PoseGNNEncoder(embedding_dim=128, dropout=0.0)
    torch.save({"model": model.state_dict(), "embedding_dim": 128}, ckpt)

    out = run(
        "benchmark.mp4",
        "user.mp4",
        "unused_model.yaml",
        "unused_ckpt.pt",
        str(cfg),
        str(run_dir),
        render_video=False,
        alignment_method="gnn_embedding",
        gnn_checkpoint=str(ckpt),
        gnn_device="cpu",
    )
    report = json.loads((out / "report.json").read_text())
    assert report["alignment"]["method"] == "gnn_embedding"
    assert report["alignment"]["feature_shape_benchmark"] == [40, 128]
    assert report["alignment"]["feature_shape_user"] == [38, 128]
    assert report["alignment"]["embedding_dim"] == 128
