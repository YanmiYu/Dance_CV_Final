from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.error.keypoint_stream import build as build_keypoint_stream
from src.mia.model import TemporalErrorDetector, load_checkpoint, save_checkpoint
from src.pipeline.run_pipeline import _resolve_lstm_section


def _toy_pose(num_frames: int = 8) -> np.ndarray:
    pose = np.zeros((num_frames, 17, 3), dtype=np.float32)
    xs = np.linspace(-0.4, 0.4, 17, dtype=np.float32)
    ys = np.linspace(-0.2, 0.2, 17, dtype=np.float32)
    pose[:, :, 0] = xs[None, :]
    pose[:, :, 1] = ys[None, :]
    pose[:, :, 2] = 1.0

    # Ensure normalization has a stable torso length.
    pose[:, 5, :2] = np.array([-0.2, 0.0], dtype=np.float32)
    pose[:, 6, :2] = np.array([0.2, 0.0], dtype=np.float32)
    pose[:, 11, :2] = np.array([-0.2, 1.0], dtype=np.float32)
    pose[:, 12, :2] = np.array([0.2, 1.0], dtype=np.float32)

    # Add a little motion so DTW sees a real sequence.
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    pose[:, 9, 0] += 0.05 * np.sin(t * np.pi)
    pose[:, 10, 0] -= 0.05 * np.sin(t * np.pi)
    return pose


def test_lstm_checkpoint_round_trips_and_infers_architecture(tmp_path: Path) -> None:
    ckpt = tmp_path / "best_model.pt"
    model = TemporalErrorDetector(hidden_size=8, num_layers=1, dropout=0.0)
    save_checkpoint(model, ckpt, epoch=3, val_f1=0.42, hidden_size=8, num_layers=1, dropout=0.0)

    loaded, meta = load_checkpoint(ckpt, device="cpu")
    x = torch.zeros(1, 4, 24)

    assert meta["epoch"] == 3
    assert loaded(x).shape == (1, 4, 6)


def test_keypoint_stream_uses_lstm_probabilities_when_checkpoint_exists(tmp_path: Path) -> None:
    ckpt = tmp_path / "best_model.pt"
    save_checkpoint(TemporalErrorDetector(), ckpt, epoch=1)

    bench = _toy_pose()
    learner = _toy_pose()
    learner[:, 9, 0] += 0.15

    stream = build_keypoint_stream(
        name="hrnet",
        bench_kp=bench,
        user_kp=learner,
        fps=10.0,
        lstm_ckpt=str(ckpt),
        device="cpu",
    )

    assert stream.part_probs is not None
    assert stream.part_probs.shape == stream.part_signal.shape
    assert np.all((stream.part_probs >= 0.0) & (stream.part_probs <= 1.0))


def test_keypoint_stream_falls_back_without_checkpoint(tmp_path: Path) -> None:
    stream = build_keypoint_stream(
        name="hrnet",
        bench_kp=_toy_pose(),
        user_kp=_toy_pose(),
        fps=10.0,
        lstm_ckpt=str(tmp_path / "missing.pt"),
        device="cpu",
    )

    assert stream.part_probs is None


def test_lstm_config_override_and_require_mode(tmp_path: Path) -> None:
    ckpt = tmp_path / "best_model.pt"
    ckpt.write_bytes(b"placeholder")
    cfg = {"lstm": {"enabled": False, "checkpoint": "missing.pt"}}

    resolved, status = _resolve_lstm_section(cfg, lstm_checkpoint=str(ckpt))

    assert resolved == str(ckpt)
    assert status["enabled"] is True
    assert status["checkpoint_exists"] is True

    with pytest.raises(FileNotFoundError):
        _resolve_lstm_section(
            {"lstm": {"enabled": True, "checkpoint": str(tmp_path / "nope.pt")}},
            require_lstm=True,
        )
