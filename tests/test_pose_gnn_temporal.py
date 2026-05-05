"""Tests for the temporal PoseGNN wrapper."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.pose_gnn_temporal import PoseGNNTemporalEncoder


def _window(batch: int = 2, frames: int = 16, channels: int = 3) -> torch.Tensor:
    x = torch.rand(batch, frames, 17, channels)
    x[..., 2] = torch.rand(batch, frames, 17)
    return x


def test_temporal_encoder_output_shape_and_l2_norm() -> None:
    model = PoseGNNTemporalEncoder(out_embedding_dim=64, dropout=0.0).eval()
    with torch.no_grad():
        z = model(_window(batch=4, frames=12))
    assert z.shape == (4, 64)
    assert torch.allclose(z.norm(p=2, dim=-1), torch.ones(4), atol=1e-5)


def test_temporal_encoder_supports_mask() -> None:
    model = PoseGNNTemporalEncoder(out_embedding_dim=32, dropout=0.0).eval()
    x = _window(batch=2, frames=10)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, 5:] = False
    with torch.no_grad():
        z_full = model(x)
        z_masked = model(x, mask=mask)
    assert z_full.shape == z_masked.shape == (2, 32)
    assert not torch.allclose(z_full[0], z_masked[0])


def test_frame_encoder_state_dict_compatible_with_loader(tmp_path) -> None:
    from src.compare.embedding_features import load_pose_gnn_encoder

    model = PoseGNNTemporalEncoder(frame_embedding_dim=128, out_embedding_dim=128).eval()
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model": model.frame_encoder_state_dict(),
            "embedding_dim": int(model.frame_encoder.embedding_dim),
            "temporal_model": model.state_dict(),
            "temporal_embedding_dim": int(model.embedding_dim),
            "format_version": 2,
        },
        ckpt_path,
    )
    enc, _device = load_pose_gnn_encoder(ckpt_path, device="cpu")
    out = enc(torch.rand(3, 17, 3))
    assert out.shape == (3, 128)
