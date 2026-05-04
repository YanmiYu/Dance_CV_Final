"""Tests for the temporal pose-GNN wrapper."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.pose_gnn_temporal import PoseGNNTemporalEncoder


def _window(B: int = 2, T: int = 16, C: int = 3) -> torch.Tensor:
    x = torch.rand(B, T, 17, C)
    x[..., 2] = torch.rand(B, T, 17)  # confidence
    return x


def test_temporal_encoder_output_shape_and_l2_norm():
    model = PoseGNNTemporalEncoder(out_embedding_dim=64, dropout=0.0).eval()
    with torch.no_grad():
        z = model(_window(B=4, T=12))
    assert z.shape == (4, 64)
    norms = z.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_temporal_encoder_supports_mask():
    model = PoseGNNTemporalEncoder(out_embedding_dim=32, dropout=0.0).eval()
    x = _window(B=2, T=10)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, 5:] = False  # mask out half of sample 0
    with torch.no_grad():
        z_full = model(x)
        z_masked = model(x, mask=mask)
    assert z_full.shape == z_masked.shape == (2, 32)
    # masking changes the embedding for the masked sample but not the other
    assert not torch.allclose(z_full[0], z_masked[0])


def test_temporal_encoder_backward_pass():
    model = PoseGNNTemporalEncoder(out_embedding_dim=32, dropout=0.0)
    x = _window(B=4, T=8)
    z = model(x)
    z.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_frame_encoder_state_dict_compatible_with_loader(tmp_path):
    """A saved checkpoint should load through src.compare.embedding_features."""
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
    enc, _ = load_pose_gnn_encoder(ckpt_path, device="cpu")
    # Frame-level forward should still work for the report pipeline.
    out = enc(torch.rand(3, 17, 3))
    assert out.shape == (3, 128)
