"""Tests for the minimal pose GNN encoder."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _make_pose_batch(batch_size: int = 4) -> torch.Tensor:
    x = torch.rand(batch_size, 17, 3)
    x[..., 2] = torch.rand(batch_size, 17)
    return x


def test_pose_gnn_forward_shape_and_l2_norm() -> None:
    from src.models.pose_gnn import PoseGNNEncoder

    model = PoseGNNEncoder(embedding_dim=128, dropout=0.0).eval()
    with torch.no_grad():
        z = model(_make_pose_batch(4))

    assert z.shape == (4, 128)
    norms = z.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_pose_gnn_triplet_backward_pass() -> None:
    from src.models.pose_gnn import PoseGNNEncoder

    model = PoseGNNEncoder(embedding_dim=128, dropout=0.0)
    loss_fn = torch.nn.TripletMarginLoss(margin=1.0)

    anchor = _make_pose_batch(4)
    positive = anchor + 0.01 * torch.randn_like(anchor)
    negative = torch.rand_like(anchor)

    z_anchor = model(anchor)
    z_positive = model(positive)
    z_negative = model(negative)
    loss = loss_fn(z_anchor, z_positive, z_negative)
    loss.backward()

    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
