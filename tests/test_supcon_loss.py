"""Tests for the supervised contrastive loss."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.losses.supcon import SupConLoss, positive_negative_cosine_means


def test_supcon_finite_with_balanced_batch() -> None:
    torch.manual_seed(0)
    z = torch.randn(8, 16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = SupConLoss(temperature=0.1)(z, labels)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_supcon_zero_with_no_positives() -> None:
    z = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])
    loss = SupConLoss(temperature=0.1)(z, labels)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_pos_neg_cosine_means_signal() -> None:
    base = torch.randn(2, 16)
    z = torch.cat([base[:1].repeat(3, 1), base[1:].repeat(3, 1)], dim=0)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    pos, neg = positive_negative_cosine_means(z, labels)
    assert pos > neg
