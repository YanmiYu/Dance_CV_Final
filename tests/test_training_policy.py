from __future__ import annotations

from pathlib import Path

import pytest


def test_training_policy_allows_custom_dance_when_manual_labels_exist(tmp_path: Path) -> None:
    from src.train.train_pose import _require_allowed_train_sources

    train = tmp_path / "custom_train.jsonl"
    val = tmp_path / "custom_val.jsonl"
    train.write_text("")
    val.write_text("")
    _require_allowed_train_sources(
        {"dataset_mix": {"aistpp": 0.75, "custom_dance": 0.25}},
        {"datasets": {"custom_dance": {"enabled": True, "annotations": str(train), "val_annotations": str(val)}}},
    )


def test_training_policy_rejects_missing_custom_dance_labels() -> None:
    from src.train.train_pose import _require_allowed_train_sources

    with pytest.raises(SystemExit):
        _require_allowed_train_sources(
            {"dataset_mix": {"custom_dance": 1.0}},
            {"datasets": {"custom_dance": {"enabled": True}}},
        )


def test_training_policy_rejects_unsupported_supervised_sources() -> None:
    from src.train.train_pose import _require_allowed_train_sources

    with pytest.raises(SystemExit):
        _require_allowed_train_sources({"dataset_mix": {"coco": 1.0}}, {"datasets": {}})


def test_internal_checkpoint_guard_rejects_external_path(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from src.train.engine import _load_state_from_internal_ckpt

    with pytest.raises(AssertionError):
        _load_state_from_internal_ckpt(torch.nn.Linear(1, 1), str(tmp_path / "external.pt"))


def test_internal_checkpoint_loader_rejects_git_lfs_pointer(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from src.train.engine import _load_state_from_internal_ckpt

    ckpt_path = tmp_path / "data" / "processed" / "model" / "best.pt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0123456789abcdef\n"
        "size 123\n"
    )

    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        _load_state_from_internal_ckpt(torch.nn.Linear(1, 1), str(ckpt_path))
