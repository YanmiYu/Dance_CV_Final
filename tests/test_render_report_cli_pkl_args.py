"""CLI behaviour: --model-config / --ckpt are only required when the
script must run pose inference (i.e. one or both pose PKLs are missing)."""
from __future__ import annotations

import pytest

from src.compare.render_report import _build_arg_parser, _validate_cli_args


def _parse(args: list[str]):
    return _build_arg_parser().parse_args(args)


_BASE = ["--benchmark", "bench.mp4", "--user", "user.mp4"]


def test_pkls_only_no_model_or_ckpt_required():
    args = _parse(_BASE + [
        "--bench-poses-pkl", "bench.pkl",
        "--user-poses-pkl", "user.pkl",
    ])
    assert _validate_cli_args(args) is None


def test_pkls_plus_gnn_embedding_requires_gnn_ckpt():
    args = _parse(_BASE + [
        "--bench-poses-pkl", "bench.pkl",
        "--user-poses-pkl", "user.pkl",
        "--alignment-method", "gnn_embedding",
    ])
    err = _validate_cli_args(args)
    assert err is not None
    assert "gnn-checkpoint" in err


def test_pkls_plus_gnn_embedding_with_ckpt_passes():
    args = _parse(_BASE + [
        "--bench-poses-pkl", "bench.pkl",
        "--user-poses-pkl", "user.pkl",
        "--alignment-method", "gnn_embedding",
        "--gnn-checkpoint", "ckpt.pt",
    ])
    assert _validate_cli_args(args) is None


def test_no_pkls_requires_model_config_and_ckpt():
    args = _parse(_BASE)
    err = _validate_cli_args(args)
    assert err is not None
    assert err == (
        "Either provide both pose PKLs, or provide --model-config and "
        "--ckpt to run pose inference."
    )


def test_only_one_pkl_still_requires_model_and_ckpt():
    args = _parse(_BASE + ["--bench-poses-pkl", "bench.pkl"])
    err = _validate_cli_args(args)
    assert err is not None
    assert "--model-config and --ckpt" in err


def test_legacy_video_path_works_with_model_and_ckpt():
    args = _parse(_BASE + [
        "--model-config", "configs/model.yaml",
        "--ckpt", "ckpt.pt",
    ])
    assert _validate_cli_args(args) is None


def test_lone_pkl_is_rejected_to_avoid_silent_ignore():
    """If only one of the two PKLs is supplied, reject -- otherwise the
    supplied PKL would be silently ignored and inference run on both videos."""
    args = _parse(_BASE + [
        "--bench-poses-pkl", "bench.pkl",
        "--model-config", "configs/model.yaml",
        "--ckpt", "ckpt.pt",
    ])
    err = _validate_cli_args(args)
    assert err is not None
    assert "both pose PKLs" in err
