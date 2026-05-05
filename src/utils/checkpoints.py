"""Checkpoint loading helpers shared by model entry points."""
from __future__ import annotations

from pathlib import Path
from typing import Any


_GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_git_lfs_pointer(path: str | Path) -> bool:
    """Return True when ``path`` is an unsmudged Git LFS pointer file."""
    try:
        with Path(path).open("rb") as f:
            return f.read(len(_GIT_LFS_POINTER_PREFIX)) == _GIT_LFS_POINTER_PREFIX
    except OSError:
        return False


def ensure_real_checkpoint(path: str | Path) -> None:
    """Fail early with an actionable message for missing Git LFS artifacts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if is_git_lfs_pointer(path):
        raise RuntimeError(
            "Checkpoint is a Git LFS pointer, not downloaded model weights: "
            f"{path}. Run `git lfs pull --include={path}` from the repo root, "
            "then retry."
        )


def torch_load_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> Any:
    """Load a tensor checkpoint with explicit safe defaults where available."""
    ensure_real_checkpoint(path)

    import torch

    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - depends on installed torch version
        return torch.load(str(path), map_location=map_location)
