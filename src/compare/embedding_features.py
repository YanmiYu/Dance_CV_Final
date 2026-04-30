"""Embedding features for learned pose alignment.

These helpers keep the optional GNN alignment path separate from the existing
handcrafted feature baseline. The report pipeline should pass normalized poses
from ``normalize_pose.normalize_sequence`` into ``encode_pose_sequence``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from src.models.pose_gnn import PoseGNNEncoder


def select_torch_device(name: str = "auto") -> torch.device:
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested cuda device, but CUDA is not available")
    if name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("requested mps device, but MPS is not available")
    return torch.device(name)


def load_pose_gnn_encoder(
    checkpoint_path: str | Path,
    device: str | torch.device = "auto",
) -> Tuple[PoseGNNEncoder, torch.device]:
    """Load a ``PoseGNNEncoder`` checkpoint and return ``(model, device)``."""
    device_t = select_torch_device(device) if isinstance(device, str) else device
    ckpt = torch.load(checkpoint_path, map_location=device_t)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    embedding_dim = int(ckpt.get("embedding_dim", 128)) if isinstance(ckpt, dict) else 128

    model = PoseGNNEncoder(embedding_dim=embedding_dim)
    model.load_state_dict(state)
    model.to(device_t).eval()
    return model, device_t


@torch.no_grad()
def encode_pose_sequence(
    model: PoseGNNEncoder,
    poses: np.ndarray | torch.Tensor,
    device: str | torch.device = "auto",
    batch_size: int = 256,
) -> np.ndarray:
    """Encode a pose sequence into framewise embeddings.

    Args:
        model: loaded ``PoseGNNEncoder``.
        poses: ``(T, 17, 3)`` normalized pose sequence.
        device: torch device or ``"auto"``.
        batch_size: number of frames to encode per model call.

    Returns:
        ``(T, embedding_dim)`` float32 numpy embeddings in frame order.
    """
    device_t = select_torch_device(device) if isinstance(device, str) else device
    if isinstance(poses, np.ndarray):
        x = torch.from_numpy(poses.astype(np.float32, copy=False))
    else:
        x = poses.detach().float().cpu()
    if x.dim() != 3 or x.shape[1:] != (17, 3):
        raise ValueError(f"expected poses shape (T, 17, 3), got {tuple(x.shape)}")

    outs = []
    for start in range(0, int(x.shape[0]), max(1, int(batch_size))):
        batch = x[start : start + batch_size].to(device_t)
        outs.append(model(batch).detach().cpu())
    if not outs:
        return np.zeros((0, model.embedding_dim), dtype=np.float32)
    return torch.cat(outs, dim=0).numpy().astype(np.float32, copy=False)
