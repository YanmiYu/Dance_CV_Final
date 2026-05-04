"""Embedding features for learned pose alignment.

These helpers keep the optional GNN alignment path separate from the existing
handcrafted feature baseline. The report pipeline should pass normalized poses
from ``normalize_pose.normalize_sequence`` into ``encode_pose_sequence``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

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
    """Load a ``PoseGNNEncoder`` checkpoint and return ``(model, device)``.

    Supports two checkpoint layouts:

      1. Legacy: a raw ``state_dict`` saved by the old triplet trainer, or a
         dict with key ``"model"`` and optional ``"embedding_dim"``.
      2. SupCon temporal trainer (see
         ``src.train.train_pose_gnn_supcon``): a dict with both ``"model"``
         (frame-encoder weights for compatibility with this loader) and
         ``"temporal_model"`` (full :class:`PoseGNNTemporalEncoder` weights).
         We ignore the temporal head here -- this loader returns only the
         frame-level encoder used by ``encode_pose_sequence``.
    """
    device_t = select_torch_device(device) if isinstance(device, str) else device
    ckpt = torch.load(checkpoint_path, map_location=device_t)
    if isinstance(ckpt, dict):
        # Prefer "model" (legacy + new format both put frame-encoder there).
        # Fall back to "state_dict" (some HF-style saves) and finally to the
        # whole dict (raw state_dict).
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        embedding_dim = int(ckpt.get("embedding_dim", 128))
    else:
        state = ckpt
        embedding_dim = 128

    model = PoseGNNEncoder(embedding_dim=embedding_dim)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # Common mistake: pointing at a temporal-only checkpoint.
        if isinstance(ckpt, dict) and "temporal_model" in ckpt and state is ckpt:
            raise RuntimeError(
                "Checkpoint contains 'temporal_model' but no plain frame "
                "'model' state. Re-export the frame encoder weights to use "
                "with the report pipeline."
            ) from e
        raise
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


def compute_embedding_similarity(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    aligned_a_idx: np.ndarray,
    aligned_b_idx: np.ndarray,
) -> Dict[str, float]:
    """Score how close two embedding sequences are along an alignment path.

    The encoder L2-normalizes its outputs, so we use cosine similarity (the
    natural metric for unit-norm vectors) and convert it to a 0-100 score.
    Returns a dict with the mean / median / min cosine similarity, the mean
    Euclidean distance, and the final ``score`` in ``[0, 100]``.
    """
    aligned_a_idx = np.asarray(aligned_a_idx, dtype=np.int64)
    aligned_b_idx = np.asarray(aligned_b_idx, dtype=np.int64)
    L = int(min(aligned_a_idx.shape[0], aligned_b_idx.shape[0]))
    if L == 0 or emb_a.size == 0 or emb_b.size == 0:
        return {
            "score": 0.0,
            "mean_cosine_similarity": 0.0,
            "median_cosine_similarity": 0.0,
            "min_cosine_similarity": 0.0,
            "mean_distance": 0.0,
            "num_aligned_steps": 0,
        }

    a = emb_a[aligned_a_idx[:L]].astype(np.float32, copy=False)
    b = emb_b[aligned_b_idx[:L]].astype(np.float32, copy=False)
    # Re-normalize defensively in case caller passes unnormalized rows.
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a = a / np.clip(a_norm, 1e-8, None)
    b = b / np.clip(b_norm, 1e-8, None)

    cos = np.sum(a * b, axis=1)            # (L,) in [-1, 1]
    dist = np.linalg.norm(a - b, axis=1)   # (L,) in [0, 2]
    mean_cos = float(np.mean(cos))
    median_cos = float(np.median(cos))
    min_cos = float(np.min(cos))
    mean_dist = float(np.mean(dist))
    score = float(max(0.0, mean_cos)) * 100.0   # cos in [0,1] -> score in [0,100]
    return {
        "score": score,
        "mean_cosine_similarity": mean_cos,
        "median_cosine_similarity": median_cos,
        "min_cosine_similarity": min_cos,
        "mean_distance": mean_dist,
        "num_aligned_steps": int(L),
    }


def pairwise_cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Return the full frame-to-frame cosine similarity matrix.

    ``emb_a`` is interpreted as benchmark embeddings ``(T_a, D)`` and
    ``emb_b`` as user embeddings ``(T_b, D)``. The output has shape
    ``(T_a, T_b)`` so rows are benchmark frames and columns are user frames.
    Rows are normalized defensively even though ``PoseGNNEncoder`` already
    emits L2-normalized embeddings.
    """
    emb_a = np.asarray(emb_a, dtype=np.float32)
    emb_b = np.asarray(emb_b, dtype=np.float32)
    if emb_a.ndim != 2 or emb_b.ndim != 2:
        raise ValueError(
            f"expected 2D embedding arrays, got {emb_a.shape} and {emb_b.shape}"
        )
    if emb_a.shape[1] != emb_b.shape[1]:
        raise ValueError(
            f"embedding dimensions differ: {emb_a.shape[1]} vs {emb_b.shape[1]}"
        )
    if emb_a.shape[0] == 0 or emb_b.shape[0] == 0:
        return np.zeros((emb_a.shape[0], emb_b.shape[0]), dtype=np.float32)

    a_norm = np.linalg.norm(emb_a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(emb_b, axis=1, keepdims=True)
    a = emb_a / np.clip(a_norm, 1e-8, None)
    b = emb_b / np.clip(b_norm, 1e-8, None)
    return np.clip(a @ b.T, -1.0, 1.0).astype(np.float32, copy=False)
