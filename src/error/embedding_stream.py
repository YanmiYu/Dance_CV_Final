"""Embedding-stream similarity computation for the GNN encoder.

Given two (T_b, 128) and (T_u, 128) embedding sequences plus the canonical
DTW path computed on the upstream keypoint pose (so that all streams share
a single T' time axis), produce per-frame cosine similarity in [-1, 1].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmbeddingStream:
    name: str
    cosine_sim: np.ndarray   # (T',)
    timestamps: np.ndarray   # (T',)


def build(
    name: str,
    emb_bench: np.ndarray,
    emb_user: np.ndarray,
    path: list[tuple[int, int]],
    timestamps: np.ndarray,
) -> EmbeddingStream:
    b_idx = np.array([p[0] for p in path])
    u_idx = np.array([p[1] for p in path])
    a = emb_bench[b_idx].astype(np.float32)
    b = emb_user[u_idx].astype(np.float32)
    a /= np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b /= np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    cosine = np.clip((a * b).sum(axis=1), -1.0, 1.0).astype(np.float32)
    return EmbeddingStream(name=name, cosine_sim=cosine, timestamps=timestamps)
