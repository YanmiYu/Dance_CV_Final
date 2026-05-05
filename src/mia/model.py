"""
model.py — LSTM Temporal Error Detector.

Architecture:
  Input:  (B, T', 24)   — 24-dim diff feature per aligned frame
  LSTM:   hidden=64, layers=2, unidirectional, dropout=0.3
  Head:   Linear(64 → 6)
  Output: (B, T', 6) logits — one per body part
          sigmoid → P(frame is "off") per part

Training signal: BCEWithLogitsLoss on binary labels (0=correct, 1=off).
Inference:  proba = sigmoid(logits) ; flag frame if proba > 0.5
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.utils.checkpoints import torch_load_checkpoint

from src.mia.dataset import FEATURE_DIM, N_PARTS


class TemporalErrorDetector(nn.Module):
    """Unidirectional LSTM that learns P(off) per frame per body part.

    Replaces the fixed geometric threshold: instead of asking
    "is error > 0.25?", the model asks "does this temporal pattern
    look like a mistake?" — capturing sustained drift, sudden
    mis-alignment, and noise suppression automatically.

    Parameters
    ----------
    input_size  : per-frame feature dimension (default 24)
    hidden_size : LSTM hidden units (default 64)
    num_layers  : stacked LSTM layers (default 2)
    dropout     : dropout between layers (default 0.3)
    n_parts     : body parts to classify (default 6)
    """

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_parts: int = N_PARTS,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_parts)
        self.n_parts = n_parts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T', 24)

        Returns
        -------
        logits : Tensor, shape (B, T', 6)
            Raw logits — apply sigmoid to get probabilities.
        """
        h, _ = self.lstm(x)      # (B, T', hidden)
        h = self.dropout(h)
        return self.head(h)       # (B, T', 6)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(off) in [0, 1] for each frame and body part.

        Shape: (B, T', 6)
        """
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


def save_checkpoint(model: TemporalErrorDetector, path: str | Path, **meta) -> None:
    """Save model weights and optional metadata (epoch, val_f1, etc.)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, str(path))


def _torch_load(path: str | Path, device: str) -> Any:
    """Load a checkpoint across supported PyTorch versions.

    Newer PyTorch releases support ``weights_only=True``. Older ones do not,
    so keep a small fallback for shared lab environments.
    """
    return torch_load_checkpoint(path, map_location=device)


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict or not all(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _infer_model_kwargs(
    state_dict: dict[str, torch.Tensor],
    meta: dict,
    explicit_kwargs: dict,
) -> dict:
    """Infer architecture knobs saved by this repo, falling back to tensor shapes."""
    kwargs = dict(explicit_kwargs)
    for key in ("input_size", "hidden_size", "num_layers", "dropout", "n_parts"):
        if key not in kwargs and key in meta:
            kwargs[key] = meta[key]

    weight_ih = state_dict.get("lstm.weight_ih_l0")
    head_weight = state_dict.get("head.weight")
    if weight_ih is not None:
        kwargs.setdefault("input_size", int(weight_ih.shape[1]))
        kwargs.setdefault("hidden_size", int(weight_ih.shape[0] // 4))
    if head_weight is not None:
        kwargs.setdefault("n_parts", int(head_weight.shape[0]))

    if "num_layers" not in kwargs:
        layer_ids = []
        for key in state_dict:
            if key.startswith("lstm.weight_ih_l"):
                suffix = key.removeprefix("lstm.weight_ih_l")
                if suffix.isdigit():
                    layer_ids.append(int(suffix))
        if layer_ids:
            kwargs["num_layers"] = max(layer_ids) + 1
    return kwargs


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
    **model_kwargs,
) -> tuple[TemporalErrorDetector, dict]:
    """Load model weights from a checkpoint file.

    Returns
    -------
    model : TemporalErrorDetector
    meta  : dict — everything saved alongside state_dict (epoch, val_f1, etc.)
    """
    ckpt = _torch_load(path, device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = dict(ckpt)
        state_dict = ckpt.pop("state_dict")
        meta = ckpt
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        meta = {}
    else:
        raise TypeError(f"Unsupported checkpoint format at {path!s}: {type(ckpt)!r}")

    state_dict = _strip_module_prefix(state_dict)
    model = TemporalErrorDetector(**_infer_model_kwargs(state_dict, meta, model_kwargs))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, meta
