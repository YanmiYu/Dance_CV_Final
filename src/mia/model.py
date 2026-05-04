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

import torch
import torch.nn as nn

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
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    state_dict = ckpt.pop("state_dict")
    model = TemporalErrorDetector(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, ckpt
