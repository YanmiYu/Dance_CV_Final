"""
model.py — BiGRU DeviationClassifier.
Owner: Member 2

Architecture:
  Input:  (B, T', 24)   — 24-dim diff feature per aligned frame
  BiGRU:  hidden=64, layers=2, bidirectional → effective hidden = 128
  Head:   Linear(128 → 6 × 3) → reshape (B, T', 6, 3)
  Output: (B, T', 6, 3) logits — 6 body parts × 3 classes (good/moderate/off)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from dataset import FEATURE_DIM, N_PARTS


class DeviationClassifier(nn.Module):
    """Bidirectional GRU that classifies pose deviation per frame per body part.

    Parameters
    ----------
    input_size : int
        Dimensionality of the per-frame feature vector (default 24).
    hidden_size : int
        Number of GRU hidden units per direction (default 64).
    num_layers : int
        Number of stacked GRU layers (default 2).
    dropout : float
        Dropout applied between GRU layers (default 0.3).
    n_parts : int
        Number of body parts to classify (default 6).
    n_classes : int
        Number of deviation classes: 0=good, 1=moderate, 2=off (default 3).
    """

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_parts: int = N_PARTS,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, n_parts * n_classes)
        self.n_parts = n_parts
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T', 24)

        Returns
        -------
        logits : Tensor, shape (B, T', 6, 3)
        """
        h, _ = self.gru(x)                           # (B, T', hidden*2)
        h = self.dropout(h)
        logits = self.head(h)                         # (B, T', 6*3)
        return logits.view(*logits.shape[:-1], self.n_parts, self.n_classes)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices. Shape (B, T', 6)."""
        with torch.no_grad():
            logits = self.forward(x)                 # (B, T', 6, 3)
            return logits.argmax(dim=-1)             # (B, T', 6)


def save_checkpoint(model: DeviationClassifier, path: str | Path, **meta) -> None:
    """Save model weights and optional metadata (epoch, val_f1, etc.)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, str(path))


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
    **model_kwargs,
) -> tuple[DeviationClassifier, dict]:
    """Load model weights from a checkpoint file.

    Returns
    -------
    model : DeviationClassifier
    meta  : dict — everything saved alongside state_dict (epoch, val_f1, etc.)
    """
    ckpt = torch.load(str(path), map_location=device)
    state_dict = ckpt.pop("state_dict")
    model = DeviationClassifier(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, ckpt
