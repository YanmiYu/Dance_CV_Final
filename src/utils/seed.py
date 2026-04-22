"""Deterministic seeding helpers."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy, and torch (if installed)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch optional during data-only phases
        pass
