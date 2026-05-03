"""
train.py — Training loop for the LSTM TemporalErrorDetector.

Launch via main.py:
    python main.py train --train-dir data/train/ --val-dir data/val/ \
                         --checkpoint checkpoints/best_model.pt --epochs 30

Or directly on Oscar:
    sbatch slurm_run.sh train
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DanceDeviationDataset, collate_fn, N_PARTS
from model import TemporalErrorDetector, save_checkpoint


def compute_f1(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    ignore_value: float = -1.0,
) -> float:
    """Binary F1 for the "off" class across all parts.

    Parameters
    ----------
    probs   : (N,) predicted probabilities in [0, 1]
    targets : (N,) binary ground-truth (0.0 / 1.0); -1.0 = padding, ignored
    """
    mask = targets != ignore_value
    probs   = probs[mask]
    targets = targets[mask]

    preds = probs >= threshold
    tgts  = targets >= threshold

    tp = (preds & tgts).sum().float()
    fp = (preds & ~tgts).sum().float()
    fn = (~preds & tgts).sum().float()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return (2 * prec * rec / (prec + rec + 1e-8)).item()


def train(
    train_dir: str,
    val_dir: str,
    checkpoint_path: str,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    batch_size: int = 16,
    log_path: str = "results/training_log.csv",
    device: str | None = None,
) -> None:
    """Full training loop for the TemporalErrorDetector.

    Saves the checkpoint with the highest validation F1.
    Logs per-epoch metrics to a CSV file.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device}")

    # Data
    train_ds = DanceDeviationDataset(train_dir)
    val_ds   = DanceDeviationDataset(val_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)
    print(f"[train] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Model
    model = TemporalErrorDetector(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5)

    # pos_weight upsamples the "off" class to compensate for class imbalance
    pos_weight = torch.tensor([3.0] * N_PARTS, device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    # Logging
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_f1", "elapsed_s"])

    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ---- Train ----
        model.train()
        total_loss = 0.0
        for features, labels, lengths in train_loader:
            features = features.to(device)   # (B, T_max, 24)
            labels   = labels.to(device)     # (B, T_max, 6)  float 0/1 or -1 padding

            logits = model(features)          # (B, T_max, 6)
            mask   = labels >= 0             # True where not padding
            raw    = criterion(logits, labels.clamp(min=0))  # (B, T_max, 6)
            loss   = raw[mask].mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        all_probs, all_targets = [], []
        with torch.no_grad():
            for features, labels, lengths in val_loader:
                features = features.to(device)
                labels   = labels.to(device)
                logits   = model(features)
                mask     = labels >= 0
                raw      = criterion(logits, labels.clamp(min=0))
                val_loss += raw[mask].mean().item()

                probs = torch.sigmoid(logits)
                all_probs.append(probs.view(-1).cpu())
                all_targets.append(labels.view(-1).cpu())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = compute_f1(torch.cat(all_probs), torch.cat(all_targets))
        scheduler.step(val_f1)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"val_f1={val_f1:.4f}  "
            f"({elapsed:.0f}s)"
        )
        writer.writerow([epoch, round(avg_train_loss, 5), round(avg_val_loss, 5),
                         round(val_f1, 5), round(elapsed, 1)])
        log_file.flush()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(
                model, checkpoint_path,
                epoch=epoch,
                val_f1=val_f1,
                val_loss=avg_val_loss,
            )
            print(f"  New best checkpoint saved (val_f1={val_f1:.4f})")

    log_file.close()
    print(f"\n[train] Done. Best val F1: {best_val_f1:.4f}")
    print(f"[train] Checkpoint: {checkpoint_path}")
    print(f"[train] Log:        {log_path}")
