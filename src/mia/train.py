"""Training loop for Mia's LSTM temporal error detector.

This is the namespaced version of the standalone Mia-branch trainer. It trains
``TemporalErrorDetector`` on ``.npz`` samples produced by
``scripts/build_lstm_dataset.py`` or by any process that writes:

    features: (T, 24) float32
    labels:   (T, 6)  float32, with -1.0 reserved for padding
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.mia.dataset import DanceDeviationDataset, N_PARTS, collate_fn
from src.mia.model import TemporalErrorDetector, save_checkpoint


def compute_f1(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    ignore_value: float = -1.0,
) -> float:
    """Binary F1 for the "off" class across every frame and body part."""
    mask = targets != ignore_value
    probs = probs[mask]
    targets = targets[mask]
    if targets.numel() == 0:
        return 0.0

    preds = probs >= threshold
    tgts = targets >= threshold

    tp = (preds & tgts).sum().float()
    fp = (preds & ~tgts).sum().float()
    fn = (~preds & tgts).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float((2 * precision * recall / (precision + recall + 1e-8)).item())


def _default_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    log_path: str = "results/lstm_training_log.csv",
    device: str | None = None,
    num_workers: int = 0,
) -> dict:
    """Train the LSTM and save the best validation-F1 checkpoint."""
    device = _default_device(device)
    print(f"[mia.train] device={device}")

    train_ds = DanceDeviationDataset(train_dir)
    val_ds = DanceDeviationDataset(val_dir)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    print(f"[mia.train] train={len(train_ds)} val={len(val_ds)}")

    model = TemporalErrorDetector(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=5,
        factor=0.5,
    )

    pos_weight = torch.tensor([3.0] * N_PARTS, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    best_epoch = 0

    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_f1", "elapsed_s"])

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            model.train()
            total_loss = 0.0
            for features, labels, _lengths in train_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                mask = labels >= 0
                raw_loss = criterion(logits, labels.clamp(min=0))
                loss = raw_loss[mask].mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += float(loss.item())

            avg_train_loss = total_loss / max(1, len(train_loader))

            model.eval()
            val_loss = 0.0
            all_probs: list[torch.Tensor] = []
            all_targets: list[torch.Tensor] = []
            with torch.no_grad():
                for features, labels, _lengths in val_loader:
                    features = features.to(device)
                    labels = labels.to(device)
                    logits = model(features)
                    mask = labels >= 0
                    raw_loss = criterion(logits, labels.clamp(min=0))
                    val_loss += float(raw_loss[mask].mean().item())

                    all_probs.append(torch.sigmoid(logits).reshape(-1).cpu())
                    all_targets.append(labels.reshape(-1).cpu())

            avg_val_loss = val_loss / max(1, len(val_loader))
            val_f1 = compute_f1(torch.cat(all_probs), torch.cat(all_targets))
            scheduler.step(val_f1)

            elapsed = time.time() - t0
            print(
                f"epoch={epoch:03d}/{epochs} "
                f"train_loss={avg_train_loss:.4f} "
                f"val_loss={avg_val_loss:.4f} "
                f"val_f1={val_f1:.4f} "
                f"elapsed={elapsed:.0f}s"
            )
            writer.writerow(
                [
                    epoch,
                    round(avg_train_loss, 5),
                    round(avg_val_loss, 5),
                    round(val_f1, 5),
                    round(elapsed, 1),
                ]
            )
            log_file.flush()

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                save_checkpoint(
                    model,
                    checkpoint_path,
                    epoch=epoch,
                    val_f1=val_f1,
                    val_loss=avg_val_loss,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                print(f"  saved best checkpoint: val_f1={val_f1:.4f}")

    summary = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "checkpoint": checkpoint_path,
        "log": log_path,
    }
    print(f"[mia.train] done: {summary}")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Mia LSTM temporal error detector")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--checkpoint", default="checkpoints/lstm/best_model.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--log", default="results/lstm_training_log.csv")
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        log_path=args.log,
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
