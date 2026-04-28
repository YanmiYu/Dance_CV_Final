"""Training script for SimpleBaseline on COCO keypoints.

Usage (local):
    python src/train.py --coco_root /path/to/coco --epochs 30 --batch_size 32

Usage (Oscar CCV):
    See slurm/train_simple_baseline.sbatch
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.simple_baseline import SimpleBaseline
from datasets.coco_dataset import COCOPoseDataset


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for i, (images, heatmaps_gt) in enumerate(loader):
        images      = images.to(device)
        heatmaps_gt = heatmaps_gt.to(device)

        heatmaps_pred = model(images)
        loss = criterion(heatmaps_pred, heatmaps_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 100 == 0:
            print(f"  epoch {epoch}  step {i}/{len(loader)}  loss {loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for images, heatmaps_gt in loader:
        images      = images.to(device)
        heatmaps_gt = heatmaps_gt.to(device)
        heatmaps_pred = model(images)
        total_loss += criterion(heatmaps_pred, heatmaps_gt).item()
    return total_loss / len(loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root",  required=True, help="Path to COCO root dir")
    parser.add_argument("--out_dir",    default="checkpoints/simple_baseline")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--workers",    type=int,   default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Datasets ────────────────────────────────────────────────────────────
    train_dataset = COCOPoseDataset(args.coco_root, split="train", augment=True)
    val_dataset   = COCOPoseDataset(args.coco_root, split="val",   augment=False)
    print(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    # ── Model, loss, optimiser ───────────────────────────────────────────────
    model     = SimpleBaseline(num_joints=17).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Drop LR by 10x at epoch 20 and 25
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 25], gamma=0.1)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {total_params:.1f}M")

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss   = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed:.0f}s")

        # Save checkpoint every epoch
        ckpt_path = out_dir / f"epoch_{epoch:03d}.pth"
        torch.save({
            "epoch":      epoch,
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "val_loss":   val_loss,
        }, ckpt_path)

        # Keep track of best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best.pth")
            print(f"  → new best saved (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {out_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
