"""Train a pose GNN embedding with triplet loss.

This is separate from the existing heatmap pose-estimation training pipeline.
It consumes processed AIST++ per-video keypoints from
``data/labels/aistpp/keypoints2d_raw`` and trains a graph encoder to place
nearby frames closer than negatives.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.aist_embedding_dataset import AISTEmbeddingDataset
from src.models.pose_gnn import PoseGNNEncoder


def _select_device(name: str) -> torch.device:
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested --device cuda, but CUDA is not available")
    if name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("requested --device mps, but MPS is not available")
    return torch.device(name)


def _resolve_negative_strategy(keypoints_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    n_videos = len(list(keypoints_dir.glob("*.pkl")))
    return "different_video" if n_videos > 1 else "far_frame"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> Dict[str, float]:
    keypoints_dir = Path(args.keypoints_dir)
    negative_strategy = _resolve_negative_strategy(keypoints_dir, args.negative_strategy)
    device = _select_device(args.device)
    _seed_everything(int(args.seed))

    dataset = AISTEmbeddingDataset(
        keypoints_dir,
        positive_range=int(args.positive_range),
        negative_strategy=negative_strategy,
        seed=int(args.seed),
        verbose=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        drop_last=False,
    )

    model = PoseGNNEncoder(embedding_dim=int(args.embedding_dim), dropout=float(args.dropout)).to(device)
    loss_fn = torch.nn.TripletMarginLoss(margin=float(args.margin), p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    print(f"device={device} negative_strategy={negative_strategy}")
    last_stats: Dict[str, float] = {"loss": float("nan"), "pos_dist": float("nan"), "neg_dist": float("nan")}

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        pos_sum = 0.0
        neg_sum = 0.0
        n_batches = 0
        n_samples = 0

        for batch in loader:
            anchor = batch["anchor"].to(device)
            positive = batch["positive"].to(device)
            negative = batch["negative"].to(device)

            z_anchor = model(anchor)
            z_positive = model(positive)
            z_negative = model(negative)
            loss = loss_fn(z_anchor, z_positive, z_negative)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pos_dist = F.pairwise_distance(z_anchor, z_positive, p=2)
                neg_dist = F.pairwise_distance(z_anchor, z_negative, p=2)
                batch_n = int(anchor.shape[0])
                loss_sum += float(loss.item()) * batch_n
                pos_sum += float(pos_dist.mean().item()) * batch_n
                neg_sum += float(neg_dist.mean().item()) * batch_n
                n_batches += 1
                n_samples += batch_n

        denom = max(n_samples, 1)
        last_stats = {
            "loss": loss_sum / denom,
            "pos_dist": pos_sum / denom,
            "neg_dist": neg_sum / denom,
        }
        print(
            f"epoch {epoch}/{args.epochs} "
            f"loss={last_stats['loss']:.6f} "
            f"pos_dist={last_stats['pos_dist']:.6f} "
            f"neg_dist={last_stats['neg_dist']:.6f} "
            f"batches={n_batches}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "embedding_dim": int(args.embedding_dim),
            "negative_strategy": negative_strategy,
            "keypoints_dir": str(keypoints_dir),
            "epochs": int(args.epochs),
            "last_stats": last_stats,
        },
        out,
    )
    print(f"checkpoint saved -> {out}")
    return last_stats


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a minimal GNN pose embedding encoder.")
    p.add_argument("--keypoints-dir", default="data/labels/aistpp/keypoints2d_raw", type=Path)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--epochs", default=5, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--embedding-dim", default=128, type=int)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--out", default="checkpoints/pose_gnn_encoder.pt", type=Path)
    p.add_argument(
        "--negative-strategy",
        default="auto",
        choices=["auto", "different_video", "far_frame"],
        help="auto uses different_video when multiple keypoint videos exist, otherwise far_frame",
    )
    p.add_argument("--positive-range", default=5, type=int)
    p.add_argument("--margin", default=1.0, type=float)
    p.add_argument("--dropout", default=0.1, type=float)
    p.add_argument("--num-workers", default=0, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
