"""Supervised contrastive training for the Basic Dance pose embedding model.

Usage:
    python -m src.train.train_pose_gnn_supcon \
        --config configs/train/train_pose_gnn_supcon_basicdance.yaml

Outputs (under ``output_dir``):
    checkpoints/best.pt           best-by-val-supcon-loss
    checkpoints/last.pt           latest epoch
    metrics.json                  best-epoch summary
    train_log.csv                 per-epoch metrics
    config_resolved.yaml          the merged/runtime config snapshot
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.compare.embedding_features import select_torch_device
from src.datasets.balanced_batch_sampler import DanceLabelBalancedBatchSampler
from src.datasets.basic_dance_index import build_index_csv, load_index_csv
from src.datasets.basic_dance_supcon_dataset import (
    BasicDanceSupConDataset,
    split_index_rows,
    supcon_collate,
)
from src.losses.supcon import SupConLoss, positive_negative_cosine_means
from src.models.pose_gnn_temporal import PoseGNNTemporalEncoder
from src.utils.config import load_yaml
from src.utils.io import ensure_dir, write_json


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    pkl_root: str
    index_csv: str
    output_dir: str
    situation: str
    use_cameras: Optional[Sequence[str]]
    split_mode: str
    val_ratio: float
    val_cameras: Optional[Sequence[str]]
    seed: int
    window_size: int
    window_stride: int
    min_confidence: float
    in_features: int
    embedding_dim: int
    frame_embedding_dim: int
    dropout: float
    pooling: str
    temperature: float
    n_labels_per_batch: int
    n_samples_per_label: int
    batches_per_epoch: Optional[int]
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    device: str
    num_workers: int
    save_every: int
    eval_every: int
    log_every: int
    deploy_checkpoint: Optional[str]


def _to_train_config(cfg: dict) -> TrainConfig:
    return TrainConfig(
        pkl_root=str(cfg["pkl_root"]),
        index_csv=str(cfg["index_csv"]),
        output_dir=str(cfg["output_dir"]),
        situation=str(cfg.get("situation", "sBM")),
        use_cameras=cfg.get("use_cameras"),
        split_mode=str(cfg.get("split_mode", "dance_label")),
        val_ratio=float(cfg.get("val_ratio", 0.2)),
        val_cameras=cfg.get("val_cameras"),
        seed=int(cfg.get("seed", 42)),
        window_size=int(cfg.get("window_size", 32)),
        window_stride=int(cfg.get("window_stride", 8)),
        min_confidence=float(cfg.get("min_confidence", 0.2)),
        in_features=int(cfg.get("in_features", 3)),
        embedding_dim=int(cfg.get("embedding_dim", 128)),
        frame_embedding_dim=int(cfg.get("frame_embedding_dim", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        pooling=str(cfg.get("pooling", "mean")),
        temperature=float(cfg.get("temperature", 0.07)),
        n_labels_per_batch=int(cfg.get("n_labels_per_batch", 16)),
        n_samples_per_label=int(cfg.get("n_samples_per_label", 4)),
        batches_per_epoch=(int(cfg["batches_per_epoch"])
                           if cfg.get("batches_per_epoch") else None),
        epochs=int(cfg.get("epochs", 60)),
        learning_rate=float(cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(cfg.get("weight_decay", 1.0e-4)),
        grad_clip=float(cfg.get("grad_clip", 1.0)),
        device=str(cfg.get("device", "auto")),
        num_workers=int(cfg.get("num_workers", 4)),
        save_every=int(cfg.get("save_every", 5)),
        eval_every=int(cfg.get("eval_every", 1)),
        log_every=int(cfg.get("log_every", 10)),
        deploy_checkpoint=(str(cfg["deploy_checkpoint"])
                           if cfg.get("deploy_checkpoint") else None),
    )


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _embed_dataset(
    model: PoseGNNTemporalEncoder,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (embeddings, dance_label_ids, genre_label_ids)."""
    model.eval()
    embs: List[torch.Tensor] = []
    d_labs: List[torch.Tensor] = []
    g_labs: List[torch.Tensor] = []
    for batch in loader:
        x = batch["pose_window"].to(device, non_blocking=True)
        m = batch["mask"].any(dim=-1).to(device)  # (B, T) frame-level mask
        z = model(x, mask=m)
        embs.append(z.detach().cpu())
        d_labs.append(batch["dance_label_id"])
        g_labs.append(batch["genre_label_id"])
    if not embs:
        return (np.zeros((0, model.embedding_dim), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64))
    return (torch.cat(embs).numpy(),
            torch.cat(d_labs).numpy(),
            torch.cat(g_labs).numpy())


def _retrieval_topk_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ks: Sequence[int] = (1, 5),
) -> Dict[int, float]:
    """Top-k retrieval accuracy: % of items whose top-k neighbours include
    at least one with the same label (excluding self)."""
    n = embeddings.shape[0]
    if n < 2:
        return {int(k): 0.0 for k in ks}
    z = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8, None)
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)

    out: Dict[int, float] = {}
    max_k = min(max(ks), n - 1)
    nn_idx = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]
    # sort that small slice
    rows = np.arange(n)[:, None]
    sorted_order = np.argsort(-sim[rows, nn_idx], axis=1)
    nn_idx = nn_idx[rows, sorted_order]
    nn_labels = labels[nn_idx]                    # (n, max_k)

    for k in ks:
        kk = min(int(k), max_k)
        hit = (nn_labels[:, :kk] == labels[:, None]).any(axis=1)
        out[int(k)] = float(hit.mean())
    return out


def evaluate(
    model: PoseGNNTemporalEncoder,
    val_loader: DataLoader,
    loss_fn: SupConLoss,
    device: torch.device,
) -> Dict[str, float]:
    embs, d_labs, g_labs = _embed_dataset(model, val_loader, device)
    if embs.shape[0] == 0:
        return {
            "val_supcon_loss": 0.0,
            "val_mean_pos_cos": 0.0,
            "val_mean_neg_cos": 0.0,
            "val_retrieval_top1_dance_label": 0.0,
            "val_retrieval_top5_dance_label": 0.0,
            "val_retrieval_top1_genre": 0.0,
            "val_retrieval_top5_genre": 0.0,
            "val_num_samples": 0,
        }

    # SupCon loss + cosine stats on the full val set (single batch).
    z_t = torch.from_numpy(embs).to(device)
    d_t = torch.from_numpy(d_labs).to(device)
    val_loss = float(loss_fn(z_t, d_t).item())
    mean_pos, mean_neg = positive_negative_cosine_means(z_t, d_t)

    dance_topk = _retrieval_topk_accuracy(embs, d_labs, ks=(1, 5))
    genre_topk = _retrieval_topk_accuracy(embs, g_labs, ks=(1, 5))
    return {
        "val_supcon_loss": val_loss,
        "val_mean_pos_cos": mean_pos,
        "val_mean_neg_cos": mean_neg,
        "val_retrieval_top1_dance_label": dance_topk[1],
        "val_retrieval_top5_dance_label": dance_topk[5],
        "val_retrieval_top1_genre": genre_topk[1],
        "val_retrieval_top5_genre": genre_topk[5],
        "val_num_samples": int(embs.shape[0]),
    }


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #

def _save_checkpoint(
    path: Path,
    model: PoseGNNTemporalEncoder,
    cfg: TrainConfig,
    epoch: int,
    extra: Optional[dict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        # "model" key is what src.compare.embedding_features.load_pose_gnn_encoder
        # already loads -- saving the *frame* encoder there keeps render_report
        # working without code changes.
        "model": model.frame_encoder_state_dict(),
        "embedding_dim": int(model.frame_encoder.embedding_dim),
        # New keys for the SupCon temporal model:
        "temporal_model": model.state_dict(),
        "temporal_embedding_dim": int(model.embedding_dim),
        "frame_embedding_dim": int(model.frame_embedding_dim),
        "in_features": int(model.in_features),
        "pooling": str(model.pooling),
        "epoch": int(epoch),
        "format_version": 2,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #

def run(cfg: TrainConfig) -> Path:
    out_root = ensure_dir(cfg.output_dir)
    ckpt_dir = ensure_dir(out_root / "checkpoints")
    log_path = out_root / "train_log.csv"
    metrics_path = out_root / "metrics.json"
    cfg_snapshot_path = out_root / "config_resolved.json"

    # Build / load index.
    index_csv = Path(cfg.index_csv)
    if not index_csv.exists():
        n_rows = build_index_csv(
            cfg.pkl_root, index_csv,
            situation=cfg.situation,
            use_cameras=cfg.use_cameras,
        )
        print(f"[index] built {n_rows} rows -> {index_csv}")
    rows = load_index_csv(index_csv)
    if cfg.use_cameras:
        rows = [r for r in rows if r["camera"] in set(cfg.use_cameras)]
    print(f"[index] loaded {len(rows)} rows")

    train_rows, val_rows = split_index_rows(
        rows,
        split_mode=cfg.split_mode,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        val_cameras=cfg.val_cameras,
    )
    print(f"[split:{cfg.split_mode}] train={len(train_rows)}  val={len(val_rows)}")

    # Datasets share the label maps from train (val labels may be unseen
    # under dance_label split; we extend with val-only labels too).
    all_dance = sorted({r["dance_label"] for r in rows})
    all_genre = sorted({r["genre_label"] for r in rows})
    dance_to_id = {l: i for i, l in enumerate(all_dance)}
    genre_to_id = {l: i for i, l in enumerate(all_genre)}

    train_ds = BasicDanceSupConDataset(
        train_rows,
        window_size=cfg.window_size,
        window_stride=cfg.window_stride,
        min_confidence=cfg.min_confidence,
        random_start=True,
        dance_label_to_id=dance_to_id,
        genre_label_to_id=genre_to_id,
        seed=cfg.seed,
    )
    val_ds = BasicDanceSupConDataset(
        val_rows,
        window_size=cfg.window_size,
        window_stride=cfg.window_stride,
        min_confidence=cfg.min_confidence,
        random_start=False,                    # deterministic windows
        dance_label_to_id=dance_to_id,
        genre_label_to_id=genre_to_id,
        seed=cfg.seed,
    )
    print(f"[dataset] train_windows={len(train_ds)}  val_windows={len(val_ds)}")

    sampler = DanceLabelBalancedBatchSampler(
        labels=train_ds.labels(),
        n_labels_per_batch=cfg.n_labels_per_batch,
        n_samples_per_label=cfg.n_samples_per_label,
        num_batches=cfg.batches_per_epoch,
        seed=cfg.seed,
    )
    print(f"[sampler] batches/epoch={len(sampler)} batch_size={sampler.batch_size}")

    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=supcon_collate,
        pin_memory=True,
    )
    val_batch_size = max(sampler.batch_size, 32)
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=supcon_collate,
        pin_memory=True,
    )

    device = select_torch_device(cfg.device)
    print(f"[device] {device}")

    model = PoseGNNTemporalEncoder(
        frame_embedding_dim=cfg.frame_embedding_dim,
        out_embedding_dim=cfg.embedding_dim,
        dropout=cfg.dropout,
        in_features=cfg.in_features,
        pooling=cfg.pooling,
    ).to(device)
    loss_fn = SupConLoss(temperature=cfg.temperature).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Snapshot the resolved config for reproducibility.
    write_json(cfg_snapshot_path, {k: v for k, v in cfg.__dict__.items()})

    # ---------- training loop ----------
    log_rows: List[dict] = []
    best_val = float("inf")
    best_summary: Optional[dict] = None
    best_epoch = -1

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for step, batch in enumerate(train_loader):
            x = batch["pose_window"].to(device, non_blocking=True)
            m = batch["mask"].any(dim=-1).to(device)
            d_labs = batch["dance_label_id"].to(device, non_blocking=True)

            z = model(x, mask=m)
            loss = loss_fn(z, d_labs)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            running += float(loss.item())
            n_batches += 1
            if cfg.log_every and step % cfg.log_every == 0:
                print(f"  epoch {epoch:3d} step {step:4d}  loss={float(loss.item()):.4f}")

        train_loss = running / max(1, n_batches)
        epoch_secs = time.time() - t0

        row = {"epoch": epoch, "train_supcon_loss": train_loss, "epoch_secs": epoch_secs}
        if cfg.eval_every and epoch % cfg.eval_every == 0:
            metrics = evaluate(model, val_loader, loss_fn, device)
            row.update(metrics)
            print(
                f"[epoch {epoch:3d}] train={train_loss:.4f} "
                f"val={metrics['val_supcon_loss']:.4f} "
                f"top1_dance={metrics['val_retrieval_top1_dance_label']:.3f} "
                f"top5_dance={metrics['val_retrieval_top5_dance_label']:.3f} "
                f"top1_genre={metrics['val_retrieval_top1_genre']:.3f} "
                f"({epoch_secs:.1f}s)"
            )
            v = metrics["val_supcon_loss"]
            if v < best_val:
                best_val = v
                best_epoch = epoch
                best_summary = dict(row)
                _save_checkpoint(ckpt_dir / "best.pt", model, cfg, epoch,
                                 extra={"val_metrics": metrics})
        else:
            print(f"[epoch {epoch:3d}] train={train_loss:.4f} ({epoch_secs:.1f}s)")

        log_rows.append(row)
        # Save last every epoch (cheap), and an explicit periodic copy.
        _save_checkpoint(ckpt_dir / "last.pt", model, cfg, epoch)
        if cfg.save_every and epoch % cfg.save_every == 0:
            _save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", model, cfg, epoch)

        # Persist log every epoch.
        _write_log_csv(log_path, log_rows)

    # ---------- finalise ----------
    summary = {
        "best_epoch": best_epoch,
        "best_val_supcon_loss": best_val,
        "best": best_summary,
        "num_epochs": cfg.epochs,
        "split_mode": cfg.split_mode,
        "embedding_dim": cfg.embedding_dim,
        "frame_embedding_dim": cfg.frame_embedding_dim,
        "config": {k: v for k, v in cfg.__dict__.items()},
    }
    write_json(metrics_path, summary)

    if cfg.deploy_checkpoint and (ckpt_dir / "best.pt").exists():
        deploy = Path(cfg.deploy_checkpoint)
        deploy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt_dir / "best.pt", deploy)
        print(f"[deploy] copied best -> {deploy}")

    return out_root


def _write_log_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _main() -> None:
    p = argparse.ArgumentParser(description="SupCon training for Basic Dance pose embeddings.")
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    raw = load_yaml(args.config)
    if args.epochs is not None:
        raw["epochs"] = args.epochs
    if args.device is not None:
        raw["device"] = args.device
    if args.num_workers is not None:
        raw["num_workers"] = args.num_workers
    if args.output_dir is not None:
        raw["output_dir"] = args.output_dir
    cfg = _to_train_config(raw)
    out = run(cfg)
    print(f"done -> {out}")


if __name__ == "__main__":
    _main()
