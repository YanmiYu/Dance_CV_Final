"""Supervised contrastive training for Basic Dance PoseGNN embeddings.

This is deliberately separate from the integrated inference/fusion pipeline.
It trains the old PoseGNN-based embedding branch on PKL pose windows only.
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
from torch.utils.data import DataLoader

from src.compare.embedding_features import select_torch_device
from src.datasets.balanced_batch_sampler import DanceLabelBalancedBatchSampler
from src.datasets.basic_dance_index import build_index_csv, load_index_csv
from src.datasets.basic_dance_supcon_dataset import (
    BasicDanceSupConDataset,
    filter_labels_with_min_rows,
    split_index_rows,
    supcon_collate,
)
from src.losses.supcon import SupConLoss, positive_negative_cosine_means
from src.models.pose_gnn_temporal import PoseGNNTemporalEncoder
from src.utils.config import load_yaml
from src.utils.io import ensure_dir, write_json


@dataclass
class TrainConfig:
    pkl_root: str
    index_csv: str
    output_dir: str
    situation: str
    use_cameras: Optional[Sequence[str]]
    genres: Optional[Sequence[str] | str]
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
    min_train_rows_per_label: int
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    device: str
    num_workers: int
    save_every: int
    eval_every: int
    log_every: int
    pair_eval_max_pairs_per_type: int
    deploy_checkpoint: Optional[str]
    rebuild_index: bool


def _to_train_config(cfg: dict) -> TrainConfig:
    return TrainConfig(
        pkl_root=str(cfg["pkl_root"]),
        index_csv=str(cfg["index_csv"]),
        output_dir=str(cfg["output_dir"]),
        situation=str(cfg.get("situation", "sBM")),
        use_cameras=cfg.get("use_cameras"),
        genres=cfg.get("genres"),
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
        batches_per_epoch=(int(cfg["batches_per_epoch"]) if cfg.get("batches_per_epoch") else None),
        min_train_rows_per_label=int(cfg.get("min_train_rows_per_label", 2)),
        epochs=int(cfg.get("epochs", 60)),
        learning_rate=float(cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(cfg.get("weight_decay", 1.0e-4)),
        grad_clip=float(cfg.get("grad_clip", 1.0)),
        device=str(cfg.get("device", "auto")),
        num_workers=int(cfg.get("num_workers", 4)),
        save_every=int(cfg.get("save_every", 5)),
        eval_every=int(cfg.get("eval_every", 1)),
        log_every=int(cfg.get("log_every", 10)),
        pair_eval_max_pairs_per_type=int(cfg.get("pair_eval_max_pairs_per_type", 50000)),
        deploy_checkpoint=(str(cfg["deploy_checkpoint"]) if cfg.get("deploy_checkpoint") else None),
        rebuild_index=bool(cfg.get("rebuild_index", False)),
    )


def _normalize_genres(genres: Optional[Sequence[str] | str]) -> Optional[set[str]]:
    if genres is None:
        return None
    if isinstance(genres, str):
        if genres.lower() == "all":
            return None
        genres = [genres]
    out = set()
    for genre in genres:
        text = str(genre)
        if text.lower() == "all":
            return None
        out.add(text if text.startswith("g") else f"g{text}")
    return out


def _filter_rows_for_config(rows: Sequence[dict], cfg: TrainConfig) -> List[dict]:
    keep_cams = set(cfg.use_cameras or [])
    keep_genres = _normalize_genres(cfg.genres)
    out: List[dict] = []
    for row in rows:
        if cfg.situation and row.get("situation") != cfg.situation:
            continue
        if keep_cams and row.get("camera") not in keep_cams:
            continue
        if keep_genres is not None and row.get("genre_label") not in keep_genres:
            continue
        out.append(row)
    return out


@torch.no_grad()
def _embed_dataset(
    model: PoseGNNTemporalEncoder,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    model.eval()
    embs: List[torch.Tensor] = []
    d_labs: List[torch.Tensor] = []
    g_labs: List[torch.Tensor] = []
    metas: List[dict] = []
    for batch in loader:
        x = batch["pose_window"].to(device, non_blocking=True)
        m = batch["mask"].any(dim=-1).to(device)
        z = model(x, mask=m)
        embs.append(z.detach().cpu())
        d_labs.append(batch["dance_label_id"])
        g_labs.append(batch["genre_label_id"])
        metas.extend(batch["meta"])
    if not embs:
        return (
            np.zeros((0, model.embedding_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            [],
        )
    return (
        torch.cat(embs).numpy(),
        torch.cat(d_labs).numpy(),
        torch.cat(g_labs).numpy(),
        metas,
    )


def _retrieval_topk_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ks: Sequence[int] = (1, 5),
) -> Dict[int, float]:
    n = embeddings.shape[0]
    if n < 2:
        return {int(k): 0.0 for k in ks}
    z = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8, None)
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    max_k = min(max(ks), n - 1)
    nn_idx = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]
    rows = np.arange(n)[:, None]
    sorted_order = np.argsort(-sim[rows, nn_idx], axis=1)
    nn_idx = nn_idx[rows, sorted_order]
    nn_labels = labels[nn_idx]
    out: Dict[int, float] = {}
    for k in ks:
        kk = min(int(k), max_k)
        out[int(k)] = float((nn_labels[:, :kk] == labels[:, None]).any(axis=1).mean())
    return out


def _reservoir_add(
    pairs: List[tuple[int, int]],
    seen: int,
    pair: tuple[int, int],
    *,
    max_pairs: int,
    rng: np.random.Generator,
) -> int:
    seen += 1
    if len(pairs) < max_pairs:
        pairs.append(pair)
    else:
        j = int(rng.integers(0, seen))
        if j < max_pairs:
            pairs[j] = pair
    return seen


def _pair_stats(z: np.ndarray, pairs: Sequence[tuple[int, int]]) -> dict:
    if not pairs:
        return {
            "mean_cosine_similarity": None,
            "median_cosine_similarity": None,
            "std_cosine_similarity": None,
            "min_cosine_similarity": None,
            "max_cosine_similarity": None,
            "num_pairs": 0,
        }
    vals = np.array([float(np.dot(z[i], z[j])) for i, j in pairs], dtype=np.float32)
    return {
        "mean_cosine_similarity": float(vals.mean()),
        "median_cosine_similarity": float(np.median(vals)),
        "std_cosine_similarity": float(vals.std()),
        "min_cosine_similarity": float(vals.min()),
        "max_cosine_similarity": float(vals.max()),
        "num_pairs": int(vals.shape[0]),
    }


def _gap(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a - b)


def evaluate_pair_similarity(
    embeddings: np.ndarray,
    metas: Sequence[dict],
    *,
    max_pairs_per_type: int = 50000,
    seed: int = 0,
) -> dict:
    """Evaluate explicit pair-type cosine similarities using embeddings only."""
    if embeddings.shape[0] != len(metas):
        raise ValueError("embeddings and metas must have the same length")
    if embeddings.shape[0] < 2:
        empty = _pair_stats(np.zeros((0, 1), dtype=np.float32), [])
        return {
            "pair_types": {
                "same_dance_label_same_or_diff_dancer": empty,
                "same_choreography_different_dancer": empty,
                "same_genre_different_choreography": empty,
                "different_genre": empty,
            },
            "gaps": {},
        }

    rng = np.random.default_rng(int(seed))
    z = embeddings.astype(np.float32, copy=False)
    z = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-8, None)
    max_pairs = max(1, int(max_pairs_per_type))

    by_dance: dict[str, List[int]] = {}
    by_genre: dict[str, List[int]] = {}
    for i, meta in enumerate(metas):
        by_dance.setdefault(str(meta["dance_label"]), []).append(i)
        by_genre.setdefault(str(meta["genre_label"]), []).append(i)

    pair_buckets = {
        "same_dance_label_same_or_diff_dancer": [],
        "same_choreography_different_dancer": [],
        "same_genre_different_choreography": [],
        "different_genre": [],
    }
    seen = {key: 0 for key in pair_buckets}

    for indices in by_dance.values():
        for a_pos in range(len(indices)):
            i = indices[a_pos]
            for j in indices[a_pos + 1 :]:
                seen["same_dance_label_same_or_diff_dancer"] = _reservoir_add(
                    pair_buckets["same_dance_label_same_or_diff_dancer"],
                    seen["same_dance_label_same_or_diff_dancer"],
                    (i, j),
                    max_pairs=max_pairs,
                    rng=rng,
                )
                if metas[i].get("dancer") != metas[j].get("dancer"):
                    seen["same_choreography_different_dancer"] = _reservoir_add(
                        pair_buckets["same_choreography_different_dancer"],
                        seen["same_choreography_different_dancer"],
                        (i, j),
                        max_pairs=max_pairs,
                        rng=rng,
                    )

    for genre, indices in by_genre.items():
        by_label: dict[str, List[int]] = {}
        for idx in indices:
            by_label.setdefault(str(metas[idx]["dance_label"]), []).append(idx)
        labels = sorted(by_label)
        for a_pos, label_a in enumerate(labels):
            for label_b in labels[a_pos + 1 :]:
                for i in by_label[label_a]:
                    for j in by_label[label_b]:
                        seen["same_genre_different_choreography"] = _reservoir_add(
                            pair_buckets["same_genre_different_choreography"],
                            seen["same_genre_different_choreography"],
                            (i, j),
                            max_pairs=max_pairs,
                            rng=rng,
                        )

    genres = sorted(by_genre)
    for a_pos, genre_a in enumerate(genres):
        for genre_b in genres[a_pos + 1 :]:
            for i in by_genre[genre_a]:
                for j in by_genre[genre_b]:
                    seen["different_genre"] = _reservoir_add(
                        pair_buckets["different_genre"],
                        seen["different_genre"],
                        (i, j),
                        max_pairs=max_pairs,
                        rng=rng,
                    )

    pair_types = {key: _pair_stats(z, pairs) for key, pairs in pair_buckets.items()}
    same = pair_types["same_dance_label_same_or_diff_dancer"]["mean_cosine_similarity"]
    same_choreo = pair_types["same_choreography_different_dancer"]["mean_cosine_similarity"]
    same_genre_diff = pair_types["same_genre_different_choreography"]["mean_cosine_similarity"]
    diff_genre = pair_types["different_genre"]["mean_cosine_similarity"]
    return {
        "pair_types": pair_types,
        "gaps": {
            "same_dance_label_mean_minus_same_genre_different_choreography_mean": _gap(
                same, same_genre_diff
            ),
            "same_dance_label_mean_minus_different_genre_mean": _gap(same, diff_genre),
            "same_choreography_different_dancer_mean_minus_same_genre_different_choreography_mean": _gap(
                same_choreo, same_genre_diff
            ),
        },
        "max_pairs_per_type": max_pairs,
    }


def _flatten_pair_metrics(pair_eval: dict) -> dict:
    flat = {}
    for name, stats in pair_eval.get("pair_types", {}).items():
        flat[f"val_pair_{name}_mean_cosine"] = stats.get("mean_cosine_similarity")
        flat[f"val_pair_{name}_num_pairs"] = stats.get("num_pairs", 0)
    for name, value in pair_eval.get("gaps", {}).items():
        flat[f"val_gap_{name}"] = value
    return flat


def evaluate(
    model: PoseGNNTemporalEncoder,
    val_loader: DataLoader,
    loss_fn: SupConLoss,
    device: torch.device,
    *,
    pair_eval_max_pairs_per_type: int = 50000,
    seed: int = 0,
) -> Dict:
    embs, d_labs, g_labs, metas = _embed_dataset(model, val_loader, device)
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
            "pair_similarity": evaluate_pair_similarity(embs, metas),
        }

    z_t = torch.from_numpy(embs).to(device)
    d_t = torch.from_numpy(d_labs).to(device)
    val_loss = float(loss_fn(z_t, d_t).item())
    mean_pos, mean_neg = positive_negative_cosine_means(z_t, d_t)
    dance_topk = _retrieval_topk_accuracy(embs, d_labs, ks=(1, 5))
    genre_topk = _retrieval_topk_accuracy(embs, g_labs, ks=(1, 5))
    pair_eval = evaluate_pair_similarity(
        embs,
        metas,
        max_pairs_per_type=pair_eval_max_pairs_per_type,
        seed=seed,
    )
    metrics = {
        "val_supcon_loss": val_loss,
        "val_mean_pos_cos": mean_pos,
        "val_mean_neg_cos": mean_neg,
        "val_retrieval_top1_dance_label": dance_topk[1],
        "val_retrieval_top5_dance_label": dance_topk[5],
        "val_retrieval_top1_genre": genre_topk[1],
        "val_retrieval_top5_genre": genre_topk[5],
        "val_num_samples": int(embs.shape[0]),
        "pair_similarity": pair_eval,
    }
    metrics.update(_flatten_pair_metrics(pair_eval))
    return metrics


def _save_checkpoint(
    path: Path,
    model: PoseGNNTemporalEncoder,
    cfg: TrainConfig,
    epoch: int,
    extra: Optional[dict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.frame_encoder_state_dict(),
        "embedding_dim": int(model.frame_encoder.embedding_dim),
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


def run(cfg: TrainConfig) -> Path:
    out_root = ensure_dir(cfg.output_dir)
    ckpt_dir = ensure_dir(out_root / "checkpoints")
    log_path = out_root / "train_log.csv"
    metrics_path = out_root / "metrics.json"
    pair_eval_path = out_root / "pair_similarity_eval.json"
    cfg_snapshot_path = out_root / "config_resolved.json"

    index_csv = Path(cfg.index_csv)
    if cfg.rebuild_index or not index_csv.exists():
        n_rows = build_index_csv(
            cfg.pkl_root,
            index_csv,
            situation=cfg.situation,
            use_cameras=cfg.use_cameras,
            genres=cfg.genres,
        )
        print(f"[index] built {n_rows} rows -> {index_csv}")

    rows = _filter_rows_for_config(load_index_csv(index_csv), cfg)
    if not rows:
        raise RuntimeError("index is empty after situation/camera/genre filtering")
    genres = sorted({r["genre_label"] for r in rows})
    print(f"[index] loaded {len(rows)} rows genres={genres}")

    train_rows, val_rows = split_index_rows(
        rows,
        split_mode=cfg.split_mode,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        val_cameras=cfg.val_cameras,
    )
    before = len(train_rows)
    train_rows = filter_labels_with_min_rows(train_rows, cfg.min_train_rows_per_label)
    dropped = before - len(train_rows)
    print(
        f"[split:{cfg.split_mode}] train={len(train_rows)} val={len(val_rows)} "
        f"dropped_train_rows_without_positives={dropped}"
    )
    if not train_rows:
        raise RuntimeError("no training rows remain after positive-label filtering")
    if not val_rows:
        raise RuntimeError("validation split is empty")

    all_dance = sorted({r["dance_label"] for r in rows})
    all_genre = sorted({r["genre_label"] for r in rows})
    dance_to_id = {label: i for i, label in enumerate(all_dance)}
    genre_to_id = {label: i for i, label in enumerate(all_genre)}

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
        random_start=False,
        dance_label_to_id=dance_to_id,
        genre_label_to_id=genre_to_id,
        seed=cfg.seed,
    )
    print(f"[dataset] train_items={len(train_ds)} val_windows={len(val_ds)}")

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
    val_loader = DataLoader(
        val_ds,
        batch_size=max(sampler.batch_size, 32),
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
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    cfg_snapshot = {k: v for k, v in cfg.__dict__.items()}
    cfg_snapshot["num_index_rows_after_filter"] = len(rows)
    cfg_snapshot["num_train_rows_after_positive_filter"] = len(train_rows)
    cfg_snapshot["num_val_rows"] = len(val_rows)
    cfg_snapshot["genres_discovered"] = genres
    cfg_snapshot["positive_filter_behavior"] = (
        "training rows whose dance_label has fewer than "
        f"{cfg.min_train_rows_per_label} rows are filtered before sampling"
    )
    write_json(cfg_snapshot_path, cfg_snapshot)

    log_rows: List[dict] = []
    best_val = float("inf")
    best_summary: Optional[dict] = None
    best_pair_eval: Optional[dict] = None
    best_epoch = -1

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for step, batch in enumerate(train_loader):
            x = batch["pose_window"].to(device, non_blocking=True)
            m = batch["mask"].any(dim=-1).to(device)
            labels = batch["dance_label_id"].to(device, non_blocking=True)
            z = model(x, mask=m)
            loss = loss_fn(z, labels)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            running += float(loss.item())
            n_batches += 1
            if cfg.log_every and step % cfg.log_every == 0:
                print(f"  epoch {epoch:3d} step {step:4d} loss={float(loss.item()):.4f}")

        row = {
            "epoch": epoch,
            "train_supcon_loss": running / max(1, n_batches),
            "epoch_secs": time.time() - t0,
        }
        if cfg.eval_every and epoch % cfg.eval_every == 0:
            metrics = evaluate(
                model,
                val_loader,
                loss_fn,
                device,
                pair_eval_max_pairs_per_type=cfg.pair_eval_max_pairs_per_type,
                seed=cfg.seed + epoch,
            )
            pair_eval = metrics.pop("pair_similarity")
            row.update(metrics)
            write_json(pair_eval_path, pair_eval)
            print(
                f"[epoch {epoch:3d}] train={row['train_supcon_loss']:.4f} "
                f"val={metrics['val_supcon_loss']:.4f} "
                f"top1_dance={metrics['val_retrieval_top1_dance_label']:.3f} "
                f"top1_genre={metrics['val_retrieval_top1_genre']:.3f}"
            )
            if metrics["val_supcon_loss"] < best_val:
                best_val = metrics["val_supcon_loss"]
                best_epoch = epoch
                best_summary = dict(row)
                best_pair_eval = pair_eval
                _save_checkpoint(ckpt_dir / "best.pt", model, cfg, epoch, extra={"val_metrics": metrics})
        else:
            print(f"[epoch {epoch:3d}] train={row['train_supcon_loss']:.4f}")

        log_rows.append(row)
        _save_checkpoint(ckpt_dir / "last.pt", model, cfg, epoch)
        if cfg.save_every and epoch % cfg.save_every == 0:
            _save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", model, cfg, epoch)
        _write_log_csv(log_path, log_rows)

    summary = {
        "best_epoch": best_epoch,
        "best_val_supcon_loss": best_val if np.isfinite(best_val) else None,
        "best": best_summary,
        "best_pair_similarity": best_pair_eval,
        "num_epochs": cfg.epochs,
        "split_mode": cfg.split_mode,
        "embedding_dim": cfg.embedding_dim,
        "frame_embedding_dim": cfg.frame_embedding_dim,
        "num_index_rows_after_filter": len(rows),
        "num_train_rows_after_positive_filter": len(train_rows),
        "num_val_rows": len(val_rows),
        "genres_discovered": genres,
        "positive_filter_behavior": cfg_snapshot["positive_filter_behavior"],
        "config": cfg_snapshot,
    }
    write_json(metrics_path, summary)
    if best_pair_eval is not None:
        write_json(pair_eval_path, best_pair_eval)

    if cfg.deploy_checkpoint and (ckpt_dir / "best.pt").exists():
        deploy = Path(cfg.deploy_checkpoint)
        deploy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt_dir / "best.pt", deploy)
        print(f"[deploy] copied best -> {deploy}")

    return out_root


def _write_log_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _main() -> None:
    p = argparse.ArgumentParser(description="SupCon training for Basic Dance PoseGNN embeddings.")
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batches-per-epoch", type=int, default=None)
    p.add_argument("--rebuild-index", action="store_true")
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
    if args.batches_per_epoch is not None:
        raw["batches_per_epoch"] = args.batches_per_epoch
    if args.rebuild_index:
        raw["rebuild_index"] = True

    cfg = _to_train_config(raw)
    out = run(cfg)
    print(f"done -> {out}")


if __name__ == "__main__":
    _main()
