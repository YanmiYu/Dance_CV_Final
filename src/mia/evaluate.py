"""Evaluate Mia's LSTM temporal error detector on held-out ``.npz`` samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.mia.dataset import DanceDeviationDataset, N_PARTS, PART_ORDER, collate_fn
from src.mia.model import load_checkpoint
from src.mia.scoring import THRESHOLD_MODERATE


def _precision_recall_f1(targets: np.ndarray, preds: np.ndarray) -> tuple[float, float, float]:
    targets = targets.astype(bool)
    preds = preds.astype(bool)
    tp = float(np.logical_and(preds, targets).sum())
    fp = float(np.logical_and(preds, ~targets).sum())
    fn = float(np.logical_and(~preds, targets).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _roc_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary ROC AUC using average ranks, returning NaN if undefined."""
    targets = targets.astype(int)
    n_pos = int((targets == 1).sum())
    n_neg = int((targets == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.zeros(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    sum_pos_ranks = float(ranks[targets == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _default_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(
    test_dir: str,
    checkpoint_path: str,
    out_path: str = "results/lstm_test_metrics.json",
    prob_threshold: float = 0.5,
    device: str | None = None,
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict:
    """Run held-out evaluation and write precision/recall/F1 metrics to JSON."""
    device = _default_device(device)
    print(f"[mia.evaluate] device={device}")

    test_ds = DanceDeviationDataset(test_dir)
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    print(f"[mia.evaluate] test={len(test_ds)}")

    model, ckpt_meta = load_checkpoint(checkpoint_path, device=device)
    epoch = ckpt_meta.get("epoch", "?")
    val_f1 = ckpt_meta.get("val_f1")
    val_f1_str = f"{val_f1:.4f}" if isinstance(val_f1, float) else "?"
    print(f"[mia.evaluate] checkpoint epoch={epoch} val_f1={val_f1_str}")

    all_probs_by_part: list[list[np.ndarray]] = [[] for _ in range(N_PARTS)]
    all_targets_by_part: list[list[np.ndarray]] = [[] for _ in range(N_PARTS)]

    model.eval()
    with torch.no_grad():
        for features, labels, _lengths in loader:
            features = features.to(device)
            labels = labels.to(device)
            probs = torch.sigmoid(model(features))

            for p_idx in range(N_PARTS):
                mask = labels[:, :, p_idx] >= 0
                all_probs_by_part[p_idx].append(
                    probs[:, :, p_idx][mask].detach().cpu().numpy()
                )
                all_targets_by_part[p_idx].append(
                    labels[:, :, p_idx][mask].detach().cpu().numpy()
                )

    part_metrics: dict[str, dict[str, float]] = {}
    all_probs_flat: list[np.ndarray] = []
    all_targets_flat: list[np.ndarray] = []

    for p_idx, part in enumerate(PART_ORDER):
        part_probs = np.concatenate(all_probs_by_part[p_idx])
        part_targets = np.concatenate(all_targets_by_part[p_idx]).astype(int)
        part_preds = (part_probs >= prob_threshold).astype(int)

        precision, recall, f1 = _precision_recall_f1(part_targets, part_preds)
        accuracy = float((part_preds == part_targets).mean())
        auc = _roc_auc(part_targets, part_probs)

        part_metrics[part] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "accuracy": round(accuracy, 4),
            "auc": round(auc, 4),
        }
        all_probs_flat.append(part_probs)
        all_targets_flat.append(part_targets)

    probs_all = np.concatenate(all_probs_flat)
    targets_all = np.concatenate(all_targets_flat)
    preds_all = (probs_all >= prob_threshold).astype(int)
    precision_all, recall_all, f1_all = _precision_recall_f1(targets_all, preds_all)
    accuracy_all = float((preds_all == targets_all).mean())

    baseline_preds: list[np.ndarray] = []
    baseline_targets: list[np.ndarray] = []
    for features, labels, _lengths in loader:
        for p_idx in range(N_PARTS):
            mask = labels[:, :, p_idx] >= 0
            mean_err = features[:, :, p_idx * 4 + 3]
            baseline_preds.append((mean_err[mask].numpy() >= THRESHOLD_MODERATE).astype(int))
            baseline_targets.append(labels[:, :, p_idx][mask].numpy().astype(int))

    base_preds = np.concatenate(baseline_preds)
    base_targets = np.concatenate(baseline_targets)
    _, _, base_f1 = _precision_recall_f1(base_targets, base_preds)
    base_accuracy = float((base_preds == base_targets).mean())

    metrics = {
        "overall": {
            "precision": round(float(precision_all), 4),
            "recall": round(float(recall_all), 4),
            "f1": round(float(f1_all), 4),
            "accuracy": round(accuracy_all, 4),
        },
        "per_part": part_metrics,
        "baseline": {
            "threshold": THRESHOLD_MODERATE,
            "f1": round(float(base_f1), 4),
            "accuracy": round(base_accuracy, 4),
        },
        "checkpoint_meta": {k: str(v) for k, v in ckpt_meta.items()},
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(metrics, indent=2))
    print(f"[mia.evaluate] wrote {out_path}")
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Mia LSTM temporal error detector")
    p.add_argument("--test-dir", required=True)
    p.add_argument("--checkpoint", default="checkpoints/lstm/best_model.pt")
    p.add_argument("--out", default="results/lstm_test_metrics.json")
    p.add_argument("--prob-threshold", type=float, default=0.5)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    evaluate(
        test_dir=args.test_dir,
        checkpoint_path=args.checkpoint,
        out_path=args.out,
        prob_threshold=args.prob_threshold,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
