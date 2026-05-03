"""
test.py — Evaluation of the trained LSTM TemporalErrorDetector on the held-out test split.

Reports per-part binary precision, recall, F1, and accuracy, plus
a fixed-threshold baseline for comparison.

Launch via main.py:
    python main.py test --test-dir data/test/ \
                        --checkpoint checkpoints/best_model.pt \
                        --out results/test_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from dataset import DanceDeviationDataset, collate_fn, PART_ORDER, N_PARTS
from model import load_checkpoint
from scoring import THRESHOLD_MODERATE


def evaluate(
    test_dir: str,
    checkpoint_path: str,
    out_path: str = "results/test_metrics.json",
    prob_threshold: float = 0.5,
    device: str | None = None,
) -> dict:
    """Run evaluation on the test split and save metrics to JSON.

    Returns the metrics dictionary.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test] Device: {device}")

    test_ds = DanceDeviationDataset(test_dir)
    loader  = DataLoader(test_ds, batch_size=16, shuffle=False,
                         collate_fn=collate_fn, num_workers=2)
    print(f"[test] Test samples: {len(test_ds)}")

    model, ckpt_meta = load_checkpoint(checkpoint_path, device=device)
    epoch  = ckpt_meta.get("epoch", "?")
    val_f1 = ckpt_meta.get("val_f1")
    f1_str = f"{val_f1:.4f}" if isinstance(val_f1, float) else "?"
    print(f"[test] Loaded checkpoint from epoch {epoch} (val_f1={f1_str})")

    # Collect per-part predictions across the whole test set
    all_probs_by_part   = [[] for _ in range(N_PARTS)]
    all_targets_by_part = [[] for _ in range(N_PARTS)]

    model.eval()
    with torch.no_grad():
        for features, labels, lengths in loader:
            features = features.to(device)
            logits   = model(features)              # (B, T_max, 6)
            probs    = torch.sigmoid(logits)        # (B, T_max, 6)

            for p in range(N_PARTS):
                mask = labels[:, :, p] >= 0        # not padding
                all_probs_by_part[p].append(
                    probs[:, :, p][mask].cpu().numpy())
                all_targets_by_part[p].append(
                    labels[:, :, p][mask].cpu().numpy())

    # ---- Per-body-part metrics ----
    part_metrics = {}
    print("\n=== Per-Body-Part Metrics (model, threshold=0.5) ===")
    all_probs_flat   = []
    all_targets_flat = []

    for p, part in enumerate(PART_ORDER):
        p_probs   = np.concatenate(all_probs_by_part[p])
        p_targets = np.concatenate(all_targets_by_part[p]).astype(int)
        p_preds   = (p_probs >= prob_threshold).astype(int)

        prec, rec, f1, _ = precision_recall_fscore_support(
            p_targets, p_preds, average="binary", zero_division=0)
        acc  = float((p_preds == p_targets).mean())
        try:
            auc = float(roc_auc_score(p_targets, p_probs))
        except ValueError:
            auc = float("nan")

        part_metrics[part] = {
            "precision": round(float(prec), 4),
            "recall":    round(float(rec),  4),
            "f1":        round(float(f1),   4),
            "accuracy":  round(acc,          4),
            "auc":       round(auc,          4),
        }
        print(f"  {part:12s}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  "
              f"Acc={acc:.4f}  AUC={auc:.4f}")

        all_probs_flat.append(p_probs)
        all_targets_flat.append(p_targets)

    # ---- Overall (all parts pooled) ----
    probs_all   = np.concatenate(all_probs_flat)
    targets_all = np.concatenate(all_targets_flat)
    preds_all   = (probs_all >= prob_threshold).astype(int)
    prec_all, rec_all, f1_all, _ = precision_recall_fscore_support(
        targets_all, preds_all, average="binary", zero_division=0)
    acc_all = float((preds_all == targets_all).mean())

    print(f"\n=== Overall (all parts pooled) ===")
    print(f"  P={prec_all:.4f}  R={rec_all:.4f}  F1={f1_all:.4f}  Acc={acc_all:.4f}")

    # ---- Fixed-threshold baseline ----
    # Re-derive predictions from the mean_joint_err feature column (col +3)
    base_preds_flat   = []
    base_targets_flat = []
    model.eval()
    with torch.no_grad():
        for features, labels, lengths in loader:
            for p in range(N_PARTS):
                mask = labels[:, :, p] >= 0
                mean_err = features[:, :, p * 4 + 3]
                base_preds_flat.append(
                    (mean_err[mask].numpy() >= THRESHOLD_MODERATE).astype(int))
                base_targets_flat.append(
                    labels[:, :, p][mask].numpy().astype(int))

    base_preds   = np.concatenate(base_preds_flat)
    base_targets = np.concatenate(base_targets_flat)
    _, _, base_f1, _ = precision_recall_fscore_support(
        base_targets, base_preds, average="binary", zero_division=0)
    base_acc = float((base_preds == base_targets).mean())
    print(f"\n=== Threshold Baseline (error ≥ {THRESHOLD_MODERATE}) ===")
    print(f"  F1={base_f1:.4f}  Acc={base_acc:.4f}")

    # ---- Save ----
    metrics = {
        "overall": {
            "precision": round(float(prec_all), 4),
            "recall":    round(float(rec_all),  4),
            "f1":        round(float(f1_all),   4),
            "accuracy":  round(acc_all,          4),
        },
        "per_part": part_metrics,
        "baseline": {
            "f1":       round(float(base_f1),  4),
            "accuracy": round(base_acc,         4),
        },
        "checkpoint_meta": {k: str(v) for k, v in ckpt_meta.items()},
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(metrics, indent=2))
    print(f"\n[test] Metrics saved → {out_path}")
    return metrics
