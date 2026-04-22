"""
dataset.py — Difference feature builder and PyTorch Dataset for the BiGRU classifier.
Owner: Member 2

Feature vector per aligned frame: shape (24,)
  For each of 6 body parts:
    [mean_x_err, mean_y_err, max_joint_err, mean_joint_err]  →  4 features
  6 parts × 4 features = 24 dimensions

Label per frame per body part: int in {0, 1, 2}
  0 = good      (error < THRESHOLD_GOOD)
  1 = moderate  (THRESHOLD_GOOD ≤ error < THRESHOLD_MODERATE)
  2 = off       (error ≥ THRESHOLD_MODERATE)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scoring import BODY_PARTS, THRESHOLD_GOOD, THRESHOLD_MODERATE

# Ordered list so features are always in the same column order
PART_ORDER = ["LEFT_ARM", "RIGHT_ARM", "LEFT_LEG", "RIGHT_LEG", "TORSO", "HEAD"]
N_PARTS = len(PART_ORDER)
N_FEATURES_PER_PART = 4
FEATURE_DIM = N_PARTS * N_FEATURES_PER_PART  # 24


def build_diff_features(
    bench_aligned: np.ndarray,
    user_aligned: np.ndarray,
) -> np.ndarray:
    """Compute the 24-dim difference feature vector for each aligned frame.

    Parameters
    ----------
    bench_aligned, user_aligned : np.ndarray, shape (T', 17, 3)
        DTW-aligned normalized keypoint sequences.

    Returns
    -------
    np.ndarray, shape (T', 24)
    """
    diff_xy = bench_aligned[:, :, :2] - user_aligned[:, :, :2]  # (T', 17, 2)
    joint_errors = np.linalg.norm(diff_xy, axis=2)              # (T', 17)

    features = np.zeros((len(bench_aligned), FEATURE_DIM), dtype=np.float32)
    for p_idx, part in enumerate(PART_ORDER):
        joints = BODY_PARTS[part]
        part_err = joint_errors[:, joints]          # (T', n_joints)
        col = p_idx * N_FEATURES_PER_PART
        features[:, col + 0] = diff_xy[:, joints, 0].mean(axis=1)  # mean x err
        features[:, col + 1] = diff_xy[:, joints, 1].mean(axis=1)  # mean y err
        features[:, col + 2] = part_err.max(axis=1)                 # max joint err
        features[:, col + 3] = part_err.mean(axis=1)                # mean joint err
    return features


def build_labels(
    bench_aligned: np.ndarray,
    user_aligned: np.ndarray,
) -> np.ndarray:
    """Generate per-frame per-part class labels using geometric thresholds.

    Returns
    -------
    np.ndarray, shape (T', 6)  — dtype int64; values in {0, 1, 2}
    """
    diff_xy = bench_aligned[:, :, :2] - user_aligned[:, :, :2]
    joint_errors = np.linalg.norm(diff_xy, axis=2)  # (T', 17)

    labels = np.zeros((len(bench_aligned), N_PARTS), dtype=np.int64)
    for p_idx, part in enumerate(PART_ORDER):
        joints = BODY_PARTS[part]
        mean_err = joint_errors[:, joints].mean(axis=1)  # (T',)
        labels[:, p_idx] = np.where(
            mean_err >= THRESHOLD_MODERATE, 2,
            np.where(mean_err >= THRESHOLD_GOOD, 1, 0)
        )
    return labels


def save_sample(
    features: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
) -> None:
    """Save a (features, labels) pair to a .npz file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), features=features, labels=labels)


def load_sample(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a (features, labels) pair from a .npz file."""
    data = np.load(str(path))
    return data["features"], data["labels"]


class DanceDeviationDataset:
    """PyTorch Dataset over a directory of .npz sample files.

    Each sample is one aligned pair:
      features: Tensor (T', 24)
      labels:   Tensor (T', 6)  — int64

    torch is imported lazily so the rest of the pipeline can run without it.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.files = sorted(Path(data_dir).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        import torch
        features, labels = load_sample(self.files[idx])
        return (
            torch.from_numpy(features).float(),   # (T', 24)
            torch.from_numpy(labels).long(),       # (T', 6)
        )


def collate_fn(batch):
    """Pad sequences in a batch to the same length.

    Returns
    -------
    features  : (B, T_max, 24)
    labels    : (B, T_max, 6)
    lengths   : (B,)  — original sequence lengths before padding
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    feat_list, lbl_list = zip(*batch)
    lengths  = torch.tensor([f.shape[0] for f in feat_list])
    features = pad_sequence(feat_list, batch_first=True)
    labels   = pad_sequence(lbl_list,  batch_first=True, padding_value=-1)
    return features, labels, lengths
