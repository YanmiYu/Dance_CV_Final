"""Temporal pose-window dataset for supervised contrastive GNN training."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.compare.normalize_pose import NormalizeConfig, normalize_sequence


def _load_camera_sequence(path: str, camera_index: int) -> np.ndarray:
    """Load one ``(T, 17, 3)`` camera sequence from a single or cAll PKL."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    kp = data["keypoints2d"] if isinstance(data, dict) and "keypoints2d" in data else data
    arr = np.asarray(kp)
    if arr.ndim == 4:
        if camera_index < 0 or camera_index >= arr.shape[0]:
            raise IndexError(f"camera_index {camera_index} out of range for {arr.shape}")
        arr = arr[camera_index]
    if arr.ndim != 3 or arr.shape[1:] != (17, 3):
        raise ValueError(f"unexpected keypoints2d shape: {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32, copy=False)
    return arr


def _build_label_maps(rows: Sequence[dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    dance_labels = sorted({r["dance_label"] for r in rows})
    genre_labels = sorted({r["genre_label"] for r in rows})
    return (
        {label: i for i, label in enumerate(dance_labels)},
        {label: i for i, label in enumerate(genre_labels)},
    )


def split_index_rows(
    rows: Sequence[dict],
    *,
    split_mode: str = "dance_label",
    val_ratio: float = 0.2,
    seed: int = 42,
    val_cameras: Optional[Sequence[str]] = None,
) -> Tuple[List[dict], List[dict]]:
    """Split rows into train/val without leakage for the chosen split mode."""
    rng = np.random.default_rng(int(seed))
    rows = list(rows)
    if split_mode == "dance_label":
        labels = sorted({r["dance_label"] for r in rows})
        rng.shuffle(labels)
        n_val = max(1, int(round(len(labels) * float(val_ratio)))) if labels else 0
        val_labels = set(labels[:n_val])
        train = [r for r in rows if r["dance_label"] not in val_labels]
        val = [r for r in rows if r["dance_label"] in val_labels]
    elif split_mode == "camera":
        all_cams = sorted({r["camera"] for r in rows})
        if val_cameras is None:
            n_val = max(1, int(round(len(all_cams) * float(val_ratio)))) if all_cams else 0
            val_cameras = all_cams[-n_val:]
        val_cams = set(val_cameras)
        train = [r for r in rows if r["camera"] not in val_cams]
        val = [r for r in rows if r["camera"] in val_cams]
    else:
        raise ValueError(f"unknown split_mode: {split_mode!r}")
    return train, val


def filter_labels_with_min_rows(rows: Sequence[dict], min_rows_per_label: int = 2) -> List[dict]:
    """Drop dance labels that do not have enough rows for SupCon positives.

    Training uses one random window per dataset row, so this filter guarantees
    the balanced sampler has at least two distinct items per retained label.
    """
    if min_rows_per_label <= 1:
        return list(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["dance_label"]] = counts.get(row["dance_label"], 0) + 1
    keep = {label for label, count in counts.items() if count >= min_rows_per_label}
    return [row for row in rows if row["dance_label"] in keep]


class BasicDanceSupConDataset(Dataset):
    """Yield normalized temporal pose windows for SupCon training."""

    def __init__(
        self,
        rows: Sequence[dict],
        *,
        window_size: int = 32,
        window_stride: int = 8,
        min_confidence: float = 0.2,
        random_start: bool = True,
        normalize: bool = True,
        dance_label_to_id: Optional[Dict[str, int]] = None,
        genre_label_to_id: Optional[Dict[str, int]] = None,
        seed: int = 0,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if window_stride <= 0:
            raise ValueError("window_stride must be positive")
        rows = [r for r in rows if int(r.get("num_frames", 0)) >= window_size]
        if not rows:
            raise ValueError("no rows have enough frames for the requested window_size")

        self._rows = list(rows)
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.min_confidence = float(min_confidence)
        self.random_start = bool(random_start)
        self.normalize = bool(normalize)
        self._rng = np.random.default_rng(int(seed))
        self._normalize_cfg = NormalizeConfig(min_visibility=self.min_confidence)

        if dance_label_to_id is None or genre_label_to_id is None:
            d_map, g_map = _build_label_maps(self._rows)
            dance_label_to_id = dance_label_to_id or d_map
            genre_label_to_id = genre_label_to_id or g_map
        self.dance_label_to_id = dict(dance_label_to_id)
        self.genre_label_to_id = dict(genre_label_to_id)

        self._cache: Dict[Tuple[str, int], np.ndarray] = {}
        self._items: List[Tuple[int, int]] = []
        if self.random_start:
            self._items = [(i, -1) for i in range(len(self._rows))]
        else:
            for i, row in enumerate(self._rows):
                T = int(row["num_frames"])
                last_start = T - self.window_size
                for start in range(0, last_start + 1, self.window_stride):
                    self._items.append((i, start))
                if last_start % self.window_stride != 0:
                    self._items.append((i, last_start))

    @property
    def num_dance_labels(self) -> int:
        return len(self.dance_label_to_id)

    @property
    def num_genre_labels(self) -> int:
        return len(self.genre_label_to_id)

    def labels(self) -> List[int]:
        return [self.dance_label_to_id[self._rows[i]["dance_label"]] for i, _ in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        row_i, fixed_start = self._items[idx]
        row = self._rows[row_i]
        sequence = self._get_sequence(row)
        T = sequence.shape[0]
        if fixed_start < 0:
            last = T - self.window_size
            start = int(self._rng.integers(0, last + 1)) if last > 0 else 0
        else:
            start = int(fixed_start)
        window = sequence[start : start + self.window_size]
        mask = window[..., 2] >= self.min_confidence

        return {
            "pose_window": torch.from_numpy(window.astype(np.float32, copy=False)),
            "mask": torch.from_numpy(mask.astype(bool)),
            "dance_label_id": torch.tensor(self.dance_label_to_id[row["dance_label"]], dtype=torch.long),
            "genre_label_id": torch.tensor(self.genre_label_to_id[row["genre_label"]], dtype=torch.long),
            "meta": {
                "path": row["path"],
                "stem": row["stem"],
                "camera": row["camera"],
                "dancer": row["dancer"],
                "music_id": row["music_id"],
                "choreography_id": row["choreography_id"],
                "dance_label": row["dance_label"],
                "genre_label": row["genre_label"],
                "window_start": start,
            },
        }

    def _get_sequence(self, row: dict) -> np.ndarray:
        key = (row["path"], int(row["camera_index"]))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        seq = _load_camera_sequence(row["path"], int(row["camera_index"]))
        if self.normalize:
            seq, _mask = normalize_sequence(seq, self._normalize_cfg)
        self._cache[key] = seq
        return seq


def supcon_collate(batch: Sequence[Dict]) -> Dict:
    return {
        "pose_window": torch.stack([b["pose_window"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "dance_label_id": torch.stack([b["dance_label_id"] for b in batch], dim=0),
        "genre_label_id": torch.stack([b["genre_label_id"] for b in batch], dim=0),
        "meta": [b["meta"] for b in batch],
    }
