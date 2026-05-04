"""Temporal pose-window dataset for supervised contrastive training.

Each item is one window of ``window_size`` frames drawn from one logical
camera view of one PKL. Windows from the same ``dance_label`` are positives,
windows from different labels are negatives.

Returns a dict per item with:

  * ``pose_window``     ``torch.float32`` of shape ``(T, 17, 3)``
                        (already normalized; channel 2 is confidence)
  * ``mask``            ``torch.bool``    of shape ``(T, 17)`` -- True for
                        joints above the confidence threshold
  * ``dance_label_id``  ``torch.long`` scalar
  * ``genre_label_id``  ``torch.long`` scalar
  * ``meta``            ``dict`` with ``stem``, ``camera``, ``window_start``
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.compare.normalize_pose import NormalizeConfig, normalize_sequence
from src.datasets.basic_dance_index import load_index_csv


def _load_camera_sequence(path: str, camera_index: int) -> np.ndarray:
    """Return a ``(T, 17, 3)`` pose sequence for the given camera.

    Handles both consolidated ``(C, T, 17, 3)`` PKLs and single-camera
    ``(T, 17, 3)`` PKLs.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    kp = data["keypoints2d"] if isinstance(data, dict) and "keypoints2d" in data else data
    arr = np.asarray(kp)
    if arr.ndim == 4:
        if camera_index < 0 or camera_index >= arr.shape[0]:
            raise IndexError(
                f"camera_index {camera_index} out of range for shape {arr.shape}"
            )
        arr = arr[camera_index]
    if arr.ndim != 3 or arr.shape[1:] != (17, 3):
        raise ValueError(f"unexpected keypoints2d shape: {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    # The raw AIST++ PKLs occasionally contain all-NaN frames where pose
    # estimation failed. Replace those with zero coords + zero confidence so
    # downstream normalization stays finite; the confidence mask will hide
    # the joints from the SupCon objective.
    if not np.isfinite(arr).all():
        arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32, copy=False)
    return arr


def _build_label_maps(rows: Sequence[dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    dance_labels = sorted({r["dance_label"] for r in rows})
    genre_labels = sorted({r["genre_label"] for r in rows})
    return (
        {lab: i for i, lab in enumerate(dance_labels)},
        {lab: i for i, lab in enumerate(genre_labels)},
    )


def split_index_rows(
    rows: Sequence[dict],
    *,
    split_mode: str = "dance_label",
    val_ratio: float = 0.2,
    seed: int = 42,
    val_cameras: Optional[Sequence[str]] = None,
) -> Tuple[List[dict], List[dict]]:
    """Split rows into (train, val) with no leakage.

    ``split_mode``:
      * ``"dance_label"`` — partitions distinct dance_label values; tests
        generalisation to unseen choreographies.
      * ``"camera"``      — same dance_label can appear in train and val,
        but with different cameras; tests view-invariance. ``val_cameras``
        controls which cameras go to val (default: last 2 cameras).
    """
    rng = np.random.default_rng(int(seed))
    if split_mode == "dance_label":
        labels = sorted({r["dance_label"] for r in rows})
        rng.shuffle(labels)
        n_val = max(1, int(round(len(labels) * float(val_ratio))))
        val_labels = set(labels[:n_val])
        train = [r for r in rows if r["dance_label"] not in val_labels]
        val = [r for r in rows if r["dance_label"] in val_labels]
    elif split_mode == "camera":
        all_cams = sorted({r["camera"] for r in rows})
        if val_cameras is None:
            n_val = max(1, int(round(len(all_cams) * float(val_ratio))))
            val_cameras = all_cams[-n_val:]
        val_cams = set(val_cameras)
        train = [r for r in rows if r["camera"] not in val_cams]
        val = [r for r in rows if r["camera"] in val_cams]
    else:
        raise ValueError(f"unknown split_mode: {split_mode!r}")
    return train, val


class BasicDanceSupConDataset(Dataset):
    """Yields temporal pose windows for SupCon training.

    Args:
        rows: index rows (e.g. from :func:`load_index_csv`).
        window_size: number of frames per window.
        window_stride: distance between consecutive window starts when
            enumerating windows in deterministic mode (training uses random
            starts within the clip when ``random_start=True``).
        min_confidence: joints with confidence below this are masked. We
            still feed them to the model (zero-masked), so the GNN can rely
            on neighbours via the COCO graph.
        random_start: training mode -- pick a random start within the
            valid range each call. Validation mode pre-enumerates windows.
        normalize: whether to apply :func:`normalize_sequence` before
            window slicing (keeps the model in the same coordinate space
            as render_report).
        dance_label_to_id / genre_label_to_id: optional precomputed maps.
    """

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

        # Cache: path -> (camera_index -> normalized (T,17,3) ndarray).
        self._cache: Dict[Tuple[str, int], np.ndarray] = {}

        # In deterministic mode, expand each row into multiple windows by stride.
        self._items: List[Tuple[int, int]] = []
        if self.random_start:
            self._items = [(i, -1) for i in range(len(self._rows))]
        else:
            for i, row in enumerate(self._rows):
                T = int(row["num_frames"])
                last_start = T - self.window_size
                if last_start < 0:
                    continue
                for s in range(0, last_start + 1, self.window_stride):
                    self._items.append((i, s))
                if last_start % self.window_stride != 0:
                    self._items.append((i, last_start))

    # ----- public API ----------------------------------------------------

    @property
    def num_dance_labels(self) -> int:
        return len(self.dance_label_to_id)

    @property
    def num_genre_labels(self) -> int:
        return len(self.genre_label_to_id)

    def labels(self) -> List[int]:
        """Per-item dance_label_id (used by the balanced sampler)."""
        return [self.dance_label_to_id[self._rows[i]["dance_label"]] for i, _ in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        row_i, fixed_start = self._items[idx]
        row = self._rows[row_i]
        sequence = self._get_sequence(row)
        T = sequence.shape[0]
        if T < self.window_size:
            raise RuntimeError(
                f"sequence shorter than window_size after normalization: {T} < {self.window_size}"
            )

        if fixed_start < 0:
            last = T - self.window_size
            start = int(self._rng.integers(0, last + 1)) if last > 0 else 0
        else:
            start = int(fixed_start)
        end = start + self.window_size
        window = sequence[start:end]              # (T, 17, 3)
        conf = window[..., 2]
        mask = conf >= self.min_confidence

        return {
            "pose_window": torch.from_numpy(window.astype(np.float32, copy=False)),
            "mask": torch.from_numpy(mask.astype(bool)),
            "dance_label_id": torch.tensor(
                self.dance_label_to_id[row["dance_label"]], dtype=torch.long
            ),
            "genre_label_id": torch.tensor(
                self.genre_label_to_id[row["genre_label"]], dtype=torch.long
            ),
            "meta": {
                "stem": row["stem"],
                "camera": row["camera"],
                "dance_label": row["dance_label"],
                "genre_label": row["genre_label"],
                "window_start": start,
            },
        }

    # ----- helpers -------------------------------------------------------

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
    """Default collate that stacks tensors and keeps ``meta`` as a list."""
    pose = torch.stack([b["pose_window"] for b in batch], dim=0)
    mask = torch.stack([b["mask"] for b in batch], dim=0)
    dance_ids = torch.stack([b["dance_label_id"] for b in batch], dim=0)
    genre_ids = torch.stack([b["genre_label_id"] for b in batch], dim=0)
    metas = [b["meta"] for b in batch]
    return {
        "pose_window": pose,
        "mask": mask,
        "dance_label_id": dance_ids,
        "genre_label_id": genre_ids,
        "meta": metas,
    }
