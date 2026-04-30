"""Triplet dataset for AIST++ pose-embedding training.

This dataset reads the per-video AIST++ keypoint files produced by
``scripts.link_aist_keypoints``:

    data/labels/aistpp/keypoints2d_raw/<video_stem>.pkl

Each file is expected to contain a ``keypoints2d`` array of shape
``(T, 17, 3)`` with ``x, y, confidence``. The x/y coordinates are normalized
per video to [0, 1] when they appear to be pixel coordinates. Confidence is
kept as the third node feature so the future GNN can learn to discount weak
joints without changing the graph input shape.
"""
from __future__ import annotations

import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


PoseIndex = Tuple[str, int]
TripletIndices = Tuple[PoseIndex, PoseIndex, PoseIndex]


_CHORE_RE = re.compile(r"_ch(?P<chore>\d+)$")


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    keypoints: np.ndarray  # (T, 17, 3), normalized x/y + confidence
    choreography: Optional[str]

    @property
    def num_frames(self) -> int:
        return int(self.keypoints.shape[0])


def _load_keypoints(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        if "keypoints2d" not in obj:
            raise ValueError(f"{path} is missing key 'keypoints2d'")
        kps = obj["keypoints2d"]
    else:
        kps = obj
    arr = np.asarray(kps, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (17, 3):
        raise ValueError(f"Expected keypoints shape (T, 17, 3), got {arr.shape} in {path}")
    return arr


def _normalize_xy_keep_conf(kps: np.ndarray) -> np.ndarray:
    """Return a copy with x/y in [0, 1] and confidence kept as channel 2."""
    out = kps.astype(np.float32, copy=True)
    xy = out[..., :2]
    finite = np.isfinite(xy)
    if not finite.any():
        out[..., :2] = 0.0
    else:
        valid_xy = xy[finite]
        already_unit = float(np.nanmin(valid_xy)) >= 0.0 and float(np.nanmax(valid_xy)) <= 1.0
        if not already_unit:
            for dim in range(2):
                vals = xy[..., dim]
                mask = np.isfinite(vals)
                if not mask.any():
                    out[..., dim] = 0.0
                    continue
                lo = float(np.nanmin(vals[mask]))
                hi = float(np.nanmax(vals[mask]))
                denom = max(hi - lo, 1e-6)
                out[..., dim] = (vals - lo) / denom
        out[..., :2] = np.nan_to_num(out[..., :2], nan=0.0, posinf=1.0, neginf=0.0)
        out[..., :2] = np.clip(out[..., :2], 0.0, 1.0)

    out[..., 2] = np.nan_to_num(out[..., 2], nan=0.0, posinf=1.0, neginf=0.0)
    out[..., 2] = np.clip(out[..., 2], 0.0, 1.0)
    return out


def _parse_choreography(video_id: str) -> Optional[str]:
    m = _CHORE_RE.search(video_id)
    return m.group("chore") if m else None


def build_pose_index(records: Sequence[VideoRecord]) -> List[PoseIndex]:
    """Flatten videos into ``(video_id, frame_index)`` samples."""
    index: List[PoseIndex] = []
    for rec in records:
        index.extend((rec.video_id, t) for t in range(rec.num_frames))
    return index


class AISTEmbeddingDataset(Dataset):
    """AIST++ temporal triplet dataset for pose-embedding training.

    ``negative_strategy`` supports:
      - ``"far_frame"``: same video, frame distance greater than 30 if possible.
      - ``"different_video"``: sample another video, preferring different choreo.
    """

    def __init__(
        self,
        keypoints_dir: str | Path,
        window_size: int = 1,
        positive_range: int = 5,
        negative_strategy: str = "far_frame",
        *,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self.keypoints_dir = Path(keypoints_dir)
        self.window_size = int(window_size)
        self.positive_range = int(positive_range)
        self.negative_strategy = str(negative_strategy)
        self.rng = random.Random(seed)

        if self.window_size != 1:
            raise NotImplementedError("window_size > 1 will be added when sequence embeddings are introduced")
        if self.positive_range < 1:
            raise ValueError("positive_range must be >= 1")
        if self.negative_strategy not in {"far_frame", "different_video"}:
            raise ValueError(f"unknown negative_strategy: {self.negative_strategy!r}")

        self.records = self._load_records(self.keypoints_dir)
        if not self.records:
            raise ValueError(f"No .pkl keypoint files found under {self.keypoints_dir}")
        self.video_to_idx: Dict[str, int] = {rec.video_id: i for i, rec in enumerate(self.records)}
        self.index = build_pose_index(self.records)

        if verbose:
            self.print_debug_summary()

    @staticmethod
    def _load_records(keypoints_dir: Path) -> List[VideoRecord]:
        records: List[VideoRecord] = []
        for path in sorted(keypoints_dir.glob("*.pkl")):
            kps = _normalize_xy_keep_conf(_load_keypoints(path))
            if kps.shape[0] == 0:
                continue
            records.append(
                VideoRecord(
                    video_id=path.stem,
                    keypoints=kps,
                    choreography=_parse_choreography(path.stem),
                )
            )
        return records

    def print_debug_summary(self) -> None:
        total = sum(rec.num_frames for rec in self.records)
        avg = total / max(len(self.records), 1)
        print(
            f"AISTEmbeddingDataset: videos={len(self.records)} "
            f"total_frames={total} avg_frames_per_video={avg:.1f}"
        )

    def __len__(self) -> int:
        return len(self.index)

    def _pose(self, video_id: str, frame_index: int) -> torch.Tensor:
        rec = self.records[self.video_to_idx[video_id]]
        return torch.from_numpy(rec.keypoints[int(frame_index)].astype(np.float32, copy=False))

    def _sample_positive(self, video_id: str, t: int) -> PoseIndex:
        rec = self.records[self.video_to_idx[video_id]]
        lo = max(0, t - self.positive_range)
        hi = min(rec.num_frames - 1, t + self.positive_range)
        candidates = [i for i in range(lo, hi + 1) if i != t]
        if not candidates:
            raise ValueError(f"Cannot sample positive for single-frame video {video_id}")
        return video_id, self.rng.choice(candidates)

    def _sample_far_frame_negative(self, video_id: str, t: int, min_gap: int = 30) -> PoseIndex:
        rec = self.records[self.video_to_idx[video_id]]
        candidates = [i for i in range(rec.num_frames) if abs(i - t) > min_gap]
        if candidates:
            return video_id, self.rng.choice(candidates)

        if len(self.records) > 1:
            return self._sample_different_video_negative(video_id)

        fallback = [i for i in range(rec.num_frames) if i != t]
        if not fallback:
            raise ValueError(f"Cannot sample negative for single-frame video {video_id}")
        return video_id, self.rng.choice(fallback)

    def _sample_different_video_negative(self, video_id: str) -> PoseIndex:
        anchor_rec = self.records[self.video_to_idx[video_id]]
        candidates = [
            rec for rec in self.records
            if rec.video_id != video_id and rec.choreography != anchor_rec.choreography
        ]
        if not candidates:
            candidates = [rec for rec in self.records if rec.video_id != video_id]
        if not candidates:
            raise ValueError("Cannot sample different-video negative with only one loaded video")
        rec = self.rng.choice(candidates)
        return rec.video_id, self.rng.randrange(rec.num_frames)

    def sample_triplet_indices(self, idx: int) -> TripletIndices:
        anchor = self.index[int(idx)]
        video_id, t = anchor
        positive = self._sample_positive(video_id, t)
        if self.negative_strategy == "different_video":
            negative = self._sample_different_video_negative(video_id)
        else:
            negative = self._sample_far_frame_negative(video_id, t)
        return anchor, positive, negative

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor_idx, positive_idx, negative_idx = self.sample_triplet_indices(idx)
        return {
            "anchor": self._pose(*anchor_idx),
            "positive": self._pose(*positive_idx),
            "negative": self._pose(*negative_idx),
        }
