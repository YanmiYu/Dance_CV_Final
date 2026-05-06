"""Balanced batch sampler for supervised contrastive training."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


class DanceLabelBalancedBatchSampler(Sampler[List[int]]):
    """Sample N dance labels and K samples per label in every batch."""

    def __init__(
        self,
        labels: Sequence[int],
        *,
        n_labels_per_batch: int,
        n_samples_per_label: int,
        num_batches: int | None = None,
        seed: int = 0,
        drop_singletons: bool = True,
    ) -> None:
        labels = [int(label) for label in labels]
        self.n_labels_per_batch = int(n_labels_per_batch)
        self.n_samples_per_label = int(n_samples_per_label)
        if self.n_labels_per_batch <= 0 or self.n_samples_per_label <= 0:
            raise ValueError("n_labels_per_batch and n_samples_per_label must be > 0")

        pool: dict[int, List[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            pool[label].append(i)
        if drop_singletons:
            pool = {label: idxs for label, idxs in pool.items() if len(idxs) >= 2}
        if len(pool) < self.n_labels_per_batch:
            raise ValueError(
                f"need at least {self.n_labels_per_batch} labels with "
                f"{2 if drop_singletons else 1}+ samples; got {len(pool)}"
            )

        self._pool = pool
        self._unique_labels = sorted(pool.keys())
        if num_batches is None:
            usable = sum(len(idxs) for idxs in pool.values())
            per_batch = max(1, self.n_labels_per_batch * self.n_samples_per_label)
            num_batches = max(1, usable // per_batch)
        self.num_batches = int(num_batches)
        self._rng = np.random.default_rng(int(seed))

    @property
    def batch_size(self) -> int:
        return self.n_labels_per_batch * self.n_samples_per_label

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        labels = np.array(self._unique_labels)
        for _ in range(self.num_batches):
            chosen = self._rng.choice(labels, size=self.n_labels_per_batch, replace=False)
            batch: List[int] = []
            for label in chosen:
                pool = self._pool[int(label)]
                replace = len(pool) < self.n_samples_per_label
                idxs = self._rng.choice(pool, size=self.n_samples_per_label, replace=replace)
                batch.extend(int(i) for i in idxs)
            yield batch
