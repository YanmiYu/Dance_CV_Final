"""Flip joint remapping (left/right swap) for COCO-17."""
from __future__ import annotations

import numpy as np

from src.datasets.common import COCO_17_FLIP_PAIRS, flip_keypoints


def test_flip_swaps_all_paired_joints_and_is_involutive():
    W = 640
    rng = np.random.default_rng(0)
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[:, 0] = rng.uniform(0, W, size=17)
    kps[:, 1] = rng.uniform(0, 480, size=17)
    kps[:, 2] = 1.0

    flipped = flip_keypoints(kps, image_width=W)
    for a, b in COCO_17_FLIP_PAIRS:
        assert np.allclose(flipped[a, 1:], kps[b, 1:]), f"pair {a}<->{b} y/vis not swapped"
        assert np.isclose(flipped[a, 0], (W - 1) - kps[b, 0])

    # Flipping twice returns the original.
    back = flip_keypoints(flipped, image_width=W)
    assert np.allclose(back, kps, atol=1e-3)
