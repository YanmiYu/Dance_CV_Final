"""Heatmap target generation + argmax decoding."""
from __future__ import annotations

import numpy as np
import torch

from src.datasets.coco_pose_dataset import gaussian_heatmap
from src.models.decode import argmax_heatmaps


def test_gaussian_heatmap_peaks_are_at_joints():
    J = 17
    H, W = 64, 48
    joints = np.array([[10.5, 20.0]] * J, dtype=np.float32)
    vis = np.ones(J, dtype=np.float32)
    hm, w = gaussian_heatmap(joints, vis, (H, W), sigma=2.0)
    assert hm.shape == (J, H, W)
    assert w.shape == (J,)
    for j in range(J):
        y, x = np.unravel_index(np.argmax(hm[j]), hm[j].shape)
        assert abs(x - 10) <= 1 and abs(y - 20) <= 1


def test_argmax_heatmaps_returns_xy():
    J = 17
    H, W = 64, 48
    hm = np.zeros((1, J, H, W), dtype=np.float32)
    for j in range(J):
        hm[0, j, 5 + j, 3 + j] = 1.0
    coords, vals = argmax_heatmaps(torch.from_numpy(hm))
    coords = coords.numpy()
    vals = vals.numpy()
    for j in range(J):
        assert int(coords[0, j, 0]) == 3 + j
        assert int(coords[0, j, 1]) == 5 + j
        assert vals[0, j, 0] == 1.0
