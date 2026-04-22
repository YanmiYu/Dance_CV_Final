"""Affine transform + inverse transform round-trip (geometry module)."""
from __future__ import annotations

import numpy as np

from src.datasets.coco_pose_dataset import affine_transform_point, get_affine_transform


def test_affine_forward_inverse_round_trip():
    center = np.array([320.0, 240.0], dtype=np.float32)
    scale = np.array([2.0, 2.5], dtype=np.float32)
    input_size = (256, 192)
    M = get_affine_transform(center, scale, rot_deg=15.0, output_size=input_size)
    Minv = get_affine_transform(center, scale, rot_deg=15.0, output_size=input_size, inv=True)

    for pt in [(0.0, 0.0), (100.0, 100.0), (200.0, 50.0), (center[0], center[1])]:
        pt = np.array(pt, dtype=np.float32)
        p2 = affine_transform_point(pt, M)
        p3 = affine_transform_point(p2, Minv)
        assert np.allclose(pt, p3, atol=1e-3), f"round trip failed for {pt}: got {p3}"


def test_center_maps_to_output_center():
    center = np.array([320.0, 240.0], dtype=np.float32)
    scale = np.array([2.0, 2.5], dtype=np.float32)
    H, W = (256, 192)
    M = get_affine_transform(center, scale, rot_deg=0.0, output_size=(H, W))
    out = affine_transform_point(center, M)
    assert np.allclose(out, np.array([W / 2, H / 2], dtype=np.float32), atol=1e-3)
