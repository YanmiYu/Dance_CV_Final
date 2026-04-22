"""Heatmap decoding: argmax with quarter-pixel refinement, and inverse affine."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def argmax_heatmaps(heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (coords (B, J, 2), maxvals (B, J, 1)) in heatmap coords."""
    assert heatmaps.dim() == 4, "expect (B, J, H, W)"
    B, J, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, J, -1)
    maxvals, idx = flat.max(dim=-1)
    coords = torch.stack([idx % W, idx // W], dim=-1).float()   # (B, J, 2) [x, y]
    return coords, maxvals.unsqueeze(-1)


def refine_coords_quarter_pixel(
    heatmaps: torch.Tensor, coords: torch.Tensor
) -> torch.Tensor:
    """Quarter-pixel refinement using the sign of the gradient around the peak."""
    B, J, H, W = heatmaps.shape
    coords = coords.clone()
    hm = heatmaps.detach().cpu().numpy()
    c = coords.detach().cpu().numpy()
    for b in range(B):
        for j in range(J):
            x, y = int(c[b, j, 0]), int(c[b, j, 1])
            if 1 < x < W - 1 and 1 < y < H - 1:
                dx = 0.25 * np.sign(hm[b, j, y, x + 1] - hm[b, j, y, x - 1])
                dy = 0.25 * np.sign(hm[b, j, y + 1, x] - hm[b, j, y - 1, x])
                c[b, j, 0] += dx
                c[b, j, 1] += dy
    return torch.from_numpy(c).to(coords.device)


def decode_heatmaps_to_image(
    heatmaps: torch.Tensor,
    centers: np.ndarray,     # (B, 2) original-image coords
    scales: np.ndarray,      # (B, 2)
    input_size: Tuple[int, int],    # (H, W)
    heatmap_size: Tuple[int, int],  # (H, W)
    pixel_std: float = 200.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode heatmaps to original image coordinates.

    Returns ``(coords (B, J, 2), maxvals (B, J))`` in pixel space.
    """
    from src.datasets.coco_pose_dataset import get_affine_transform

    coords, maxvals = argmax_heatmaps(heatmaps)
    coords = refine_coords_quarter_pixel(heatmaps, coords)

    B, J, _ = coords.shape
    coords = coords.detach().cpu().numpy()
    maxvals = maxvals.detach().cpu().numpy().squeeze(-1)

    Hh, Hw = heatmap_size
    H, W = input_size
    fx, fy = W / Hw, H / Hh

    out = np.zeros_like(coords)
    for b in range(B):
        M_inv = get_affine_transform(
            np.asarray(centers[b], dtype=np.float32),
            np.asarray(scales[b], dtype=np.float32),
            rot_deg=0.0,
            output_size=(H, W),
            pixel_std=pixel_std,
            inv=True,
        )
        for j in range(J):
            x_in = coords[b, j, 0] * fx
            y_in = coords[b, j, 1] * fy
            p = np.array([x_in, y_in, 1.0], dtype=np.float32)
            out[b, j] = (M_inv @ p)[:2]
    return out, maxvals
