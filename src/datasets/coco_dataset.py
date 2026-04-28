"""COCO keypoints dataset for training SimpleBaseline.

Expected directory layout:
    coco/
        annotations/
            person_keypoints_train2017.json
            person_keypoints_val2017.json
        images/
            train2017/
            val2017/

Each sample returns:
    image   : (3, 256, 192)  float32 tensor, normalised with ImageNet stats
    heatmaps: (17, 64, 48)   float32 tensor, one Gaussian blob per visible joint
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_SIZE = (256, 192)   # (height, width) fed to the network
HEATMAP_SIZE = (64, 48)   # (height, width) output by DeconvHead
NUM_JOINTS = 17
SIGMA = 2                 # Gaussian radius in heatmap pixels


def _gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: int) -> np.ndarray:
    """Return a (h, w) float32 array with a Gaussian peak at (cx, cy)."""
    hm = np.zeros((h, w), dtype=np.float32)
    # Only draw if centre is inside the map
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return hm
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xs, ys = np.meshgrid(xs, ys)
    hm = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return hm.astype(np.float32)


def _make_heatmaps(
    joints: np.ndarray,       # (17, 3) — (x, y, visibility)
    hm_h: int,
    hm_w: int,
) -> np.ndarray:
    """Convert joint coords (in heatmap space) to (17, hm_h, hm_w) heatmaps."""
    heatmaps = np.zeros((NUM_JOINTS, hm_h, hm_w), dtype=np.float32)
    for j in range(NUM_JOINTS):
        if joints[j, 2] > 0:   # visible or occluded but labelled
            heatmaps[j] = _gaussian_heatmap(hm_h, hm_w, joints[j, 0], joints[j, 1], SIGMA)
    return heatmaps


class COCOPoseDataset(Dataset):
    """Top-down COCO keypoints dataset.

    Parameters
    ----------
    root : str | Path
        Path to the COCO root directory (contains 'annotations/' and 'images/').
    split : {"train", "val"}
        Which split to load.
    augment : bool
        If True, apply random horizontal flip during training.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        augment: bool = True,
    ) -> None:
        self.root    = Path(root)
        self.split   = split
        self.augment = augment and split == "train"

        ann_file = self.root / "annotations" / f"person_keypoints_{split}2017.json"
        with open(ann_file) as f:
            data = json.load(f)

        # Build id → file_name lookup
        id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

        # Keep only annotations that have keypoint labels and a valid bbox
        self.samples: list[dict] = []
        for ann in data["annotations"]:
            if ann.get("num_keypoints", 0) == 0:
                continue
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w < 2 or h < 2:
                continue
            self.samples.append({
                "image_path": str(
                    self.root / "images" / f"{split}2017" / id_to_file[ann["image_id"]]
                ),
                "bbox": [x, y, w, h],
                "keypoints": ann["keypoints"],  # flat list: [x,y,v, x,y,v, ...]
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # ── 1. Load and crop image ──────────────────────────────────────────
        img = cv2.imread(sample["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x, y, w, h = sample["bbox"]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))
        crop = img[y1:y2, x1:x2]

        crop_h, crop_w = crop.shape[:2]
        img_h, img_w   = IMAGE_SIZE

        crop_resized = cv2.resize(crop, (img_w, img_h))  # (192, 256) → (W, H)

        # ── 2. Transform keypoints into heatmap space ───────────────────────
        kps_raw = np.array(sample["keypoints"], dtype=np.float32).reshape(NUM_JOINTS, 3)
        # Shift to crop origin
        kps_raw[:, 0] -= x1
        kps_raw[:, 1] -= y1
        # Scale to heatmap size
        scale_x = HEATMAP_SIZE[1] / max(crop_w, 1)
        scale_y = HEATMAP_SIZE[0] / max(crop_h, 1)
        joints = kps_raw.copy()
        joints[:, 0] *= scale_x
        joints[:, 1] *= scale_y
        # Zero out visibility for joints outside the crop
        joints[(kps_raw[:, 2] == 0), 2] = 0

        # ── 3. Optional horizontal flip ─────────────────────────────────────
        # COCO flip pairs: (L_shoulder↔R_shoulder), etc.
        FLIP_PAIRS = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
        if self.augment and np.random.rand() < 0.5:
            crop_resized = crop_resized[:, ::-1, :].copy()
            joints[:, 0] = HEATMAP_SIZE[1] - 1 - joints[:, 0]
            for left, right in FLIP_PAIRS:
                joints[[left, right]] = joints[[right, left]]

        # ── 4. Build GT heatmaps ────────────────────────────────────────────
        heatmaps = _make_heatmaps(joints, HEATMAP_SIZE[0], HEATMAP_SIZE[1])

        # ── 5. Normalise image and convert to tensor ─────────────────────────
        image = crop_resized.astype(np.float32) / 255.0
        image = (image - _MEAN) / _STD                    # (H, W, 3)
        image = torch.from_numpy(image.transpose(2, 0, 1))  # (3, H, W)
        heatmaps = torch.from_numpy(heatmaps)               # (17, 64, 48)

        return image, heatmaps
