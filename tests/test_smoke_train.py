"""Smoke test A: tiny labeled image batch -> loss decreases over a few steps.

We synthesize a minimal JSONL dataset in a tmp dir, then run 5 SGD steps on
the Simple Baseline and assert the loss goes down.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

torch_installed = True
try:
    import torch.nn as nn  # noqa: F401
except Exception:
    torch_installed = False


@pytest.mark.skipif(not torch_installed, reason="torch not installed")
def test_simple_baseline_loss_decreases_on_synthetic_batch(tmp_path: Path) -> None:
    from src.datasets.coco_pose_dataset import PoseJsonlDataset, PoseAugConfig
    from src.models.losses import JointMSELoss
    from src.models.simple_baseline import SimpleBaselinePose
    from src.train.engine import _TorchDatasetAdapter, _collate

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    jsonl = tmp_path / "train.jsonl"

    rng = np.random.default_rng(0)
    with jsonl.open("w") as f:
        for i in range(8):
            img = (rng.uniform(0, 255, size=(256, 192, 3))).astype(np.uint8)
            img_path = images_dir / f"img_{i}.jpg"
            cv2.imwrite(str(img_path), img)
            kps = []
            for j in range(17):
                x = float(rng.uniform(30, 160))
                y = float(rng.uniform(30, 220))
                kps.append([x, y, 2.0])
            rec = {
                "image_path": str(img_path),
                "image_id": f"syn_{i}",
                "dataset_name": "synth",
                "bbox_xyxy": [10, 10, 180, 240],
                "keypoints_xyv": kps,
                "center": [95.0, 125.0],
                "scale": [1.0, 1.3],
                "meta": {},
            }
            f.write(json.dumps(rec))
            f.write("\n")

    ds = PoseJsonlDataset(
        jsonl,
        input_size=(128, 96),
        heatmap_size=(64, 48),
        sigma=2.0,
        is_train=False,
        aug=PoseAugConfig(),
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(_TorchDatasetAdapter(ds), batch_size=4, shuffle=False, collate_fn=_collate)

    model_cfg = {
        "name": "simple_baseline",
        "pretrained": False,
        "backbone": {"stem_channels": 16, "stages": [
            {"channels": 16, "blocks": 1, "stride": 1},
            {"channels": 32, "blocks": 1, "stride": 2},
        ]},
        "head": {"num_deconv_layers": 2, "deconv_channels": [32, 32], "deconv_kernels": [4, 4], "final_kernel": 1, "num_joints": 17},
    }
    model = SimpleBaselinePose(model_cfg)
    loss_fn = JointMSELoss(use_target_weight=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    it = iter(loader)
    batch = next(it)
    for step in range(5):
        preds = model(batch["image"])
        loss = loss_fn(preds, batch["heatmaps"], batch["target_weight"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
