"""Smoke tests for the single-person detector pipeline.

Covers:
  * model forward shape
  * pretrained=True is refused (docs/project_decisions.md section 1)
  * DetectorDataset target construction is geometrically correct
  * a handful of SGD steps on a tiny synthetic dataset reduce the loss
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _tiny_model_cfg():
    """Small config that preserves the real /4 output stride.

    Four stages with strides [1, 2, 2, 2] (total /8) plus three deconvs
    (total *8) cancel against the /4 stem, giving an input->output stride
    of 4 -- identical to the production config, just narrower channels.
    """
    return {
        "name": "person_detector",
        "pretrained": False,
        "backbone": {"stem_channels": 16, "stages": [
            {"channels": 16, "blocks": 1, "stride": 1},
            {"channels": 32, "blocks": 1, "stride": 2},
            {"channels": 32, "blocks": 1, "stride": 2},
            {"channels": 32, "blocks": 1, "stride": 2},
        ]},
        "head": {"deconv_channels": [32, 32, 32], "deconv_kernels": [4, 4, 4]},
    }


def test_person_detector_forward_shape():
    from src.models.person_detector import PersonDetector
    model = PersonDetector(_tiny_model_cfg()).eval()
    assert model.output_stride == 4
    with torch.no_grad():
        y = model(torch.randn(2, 3, 128, 128))
    assert y.shape == (2, 3, 32, 32)


def test_pretrained_true_refused():
    from src.models.person_detector import PersonDetector
    cfg = _tiny_model_cfg()
    cfg["pretrained"] = True
    with pytest.raises(AssertionError):
        PersonDetector(cfg)


def _write_syn_jsonl(tmp_path: Path, n: int = 8) -> Path:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    jsonl = tmp_path / "train.jsonl"
    rng = np.random.default_rng(0)
    with jsonl.open("w") as f:
        for i in range(n):
            img = (rng.uniform(0, 255, size=(320, 320, 3))).astype(np.uint8)
            img_path = images_dir / f"img_{i}.jpg"
            cv2.imwrite(str(img_path), img)
            cx, cy = 160.0, 160.0
            bw, bh = 120.0, 200.0
            rec = {
                "image_path": str(img_path),
                "image_id": f"syn_{i}",
                "dataset_name": "synth",
                "bbox_xyxy": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                "keypoints_xyv": [[0.0, 0.0, 0.0]] * 17,
                "center": [cx, cy],
                "scale": [1.0, 1.0],
                "meta": {},
            }
            f.write(json.dumps(rec))
            f.write("\n")
    return jsonl


def test_dataset_targets_are_geometrically_correct(tmp_path: Path):
    from src.datasets.detector_dataset import DetectorDataset, DetectorAugConfig

    jsonl = _write_syn_jsonl(tmp_path, n=2)
    ds = DetectorDataset(
        jsonl,
        input_size=(128, 128),
        output_stride=4,
        is_train=False,
        aug=DetectorAugConfig(),
    )
    s = ds[0]
    assert s["image"].shape == (3, 128, 128)
    assert s["center_heatmap"].shape == (1, 32, 32)
    assert s["size_target"].shape == (2, 32, 32)
    # GT bbox in a 320x320 frame resized to 128x128: center at (64, 64).
    # Cell in the 32x32 heatmap -> (16, 16). Expect peak == 1 there.
    peak_yx = np.unravel_index(s["center_heatmap"][0].argmax(), s["center_heatmap"][0].shape)
    assert abs(int(peak_yx[0]) - 16) <= 1
    assert abs(int(peak_yx[1]) - 16) <= 1
    assert pytest.approx(s["center_heatmap"][0].max(), rel=1e-5) == 1.0
    assert s["size_mask"].sum() == 1.0
    # Size target at the center cell ~ (bw/128, bh/128) = (120/320 * 128 /128, ...) ->
    # after resizing the bbox is (120 * 128/320, 200 * 128/320) = (48, 80); normalized: (0.375, 0.625).
    iy, ix = np.unravel_index(s["size_mask"][0].argmax(), s["size_mask"][0].shape)
    assert abs(float(s["size_target"][0, iy, ix]) - 0.375) < 0.05
    assert abs(float(s["size_target"][1, iy, ix]) - 0.625) < 0.05


def _make_synthetic_frame_with_person(size: int = 320, bg_color: int = 230, person_color: int = 40) -> tuple:
    """Build a synthetic frame with a clearly-foreground rectangle on a clean BG.

    Returns (image_uint8, bbox_xyxy). Mirrors the AIST setup: uniform
    light background + dark "person" -- ideal for the color-silhouette
    extractor to work on without needing real AIST frames in tests.
    """
    img = np.full((size, size, 3), bg_color, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    bw, bh = size // 3, int(size * 0.6)
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = cx + bw // 2, cy + bh // 2
    img[y1:y2, x1:x2] = person_color
    return img, (float(x1), float(y1), float(x2), float(y2))


def _write_clean_bg_jsonl(tmp_path: Path, n: int = 4) -> Path:
    """Same shape as _write_syn_jsonl but uses a clean background so the
    color silhouette has something meaningful to extract."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)
    jsonl = tmp_path / "train.jsonl"
    with jsonl.open("w") as f:
        for i in range(n):
            img, bbox = _make_synthetic_frame_with_person(size=320)
            img_path = images_dir / f"img_{i}.jpg"
            cv2.imwrite(str(img_path), img)
            rec = {
                "image_path": str(img_path),
                "image_id": f"syn_{i}",
                "dataset_name": "synth",
                "bbox_xyxy": list(bbox),
                "keypoints_xyv": [[0.0, 0.0, 0.0]] * 17,
                "center": [160.0, 160.0],
                "scale": [1.0, 1.0],
                "meta": {},
            }
            f.write(json.dumps(rec))
            f.write("\n")
    return jsonl


def test_external_background_library_is_used_when_present(tmp_path: Path):
    """When external_bg_dir exists, _make_bg('frame') must return one of those textures."""
    from src.datasets.detector_dataset import DetectorDataset, DetectorAugConfig

    bg_dir = tmp_path / "bgs"
    bg_dir.mkdir()
    sentinel_color = (200, 100, 50)
    for i in range(3):
        img = np.full((128, 128, 3), sentinel_color, dtype=np.uint8)
        cv2.imwrite(str(bg_dir / f"bg_{i}.jpg"), img)

    jsonl = _write_clean_bg_jsonl(tmp_path, n=2)
    aug = DetectorAugConfig(external_bg_dir=str(bg_dir))
    ds = DetectorDataset(jsonl, input_size=(128, 128), output_stride=4, is_train=True, aug=aug)
    assert len(ds._external_bgs) == 3
    bg = ds._make_bg("frame")
    assert bg.shape == (128, 128, 3)
    # JPEG round-trip is lossy; allow a small tolerance and use median to
    # ignore any single-pixel artifacts.
    median = np.median(bg.reshape(-1, 3), axis=0)
    assert np.linalg.norm(median - np.asarray(sentinel_color, dtype=np.float32)) < 20


def test_external_background_dir_missing_falls_back_silently(tmp_path: Path):
    from src.datasets.detector_dataset import DetectorDataset, DetectorAugConfig

    jsonl = _write_clean_bg_jsonl(tmp_path, n=2)
    aug = DetectorAugConfig(external_bg_dir=str(tmp_path / "does_not_exist"))
    ds = DetectorDataset(jsonl, input_size=(128, 128), output_stride=4, is_train=True, aug=aug)
    assert ds._external_bgs == []
    bg = ds._make_bg("frame")
    assert bg.shape == (128, 128, 3)


def test_color_silhouette_finds_dark_person_on_clean_bg():
    from src.datasets.detector_dataset import DetectorDataset, DetectorAugConfig

    img, bbox = _make_synthetic_frame_with_person(size=256)
    aug = DetectorAugConfig(silhouette_method="color", silhouette_thresh=28.0)
    ds = DetectorDataset.__new__(DetectorDataset)
    ds.aug = aug
    ds.input_size = (256, 256)
    mask = ds._silhouette_color(img, np.asarray(bbox, dtype=np.float32))
    assert mask is not None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    inside_ratio = mask[y1:y2, x1:x2].mean()
    outside_mask = mask.copy()
    outside_mask[y1:y2, x1:x2] = 0
    outside_ratio = outside_mask.sum() / max(1, H * W - (x2 - x1) * (y2 - y1))
    assert inside_ratio > 0.8, f"silhouette did not cover the person: {inside_ratio:.3f}"
    assert outside_ratio < 0.01, f"silhouette leaked outside bbox: {outside_ratio:.3f}"


def test_silhouette_composite_keeps_person_replaces_background(tmp_path: Path):
    """End-to-end check: the dancer pixels should survive the paste, the
    background pixels should change."""
    from src.datasets.detector_dataset import DetectorDataset, DetectorAugConfig

    jsonl = _write_clean_bg_jsonl(tmp_path, n=1)
    bg_dir = tmp_path / "bgs"
    bg_dir.mkdir()
    bg_color = (10, 10, 200)
    cv2.imwrite(str(bg_dir / "red.jpg"), np.full((256, 256, 3), bg_color, dtype=np.uint8))

    aug = DetectorAugConfig(
        # Force the noisy-bg path with the external "frame" mode every time.
        noisy_bg_prob=1.0,
        noisy_bg_weights=(0.0, 0.0, 1.0),
        external_bg_dir=str(bg_dir),
        silhouette_method="color",
        silhouette_thresh=28.0,
        feather_ratio=0.0,
        # Disable everything else so the test is deterministic.
        color_jitter_brightness=0.0, color_jitter_contrast=0.0,
        gaussian_blur_prob=0.0, motion_blur_prob=0.0, jpeg_prob=0.0,
        horizontal_flip_prob=0.0, scale_jitter=(1.0, 1.0),
        translation_jitter_ratio=0.0,
    )
    ds = DetectorDataset(jsonl, input_size=(256, 256), output_stride=4, is_train=True, aug=aug)
    sample = ds[0]
    img = (sample["image"].transpose(1, 2, 0) * 255.0).astype(np.uint8)
    bbox = sample["bbox_xyxy_input"]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Sample bg pixels well outside the person -- they should now look like the red bg.
    corner = img[5:25, 5:25].mean(axis=(0, 1))
    assert np.linalg.norm(corner - np.asarray(bg_color, dtype=np.float32)) < 25, (
        f"outside-bbox region was not replaced by the external bg: corner={corner}"
    )
    # Sample person pixels (interior of the bbox, well inside the silhouette) -- should still be dark.
    cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
    interior = img[cy - 5 : cy + 5, cx - 5 : cx + 5].mean(axis=(0, 1))
    assert interior.mean() < 80, f"person region was overwritten by the bg: interior={interior}"


def test_detector_loss_decreases_on_synthetic_batch(tmp_path: Path):
    from src.datasets.detector_dataset import DetectorDataset, DetectorAugConfig
    from src.models.person_detector import PersonDetector
    from src.train.engine import _TorchDatasetAdapter, _collate
    from src.train.train_detector import DetectorLoss

    jsonl = _write_syn_jsonl(tmp_path, n=4)
    aug = DetectorAugConfig(
        color_jitter_brightness=0, color_jitter_contrast=0,
        gaussian_blur_prob=0, motion_blur_prob=0, jpeg_prob=0,
        horizontal_flip_prob=0, scale_jitter=(1.0, 1.0), translation_jitter_ratio=0,
        noisy_bg_prob=0,
    )
    ds = DetectorDataset(jsonl, input_size=(128, 128), output_stride=4, is_train=True, aug=aug)
    from torch.utils.data import DataLoader
    loader = DataLoader(_TorchDatasetAdapter(ds), batch_size=4, shuffle=False, collate_fn=_collate)

    model = PersonDetector(_tiny_model_cfg())
    loss_fn = DetectorLoss(center_weight=1.0, size_weight=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = next(iter(loader))
    losses = []
    for _ in range(5):
        preds = model(batch["image"])
        out = loss_fn(preds, batch["center_heatmap"], batch["size_target"], batch["size_mask"])
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(float(out["loss"].item()))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
