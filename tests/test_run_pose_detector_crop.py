from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.infer.detector_crop import PersonDetection

torch = pytest.importorskip("torch")


class _FakeDetector:
    def detect(self, frame_bgr: np.ndarray):
        return [PersonDetection((12, 10, 84, 120), 0.95)]


def _write_tiny_video(path: Path, *, size=(96, 128), n_frames: int = 3) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    assert writer.isOpened()
    try:
        for i in range(n_frames):
            frame = np.full((size[1], size[0], 3), 40 + i * 20, dtype=np.uint8)
            cv2.rectangle(frame, (20, 20), (70, 110), (220, 220, 220), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_run_pose_detector_union_writes_same_outputs_and_metadata(tmp_path: Path) -> None:
    from src.infer.run_pose_on_video import run
    from src.models.simple_baseline import SimpleBaselinePose

    model_cfg = {
        "name": "simple_baseline",
        "pretrained": False,
        "backbone": {
            "stem_channels": 8,
            "stages": [
                {"channels": 8, "blocks": 1, "stride": 1},
                {"channels": 16, "blocks": 1, "stride": 2},
            ],
        },
        "head": {
            "num_deconv_layers": 2,
            "deconv_channels": [16, 16],
            "deconv_kernels": [4, 4],
            "final_kernel": 1,
            "num_joints": 17,
        },
    }
    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text(json.dumps(model_cfg))

    ckpt_dir = tmp_path / "data/processed/test_detector_crop"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "best.pt"
    model = SimpleBaselinePose(model_cfg)
    torch.save({"model": model.state_dict()}, ckpt_path)

    video = tmp_path / "clip.mp4"
    _write_tiny_video(video)
    out_dir = tmp_path / "pred"

    run(
        str(video),
        str(cfg_path),
        str(ckpt_path),
        str(out_dir),
        input_size=(128, 96),
        heatmap_size=(64, 48),
        crop_mode="detector_union",
        detector=_FakeDetector(),
        detector_sample_stride=1,
        detector_max_samples=2,
        device="cpu",
    )

    poses = np.load(out_dir / "poses.npy")
    bboxes = np.load(out_dir / "bboxes.npy")
    meta = json.loads((out_dir / "meta.json").read_text())

    assert poses.shape == (3, 17, 3)
    assert bboxes.shape == (3, 4)
    assert np.allclose(bboxes[0], bboxes[1])
    assert meta["crop_mode"] == "detector_union"
    assert meta["crop"]["detected_count"] == 2
    assert meta["crop"]["detector"]["type"] == "_FakeDetector"
