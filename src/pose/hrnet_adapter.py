"""HRNet pose adapter — wraps src.infer.run_pose_on_video.run() so the
integrate pipeline can call a uniform PoseRunResult-returning interface.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from src.infer.run_pose_on_video import run as _run_pose_on_video
from src.pose.base import PoseRunResult


def run_video(
    video_path: str,
    *,
    model_config: str = "configs/model/hrnet_w32.yaml",
    checkpoint: str = "data/processed/train_hrnet_w32/best.pt",
    device: Optional[str] = None,
    name: str = "hrnet_w32",
    out_dir: Optional[str] = None,
    inference: Optional[dict] = None,
) -> PoseRunResult:
    """Run HRNet pose inference on a single video and return a PoseRunResult.

    `out_dir` is a working directory where poses.npy / bboxes.npy / meta.json
    are written. If None, a temp directory is used and removed afterwards.
    """
    keep_dir = out_dir is not None
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="pose_hrnet_")

    try:
        inf = inference or {}
        _run_pose_on_video(
            video_path=video_path,
            model_config_path=model_config,
            ckpt_path=checkpoint,
            out_dir=out_dir,
            device=device,
            crop_mode=inf.get("crop_mode", "detector_union"),
            detector_backend=inf.get("detector_backend", "torchvision"),
            detector_model=inf.get("detector_model"),
            detector_sample_stride=int(inf.get("detector_sample_stride", 10)),
            detector_max_samples=int(inf.get("detector_max_samples", 80)),
            detector_score_threshold=float(inf.get("detector_score_threshold", 0.7)),
            detector_pad_ratio=float(inf.get("detector_pad_ratio", 0.35)),
            detector_min_detection_rate=float(inf.get("detector_min_detection_rate", 0.6)),
            detector_min_edge_margin=float(inf.get("detector_min_edge_margin", 0.03)),
            detector_max_edge_contact_rate=float(inf.get("detector_max_edge_contact_rate", 0.0)),
        )
        out = Path(out_dir)
        poses = np.load(out / "poses.npy")
        bboxes = np.load(out / "bboxes.npy")
        meta = json.loads((out / "meta.json").read_text())
    finally:
        if not keep_dir:
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)

    return PoseRunResult(
        name=name,
        poses=poses,
        embeddings=None,
        bboxes=bboxes,
        fps=float(meta.get("fps") or 30.0),
        num_frames=int(meta.get("num_frames") or len(poses)),
        width=int(meta.get("width") or 0),
        height=int(meta.get("height") or 0),
        extra={"meta": meta, "checkpoint": checkpoint, "model_config": model_config},
    )
