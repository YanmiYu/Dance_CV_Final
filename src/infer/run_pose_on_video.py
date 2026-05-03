"""Run pose inference on a video (single person).

Outputs (in ``--out-dir``):
  poses.npy        shape (T, 17, 3)  -- (x, y, confidence) in original-image coords
  bboxes.npy       shape (T, 4)      -- (x1, y1, x2, y2) used per frame
  meta.json        fps, size, ckpt, normalization params
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from src.datasets.coco_pose_dataset import get_affine_transform
from src.datasets.common import NUM_JOINTS, bbox_to_center_scale
from src.infer.bbox_smoother import EMABBoxSmoother
from src.infer.detector_crop import TorchVisionPersonDetector, build_detector_union_crop
from src.infer.motion_crop import MotionCropper
from src.models.decode import decode_heatmaps_to_image
from src.train.engine import build_model, _load_state_from_internal_ckpt  # noqa: F401
from src.utils.config import load_yaml
from src.utils.io import ensure_dir
from src.utils.video import ffprobe_meta


def _prep_input(frame: np.ndarray, bbox_xyxy, input_size, pixel_std: float = 200.0):
    H, W = input_size
    x1, y1, x2, y2 = bbox_xyxy
    center, scale = bbox_to_center_scale((x1, y1, x2, y2), aspect_ratio=W / H, pixel_std=pixel_std)
    M = get_affine_transform(np.asarray(center, dtype=np.float32),
                             np.asarray(scale, dtype=np.float32),
                             rot_deg=0.0,
                             output_size=(H, W),
                             pixel_std=pixel_std)
    crop = cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR)
    x = torch.from_numpy(crop.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
    return x, center, scale


def run(
    video_path: str,
    model_config_path: str,
    ckpt_path: str,
    out_dir: str,
    *,
    input_size=(256, 192),
    heatmap_size=(64, 48),
    init_bbox: Optional[tuple] = None,
    crop_mode: str = "detector_union",
    detector=None,
    detector_sample_stride: int = 10,
    detector_max_samples: int = 80,
    detector_score_threshold: float = 0.7,
    detector_pad_ratio: float = 0.35,
    detector_min_detection_rate: float = 0.6,
    detector_min_edge_margin: float = 0.03,
    detector_max_edge_contact_rate: float = 0.0,
    device: Optional[str] = None,
):
    out_dir = ensure_dir(out_dir)
    model_cfg = load_yaml(model_config_path)
    if model_cfg.get("pretrained", False):
        raise SystemExit("pretrained=true is forbidden. See docs/project_decisions.md.")

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(model_cfg).to(device_t).eval()
    _load_state_from_internal_ckpt(model, ckpt_path)

    meta = ffprobe_meta(video_path)
    crop_mode = crop_mode.lower()
    fixed_bbox: Optional[np.ndarray] = None
    cropper: Optional[MotionCropper] = None
    smoother: Optional[EMABBoxSmoother] = None
    crop_meta: dict = {"mode": crop_mode}

    if crop_mode == "detector_union":
        det = detector or TorchVisionPersonDetector(
            score_threshold=detector_score_threshold,
            device=str(device_t),
        )
        crop_result = build_detector_union_crop(
            video_path,
            det,
            input_size=input_size,
            sample_stride=detector_sample_stride,
            max_samples=detector_max_samples,
            pad_ratio=detector_pad_ratio,
            min_detection_rate=detector_min_detection_rate,
            min_edge_margin_ratio=detector_min_edge_margin,
            max_edge_contact_rate=detector_max_edge_contact_rate,
            fallback_bbox=tuple(init_bbox) if init_bbox is not None else None,
        )
        fixed_bbox = np.asarray(crop_result.bbox_xyxy, dtype=np.float32)
        crop_meta = crop_result.to_meta()
        crop_meta["detector"] = {
            "type": type(det).__name__,
            "score_threshold": float(detector_score_threshold),
        }
    elif crop_mode == "manual":
        if init_bbox is None:
            raise ValueError("--crop-mode manual requires --init-bbox x1 y1 x2 y2")
        fixed_bbox = np.asarray(init_bbox, dtype=np.float32)
        crop_meta = {
            "mode": "manual",
            "bbox_xyxy": [float(v) for v in fixed_bbox],
        }
    elif crop_mode == "motion":
        cropper = MotionCropper()
        smoother = EMABBoxSmoother(alpha=0.35)
        if init_bbox is not None:
            smoother.update(init_bbox)
        crop_meta = {
            "mode": "motion",
            "init_bbox": [float(v) for v in init_bbox] if init_bbox is not None else None,
        }
    else:
        raise ValueError(f"Unknown crop_mode: {crop_mode!r}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    poses: list[np.ndarray] = []
    bboxes: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fixed_bbox is not None:
                smoothed = fixed_bbox.copy()
            else:
                assert cropper is not None and smoother is not None
                prop = cropper.propose(frame)
                smoothed = smoother.update(prop) if prop is not None else smoother.update(None)
                if smoothed is None:
                    h, w = frame.shape[:2]
                    smoothed = np.array(MotionCropper._center_fallback(h, w), dtype=np.float32)

            x, center, scale = _prep_input(frame, smoothed, input_size)
            with torch.no_grad():
                hm = model(x.to(device_t))
            coords, vals = decode_heatmaps_to_image(
                hm,
                centers=np.asarray([center], dtype=np.float32),
                scales=np.asarray([scale], dtype=np.float32),
                input_size=input_size,
                heatmap_size=heatmap_size,
            )
            # coords: (1, 17, 2); vals: (1, 17)
            conf = vals[0].astype(np.float32)
            kps = np.concatenate([coords[0].astype(np.float32), conf[:, None]], axis=-1)
            poses.append(kps)
            bboxes.append(np.asarray(smoothed, dtype=np.float32))
    finally:
        cap.release()

    poses_arr = np.stack(poses, axis=0) if poses else np.zeros((0, NUM_JOINTS, 3), dtype=np.float32)
    bboxes_arr = np.stack(bboxes, axis=0) if bboxes else np.zeros((0, 4), dtype=np.float32)
    np.save(out_dir / "poses.npy", poses_arr)
    np.save(out_dir / "bboxes.npy", bboxes_arr)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "video_path": str(video_path),
                "ckpt": str(ckpt_path),
                "model_config": str(model_config_path),
                "input_size": list(input_size),
                "heatmap_size": list(heatmap_size),
                "fps": meta.fps,
                "num_frames": meta.num_frames,
                "width": meta.width,
                "height": meta.height,
                "crop_mode": crop_mode,
                "crop": crop_meta,
            },
            indent=2,
        )
    )
    print(f"wrote poses to {out_dir}/poses.npy shape={poses_arr.shape}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run per-frame pose inference on a single-person dance video.")
    p.add_argument("--video", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--ckpt", required=True, help="must live inside data/processed/")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--init-bbox", nargs=4, type=float, default=None, help="x1 y1 x2 y2 fallback bbox")
    p.add_argument("--input-size", nargs=2, type=int, default=[256, 192])
    p.add_argument("--heatmap-size", nargs=2, type=int, default=[64, 48])
    p.add_argument("--crop-mode", choices=["detector_union", "motion", "manual"], default="detector_union")
    p.add_argument("--detector-sample-stride", type=int, default=10)
    p.add_argument("--detector-max-samples", type=int, default=80)
    p.add_argument("--detector-score-threshold", type=float, default=0.7)
    p.add_argument("--detector-pad-ratio", type=float, default=0.35)
    p.add_argument("--detector-min-detection-rate", type=float, default=0.6)
    p.add_argument("--detector-min-edge-margin", type=float, default=0.03)
    p.add_argument("--detector-max-edge-contact-rate", type=float, default=0.0)
    p.add_argument("--device", default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run(
        args.video,
        args.model_config,
        args.ckpt,
        args.out_dir,
        input_size=tuple(args.input_size),
        heatmap_size=tuple(args.heatmap_size),
        init_bbox=tuple(args.init_bbox) if args.init_bbox else None,
        crop_mode=args.crop_mode,
        detector_sample_stride=args.detector_sample_stride,
        detector_max_samples=args.detector_max_samples,
        detector_score_threshold=args.detector_score_threshold,
        detector_pad_ratio=args.detector_pad_ratio,
        detector_min_detection_rate=args.detector_min_detection_rate,
        detector_min_edge_margin=args.detector_min_edge_margin,
        detector_max_edge_contact_rate=args.detector_max_edge_contact_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
