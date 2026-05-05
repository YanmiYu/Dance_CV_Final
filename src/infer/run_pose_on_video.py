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
from src.infer.detector_crop import build_detector_union_crop, build_person_detector
from src.infer.motion_crop import MotionCropper
from src.models.decode import decode_heatmaps_to_image
from src.train.engine import build_model, _load_state_from_internal_ckpt  # noqa: F401
from src.utils.config import load_yaml
from src.utils.io import ensure_dir
from src.utils.video import ffprobe_meta


def _size_tuple(value, fallback) -> tuple[int, int]:
    if value is None:
        return tuple(int(v) for v in fallback)
    if len(value) != 2:
        raise ValueError(f"expected a 2-item size, got {value!r}")
    return int(value[0]), int(value[1])


def _crop_to_tensor(
    crop: np.ndarray,
    *,
    color_order: str = "bgr",
    image_mean: Optional[list[float]] = None,
    image_std: Optional[list[float]] = None,
) -> torch.Tensor:
    order = str(color_order).lower()
    if order == "rgb":
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    elif order != "bgr":
        raise ValueError(f"Unsupported color_order: {color_order!r}")

    arr = crop.astype(np.float32) / 255.0
    if image_mean is not None or image_std is not None:
        mean = np.asarray(image_mean if image_mean is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        std = np.asarray(image_std if image_std is not None else [1.0, 1.0, 1.0], dtype=np.float32)
        arr = (arr - mean.reshape(1, 1, 3)) / np.clip(std.reshape(1, 1, 3), 1e-8, None)
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def _prep_input(
    frame: np.ndarray,
    bbox_xyxy,
    input_size,
    pixel_std: float = 200.0,
    *,
    crop_transform: str = "affine",
    color_order: str = "bgr",
    image_mean: Optional[list[float]] = None,
    image_std: Optional[list[float]] = None,
):
    H, W = input_size
    x1, y1, x2, y2 = bbox_xyxy
    if crop_transform == "direct_resize":
        frame_h, frame_w = frame.shape[:2]
        x1_i = int(max(0, min(frame_w - 1, round(float(x1)))))
        y1_i = int(max(0, min(frame_h - 1, round(float(y1)))))
        x2_i = int(max(x1_i + 1, min(frame_w, round(float(x2)))))
        y2_i = int(max(y1_i + 1, min(frame_h, round(float(y2)))))
        crop = frame[y1_i:y2_i, x1_i:x2_i]
        if crop.size == 0:
            x1_i, y1_i, x2_i, y2_i = 0, 0, frame_w, frame_h
            crop = frame
        crop = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
        x = _crop_to_tensor(crop, color_order=color_order, image_mean=image_mean, image_std=image_std)
        return x, {"mode": "direct_resize", "bbox": np.array([x1_i, y1_i, x2_i, y2_i], dtype=np.float32)}

    if crop_transform != "affine":
        raise ValueError(f"Unknown crop_transform: {crop_transform!r}")

    center, scale = bbox_to_center_scale((x1, y1, x2, y2), aspect_ratio=W / H, pixel_std=pixel_std)
    M = get_affine_transform(np.asarray(center, dtype=np.float32),
                             np.asarray(scale, dtype=np.float32),
                             rot_deg=0.0,
                             output_size=(H, W),
                             pixel_std=pixel_std)
    crop = cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR)
    x = _crop_to_tensor(crop, color_order=color_order, image_mean=image_mean, image_std=image_std)
    return x, {"mode": "affine", "center": center, "scale": scale}


def _decode_heatmaps_to_bbox(heatmaps: torch.Tensor, bbox_xyxy, heatmap_size) -> np.ndarray:
    """Decode heatmap argmaxes using the branch SimpleBaseline crop convention."""
    hm = heatmaps.detach().cpu().numpy()
    if hm.ndim == 4:
        hm = hm[0]
    num_joints, hm_h, hm_w = hm.shape
    expected_h, expected_w = _size_tuple(heatmap_size, (hm_h, hm_w))
    if (hm_h, hm_w) != (expected_h, expected_w):
        raise ValueError(
            f"heatmap output {(hm_h, hm_w)} does not match configured {(expected_h, expected_w)}"
        )
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    crop_w = max(1.0, x2 - x1)
    crop_h = max(1.0, y2 - y1)

    flat = hm.reshape(num_joints, -1)
    idx = flat.argmax(axis=1)
    peak_vals = flat[np.arange(num_joints), idx].astype(np.float32)
    max_peak = float(peak_vals.max()) if float(peak_vals.max()) > 0.0 else 1.0

    kps = np.zeros((num_joints, 3), dtype=np.float32)
    for j in range(num_joints):
        hm_x = float(idx[j] % hm_w)
        hm_y = float(idx[j] // hm_w)
        kps[j, 0] = (hm_x / float(hm_w)) * crop_w + x1
        kps[j, 1] = (hm_y / float(hm_h)) * crop_h + y1
        kps[j, 2] = float(np.clip(peak_vals[j] / max_peak, 0.0, 1.0))
    return kps


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


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
    detector_backend: str = "torchvision",
    detector_model: Optional[str] = None,
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
    model_inference_cfg = model_cfg.get("inference", {}) or {}
    input_size = _size_tuple(model_inference_cfg.get("input_size"), input_size)
    heatmap_size = _size_tuple(model_inference_cfg.get("heatmap_size"), heatmap_size)
    crop_transform = str(model_inference_cfg.get("crop_transform", "affine")).lower()
    color_order = str(model_inference_cfg.get("color_order", "bgr")).lower()
    image_mean = model_inference_cfg.get("image_mean")
    image_std = model_inference_cfg.get("image_std")

    device_t = _resolve_device(device)
    model = build_model(model_cfg).to(device_t).eval()
    _load_state_from_internal_ckpt(model, ckpt_path)

    meta = ffprobe_meta(video_path)
    crop_mode = crop_mode.lower()
    fixed_bbox: Optional[np.ndarray] = None
    cropper: Optional[MotionCropper] = None
    smoother: Optional[EMABBoxSmoother] = None
    crop_meta: dict = {"mode": crop_mode}

    if crop_mode == "detector_union":
        det = detector or build_person_detector(
            detector_backend,
            score_threshold=detector_score_threshold,
            device=str(device_t),
            model_name=detector_model,
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
            "backend": detector_backend,
            "model": getattr(det, "model_name", detector_model),
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

            x, decode_info = _prep_input(
                frame,
                smoothed,
                input_size,
                crop_transform=crop_transform,
                color_order=color_order,
                image_mean=image_mean,
                image_std=image_std,
            )
            with torch.no_grad():
                hm = model(x.to(device_t))
            if decode_info["mode"] == "direct_resize":
                kps = _decode_heatmaps_to_bbox(hm, decode_info["bbox"], heatmap_size)
            else:
                coords, vals = decode_heatmaps_to_image(
                    hm,
                    centers=np.asarray([decode_info["center"]], dtype=np.float32),
                    scales=np.asarray([decode_info["scale"]], dtype=np.float32),
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
                "model_inference": {
                    "crop_transform": crop_transform,
                    "color_order": color_order,
                    "image_mean": image_mean,
                    "image_std": image_std,
                },
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
    p.add_argument("--detector-backend", choices=["torchvision", "yolov8"], default="torchvision")
    p.add_argument("--detector-model", default=None, help="YOLOv8 model path/name, e.g. yolov8n.pt")
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
        detector_backend=args.detector_backend,
        detector_model=args.detector_model,
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
