"""Audit a dance clip for full-body detector crop quality.

The pretrained detector is used only for person bounding boxes. Its outputs
are never used as keypoint labels.

Example:

    python -m scripts.audit_capture \
        --video data/Inputdata/user_video3.mp4 \
        --out-dir data/processed/detector/user_video3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.infer.detector_crop import TorchVisionPersonDetector, build_detector_union_crop
from src.utils.io import ensure_dir
from src.utils.viz import draw_bbox


def _preview_sample_indices(samples: list[dict], preview_count: int) -> list[int]:
    if not samples or preview_count <= 0:
        return []
    keep = np.linspace(0, len(samples) - 1, min(preview_count, len(samples)), dtype=int)
    return [int(samples[int(i)]["frame_index"]) for i in keep]


def write_capture_audit(
    video: str | Path,
    out_dir: str | Path,
    *,
    detector=None,
    input_size=(256, 192),
    sample_stride: int = 10,
    max_samples: int = 80,
    score_threshold: float = 0.7,
    pad_ratio: float = 0.35,
    min_detection_rate: float = 0.6,
    min_edge_margin: float = 0.03,
    max_edge_contact_rate: float = 0.0,
    preview_count: int = 12,
    device: Optional[str] = None,
) -> Path:
    out_dir = ensure_dir(out_dir)
    det = detector or TorchVisionPersonDetector(score_threshold=score_threshold, device=device)
    result = build_detector_union_crop(
        video,
        det,
        input_size=input_size,
        sample_stride=sample_stride,
        max_samples=max_samples,
        pad_ratio=pad_ratio,
        min_detection_rate=min_detection_rate,
        min_edge_margin_ratio=min_edge_margin,
        max_edge_contact_rate=max_edge_contact_rate,
    )
    audit = result.to_meta()
    audit["video_path"] = str(video)
    audit["status"] = "accept" if result.accepted else "review_or_rerecord"
    audit["detector"] = {
        "type": type(det).__name__,
        "score_threshold": float(score_threshold),
    }

    audit_path = out_dir / "capture_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))

    preview_dir = ensure_dir(out_dir / "previews")
    sample_by_idx = {int(s["frame_index"]): s for s in result.samples}
    cap = cv2.VideoCapture(str(video))
    try:
        for frame_idx in _preview_sample_indices(result.samples, preview_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = draw_bbox(frame, result.bbox_xyxy, color=(0, 255, 0), thickness=2)
            sample = sample_by_idx.get(frame_idx, {})
            if sample.get("selected_bbox_xyxy") is not None:
                frame = draw_bbox(frame, sample["selected_bbox_xyxy"], color=(0, 220, 255), thickness=2)
            label = f"{audit['status']} frame={frame_idx}"
            cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(preview_dir / f"frame_{frame_idx:06d}.jpg"), frame)
    finally:
        cap.release()

    print(f"capture audit -> {audit_path}")
    print(f"preview frames -> {preview_dir}")
    return audit_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit full-body capture quality with detector bboxes.")
    p.add_argument("--video", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--input-size", nargs=2, type=int, default=[256, 192], metavar=("H", "W"))
    p.add_argument("--sample-stride", type=int, default=10)
    p.add_argument("--max-samples", type=int, default=80)
    p.add_argument("--score-threshold", type=float, default=0.7)
    p.add_argument("--pad-ratio", type=float, default=0.35)
    p.add_argument("--min-detection-rate", type=float, default=0.6)
    p.add_argument("--min-edge-margin", type=float, default=0.03)
    p.add_argument("--max-edge-contact-rate", type=float, default=0.0)
    p.add_argument("--preview-count", type=int, default=12)
    p.add_argument("--device", default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = args.out_dir or str(Path("data/processed/detector") / Path(args.video).stem)
    write_capture_audit(
        args.video,
        out_dir,
        input_size=tuple(args.input_size),
        sample_stride=args.sample_stride,
        max_samples=args.max_samples,
        score_threshold=args.score_threshold,
        pad_ratio=args.pad_ratio,
        min_detection_rate=args.min_detection_rate,
        min_edge_margin=args.min_edge_margin,
        max_edge_contact_rate=args.max_edge_contact_rate,
        preview_count=args.preview_count,
        device=args.device,
    )


if __name__ == "__main__":
    main()
