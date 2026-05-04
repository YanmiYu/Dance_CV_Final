"""Detector-backed single-person crop proposals.

The pretrained detector in this module is used only to find person bounding
boxes for capture QA and inference crops. It must never provide keypoint
labels or initialize pose/keypoint model weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class PersonDetection:
    bbox_xyxy: BBox
    score: float


@dataclass
class DetectorUnionResult:
    bbox_xyxy: BBox
    raw_union_bbox_xyxy: Optional[BBox]
    sample_count: int
    detected_count: int
    detection_rate: float
    edge_contact_rate: float
    union_edge_margins: Optional[dict]
    accepted: bool
    fallback_reason: Optional[str]
    config: dict
    samples: List[dict]

    def to_meta(self) -> dict:
        out = asdict(self)
        out["mode"] = "detector_union"
        return out


class PersonDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> Sequence[PersonDetection]:
        ...


def _configure_detector_weight_download_env() -> None:
    """Make first-run torchvision detector downloads work from this repo.

    Torch defaults to ``~/.cache/torch``. In sandboxed/local project runs that
    path may be unwritable, so default to an ignored cache under data/processed.
    Python framework installs on macOS can also have an empty OpenSSL trust
    path; certifi gives urllib a usable CA bundle for the one-time download.
    """
    if not os.environ.get("TORCH_HOME"):
        repo_root = Path(__file__).resolve().parents[2]
        torch_home = repo_root / "data" / "processed" / "torch_cache"
        try:
            torch_home.mkdir(parents=True, exist_ok=True)
            os.environ["TORCH_HOME"] = str(torch_home)
        except OSError:
            pass

    if not os.environ.get("SSL_CERT_FILE"):
        try:
            import certifi

            os.environ["SSL_CERT_FILE"] = certifi.where()
        except Exception:
            pass


def clamp_bbox(bbox: Iterable[float], width: int, height: int) -> BBox:
    """Clamp an xyxy bbox to image bounds while preserving at least 1px size."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = min(max(x1, 0.0), float(max(width - 1, 0)))
    y1 = min(max(y1, 0.0), float(max(height - 1, 0)))
    x2 = min(max(x2, x1 + 1.0), float(max(width, 1)))
    y2 = min(max(y2, y1 + 1.0), float(max(height, 1)))
    return (x1, y1, x2, y2)


def bbox_area(bbox: Iterable[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = bbox_area((ix1, iy1, ix2, iy2))
    union = bbox_area(a) + bbox_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def union_bboxes(bboxes: Sequence[Iterable[float]]) -> BBox:
    if not bboxes:
        raise ValueError("union_bboxes requires at least one bbox")
    arr = np.asarray(list(bboxes), dtype=np.float32)
    return (
        float(arr[:, 0].min()),
        float(arr[:, 1].min()),
        float(arr[:, 2].max()),
        float(arr[:, 3].max()),
    )


def expand_bbox(
    bbox: Iterable[float],
    *,
    pad_ratio: float,
    width: int,
    height: int,
    aspect_ratio: Optional[float] = None,
) -> BBox:
    """Pad a bbox and optionally expand it to match ``width / height``."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    if aspect_ratio and aspect_ratio > 0:
        cur = bw / bh
        if cur > aspect_ratio:
            bh = bw / aspect_ratio
        else:
            bw = bh * aspect_ratio
    scale = 1.0 + 2.0 * max(0.0, float(pad_ratio))
    bw *= scale
    bh *= scale
    return clamp_bbox((cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0), width, height)


def edge_margin_ratios(bbox: Iterable[float], width: int, height: int) -> dict:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    vals = {
        "left": x1 / w,
        "top": y1 / h,
        "right": (w - x2) / w,
        "bottom": (h - y2) / h,
    }
    vals["min"] = min(vals.values())
    return vals


def select_primary_detection(
    detections: Sequence[PersonDetection],
    *,
    previous_bbox: Optional[BBox] = None,
) -> Optional[PersonDetection]:
    """Pick the most likely subject for this single-person pipeline."""
    if not detections:
        return None
    if previous_bbox is None:
        return max(detections, key=lambda d: float(d.score) * max(bbox_area(d.bbox_xyxy), 1.0))
    return max(
        detections,
        key=lambda d: (2.0 * bbox_iou(previous_bbox, d.bbox_xyxy)) + float(d.score),
    )


def sample_frame_indices(total_frames: int, *, sample_stride: int, max_samples: int) -> List[int]:
    stride = max(1, int(sample_stride))
    if total_frames <= 0:
        return []
    indices = list(range(0, int(total_frames), stride))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    if max_samples > 0 and len(indices) > max_samples:
        keep = np.linspace(0, len(indices) - 1, int(max_samples), dtype=int)
        indices = [indices[int(i)] for i in keep]
    return sorted(set(indices))


def build_union_crop_from_detections(
    detections_by_frame: Sequence[Tuple[int, Sequence[PersonDetection]]],
    *,
    frame_width: int,
    frame_height: int,
    aspect_ratio: float,
    pad_ratio: float = 0.35,
    min_detection_rate: float = 0.6,
    min_edge_margin_ratio: float = 0.03,
    max_edge_contact_rate: float = 0.0,
    fallback_bbox: Optional[BBox] = None,
) -> DetectorUnionResult:
    previous: Optional[BBox] = None
    selected_boxes: List[BBox] = []
    edge_contacts = 0
    samples: List[dict] = []

    for frame_index, detections in detections_by_frame:
        selected = select_primary_detection(detections, previous_bbox=previous)
        sample = {"frame_index": int(frame_index), "num_detections": int(len(detections))}
        if selected is not None:
            previous = clamp_bbox(selected.bbox_xyxy, frame_width, frame_height)
            selected_boxes.append(previous)
            margins = edge_margin_ratios(previous, frame_width, frame_height)
            edge_contacts += int(float(margins["min"]) < float(min_edge_margin_ratio))
            sample.update(
                {
                    "selected_bbox_xyxy": [float(v) for v in previous],
                    "selected_score": float(selected.score),
                    "edge_margins": margins,
                }
            )
        samples.append(sample)

    sample_count = len(detections_by_frame)
    detected_count = len(selected_boxes)
    detection_rate = detected_count / max(sample_count, 1)
    edge_contact_rate = edge_contacts / max(detected_count, 1)

    fallback_reason: Optional[str] = None
    raw_union: Optional[BBox] = None
    if selected_boxes:
        raw_union = clamp_bbox(union_bboxes(selected_boxes), frame_width, frame_height)
        bbox = expand_bbox(
            raw_union,
            pad_ratio=pad_ratio,
            width=frame_width,
            height=frame_height,
            aspect_ratio=aspect_ratio,
        )
        union_margins = edge_margin_ratios(raw_union, frame_width, frame_height)
    else:
        fallback_reason = "no_detector_person_boxes"
        fallback = fallback_bbox or (0.0, 0.0, float(frame_width), float(frame_height))
        bbox = expand_bbox(
            fallback,
            pad_ratio=0.0,
            width=frame_width,
            height=frame_height,
            aspect_ratio=aspect_ratio,
        )
        union_margins = None

    accepted = (
        detected_count > 0
        and detection_rate >= float(min_detection_rate)
        and edge_contact_rate <= float(max_edge_contact_rate)
    )
    if detected_count > 0 and detection_rate < float(min_detection_rate):
        fallback_reason = "low_detection_rate"
    elif detected_count > 0 and edge_contact_rate > float(max_edge_contact_rate):
        fallback_reason = "edge_contact_rate_too_high"

    config = {
        "pad_ratio": float(pad_ratio),
        "min_detection_rate": float(min_detection_rate),
        "min_edge_margin_ratio": float(min_edge_margin_ratio),
        "max_edge_contact_rate": float(max_edge_contact_rate),
        "aspect_ratio": float(aspect_ratio),
    }
    return DetectorUnionResult(
        bbox_xyxy=bbox,
        raw_union_bbox_xyxy=raw_union,
        sample_count=sample_count,
        detected_count=detected_count,
        detection_rate=float(detection_rate),
        edge_contact_rate=float(edge_contact_rate),
        union_edge_margins=union_margins,
        accepted=bool(accepted),
        fallback_reason=fallback_reason,
        config=config,
        samples=samples,
    )


def build_detector_union_crop(
    video_path: str | Path,
    detector: PersonDetector,
    *,
    input_size: Tuple[int, int] = (256, 192),
    sample_stride: int = 10,
    max_samples: int = 80,
    pad_ratio: float = 0.35,
    min_detection_rate: float = 0.6,
    min_edge_margin_ratio: float = 0.03,
    max_edge_contact_rate: float = 0.0,
    fallback_bbox: Optional[BBox] = None,
) -> DetectorUnionResult:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_frame_indices(total_frames, sample_stride=sample_stride, max_samples=max_samples)
        detections_by_frame: List[Tuple[int, Sequence[PersonDetection]]] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            detections_by_frame.append((int(idx), detector.detect(frame)))
    finally:
        cap.release()

    H, W = input_size
    result = build_union_crop_from_detections(
        detections_by_frame,
        frame_width=frame_width,
        frame_height=frame_height,
        aspect_ratio=W / H,
        pad_ratio=pad_ratio,
        min_detection_rate=min_detection_rate,
        min_edge_margin_ratio=min_edge_margin_ratio,
        max_edge_contact_rate=max_edge_contact_rate,
        fallback_bbox=fallback_bbox,
    )
    result.config.update(
        {
            "sample_stride": int(sample_stride),
            "max_samples": int(max_samples),
            "video_frame_width": int(frame_width),
            "video_frame_height": int(frame_height),
            "video_total_frames": int(total_frames),
        }
    )
    return result


class TorchVisionPersonDetector:
    """COCO person detector wrapper using torchvision detection weights."""

    def __init__(
        self,
        *,
        score_threshold: float = 0.7,
        device: Optional[str] = None,
        model_name: str = "fasterrcnn_resnet50_fpn_v2",
    ) -> None:
        self.score_threshold = float(score_threshold)
        self.model_name = model_name
        _configure_detector_weight_download_env()
        try:
            import torch
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_V2_Weights,
                fasterrcnn_resnet50_fpn_v2,
            )
        except Exception as e:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "torchvision detection is required for detector_union crop mode. "
                "Install torchvision or run with --crop-mode motion/manual."
            ) from e

        self._torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if model_name != "fasterrcnn_resnet50_fpn_v2":
            raise ValueError(f"Unsupported detector model: {model_name}")
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        try:
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        except Exception as e:  # pragma: no cover - network/cache dependent
            raise RuntimeError(
                "Could not load torchvision COCO person-detector weights. "
                "Run with a writable TORCH_HOME and network access once, or pre-place "
                "the weights in torch's model cache, then retry detector_union mode."
            ) from e
        self.model.to(self.device).eval()

    def detect(self, frame_bgr: np.ndarray) -> Sequence[PersonDetection]:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = self._torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        with self._torch.no_grad():
            out = self.model([x.to(self.device)])[0]
        boxes = out["boxes"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        people: List[PersonDetection] = []
        for box, label, score in zip(boxes, labels, scores):
            if int(label) == 1 and float(score) >= self.score_threshold:
                people.append(PersonDetection(tuple(float(v) for v in box), float(score)))
        return people


class YOLOv8PersonDetector:
    """COCO person detector wrapper using Ultralytics YOLOv8."""

    def __init__(
        self,
        *,
        score_threshold: float = 0.7,
        model_name: str = "yolov8n.pt",
        device: Optional[str] = None,
    ) -> None:
        self.score_threshold = float(score_threshold)
        self.model_name = model_name
        self.device = device
        try:
            from ultralytics import YOLO
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Ultralytics is required for YOLOv8 detector crops. "
                "Install `ultralytics` or set detector_backend: torchvision."
            ) from e

        try:
            self.model = YOLO(model_name)
        except Exception as e:  # pragma: no cover - network/cache dependent
            raise RuntimeError(
                f"Could not load YOLOv8 model {model_name!r}. Make sure the "
                "weights are available locally or allow Ultralytics to download them once."
            ) from e

    def detect(self, frame_bgr: np.ndarray) -> Sequence[PersonDetection]:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        kwargs = {
            "classes": [0],
            "conf": self.score_threshold,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(rgb, **kwargs)
        people: List[PersonDetection] = []
        if not results:
            return people
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return people
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        for box, score in zip(xyxy, scores):
            if float(score) >= self.score_threshold:
                people.append(PersonDetection(tuple(float(v) for v in box), float(score)))
        return people


def build_person_detector(
    backend: str,
    *,
    score_threshold: float = 0.7,
    device: Optional[str] = None,
    model_name: Optional[str] = None,
) -> PersonDetector:
    """Construct a detector backend for inference-time person crops."""
    backend = backend.lower()
    if backend in {"torchvision", "fasterrcnn", "fasterrcnn_resnet50_fpn_v2"}:
        return TorchVisionPersonDetector(score_threshold=score_threshold, device=device)
    if backend in {"yolo", "yolov8", "ultralytics"}:
        return YOLOv8PersonDetector(
            score_threshold=score_threshold,
            device=device,
            model_name=model_name or "yolov8n.pt",
        )
    raise ValueError(f"Unknown detector backend: {backend!r}")
