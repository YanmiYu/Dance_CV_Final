"""Convert manual COCO-keypoints exports into the internal pose JSONL schema.

This is intended for human-labeled target-domain dance frames exported from
tools such as CVAT or Label Studio. Detector outputs and pose pseudo-labels
must not be routed through this converter as supervised labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.datasets.common import AnnotationRecord, NUM_JOINTS, bbox_to_center_scale
from src.utils.io import write_jsonl


def _is_val(group: str, val_fraction: float, seed: str) -> bool:
    h = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    return bucket < val_fraction


def _split_group(file_name: str, mode: str) -> str:
    p = Path(file_name)
    if mode == "parent":
        parent = p.parent.as_posix()
        if parent and parent != ".":
            return parent
        mode = "stem_prefix"
    if mode == "stem_prefix":
        stem = p.stem
        parts = stem.split("_")
        return "_".join(parts[:-1]) if len(parts) > 1 else stem
    if mode == "image":
        return p.as_posix()
    raise ValueError(f"Unknown split_by mode: {mode!r}")


def _resolve_image_path(images_root: Path, file_name: str) -> Path:
    p = Path(file_name)
    return p if p.is_absolute() else images_root / p


def _bbox_xywh_to_xyxy(bbox: Iterable[float]) -> List[float]:
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + max(w, 1.0), y + max(h, 1.0)]


def _bbox_from_visible_keypoints(kps_xyv: np.ndarray, margin: float = 0.15) -> Optional[List[float]]:
    visible = kps_xyv[:, 2] > 0
    if not visible.any():
        return None
    xy = kps_xyv[visible, :2]
    x1, y1 = float(np.min(xy[:, 0])), float(np.min(xy[:, 1]))
    x2, y2 = float(np.max(xy[:, 0])), float(np.max(xy[:, 1]))
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    return [x1 - w * margin, y1 - h * margin, x2 + w * margin, y2 + h * margin]


def _annotation_rank(ann: dict) -> Tuple[int, float]:
    keypoints = ann.get("keypoints") or []
    vis = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)[:, 2] if len(keypoints) == 51 else np.zeros(0)
    visible = int((vis > 0).sum())
    area = float(ann.get("area") or 0.0)
    if not area and ann.get("bbox"):
        x, y, w, h = [float(v) for v in ann["bbox"]]
        area = max(w, 0.0) * max(h, 0.0)
    return visible, area


def convert_coco_keypoints(
    annotations_json: str | Path,
    images_root: str | Path,
    out_dir: str | Path,
    *,
    dataset_name: str = "custom_dance",
    input_size: Tuple[int, int] = (256, 192),
    val_fraction: float = 0.2,
    split_seed: str = "custom-dance-split-v1",
    split_by: str = "parent",
    min_visible_keypoints: int = 8,
) -> Tuple[int, int, int]:
    annotations_json = Path(annotations_json)
    images_root = Path(images_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = json.loads(annotations_json.read_text())
    images = {img["id"]: img for img in obj.get("images", [])}
    anns_by_image: Dict[object, List[dict]] = defaultdict(list)
    for ann in obj.get("annotations", []):
        if ann.get("keypoints") is not None:
            anns_by_image[ann.get("image_id")].append(ann)

    H, W = input_size
    aspect_ratio = W / H
    train_rows: List[dict] = []
    val_rows: List[dict] = []
    skipped = 0

    for image_id, image in sorted(images.items(), key=lambda kv: str(kv[1].get("file_name", kv[0]))):
        candidates = anns_by_image.get(image_id, [])
        candidates = [ann for ann in candidates if len(ann.get("keypoints") or []) == NUM_JOINTS * 3]
        if not candidates:
            skipped += 1
            continue
        ann = max(candidates, key=_annotation_rank)
        keypoints = np.asarray(ann["keypoints"], dtype=np.float32).reshape(NUM_JOINTS, 3)
        if int((keypoints[:, 2] > 0).sum()) < int(min_visible_keypoints):
            skipped += 1
            continue

        if ann.get("bbox"):
            bbox = _bbox_xywh_to_xyxy(ann["bbox"])
        else:
            bbox = _bbox_from_visible_keypoints(keypoints)
            if bbox is None:
                skipped += 1
                continue
        center, scale = bbox_to_center_scale(bbox, aspect_ratio=aspect_ratio)
        file_name = str(image["file_name"])
        rec = AnnotationRecord(
            image_path=str(_resolve_image_path(images_root, file_name)),
            image_id=f"{dataset_name}_{image_id}",
            dataset_name=dataset_name,
            bbox_xyxy=[float(v) for v in bbox],
            keypoints_xyv=keypoints.tolist(),
            center=center,
            scale=scale,
            meta={
                "source_json": str(annotations_json),
                "source_image_id": image_id,
                "source_annotation_id": ann.get("id"),
                "split_group": _split_group(file_name, split_by),
                "width": image.get("width"),
                "height": image.get("height"),
            },
        )
        rec.validate()
        target = val_rows if _is_val(rec.meta["split_group"], val_fraction, split_seed) else train_rows
        target.append(rec.__dict__)

    n_train = write_jsonl(out_dir / "internal_train.jsonl", train_rows)
    n_val = write_jsonl(out_dir / "internal_val.jsonl", val_rows)
    return n_train, n_val, skipped


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert manual COCO-keypoints JSON into internal custom_dance JSONL.")
    p.add_argument("--annotations", required=True, type=Path, help="COCO keypoints JSON from CVAT/Label Studio")
    p.add_argument("--images-root", default="data/raw_frames/custom_dance", type=Path)
    p.add_argument("--out-dir", default="data/labels/custom_dance", type=Path)
    p.add_argument("--dataset-name", default="custom_dance")
    p.add_argument("--input-size", nargs=2, type=int, default=[256, 192], metavar=("H", "W"))
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--split-seed", default="custom-dance-split-v1")
    p.add_argument("--split-by", choices=["parent", "stem_prefix", "image"], default="parent")
    p.add_argument("--min-visible-keypoints", type=int, default=8)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    n_train, n_val, skipped = convert_coco_keypoints(
        args.annotations,
        args.images_root,
        args.out_dir,
        dataset_name=args.dataset_name,
        input_size=tuple(args.input_size),
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        split_by=args.split_by,
        min_visible_keypoints=args.min_visible_keypoints,
    )
    print(f"Wrote {n_train} train rows and {n_val} val rows -> {args.out_dir}")
    if skipped:
        print(f"Skipped {skipped} images without a usable single-person keypoint annotation")


if __name__ == "__main__":
    main()
