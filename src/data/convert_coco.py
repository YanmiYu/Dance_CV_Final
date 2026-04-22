"""Phase 5.2: convert COCO person_keypoints into the internal schema.

Input: COCO ``person_keypoints_train2017.json`` (or val).
Output: newline-delimited json (``.jsonl``) with ``AnnotationRecord`` rows.

We only keep persons with:
  * at least ``min_visible_joints`` visible keypoints (v in {1,2})
  * non-trivial bbox area

We DO NOT download COCO ourselves -- the user provides the path to the
official annotations and image directory. See ``docs/project_decisions.md``
section 6.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from src.datasets.common import (
    NUM_JOINTS,
    AnnotationRecord,
    bbox_to_center_scale,
)
from src.utils.io import read_json, write_jsonl


def _xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def _kps_coco_to_internal(kps_flat) -> list[list[float]]:
    """COCO stores (x, y, v) flat (51 values) -> (17, 3). COCO already matches our 17 order."""
    assert len(kps_flat) == 51, f"expected 51 values, got {len(kps_flat)}"
    out = []
    for i in range(NUM_JOINTS):
        x, y, v = kps_flat[3 * i : 3 * i + 3]
        out.append([float(x), float(y), float(v)])
    return out


def convert(
    coco_json: str | Path,
    images_dir: str | Path,
    out_jsonl: str | Path,
    aspect_ratio: float,
    min_visible_joints: int = 6,
    min_bbox_area: float = 1024.0,
    dataset_name: str = "coco",
) -> int:
    data = read_json(coco_json)
    images_by_id = {im["id"]: im for im in data["images"]}

    def _gen() -> Iterable[dict]:
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0) == 1:
                continue
            if ann.get("num_keypoints", 0) < min_visible_joints:
                continue
            bbox = _xywh_to_xyxy(ann["bbox"])
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < min_bbox_area:
                continue
            im = images_by_id.get(ann["image_id"])
            if im is None:
                continue
            kps = _kps_coco_to_internal(ann["keypoints"])
            center, scale = bbox_to_center_scale(bbox, aspect_ratio=aspect_ratio)
            rec = AnnotationRecord(
                image_path=str(Path(images_dir) / im["file_name"]),
                image_id=f"coco_{ann['image_id']}_{ann['id']}",
                dataset_name=dataset_name,
                bbox_xyxy=[float(v) for v in bbox],
                keypoints_xyv=kps,
                center=center,
                scale=scale,
                meta={"coco_image_id": ann["image_id"], "coco_ann_id": ann["id"]},
            )
            rec.validate()
            yield rec.__dict__

    return write_jsonl(out_jsonl, _gen())


def _main() -> None:
    p = argparse.ArgumentParser(description="Convert COCO person_keypoints JSON to internal JSONL.")
    p.add_argument("--coco-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--input-size", nargs=2, type=int, default=[256, 192], help="H W")
    args = p.parse_args()
    H, W = args.input_size
    n = convert(args.coco_json, args.images_dir, args.out, aspect_ratio=W / H)
    print(f"Wrote {n} COCO annotations -> {args.out}")


if __name__ == "__main__":
    _main()
