"""Phase 5.2: convert CrowdPose annotations into the internal (COCO-17) schema.

CrowdPose uses 14 keypoints. We map them to COCO-17; joints that do not exist
in CrowdPose (``nose``, ``left/right_eye``, ``left/right_ear``) are emitted
with visibility ``0`` (not annotated).

CrowdPose joint order (official):
  0 left_shoulder, 1 right_shoulder, 2 left_elbow, 3 right_elbow,
  4 left_wrist, 5 right_wrist, 6 left_hip, 7 right_hip,
  8 left_knee, 9 right_knee, 10 left_ankle, 11 right_ankle,
  12 head (top), 13 neck.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from src.datasets.common import (
    AnnotationRecord,
    NUM_JOINTS,
    bbox_to_center_scale,
)
from src.utils.io import read_json, write_jsonl


# Map from CrowdPose index -> COCO-17 index (or None if it has no direct equivalent).
_CP_TO_COCO17 = {
    0: 5,   # left_shoulder
    1: 6,   # right_shoulder
    2: 7,   # left_elbow
    3: 8,   # right_elbow
    4: 9,   # left_wrist
    5: 10,  # right_wrist
    6: 11,  # left_hip
    7: 12,  # right_hip
    8: 13,  # left_knee
    9: 14,  # right_knee
    10: 15, # left_ankle
    11: 16, # right_ankle
    12: 0,  # head top -> we stash into 'nose' as a rough anchor
    13: None,  # neck has no COCO-17 slot
}


def _xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def _kps_crowdpose_to_coco17(cp_kps) -> list[list[float]]:
    """Convert 14*3 flat crowdpose keypoints to 17*3 coco order."""
    out = [[0.0, 0.0, 0.0] for _ in range(NUM_JOINTS)]
    assert len(cp_kps) == 14 * 3, f"expected 42 values, got {len(cp_kps)}"
    for cp_idx in range(14):
        target = _CP_TO_COCO17.get(cp_idx)
        if target is None:
            continue
        x, y, v = cp_kps[3 * cp_idx : 3 * cp_idx + 3]
        out[target] = [float(x), float(y), float(v)]
    return out


def convert(
    json_path: str | Path,
    images_dir: str | Path,
    out_jsonl: str | Path,
    aspect_ratio: float,
    min_visible_joints: int = 6,
    min_bbox_area: float = 1024.0,
) -> int:
    data = read_json(json_path)
    images_by_id = {im["id"]: im for im in data["images"]}

    def _gen() -> Iterable[dict]:
        for ann in data["annotations"]:
            if ann.get("num_keypoints", 0) < min_visible_joints:
                continue
            bbox = _xywh_to_xyxy(ann["bbox"])
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < min_bbox_area:
                continue
            im = images_by_id.get(ann["image_id"])
            if im is None:
                continue
            kps = _kps_crowdpose_to_coco17(ann["keypoints"])
            center, scale = bbox_to_center_scale(bbox, aspect_ratio=aspect_ratio)
            rec = AnnotationRecord(
                image_path=str(Path(images_dir) / im["file_name"]),
                image_id=f"crowdpose_{ann['image_id']}_{ann['id']}",
                dataset_name="crowdpose",
                bbox_xyxy=[float(v) for v in bbox],
                keypoints_xyv=kps,
                center=center,
                scale=scale,
                meta={"cp_image_id": ann["image_id"], "cp_ann_id": ann["id"]},
            )
            rec.validate()
            yield rec.__dict__

    return write_jsonl(out_jsonl, _gen())


def _main() -> None:
    p = argparse.ArgumentParser(description="Convert CrowdPose annotations into internal (COCO-17) JSONL.")
    p.add_argument("--cp-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--input-size", nargs=2, type=int, default=[256, 192])
    args = p.parse_args()
    H, W = args.input_size
    n = convert(args.cp_json, args.images_dir, args.out, aspect_ratio=W / H)
    print(f"Wrote {n} CrowdPose annotations -> {args.out}")


if __name__ == "__main__":
    _main()
