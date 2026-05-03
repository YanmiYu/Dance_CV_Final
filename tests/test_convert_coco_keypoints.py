from __future__ import annotations

import json
from pathlib import Path

from src.data.convert_coco_keypoints import convert_coco_keypoints
from src.utils.io import read_jsonl


def _kps(visible: int = 17) -> list[float]:
    vals = []
    for j in range(17):
        v = 2 if j < visible else 0
        vals.extend([float(10 + j), float(20 + j), float(v)])
    return vals


def test_convert_coco_keypoints_writes_internal_schema_and_clip_groups(tmp_path: Path) -> None:
    coco = {
        "images": [
            {"id": 1, "file_name": "clipA/frame_000001.jpg", "width": 100, "height": 120},
            {"id": 2, "file_name": "clipB/frame_000001.jpg", "width": 100, "height": 120},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "keypoints": _kps(), "bbox": [5, 10, 80, 100]},
            {"id": 20, "image_id": 2, "category_id": 1, "keypoints": _kps(), "bbox": [6, 11, 70, 95]},
        ],
    }
    src = tmp_path / "manual.json"
    src.write_text(json.dumps(coco))
    out_dir = tmp_path / "labels"

    n_train, n_val, skipped = convert_coco_keypoints(
        src,
        tmp_path / "frames",
        out_dir,
        val_fraction=0.0,
        split_by="parent",
    )

    assert (n_train, n_val, skipped) == (2, 0, 0)
    rows = list(read_jsonl(out_dir / "internal_train.jsonl"))
    assert rows[0]["dataset_name"] == "custom_dance"
    assert rows[0]["keypoints_xyv"][0] == [10.0, 20.0, 2.0]
    assert rows[0]["meta"]["split_group"] == "clipA"
    assert rows[0]["image_path"].endswith("frames/clipA/frame_000001.jpg")


def test_convert_coco_keypoints_skips_low_visibility(tmp_path: Path) -> None:
    coco = {
        "images": [{"id": 1, "file_name": "clipA/frame_000001.jpg"}],
        "annotations": [{"id": 10, "image_id": 1, "keypoints": _kps(visible=4), "bbox": [5, 10, 80, 100]}],
    }
    src = tmp_path / "manual.json"
    src.write_text(json.dumps(coco))

    n_train, n_val, skipped = convert_coco_keypoints(src, tmp_path / "frames", tmp_path / "labels")

    assert (n_train, n_val, skipped) == (0, 0, 1)
