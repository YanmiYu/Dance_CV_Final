from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.convert_aistpp import convert_video


def test_convert_video_can_require_extracted_frame_exists(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "clip_000000.jpg").write_bytes(b"stub")

    kps = np.zeros((3, 17, 3), dtype=np.float32)
    kps[:, :, 0] = np.arange(17, dtype=np.float32) + 10.0
    kps[:, :, 1] = np.arange(17, dtype=np.float32) + 20.0
    kps[:, :, 2] = 1.0
    kps_file = tmp_path / "clip.npy"
    np.save(kps_file, kps)

    rows: list[dict] = []
    convert_video(
        video_path=tmp_path / "clip.mp4",
        frames_dir=frames_dir,
        kps_file=kps_file,
        out_rows=rows,
        aspect_ratio=192 / 256,
        frame_stride=1,
        image_width=640,
        image_height=480,
        require_frame_exists=True,
    )

    assert len(rows) == 1
    assert rows[0]["image_path"].endswith("clip_000000.jpg")
