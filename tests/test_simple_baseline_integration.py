from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")


def _small_simple_baseline_cfg() -> dict:
    return {
        "name": "simple_baseline",
        "pretrained": False,
        "backbone": {
            "stem_channels": 8,
            "stages": [
                {"channels": 8, "blocks": 1, "stride": 1},
                {"channels": 16, "blocks": 1, "stride": 2},
            ],
        },
        "head": {
            "num_deconv_layers": 1,
            "deconv_channels": [16],
            "deconv_kernels": [4],
            "final_kernel": 1,
            "num_joints": 17,
        },
    }


def test_direct_resize_preprocess_matches_simple_baseline_branch() -> None:
    from src.infer.run_pose_on_video import _prep_input

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = 20
    frame[:, :, 2] = 30

    x, info = _prep_input(
        frame,
        (1, 1, 3, 3),
        (2, 2),
        crop_transform="direct_resize",
        color_order="rgb",
        image_mean=[30 / 255.0, 20 / 255.0, 10 / 255.0],
        image_std=[1.0, 1.0, 1.0],
    )

    assert x.shape == (1, 3, 2, 2)
    assert torch.allclose(x, torch.zeros_like(x), atol=1e-6)
    assert info["mode"] == "direct_resize"
    np.testing.assert_array_equal(info["bbox"], np.array([1, 1, 3, 3], dtype=np.float32))


def test_direct_resize_heatmap_decode_maps_back_to_bbox() -> None:
    from src.infer.run_pose_on_video import _decode_heatmaps_to_bbox

    hm = torch.zeros((1, 17, 64, 48), dtype=torch.float32)
    hm[0, 0, 32, 24] = 1.0
    hm[0, 1, 16, 12] = 0.5

    kps = _decode_heatmaps_to_bbox(hm, np.array([10, 20, 202, 276], dtype=np.float32), (64, 48))

    assert kps.shape == (17, 3)
    assert kps[0, 0] == pytest.approx(106.0)
    assert kps[0, 1] == pytest.approx(148.0)
    assert kps[0, 2] == pytest.approx(1.0)
    assert kps[1, 2] == pytest.approx(0.5)


def test_internal_checkpoint_loader_accepts_state_dict_wrappers(tmp_path: Path) -> None:
    from src.train.engine import _load_state_from_internal_ckpt, build_model

    source = build_model(_small_simple_baseline_cfg())
    wrapped_state = {
        "state_dict": {
            f"module.{key}": value.detach().clone()
            for key, value in source.state_dict().items()
        }
    }

    ckpt_path = tmp_path / "data" / "processed" / "simple_baseline" / "fake.pt"
    ckpt_path.parent.mkdir(parents=True)
    torch.save(wrapped_state, ckpt_path)

    target = build_model(_small_simple_baseline_cfg())
    _load_state_from_internal_ckpt(target, str(ckpt_path))

    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)
