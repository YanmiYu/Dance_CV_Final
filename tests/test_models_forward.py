"""Shape + no-pretrained-weights sanity for both models."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _make_input():
    return torch.randn(2, 3, 256, 192)


def test_simple_baseline_forward_shape():
    from src.models.simple_baseline import SimpleBaselinePose

    cfg = {
        "name": "simple_baseline",
        "pretrained": False,
        "backbone": {"stem_channels": 16, "stages": [
            {"channels": 16, "blocks": 1, "stride": 1},
            {"channels": 32, "blocks": 1, "stride": 2},
        ]},
        "head": {"num_deconv_layers": 2, "deconv_channels": [32, 32], "deconv_kernels": [4, 4], "final_kernel": 1, "num_joints": 17},
    }
    model = SimpleBaselinePose(cfg).eval()
    with torch.no_grad():
        y = model(_make_input())
    assert y.dim() == 4
    assert y.shape[1] == 17
    assert y.shape[0] == 2


def test_hrnet_forward_shape_small():
    from src.models.hrnet import HRNetPose

    cfg = {
        "name": "hrnet_w32",
        "pretrained": False,
        "stem": {"out_channels": 16},
        "stage1": {"num_blocks": 1, "channels": 8},
        "stage2": {"num_branches": 2, "num_blocks": [1, 1], "channels": [8, 16], "num_modules": 1},
        "stage3": {"num_branches": 3, "num_blocks": [1, 1, 1], "channels": [8, 16, 32], "num_modules": 1},
        "stage4": {"num_branches": 4, "num_blocks": [1, 1, 1, 1], "channels": [8, 16, 32, 64], "num_modules": 1},
        "head": {"type": "final_conv", "in_channels": 8, "num_joints": 17},
    }
    model = HRNetPose(cfg).eval()
    with torch.no_grad():
        y = model(_make_input())
    assert y.dim() == 4
    assert y.shape[1] == 17


def test_pretrained_true_refused():
    from src.models.simple_baseline import SimpleBaselinePose
    import pytest

    with pytest.raises(AssertionError):
        SimpleBaselinePose({"name": "simple_baseline", "pretrained": True})
