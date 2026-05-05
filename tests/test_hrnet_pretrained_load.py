"""Tests for HRNet ImageNet backbone loading and differential LR groups."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


_TINY_CFG = {
    "name": "hrnet_w32",
    "pretrained": False,
    "stem": {"out_channels": 16},
    "stage1": {"num_blocks": 2, "channels": 8},
    "stage2": {"num_branches": 2, "num_blocks": [2, 2], "channels": [8, 16], "num_modules": 1},
    "stage3": {"num_branches": 3, "num_blocks": [2, 2, 2], "channels": [8, 16, 32], "num_modules": 1},
    "stage4": {"num_branches": 4, "num_blocks": [2, 2, 2, 2], "channels": [8, 16, 32, 64], "num_modules": 1},
    "head": {"type": "final_conv", "in_channels": 8, "num_joints": 17},
}


def test_remap_keys_cover_core_backbone_groups():
    from src.models.hrnet_pretrained import _remap_key

    cases = {
        "conv1.weight": "stem.0.weight",
        "bn1.running_mean": "stem.1.running_mean",
        "conv2.weight": "stem.3.weight",
        "layer1.0.conv1.weight": "stage1.0.conv1.weight",
        "transition1.1.0.0.weight": "trans_12.transitions.1.0.weight",
        "stage2.0.branches.0.0.conv1.weight": "stage2.modules_list.0.branches.0.blocks.0.conv1.weight",
        "stage3.0.fuse_layers.2.0.1.0.weight": "stage3.modules_list.0.fuse.fuse_layers.2.0.3.weight",
    }
    for official, local in cases.items():
        assert _remap_key(official) == local


def test_remap_drops_classifier_keys():
    from src.models.hrnet_pretrained import _remap_key

    assert _remap_key("classifier.weight") is None
    assert _remap_key("incre_modules.0.0.conv1.weight") is None


def test_missing_checkpoint_error_is_actionable(tmp_path: Path):
    from src.models.hrnet_pretrained import _resolve_checkpoint_path

    with pytest.raises(FileNotFoundError, match="download_hrnet_imagenet.py"):
        _resolve_checkpoint_path(tmp_path / "missing" / "hrnetv2_w32_imagenet.pth")


def test_load_hrnet_imagenet_backbone_keeps_head_random(tmp_path: Path):
    from src.models.hrnet import HRNetPose
    from src.models.hrnet_pretrained import load_hrnet_imagenet_backbone

    torch.manual_seed(0)
    source = HRNetPose(_TINY_CFG)
    source_state = source.state_dict()

    official = {}
    for key, value in source_state.items():
        if key.startswith("head."):
            continue
        official_key = _local_to_official(key)
        if official_key is not None:
            official[official_key] = value.clone()
    official["classifier.weight"] = torch.zeros(1000, 2048)

    ckpt_path = tmp_path / "fake_hrnet_imagenet.pth"
    torch.save({"state_dict": official}, ckpt_path)

    torch.manual_seed(1)
    target = HRNetPose(_TINY_CFG)
    backbone_key = "stage2.modules_list.0.branches.0.blocks.0.conv1.weight"
    head_key = next(k for k in target.state_dict() if k.startswith("head."))
    pre_head = target.state_dict()[head_key].clone()

    summary = load_hrnet_imagenet_backbone(target, ckpt_path)

    assert torch.allclose(target.state_dict()[backbone_key], source_state[backbone_key])
    assert torch.allclose(target.state_dict()[head_key], pre_head)
    assert summary["loaded"] > 0
    assert summary["dropped_classifier"] >= 1
    assert summary["missing_backbone"] == 0


def test_build_optimizer_param_groups_covers_all_params():
    from src.models.hrnet import HRNetPose
    from src.train.engine import build_optimizer

    model = HRNetPose(_TINY_CFG)
    opt = build_optimizer(
        model,
        {
            "name": "adamw",
            "weight_decay": 1e-4,
            "param_groups": {
                "backbone": {
                    "lr": 1e-4,
                    "modules": ["stem", "stage1", "trans_12", "stage2", "trans_23", "stage3", "trans_34", "stage4"],
                },
                "head": {"lr": 1e-3, "modules": ["head"]},
            },
        },
    )
    assert sorted(group["lr"] for group in opt.param_groups) == [1e-4, 1e-3]
    total_assigned = sum(p.numel() for group in opt.param_groups for p in group["params"])
    assert total_assigned == sum(p.numel() for p in model.parameters())


def _local_to_official(key: str) -> str | None:
    if key.startswith("stem.0."):
        return "conv1." + key[len("stem.0."):]
    if key.startswith("stem.1."):
        return "bn1." + key[len("stem.1."):]
    if key.startswith("stem.3."):
        return "conv2." + key[len("stem.3."):]
    if key.startswith("stem.4."):
        return "bn2." + key[len("stem.4."):]
    if key.startswith("stage1."):
        return "layer1." + key[len("stage1."):]
    if key.startswith("trans_12.transitions."):
        return _transition_to_official("transition1", 1, key[len("trans_12.transitions."):])
    if key.startswith("trans_23.transitions."):
        return _transition_to_official("transition2", 2, key[len("trans_23.transitions."):])
    if key.startswith("trans_34.transitions."):
        return _transition_to_official("transition3", 3, key[len("trans_34.transitions."):])
    if ".modules_list." in key and ".branches." in key:
        stage, rest = key.split(".modules_list.", 1)
        module, rest = rest.split(".branches.", 1)
        branch, rest = rest.split(".blocks.", 1)
        block, suffix = rest.split(".", 1)
        return f"{stage}.{module}.branches.{branch}.{block}.{suffix}"
    if ".modules_list." in key and ".fuse.fuse_layers." in key:
        stage, rest = key.split(".modules_list.", 1)
        module, rest = rest.split(".fuse.fuse_layers.", 1)
        dst, rest = rest.split(".", 1)
        src, suffix = rest.split(".", 1)
        if int(dst) > int(src):
            layer, param = suffix.split(".", 1)
            layer_idx = int(layer)
            block = layer_idx // 3
            block_layer = layer_idx % 3
            return f"{stage}.{module}.fuse_layers.{dst}.{src}.{block}.{block_layer}.{param}"
        return f"{stage}.{module}.fuse_layers.{dst}.{src}.{suffix}"
    return None


def _transition_to_official(prefix: str, new_branch: int, suffix: str) -> str:
    branch, rest = suffix.split(".", 1)
    if int(branch) == new_branch:
        layer, param = rest.split(".", 1)
        layer_idx = int(layer)
        block = layer_idx // 3
        block_layer = layer_idx % 3
        return f"{prefix}.{branch}.{block}.{block_layer}.{param}"
    return f"{prefix}.{branch}.{rest}"
