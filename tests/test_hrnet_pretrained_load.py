"""Tests for the HRNet ImageNet backbone loader and differential-LR optimizer."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


# ------------------------------ remap unit tests ------------------------------


def test_remap_keys_cover_all_backbone_groups():
    from src.models.hrnet_pretrained import _remap_key

    cases = {
        "conv1.weight": "stem.0.weight",
        "bn1.weight": "stem.1.weight",
        "bn1.running_mean": "stem.1.running_mean",
        "conv2.weight": "stem.3.weight",
        "bn2.bias": "stem.4.bias",
        "layer1.0.conv1.weight": "stage1.0.conv1.weight",
        "layer1.0.downsample.0.weight": "stage1.0.downsample.0.weight",
        "layer1.3.bn3.running_var": "stage1.3.bn3.running_var",
        "transition1.0.0.weight": "trans_12.transitions.0.0.weight",
        "transition1.1.0.weight": "trans_12.transitions.1.0.weight",
        "transition2.2.0.weight": "trans_23.transitions.2.0.weight",
        "transition3.3.1.bias": "trans_34.transitions.3.1.bias",
        "stage2.0.branches.0.0.conv1.weight": "stage2.modules_list.0.branches.0.blocks.0.conv1.weight",
        "stage2.0.branches.1.3.bn2.bias": "stage2.modules_list.0.branches.1.blocks.3.bn2.bias",
        "stage2.0.fuse_layers.0.1.0.weight": "stage2.modules_list.0.fuse.fuse_layers.0.1.0.weight",
        "stage3.2.branches.2.1.conv2.weight": "stage3.modules_list.2.branches.2.blocks.1.conv2.weight",
        "stage4.0.fuse_layers.3.0.4.weight": "stage4.modules_list.0.fuse.fuse_layers.3.0.4.weight",
    }
    for off, expected in cases.items():
        assert _remap_key(off) == expected, f"{off} -> {_remap_key(off)}, expected {expected}"


def test_remap_drops_classification_head_keys():
    from src.models.hrnet_pretrained import _remap_key

    drops = [
        "incre_modules.0.0.conv1.weight",
        "downsamp_modules.1.0.weight",
        "final_layer.weight",
        "classifier.weight",
        "classifier.bias",
    ]
    for k in drops:
        assert _remap_key(k) is None, f"expected drop for {k}"


def test_missing_checkpoint_error_is_actionable(tmp_path: Path):
    from src.models.hrnet_pretrained import _resolve_checkpoint_path

    missing = tmp_path / "pretrained" / "hrnetv2_w32_imagenet.pth"
    with pytest.raises(FileNotFoundError, match="download_hrnet_imagenet.py") as exc:
        _resolve_checkpoint_path(missing)
    assert "push_to_oscar.sh" in str(exc.value)


# --------------------- end-to-end load with synthetic ckpt --------------------


_TINY_CFG = {
    "name": "hrnet_w32",
    "pretrained": False,
    "stem": {"out_channels": 16},
    "stage1": {"num_blocks": 2, "channels": 8},
    "stage2": {"num_branches": 2, "num_blocks": [2, 2], "channels": [8, 16], "num_modules": 1},
    "stage3": {"num_branches": 3, "num_blocks": [2, 2, 2], "channels": [8, 16, 32], "num_modules": 2},
    "stage4": {"num_branches": 4, "num_blocks": [2, 2, 2, 2], "channels": [8, 16, 32, 64], "num_modules": 1},
    "head": {"type": "final_conv", "in_channels": 8, "num_joints": 17},
}


_RE_LOCAL_STAGE_BRANCH = re.compile(
    r"^stage([234])\.modules_list\.(\d+)\.branches\.(\d+)\.blocks\.(\d+)\.(.+)$"
)
_RE_LOCAL_STAGE_FUSE = re.compile(
    r"^stage([234])\.modules_list\.(\d+)\.fuse\.fuse_layers\.(\d+)\.(\d+)\.(.+)$"
)
_RE_LOCAL_STAGE1 = re.compile(r"^stage1\.(\d+)\.(.+)$")


def _local_to_official(key: str) -> str | None:
    """Inverse of `_remap_key` for backbone keys. Returns None for head/unknown."""
    if key.startswith("head."):
        return None
    if key.startswith("stem.0."):
        return "conv1." + key[len("stem.0."):]
    if key.startswith("stem.1."):
        return "bn1." + key[len("stem.1."):]
    if key.startswith("stem.3."):
        return "conv2." + key[len("stem.3."):]
    if key.startswith("stem.4."):
        return "bn2." + key[len("stem.4."):]
    if (m := _RE_LOCAL_STAGE1.match(key)) is not None:
        return f"layer1.{m.group(1)}.{m.group(2)}"
    if key.startswith("trans_12.transitions."):
        return "transition1." + key[len("trans_12.transitions."):]
    if key.startswith("trans_23.transitions."):
        return "transition2." + key[len("trans_23.transitions."):]
    if key.startswith("trans_34.transitions."):
        return "transition3." + key[len("trans_34.transitions."):]
    if (m := _RE_LOCAL_STAGE_FUSE.match(key)) is not None:
        n, mod, i, j, suf = m.groups()
        return f"stage{n}.{mod}.fuse_layers.{i}.{j}.{suf}"
    if (m := _RE_LOCAL_STAGE_BRANCH.match(key)) is not None:
        n, mod, i, j, suf = m.groups()
        return f"stage{n}.{mod}.branches.{i}.{j}.{suf}"
    return None


def test_load_hrnet_imagenet_backbone_end_to_end(tmp_path: Path):
    from src.models.hrnet import HRNetPose
    from src.models.hrnet_pretrained import load_hrnet_imagenet_backbone

    torch.manual_seed(0)
    src_model = HRNetPose(_TINY_CFG)
    src_state = src_model.state_dict()

    # Build an "official-looking" state_dict from the source model's backbone.
    official = {}
    for k, v in src_state.items():
        off = _local_to_official(k)
        if off is not None:
            official[off] = v.clone()
    # Add an ImageNet classifier head that should be silently dropped.
    official["classifier.weight"] = torch.zeros(1000, 2048)
    official["classifier.bias"] = torch.zeros(1000)
    official["incre_modules.0.0.conv1.weight"] = torch.zeros(1, 1, 1, 1)

    ckpt_path = tmp_path / "fake_hrnet_imagenet.pth"
    torch.save({"state_dict": official}, ckpt_path)

    # Build a fresh model with a different seed; backbone weights should differ.
    torch.manual_seed(1)
    tgt_model = HRNetPose(_TINY_CFG)
    snapshot_key = "stage2.modules_list.0.branches.0.blocks.0.conv1.weight"
    head_key = "head.final.weight"
    if head_key not in tgt_model.state_dict():
        head_key = next(k for k in tgt_model.state_dict() if k.startswith("head."))
    pre_backbone = tgt_model.state_dict()[snapshot_key].clone()
    pre_head = tgt_model.state_dict()[head_key].clone()

    summary = load_hrnet_imagenet_backbone(tgt_model, str(ckpt_path))

    post_backbone = tgt_model.state_dict()[snapshot_key]
    post_head = tgt_model.state_dict()[head_key]

    # Backbone tensor was overwritten with the source values exactly.
    assert torch.allclose(post_backbone, src_state[snapshot_key])
    assert not torch.allclose(post_backbone, pre_backbone)
    # Head untouched.
    assert torch.allclose(post_head, pre_head)
    # Sanity counts.
    assert summary["loaded"] > 0
    assert summary["dropped_classifier"] >= 3  # classifier.weight/bias + incre_modules
    assert summary["missing_backbone"] == 0


# ------------------------- param-group optimizer tests -------------------------


def test_build_optimizer_param_groups_covers_all_params():
    from src.models.hrnet import HRNetPose
    from src.train.engine import build_optimizer

    torch.manual_seed(0)
    model = HRNetPose(_TINY_CFG)
    optim_cfg = {
        "name": "adamw",
        "weight_decay": 1e-4,
        "param_groups": {
            "backbone": {
                "lr": 1e-4,
                "modules": ["stem", "stage1", "trans_12", "stage2", "trans_23", "stage3", "trans_34", "stage4"],
            },
            "head": {"lr": 1e-3, "modules": ["head"]},
        },
    }
    opt = build_optimizer(model, optim_cfg)
    assert len(opt.param_groups) == 2
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [1e-4, 1e-3]
    total_assigned = sum(p.numel() for g in opt.param_groups for p in g["params"])
    total_model = sum(p.numel() for p in model.parameters())
    assert total_assigned == total_model
    seen = set()
    for g in opt.param_groups:
        for p in g["params"]:
            assert id(p) not in seen
            seen.add(id(p))


def test_build_optimizer_param_groups_rejects_missing_module():
    from src.models.hrnet import HRNetPose
    from src.train.engine import build_optimizer

    model = HRNetPose(_TINY_CFG)
    optim_cfg = {
        "name": "adamw",
        "weight_decay": 1e-4,
        "param_groups": {
            "backbone": {"lr": 1e-4, "modules": ["stem", "stage1"]},  # missing the rest
            "head": {"lr": 1e-3, "modules": ["head"]},
        },
    }
    with pytest.raises(ValueError):
        build_optimizer(model, optim_cfg)


def test_build_scheduler_per_group_min_lr_clamp():
    from src.models.hrnet import HRNetPose
    from src.train.engine import build_optimizer, build_scheduler

    torch.manual_seed(0)
    model = HRNetPose(_TINY_CFG)
    optim_cfg = {
        "name": "adamw",
        "weight_decay": 1e-4,
        "param_groups": {
            "backbone": {"lr": 1e-4, "modules": ["stem", "stage1", "trans_12", "stage2", "trans_23", "stage3", "trans_34", "stage4"]},
            "head": {"lr": 1e-3, "modules": ["head"]},
        },
    }
    opt = build_optimizer(model, optim_cfg)
    sched = build_scheduler(opt, {"name": "cosine", "warmup_epochs": 0, "min_lr": 1e-6}, total_epochs=10)
    # Step to the end so cosine -> 0; both groups should clamp at min_lr exactly.
    for _ in range(20):
        sched.step()
    final = [g["lr"] for g in opt.param_groups]
    for lr in final:
        assert abs(lr - 1e-6) < 1e-9, f"group did not clamp to min_lr: {final}"
