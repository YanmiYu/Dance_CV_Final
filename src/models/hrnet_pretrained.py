"""Load ImageNet-pretrained HRNet backbone weights into ``HRNetPose``.

The official MSRA HRNet checkpoint uses different submodule names than this
repo's local implementation in ``src/models/hrnet.py``. This module remaps the
official keys to local keys and loads only the backbone (stem, stage1..4, and
transitions). The pose head remains Kaiming-initialized.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


DEFAULT_HRNET_IMAGENET_CKPT = Path("data/external/pretrained/hrnetv2_w32_imagenet.pth")
_ALTERNATE_CKPT_NAMES: Tuple[str, ...] = (
    "hrnetv2_w32_imagenet_pretrained.pth",
)

_DROP_PREFIXES: Tuple[str, ...] = (
    "incre_modules.",
    "downsamp_modules.",
    "final_layer.",
    "classifier.",
    "last_layer.",
)

_STEM_MAP: Dict[str, str] = {
    "conv1.": "stem.0.",
    "bn1.": "stem.1.",
    "conv2.": "stem.3.",
    "bn2.": "stem.4.",
}

_TRANS_MAP: Dict[str, str] = {
    "transition1.": "trans_12.transitions.",
    "transition2.": "trans_23.transitions.",
    "transition3.": "trans_34.transitions.",
}

_RE_TRANSITION_NESTED = re.compile(
    r"^transition([123])\.(\d+)\.(\d+)\.(\d+)\.(.+)$"
)
_RE_STAGE_BRANCHES = re.compile(r"^stage([234])\.(\d+)\.branches\.(\d+)\.(\d+)\.(.+)$")
_RE_STAGE_FUSE_DOWNSAMPLE = re.compile(
    r"^stage([234])\.(\d+)\.fuse_layers\.(\d+)\.(\d+)\.(\d+)\.(\d+)\.(.+)$"
)
_RE_STAGE_FUSE = re.compile(r"^stage([234])\.(\d+)\.fuse_layers\.(\d+)\.(\d+)\.(.+)$")
_RE_LAYER1 = re.compile(r"^layer1\.(\d+)\.(.+)$")


def _remap_key(key: str) -> str | None:
    """Map one official checkpoint key to the local HRNet key."""
    for prefix in _DROP_PREFIXES:
        if key.startswith(prefix):
            return None

    for official_prefix, local_prefix in _STEM_MAP.items():
        if key.startswith(official_prefix):
            return local_prefix + key[len(official_prefix):]

    if (m := _RE_LAYER1.match(key)) is not None:
        block, suffix = m.group(1), m.group(2)
        return f"stage1.{block}.{suffix}"

    if (m := _RE_TRANSITION_NESTED.match(key)) is not None:
        n = m.group(1)
        branch = m.group(2)
        step = int(m.group(3))
        layer = int(m.group(4))
        suffix = m.group(5)
        trans_name = {"1": "trans_12", "2": "trans_23", "3": "trans_34"}[n]
        local_layer = step * 3 + layer
        return f"{trans_name}.transitions.{branch}.{local_layer}.{suffix}"

    for official_prefix, local_prefix in _TRANS_MAP.items():
        if key.startswith(official_prefix):
            return local_prefix + key[len(official_prefix):]

    if (m := _RE_STAGE_FUSE_DOWNSAMPLE.match(key)) is not None:
        n, mod, i, j, step, layer, suffix = (
            m.group(1),
            m.group(2),
            m.group(3),
            m.group(4),
            int(m.group(5)),
            int(m.group(6)),
            m.group(7),
        )
        local_layer = step * 3 + layer
        return f"stage{n}.modules_list.{mod}.fuse.fuse_layers.{i}.{j}.{local_layer}.{suffix}"

    if (m := _RE_STAGE_FUSE.match(key)) is not None:
        n, mod, i, j, suffix = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        return f"stage{n}.modules_list.{mod}.fuse.fuse_layers.{i}.{j}.{suffix}"

    if (m := _RE_STAGE_BRANCHES.match(key)) is not None:
        n, mod, i, j, suffix = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        return f"stage{n}.modules_list.{mod}.branches.{i}.blocks.{j}.{suffix}"

    return None


def _build_remapped_state_dict(
    official: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    remapped: Dict[str, torch.Tensor] = {}
    dropped: List[str] = []
    unmapped: List[str] = []
    for key, value in official.items():
        if any(key.startswith(prefix) for prefix in _DROP_PREFIXES):
            dropped.append(key)
            continue
        new_key = _remap_key(key)
        if new_key is None:
            unmapped.append(key)
            continue
        remapped[new_key] = value
    return remapped, dropped, unmapped


def _is_unused_final_stage_fuse_key(model: nn.Module, key: str) -> bool:
    """True for final-stage fuse rows this pose forward path never consumes."""
    stage4 = getattr(model, "stage4", None)
    modules_list = getattr(stage4, "modules_list", None)
    try:
        final_idx = len(modules_list) - 1
    except TypeError:
        return False
    if final_idx < 0:
        return False

    prefix = f"stage4.modules_list.{final_idx}.fuse.fuse_layers."
    if not key.startswith(prefix):
        return False

    row = key[len(prefix):].split(".", 1)[0]
    return row.isdigit() and int(row) > 0


def _resolve_checkpoint_path(ckpt_path: str | Path) -> Path:
    path = Path(ckpt_path)
    if path.exists():
        return path

    for name in _ALTERNATE_CKPT_NAMES:
        alt = path.with_name(name)
        if alt.exists():
            print(f"[hrnet-pretrained] using alternate checkpoint filename: {alt}")
            return alt

    raise FileNotFoundError(
        "HRNet ImageNet checkpoint is missing.\n"
        f"Expected: {path}\n"
        "Fix locally with:\n"
        f"  python3 scripts/download_hrnet_imagenet.py --dest {path}\n"
        "If this is an Oscar run, upload it from your laptop with:\n"
        "  bash scripts/push_to_oscar.sh\n"
        "Or manually place the official MSRA HRNet-W32 ImageNet checkpoint at "
        "the expected path."
    )


def load_hrnet_imagenet_backbone(model: nn.Module, ckpt_path: str | Path) -> Dict[str, int]:
    """Load ImageNet HRNet backbone weights into ``model``.

    Loads backbone-only with ``strict=False``. Head parameters stay at their
    random initialization. Hard-fails if more than five backbone keys end up
    missing after key remapping, excluding unused final-stage fuse rows.
    """
    resolved = _resolve_checkpoint_path(ckpt_path)
    try:
        raw = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(resolved, map_location="cpu")

    if isinstance(raw, dict) and "state_dict" in raw:
        official = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        official = raw["model"]
    else:
        official = raw

    if any(key.startswith("module.") for key in official.keys()):
        official = {
            key[len("module."):] if key.startswith("module.") else key: value
            for key, value in official.items()
        }

    remapped, dropped, unmapped = _build_remapped_state_dict(official)
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    backbone_missing_all = [key for key in missing if not key.startswith("head.")]
    tolerated_missing = [
        key for key in backbone_missing_all if _is_unused_final_stage_fuse_key(model, key)
    ]
    backbone_missing = [key for key in backbone_missing_all if key not in tolerated_missing]
    if len(backbone_missing) > 5:
        sample = "\n  ".join(backbone_missing[:20])
        raise RuntimeError(
            f"[hrnet-pretrained] {len(backbone_missing)} backbone keys missing after "
            f"remap (>5 threshold). First 20:\n  {sample}"
        )

    summary = {
        "loaded": len(remapped),
        "dropped_classifier": len(dropped),
        "unmapped_official": len(unmapped),
        "missing_total": len(missing),
        "missing_backbone": len(backbone_missing),
        "missing_unused_stage4_fuse": len(tolerated_missing),
        "unexpected": len(unexpected),
    }
    print(
        f"[hrnet-pretrained] loaded={summary['loaded']} "
        f"dropped_classifier={summary['dropped_classifier']} "
        f"unmapped_official={summary['unmapped_official']} "
        f"missing_total={summary['missing_total']} "
        f"missing_backbone={summary['missing_backbone']} "
        f"missing_unused_stage4_fuse={summary['missing_unused_stage4_fuse']} "
        f"unexpected={summary['unexpected']}"
    )
    if unmapped:
        print(f"[hrnet-pretrained] unmapped sample: {unmapped[:5]}")
    return summary
