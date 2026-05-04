"""Load ImageNet-pretrained HRNet backbone weights into our custom HRNetPose.

The official MSRA HRNet checkpoint uses different submodule names than our
local implementation in ``src/models/hrnet.py``. This module remaps the
official keys to ours and loads only the BACKBONE (stem, stage1..4,
transitions). The pose head stays Kaiming-initialized.

See ``docs/project_decisions.md`` section 1 (2026-05-04 revision).
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


# Prefixes in the official ckpt that belong to the ImageNet classification
# head (and the v1/v2 fusion-then-classify path) — we never load these.
_DROP_PREFIXES: Tuple[str, ...] = (
    "incre_modules.",
    "downsamp_modules.",
    "final_layer.",
    "classifier.",
    "last_layer.",  # some HRNetV2 segmentation variants
)

# Stem mapping: official conv1/bn1/conv2/bn2 -> our nn.Sequential indices.
_STEM_MAP: Dict[str, str] = {
    "conv1.": "stem.0.",
    "bn1.": "stem.1.",
    "conv2.": "stem.3.",
    "bn2.": "stem.4.",
}

# transitionN -> trans_(N)(N+1)
_TRANS_MAP: Dict[str, str] = {
    "transition1.": "trans_12.transitions.",
    "transition2.": "trans_23.transitions.",
    "transition3.": "trans_34.transitions.",
}

# Compiled regexes for the body stages.
# Official transition branches newly introduced at each stage are nested as
# transitionN.<branch>.<downsample_step>.<conv_or_bn>.<param>. Our local
# transition stores the single downsample step as a flat Sequential.
_RE_TRANSITION_NESTED = re.compile(
    r"^transition([123])\.(\d+)\.(\d+)\.(\d+)\.(.+)$"
)
# Official: stage{N}.{m}.branches.{i}.{j}.{...}
# Local:    stage{N}.modules_list.{m}.branches.{i}.blocks.{j}.{...}
_RE_STAGE_BRANCHES = re.compile(r"^stage([234])\.(\d+)\.branches\.(\d+)\.(\d+)\.(.+)$")
# Official downsample fuse paths are nested as
# fuse_layers.<dst>.<src>.<downsample_step>.<conv_or_bn>.<param>. Our local
# fuse path is a flat Sequential with ReLUs between downsample steps.
_RE_STAGE_FUSE_DOWNSAMPLE = re.compile(
    r"^stage([234])\.(\d+)\.fuse_layers\.(\d+)\.(\d+)\.(\d+)\.(\d+)\.(.+)$"
)
# Official: stage{N}.{m}.fuse_layers.{i}.{j}.{...}
# Local:    stage{N}.modules_list.{m}.fuse.fuse_layers.{i}.{j}.{...}
_RE_STAGE_FUSE = re.compile(r"^stage([234])\.(\d+)\.fuse_layers\.(\d+)\.(\d+)\.(.+)$")
# Official stage1: layer1.{k}.{...}
# Local:           stage1.{k}.{...}
_RE_LAYER1 = re.compile(r"^layer1\.(\d+)\.(.+)$")


def _remap_key(key: str) -> str | None:
    """Map one official key to our local key. Return None to drop the key."""
    for p in _DROP_PREFIXES:
        if key.startswith(p):
            return None

    # Stem (4 fixed prefixes; check exact prefix match)
    for off_pref, loc_pref in _STEM_MAP.items():
        if key.startswith(off_pref):
            return loc_pref + key[len(off_pref):]

    # Stage 1 bottlenecks
    if (m := _RE_LAYER1.match(key)) is not None:
        k, suffix = m.group(1), m.group(2)
        return f"stage1.{k}.{suffix}"

    # Transitions
    if (m := _RE_TRANSITION_NESTED.match(key)) is not None:
        n = m.group(1)
        branch = m.group(2)
        step = int(m.group(3))
        layer = int(m.group(4))
        suffix = m.group(5)
        trans_name = {"1": "trans_12", "2": "trans_23", "3": "trans_34"}[n]
        local_layer = step * 3 + layer
        return f"{trans_name}.transitions.{branch}.{local_layer}.{suffix}"

    for off_pref, loc_pref in _TRANS_MAP.items():
        if key.startswith(off_pref):
            return loc_pref + key[len(off_pref):]

    # Stage 2/3/4 fuse_layers (must be checked BEFORE branches because both
    # share the stage{N}.{m}. prefix).
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

    # Stage 2/3/4 branches
    if (m := _RE_STAGE_BRANCHES.match(key)) is not None:
        n, mod, i, j, suffix = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        return f"stage{n}.modules_list.{mod}.branches.{i}.blocks.{j}.{suffix}"

    # Unknown / unmapped — return None to drop with logging upstream.
    return None


def _build_remapped_state_dict(
    official: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    """Return (remapped, dropped_keys, unmapped_keys)."""
    remapped: Dict[str, torch.Tensor] = {}
    dropped: List[str] = []
    unmapped: List[str] = []
    for k, v in official.items():
        if any(k.startswith(p) for p in _DROP_PREFIXES):
            dropped.append(k)
            continue
        new_k = _remap_key(k)
        if new_k is None:
            unmapped.append(k)
            continue
        remapped[new_k] = v
    return remapped, dropped, unmapped


def _is_unused_final_stage_fuse_key(model: nn.Module, key: str) -> bool:
    """Return True for final-stage fuse rows that HRNetPose never consumes.

    The MSRA ImageNet classification checkpoint was trained with the final
    stage's last module emitting only branch 0. Our pose forward also feeds only
    branch 0 to the head, so missing fuse rows for branches 1..N in that last
    module are unused parameters rather than a required initialization gap.
    """
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


def load_hrnet_imagenet_backbone(model: nn.Module, ckpt_path: str) -> Dict[str, int]:
    """Load ImageNet HRNet backbone weights into ``model``.

    Loads backbone-only with ``strict=False``. Head parameters stay at their
    random init. Hard-fails if more than 5 backbone keys end up in `missing`.

    Returns a small summary dict (counts) for logging/tests.
    """
    resolved_ckpt_path = _resolve_checkpoint_path(ckpt_path)
    try:
        raw = torch.load(resolved_ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(resolved_ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        official = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        official = raw["model"]
    else:
        official = raw

    # Some checkpoints prefix keys with "module." (DataParallel) — strip it.
    if any(k.startswith("module.") for k in official.keys()):
        official = {k[len("module."):] if k.startswith("module.") else k: v for k, v in official.items()}

    remapped, dropped, unmapped = _build_remapped_state_dict(official)

    missing, unexpected = model.load_state_dict(remapped, strict=False)

    backbone_missing_all = [k for k in missing if not k.startswith("head.")]
    tolerated_missing = [
        k for k in backbone_missing_all if _is_unused_final_stage_fuse_key(model, k)
    ]
    backbone_missing = [k for k in backbone_missing_all if k not in tolerated_missing]
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
