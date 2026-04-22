"""Config loading + shallow merge utilities.

We keep this intentionally simple: YAML files are the single source of truth
(see ``configs/`` and ``docs/project_decisions.md``). No magic substitution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(data)}")
    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive merge; ``override`` wins leaf-by-leaf."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def require(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing required config key: {key!r}")
    return cfg[key]
