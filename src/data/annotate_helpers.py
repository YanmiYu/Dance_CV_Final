"""Phase 2.2: schema-tolerant filename parser for AIST-style names and friends.

The key design rule: MISSING FIELDS SHOULD BE ``None``, NOT AN EXCEPTION.
See ``docs/project_decisions.md`` section 6.
"""
from __future__ import annotations

import re
from typing import Dict, Optional


def parse_aist_filename(filename: str, regex: Optional[re.Pattern] = None) -> Dict[str, Optional[str]]:
    """Parse an AIST-style filename best-effort.

    Returns a dict with keys ``genre``, ``situation``, ``camera``, ``dancer``,
    ``music``, ``chore`` and a boolean ``parse_ok``.

    If ``regex`` is None we use a permissive default.
    """
    default_regex = re.compile(
        r"^g(?P<genre>[A-Z0-9]+)"
        r"_s(?P<situation>[A-Z0-9]+)"
        r"_c(?P<camera>\d+)"
        r"_d(?P<dancer>\d+)"
        r"_m(?P<music>[A-Z0-9]+)"
        r"_ch(?P<chore>\d+)\.mp4$"
    )
    pat = regex or default_regex
    m = pat.match(filename)
    if not m:
        return {
            "genre": None,
            "situation": None,
            "camera": None,
            "dancer": None,
            "music": None,
            "chore": None,
            "parse_ok": False,
        }
    gd = m.groupdict()
    return {**gd, "parse_ok": True}


def parse_pair_filename(filename: str) -> Dict[str, Optional[str]]:
    """Parse our own naming convention (see docs/recording_protocol.md):

    ``<song_id>_<phrase_id>_<role>_<take>_<cam>.mp4``
    """
    stem = filename
    if stem.lower().endswith(".mp4"):
        stem = stem[:-4]
    parts = stem.split("_")
    if len(parts) < 5:
        return {
            "song_id": None, "phrase_id": None, "role": None,
            "take": None, "camera_id": None, "parse_ok": False,
        }
    song_id = "_".join(parts[:-4])
    phrase_id, role, take, camera_id = parts[-4:]
    return {
        "song_id": song_id,
        "phrase_id": phrase_id,
        "role": role,
        "take": take,
        "camera_id": camera_id,
        "parse_ok": bool(song_id),
    }
