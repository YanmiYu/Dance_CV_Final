"""Index builder for AIST++ Basic Dance (sBM) keypoint PKLs.

The training set lives in two possible layouts:

  * ``data/keypoints2d/``                  — consolidated ``cAll`` files. Each
    PKL stores ``keypoints2d`` of shape ``(num_cameras, T, 17, 3)``. We
    expand each file into one logical row per camera (c01..cN).
  * ``data/labels/aistpp/keypoints2d_raw/`` — single-camera files of shape
    ``(T, 17, 3)``. Each file becomes one row.

Filename grammar (Basic Dance):

    g<GENRE>_s<SITUATION>_c<CAMERA>_d<DANCER>_m<MUSIC>_ch<CHORE>.pkl

Where ``CAMERA`` may be ``All`` or ``cAll`` for consolidated multi-camera files.

Output CSV columns (order is stable):

    path, stem, genre, situation, camera, dancer, music_id,
    choreography_id, dance_label, genre_label, camera_index, num_frames
"""
from __future__ import annotations

import csv
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence


_AIST_PKL_RE = re.compile(
    r"^g(?P<genre>[A-Z0-9]+)"
    r"_s(?P<situation>[A-Z0-9]+)"
    r"_c(?P<camera>(?:\d+|All|all))"
    r"_d(?P<dancer>\d+)"
    r"_m(?P<music>[A-Z0-9]+)"
    r"_ch(?P<chore>\d+)\.pkl$"
)


@dataclass
class BasicDanceRecord:
    path: str
    stem: str
    genre: str           # e.g. "gBR"
    situation: str       # e.g. "sBM"
    camera: str          # e.g. "c01" (logical camera; expanded from cAll)
    dancer: str          # e.g. "d04"
    music_id: str        # e.g. "mBR0"
    choreography_id: str # e.g. "ch04"
    dance_label: str     # genre + "_" + music_id + "_" + choreography_id
    genre_label: str     # genre
    camera_index: int    # 0-based index into the cAll axis (0 for already-single)
    num_frames: int

    def to_row(self) -> dict:
        return asdict(self)


CSV_COLUMNS: Sequence[str] = (
    "path",
    "stem",
    "genre",
    "situation",
    "camera",
    "dancer",
    "music_id",
    "choreography_id",
    "dance_label",
    "genre_label",
    "camera_index",
    "num_frames",
)


def parse_basic_dance_filename(filename: str) -> Optional[dict]:
    """Parse an AIST++ Basic Dance PKL filename.

    Returns ``None`` if the name does not match. On success the dict has the
    raw fields ``genre`` (e.g. ``"BR"``), ``situation`` (``"BM"``), ``camera``
    (digits or ``"All"``), ``dancer``, ``music`` and ``chore``.
    """
    m = _AIST_PKL_RE.match(Path(filename).name)
    if not m:
        return None
    return m.groupdict()


def _is_consolidated_camera(camera_field: str) -> bool:
    return camera_field.lower() == "all"


def _read_pkl_shape(pkl_path: Path) -> tuple:
    """Return the shape of the ``keypoints2d`` array in a Basic Dance PKL.

    Supports both consolidated ``(C, T, 17, 3)`` and single ``(T, 17, 3)``.
    """
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "keypoints2d" in data:
        kp = data["keypoints2d"]
    else:
        kp = data
    if not hasattr(kp, "shape"):
        raise ValueError(f"{pkl_path}: keypoints2d has no shape attribute")
    return tuple(kp.shape)


def iter_basic_dance_records(
    pkl_root: Path | str,
    *,
    situation: str = "sBM",
    use_cameras: Optional[Sequence[str]] = None,
) -> Iterator[BasicDanceRecord]:
    """Yield one :class:`BasicDanceRecord` per (file, camera_index).

    Args:
        pkl_root: directory containing the PKL files.
        situation: keep files whose situation field equals this value
            (``"sBM"`` for Basic Dance). Pass ``""`` to keep everything.
        use_cameras: when set, restrict expanded ``cAll`` rows to these
            camera names (e.g. ``["c01", "c02"]``). For non-consolidated
            files the filename camera must be in the list.
    """
    pkl_root = Path(pkl_root)
    if not pkl_root.exists():
        raise FileNotFoundError(pkl_root)
    keep_cams = set(use_cameras) if use_cameras else None

    for pkl in sorted(pkl_root.glob("*.pkl")):
        parsed = parse_basic_dance_filename(pkl.name)
        if parsed is None:
            continue
        if situation and ("s" + parsed["situation"]) != situation:
            continue

        try:
            shape = _read_pkl_shape(pkl)
        except Exception:
            continue

        # Resolve camera axis vs. single-camera layouts.
        if _is_consolidated_camera(parsed["camera"]):
            if len(shape) != 4 or shape[2:] != (17, 3):
                # Unexpected layout -- skip rather than crash.
                continue
            num_cams, num_frames, _, _ = shape
            cam_iter = range(num_cams)
        else:
            if len(shape) != 3 or shape[1:] != (17, 3):
                continue
            num_frames = shape[0]
            cam_iter = (0,)  # single logical camera

        for cam_idx in cam_iter:
            if _is_consolidated_camera(parsed["camera"]):
                cam_name = f"c{cam_idx + 1:02d}"
            else:
                cam_name = "c" + parsed["camera"]
            if keep_cams is not None and cam_name not in keep_cams:
                continue

            genre_label = "g" + parsed["genre"]
            music_id = "m" + parsed["music"]
            chore_id = "ch" + parsed["chore"]
            dance_label = f"{genre_label}_{music_id}_{chore_id}"

            yield BasicDanceRecord(
                path=str(pkl),
                stem=pkl.stem,
                genre=genre_label,
                situation="s" + parsed["situation"],
                camera=cam_name,
                dancer="d" + parsed["dancer"],
                music_id=music_id,
                choreography_id=chore_id,
                dance_label=dance_label,
                genre_label=genre_label,
                camera_index=int(cam_idx),
                num_frames=int(num_frames),
            )


def build_index_csv(
    pkl_root: Path | str,
    out_csv: Path | str,
    *,
    situation: str = "sBM",
    use_cameras: Optional[Sequence[str]] = None,
) -> int:
    """Write the index CSV; return the number of rows written."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = list(iter_basic_dance_records(
        pkl_root, situation=situation, use_cameras=use_cameras
    ))
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_row())
    return len(rows)


def load_index_csv(csv_path: Path | str) -> List[dict]:
    """Read the index CSV back as a list of dicts."""
    csv_path = Path(csv_path)
    out: List[dict] = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["camera_index"] = int(row.get("camera_index") or 0)
            row["num_frames"] = int(row.get("num_frames") or 0)
            out.append(row)
    return out


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build the Basic Dance embedding index CSV.")
    p.add_argument(
        "--pkl-root",
        default="data/keypoints2d",
        help="Directory of AIST++ Basic Dance PKL files.",
    )
    p.add_argument(
        "--out-csv",
        default="data/processed/basic_dance_embedding_index.csv",
    )
    p.add_argument("--situation", default="sBM")
    p.add_argument(
        "--use-cameras",
        nargs="*",
        default=None,
        help="Optional whitelist, e.g. c01 c02 c03 ...",
    )
    args = p.parse_args()
    n = build_index_csv(
        args.pkl_root, args.out_csv,
        situation=args.situation,
        use_cameras=args.use_cameras,
    )
    print(f"wrote {n} rows to {args.out_csv}")


if __name__ == "__main__":
    _main()
