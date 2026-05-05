"""Index builder for AIST++ Basic Dance keypoint PKLs.

The primary training layout is ``data/keypoints2d/`` with consolidated
``cAll`` files. A consolidated PKL stores ``keypoints2d`` as
``(num_cameras, T, 17, 3)`` and is expanded into one logical row per camera.
Single-camera PKLs shaped ``(T, 17, 3)`` are also supported.

Filename grammar:

    g<GENRE>_s<SITUATION>_c<CAMERA>_d<DANCER>_m<MUSIC>_ch<CHORE>.pkl

Output CSV columns:

    path, stem, genre, situation, camera, dancer, music_id,
    choreography_id, dance_label, genre_label, camera_index, num_frames
"""
from __future__ import annotations

import csv
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence


_AIST_PKL_RE = re.compile(
    r"^g(?P<genre>[A-Z0-9]+)"
    r"_s(?P<situation>[A-Z0-9]+)"
    r"_c(?P<camera>(?:\d+|All|all))"
    r"_d(?P<dancer>\d+)"
    r"_m(?P<music>[A-Z0-9]+)"
    r"_ch(?P<chore>\d+)\.pkl$"
)


@dataclass(frozen=True)
class BasicDanceRecord:
    path: str
    stem: str
    genre: str
    situation: str
    camera: str
    dancer: str
    music_id: str
    choreography_id: str
    dance_label: str
    genre_label: str
    camera_index: int
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

    Returns raw fields without their leading prefixes, except ``camera`` which
    is either digits or ``All``.
    """
    m = _AIST_PKL_RE.match(Path(filename).name)
    if not m:
        return None
    return m.groupdict()


def _is_consolidated_camera(camera_field: str) -> bool:
    return str(camera_field).lower() == "all"


def _prefixed(value: str, prefix: str) -> str:
    value = str(value)
    return value if value.startswith(prefix) else f"{prefix}{value}"


def _normalize_cameras(use_cameras: Optional[Sequence[str]]) -> Optional[set[str]]:
    if use_cameras is None:
        return None
    cams = set()
    for cam in use_cameras:
        text = str(cam)
        cams.add(text if text.startswith("c") else f"c{text}")
    return cams


def _normalize_genres(genres: Optional[Sequence[str] | str]) -> Optional[set[str]]:
    if genres is None:
        return None
    if isinstance(genres, str):
        if genres.lower() == "all":
            return None
        genres = [genres]
    out = set()
    for genre in genres:
        text = str(genre)
        if text.lower() == "all":
            return None
        out.add(text if text.startswith("g") else f"g{text}")
    return out


def _read_pkl_shape(pkl_path: Path) -> tuple[int, ...]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    kp = data["keypoints2d"] if isinstance(data, dict) and "keypoints2d" in data else data
    if not hasattr(kp, "shape"):
        raise ValueError(f"{pkl_path}: keypoints2d has no shape")
    return tuple(int(v) for v in kp.shape)


def iter_basic_dance_records(
    pkl_root: Path | str,
    *,
    situation: str = "sBM",
    use_cameras: Optional[Sequence[str]] = None,
    genres: Optional[Sequence[str] | str] = None,
) -> Iterator[BasicDanceRecord]:
    """Yield one record per logical file/camera row.

    ``genres=None`` and ``genres="all"`` both mean all discovered genres.
    ``use_cameras=["c01"]`` keeps only c01 after ``cAll`` expansion.
    """
    pkl_root = Path(pkl_root)
    if not pkl_root.exists():
        raise FileNotFoundError(pkl_root)

    keep_cams = _normalize_cameras(use_cameras)
    keep_genres = _normalize_genres(genres)

    for pkl in sorted(pkl_root.glob("*.pkl")):
        parsed = parse_basic_dance_filename(pkl.name)
        if parsed is None:
            continue

        situation_label = _prefixed(parsed["situation"], "s")
        if situation and situation_label != situation:
            continue

        genre_label = _prefixed(parsed["genre"], "g")
        if keep_genres is not None and genre_label not in keep_genres:
            continue

        try:
            shape = _read_pkl_shape(pkl)
        except Exception:
            continue

        if _is_consolidated_camera(parsed["camera"]):
            if len(shape) != 4 or shape[2:] != (17, 3):
                continue
            num_cams, num_frames = shape[0], shape[1]
            camera_indices = range(num_cams)
        else:
            if len(shape) != 3 or shape[1:] != (17, 3):
                continue
            num_frames = shape[0]
            camera_indices = (0,)

        for cam_idx in camera_indices:
            if _is_consolidated_camera(parsed["camera"]):
                camera = f"c{cam_idx + 1:02d}"
            else:
                camera = _prefixed(parsed["camera"], "c")
            if keep_cams is not None and camera not in keep_cams:
                continue

            music_id = _prefixed(parsed["music"], "m")
            choreography_id = _prefixed(parsed["chore"], "ch")
            dance_label = f"{genre_label}_{music_id}_{choreography_id}"

            yield BasicDanceRecord(
                path=str(pkl),
                stem=pkl.stem,
                genre=genre_label,
                situation=situation_label,
                camera=camera,
                dancer=_prefixed(parsed["dancer"], "d"),
                music_id=music_id,
                choreography_id=choreography_id,
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
    genres: Optional[Sequence[str] | str] = None,
) -> int:
    """Write the index CSV and return the number of rows."""
    rows = list(
        iter_basic_dance_records(
            pkl_root,
            situation=situation,
            use_cameras=use_cameras,
            genres=genres,
        )
    )
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())
    return len(rows)


def load_index_csv(csv_path: Path | str) -> List[dict]:
    out: List[dict] = []
    with Path(csv_path).open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["camera_index"] = int(row.get("camera_index") or 0)
            row["num_frames"] = int(row.get("num_frames") or 0)
            out.append(row)
    return out


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build a Basic Dance embedding index CSV.")
    p.add_argument("--pkl-root", default="data/keypoints2d")
    p.add_argument("--out-csv", default="data/processed/basic_dance_embedding_index.csv")
    p.add_argument("--situation", default="sBM")
    p.add_argument("--use-cameras", nargs="*", default=None)
    p.add_argument(
        "--genres",
        nargs="*",
        default=None,
        help='Optional genre whitelist, e.g. gBR gPO. Omit or pass "all" for all genres.',
    )
    args = p.parse_args()
    n = build_index_csv(
        args.pkl_root,
        args.out_csv,
        situation=args.situation,
        use_cameras=args.use_cameras,
        genres=args.genres,
    )
    print(f"wrote {n} rows to {args.out_csv}")


if __name__ == "__main__":
    _main()
