"""Build Mia LSTM train/val/test ``.npz`` samples from keypoint sequences.

The input directory should contain one keypoint file per video stem, either:

    <stem>.npy
    <stem>.pkl with a ``keypoints2d`` array

Each sequence must be shaped ``(T, 17, 2|3)``. The pairing defaults mirror the
Mia branch: dancer ``d04`` is the benchmark, dancers ``d05`` and ``d06`` are
learners, and choreography ``ch01``-``ch07`` / ``ch08``-``ch09`` / ``ch10`` map
to train / val / test.
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mia.alignment import dtw_align
from src.mia.dataset import build_diff_features, build_labels, save_sample
from src.mia.normalization import normalize


def _load_keypoints(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".pkl":
        with path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            arr = obj.get("keypoints2d")
            if arr is None:
                raise ValueError(f"{path} has no 'keypoints2d' key")
        else:
            arr = obj
    else:
        raise ValueError(f"Unsupported keypoint file extension: {path}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != 17 or arr.shape[2] not in (2, 3):
        raise ValueError(f"Expected (T, 17, 2|3), got {arr.shape} for {path}")
    if arr.shape[2] == 2:
        conf = np.ones((arr.shape[0], 17, 1), dtype=np.float32)
        arr = np.concatenate([arr, conf], axis=2)
    return arr


def parse_stem(stem: str) -> dict[str, str] | None:
    """Parse an AIST-style stem such as ``gBR_sBM_c01_d04_mBR0_ch01``."""
    m = re.match(r"^(.+)_(d\d+)_(m\w+)_(ch\d+)$", stem)
    if not m:
        return None
    return {
        "prefix": m.group(1),
        "dancer": m.group(2),
        "music": m.group(3),
        "choreo": m.group(4),
    }


def choreo_split(choreo: str, train_max: int = 7, val_max: int = 9) -> str:
    n = int(choreo.removeprefix("ch"))
    if n <= train_max:
        return "train"
    if n <= val_max:
        return "val"
    return "test"


def process_pair(bench_path: Path, learner_path: Path, out_path: Path) -> bool:
    try:
        bench = _load_keypoints(bench_path)
        learner = _load_keypoints(learner_path)
        bench_al, learner_al, _ = dtw_align(normalize(bench), normalize(learner))
        features = build_diff_features(bench_al, learner_al)
        labels = build_labels(bench_al, learner_al)
        save_sample(features, labels, out_path)
        return True
    except Exception as exc:
        print(f"  [warn] {bench_path.name} vs {learner_path.name}: {exc}")
        return False


def build_dataset(
    kp_dir: Path,
    out_train: Path,
    out_val: Path,
    out_test: Path,
    benchmark_dancer: str = "d04",
    learner_dancers: tuple[str, ...] = ("d05", "d06"),
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    out_dirs = {"train": out_train, "val": out_val, "test": out_test}
    for out_dir in out_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    index: dict[tuple[str, str, str], dict[str, Path]] = defaultdict(dict)
    for path in sorted(kp_dir.iterdir()):
        if path.suffix not in {".npy", ".pkl"}:
            continue
        info = parse_stem(path.stem)
        if info is None:
            print(f"[build_lstm_dataset] skip unrecognized: {path.name}")
            continue
        key = (info["prefix"], info["music"], info["choreo"])
        index[key][info["dancer"]] = path

    counts = {
        split: {"saved": 0, "skipped_existing": 0, "failed": 0}
        for split in out_dirs
    }
    for key, dancers in sorted(index.items()):
        if benchmark_dancer not in dancers:
            continue
        prefix, music, choreo = key
        split = choreo_split(choreo)
        bench_path = dancers[benchmark_dancer]

        for learner_dancer in learner_dancers:
            learner_path = dancers.get(learner_dancer)
            if learner_path is None:
                continue
            pair_id = f"{bench_path.stem}_vs_{learner_path.stem}"
            out_path = out_dirs[split] / f"{pair_id}.npz"
            if out_path.exists() and not overwrite:
                counts[split]["skipped_existing"] += 1
                continue

            ok = process_pair(bench_path, learner_path, out_path)
            if ok:
                counts[split]["saved"] += 1
                print(f"  [{split}] {prefix}_{music}_{choreo}: {pair_id}")
            else:
                counts[split]["failed"] += 1

    return counts


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build .npz samples for Mia LSTM training")
    p.add_argument("--kp-dir", required=True, type=Path)
    p.add_argument("--out-train", default="data/lstm/train", type=Path)
    p.add_argument("--out-val", default="data/lstm/val", type=Path)
    p.add_argument("--out-test", default="data/lstm/test", type=Path)
    p.add_argument("--benchmark-dancer", default="d04")
    p.add_argument("--learner-dancers", nargs="+", default=["d05", "d06"])
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    counts = build_dataset(
        kp_dir=args.kp_dir,
        out_train=args.out_train,
        out_val=args.out_val,
        out_test=args.out_test,
        benchmark_dancer=args.benchmark_dancer,
        learner_dancers=tuple(args.learner_dancers),
        overwrite=args.overwrite,
    )
    print("")
    print("=== Summary ===")
    for split, split_counts in counts.items():
        print(f"{split:5s} {split_counts}")


if __name__ == "__main__":
    main()
