"""
scripts/build_dataset.py — Build train/val/test .npz pairs from extracted keypoints.

Reads .npy keypoint files produced by `main.py extract_all` on the videos in
`data/videos/`.  Filenames follow the AIST Dance DB convention:

    gBR_sBM_c01_d04_mBR0_ch01.npy
                    ^^^  ^^^^  ^^^^
                    dancer music choreo

Pairing rule
------------
Dancer d04 is always the benchmark.
Dancers d05 and d06 are learners.
A pair is valid when benchmark and learner share the same music and choreo codes.

  d04 vs d05 — music codes: mBR0, mBR1
  d04 vs d06 — music codes: mBR2, mBR3

Train / val / test split  (by choreography number)
---------------------------------------------------
  ch01–ch07  →  train
  ch08–ch09  →  val
  ch10       →  test

Usage
-----
    python scripts/build_dataset.py \
        --kp-dir    data/keypoints/ \
        --out-train data/train/ \
        --out-val   data/val/ \
        --out-test  data/test/
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from normalization import normalize
from alignment import dtw_align
from dataset import build_diff_features, build_labels, save_sample

BENCHMARK_DANCER = "d04"
LEARNER_DANCERS  = {"d05", "d06"}

# Choreography number → split name
def choreo_split(choreo: str) -> str:
    n = int(choreo[2:])   # "ch07" → 7
    if n <= 7:
        return "train"
    if n <= 9:
        return "val"
    return "test"


def parse_stem(stem: str) -> dict | None:
    """Parse 'gBR_sBM_c01_d04_mBR0_ch01' into its components.

    Returns None if the filename doesn't match the expected pattern.
    """
    m = re.match(r"^(.+)_(d\d+)_(m\w+)_(ch\d+)$", stem)
    if not m:
        return None
    return {
        "prefix": m.group(1),   # gBR_sBM_c01
        "dancer": m.group(2),   # d04
        "music":  m.group(3),   # mBR0
        "choreo": m.group(4),   # ch01
    }


def process_pair(bench_kp: np.ndarray, learner_kp: np.ndarray, out_path: Path) -> bool:
    """Normalize, align, build features/labels, save .npz. Returns True on success."""
    try:
        bench_al, learner_al, _ = dtw_align(normalize(bench_kp), normalize(learner_kp))
        save_sample(build_diff_features(bench_al, learner_al),
                    build_labels(bench_al, learner_al),
                    out_path)
        return True
    except Exception as exc:
        print(f"  [WARN] {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AIST train/val/test .npz dataset")
    parser.add_argument("--kp-dir",    required=True, dest="kp_dir",
                        help="Directory of .npy keypoint files (data/keypoints/)")
    parser.add_argument("--out-train", required=True, dest="out_train")
    parser.add_argument("--out-val",   required=True, dest="out_val")
    parser.add_argument("--out-test",  required=True, dest="out_test")
    args = parser.parse_args()

    kp_dir  = Path(args.kp_dir)
    out_dirs = {
        "train": Path(args.out_train),
        "val":   Path(args.out_val),
        "test":  Path(args.out_test),
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Index all .npy files by (prefix, music, choreo) → {dancer: path}
    index: dict[tuple, dict[str, Path]] = defaultdict(dict)
    for npy in sorted(kp_dir.glob("*.npy")):
        info = parse_stem(npy.stem)
        if info is None:
            print(f"[build] Skipping unrecognised filename: {npy.name}")
            continue
        key = (info["prefix"], info["music"], info["choreo"])
        index[key][info["dancer"]] = npy

    counts = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}  # [ok, fail]

    for key, dancers in sorted(index.items()):
        if BENCHMARK_DANCER not in dancers:
            continue
        bench_path = dancers[BENCHMARK_DANCER]
        prefix, music, choreo = key
        split = choreo_split(choreo)

        for learner_dancer in LEARNER_DANCERS:
            if learner_dancer not in dancers:
                continue

            learner_path = dancers[learner_dancer]
            pair_id  = f"{bench_path.stem}_vs_{learner_path.stem}"
            out_path = out_dirs[split] / f"{pair_id}.npz"

            if out_path.exists():
                counts[split][0] += 1
                continue

            bench_kp   = np.load(str(bench_path))
            learner_kp = np.load(str(learner_path))
            ok = process_pair(bench_kp, learner_kp, out_path)

            if ok:
                counts[split][0] += 1
                print(f"  [{split}] {pair_id}")
            else:
                counts[split][1] += 1

    print("\n=== Summary ===")
    for split, (ok, fail) in counts.items():
        print(f"  {split:5s}  {ok} saved, {fail} failed")
    print("build_dataset.py complete.")


if __name__ == "__main__":
    main()
