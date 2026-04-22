"""
scripts/build_dataset.py — Build train/val/test .npz files from AIST++ keypoints.
Owner: Member 3

AIST++ pre-extracted 2D keypoints can be downloaded from:
  https://aistdancedb.ongaaccel.jp/database_split/

The keypoints are stored per-sequence as .pkl files with shape (T, 17, 2) or (T, 17, 3).
This script:
  1. Reads splits.json to determine which choreography IDs go in train/val/test
  2. For each choreography, pairs sequences (one as benchmark, one as learner)
  3. Normalizes + DTW-aligns each pair
  4. Builds diff features + labels and saves as .npz

Usage:
    python scripts/build_dataset.py \
        --aist-dir  data/aist_keypoints/ \
        --splits    data/splits.json \
        --out-train data/train/ \
        --out-val   data/val/ \
        --out-test  data/test/

splits.json format:
    {
      "train": ["gBR_sBM_cAll_d04", "gHO_sBM_cAll_d06", ...],
      "val":   ["gJB_sBM_cAll_d02", ...],
      "test":  ["gKR_sBM_cAll_d01", ...]
    }
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from normalization import normalize
from alignment import dtw_align
from dataset import build_diff_features, build_labels, save_sample


def load_aist_keypoints(path: Path) -> np.ndarray:
    """Load AIST++ keypoints from a .pkl file.

    Expected shape: (T, 17, 2) or (T, 17, 3).
    If shape is (T, 17, 2), a confidence column of 1.0 is appended.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    # AIST++ may store keypoints under different keys; try common ones
    if isinstance(data, dict):
        kp = data.get("keypoints2d", data.get("kps2d", data.get("joints2d")))
        if kp is None:
            raise ValueError(f"No recognized keypoint key in dict from {path}. "
                             f"Keys found: {list(data.keys())}")
    else:
        kp = data

    kp = np.array(kp, dtype=np.float32)
    if kp.ndim == 3 and kp.shape[-1] == 2:
        conf = np.ones((*kp.shape[:2], 1), dtype=np.float32)
        kp = np.concatenate([kp, conf], axis=-1)
    return kp   # (T, 17, 3)


def process_pair(
    bench_kp: np.ndarray,
    learner_kp: np.ndarray,
    out_path: Path,
) -> bool:
    """Normalize, align, extract features/labels, save .npz. Returns True on success."""
    try:
        bench_norm  = normalize(bench_kp)
        learner_norm = normalize(learner_kp)
        bench_al, learner_al, _ = dtw_align(bench_norm, learner_norm)
        features = build_diff_features(bench_al, learner_al)
        labels   = build_labels(bench_al, learner_al)
        save_sample(features, labels, out_path)
        return True
    except Exception as exc:
        print(f"  [WARN] Failed: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AIST++ train/val/test dataset")
    parser.add_argument("--aist-dir",  required=True, help="Dir with AIST++ .pkl files")
    parser.add_argument("--splits",    required=True, help="Path to splits.json")
    parser.add_argument("--out-train", required=True)
    parser.add_argument("--out-val",   required=True)
    parser.add_argument("--out-test",  required=True)
    parser.add_argument("--fps",       type=float, default=15.0)
    args = parser.parse_args()

    aist_dir = Path(args.aist_dir)
    splits   = json.loads(Path(args.splits).read_text())
    out_dirs = {"train": Path(args.out_train),
                "val":   Path(args.out_val),
                "test":  Path(args.out_test)}

    for split_name, choreo_ids in splits.items():
        out_dir = out_dirs[split_name]
        out_dir.mkdir(parents=True, exist_ok=True)
        n_ok = n_fail = 0

        for choreo_id in choreo_ids:
            # Find all sequences with this choreography ID prefix
            seq_files = sorted(aist_dir.glob(f"{choreo_id}*.pkl"))
            if len(seq_files) < 2:
                print(f"[{split_name}] Skipping {choreo_id}: fewer than 2 sequences")
                continue

            # Use first sequence as benchmark; pair with each other sequence
            bench_path = seq_files[0]
            bench_kp   = load_aist_keypoints(bench_path)

            for learner_path in seq_files[1:]:
                pair_id  = f"{bench_path.stem}_vs_{learner_path.stem}"
                out_path = out_dir / f"{pair_id}.npz"
                if out_path.exists():
                    n_ok += 1
                    continue

                learner_kp = load_aist_keypoints(learner_path)
                ok = process_pair(bench_kp, learner_kp, out_path)
                if ok:
                    n_ok += 1
                    print(f"  [{split_name}] Saved: {pair_id}")
                else:
                    n_fail += 1

        print(f"[{split_name}] Done — {n_ok} pairs saved, {n_fail} failed.")

    print("\nbuild_dataset.py complete.")


if __name__ == "__main__":
    main()
