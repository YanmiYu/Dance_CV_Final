"""Poster figure: one benchmark vs. several user clips, on shared axes,
to show that the old GNN embedding distinguishes choreography levels.

Curves:
  A. same choreography                   (highest similarity)
  B. same genre, different choreography  (middle)
  C. different genre                     (lowest)

If no different-genre PKL exists locally, the script still produces the
figure with A and B only, and prints a warning.

Saves PNG and PDF to ``data/reports/`` by default.

Usage:
    python -m scripts.plot_old_gnn_multi_similarity
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.compare.dtw_align import DTWConfig, dtw_align
from src.compare.embedding_features import (
    encode_pose_sequence,
    load_pose_gnn_encoder,
)
from src.compare.normalize_pose import NormalizeConfig, normalize_sequence
from src.infer.temporal_smooth import SmoothConfig, smooth_sequence
from src.utils.video import ffprobe_meta


# ----- defaults --------------------------------------------------------------

DEFAULT_BENCH_PKL = "data/labels/aistpp/keypoints2d_raw/gBR_sBM_c01_d04_mBR0_ch04.pkl"
DEFAULT_BENCH_VIDEO = "data/videos/gBR_sBM_c01_d04_mBR0_ch04.mp4"
DEFAULT_GNN_CKPT = "checkpoints/pose_gnn_encoder_oscar.pt"

DEFAULT_SAME_CHORE_PKL = "data/labels/aistpp/keypoints2d_raw/gBR_sBM_c01_d04_mBR1_ch04.pkl"
DEFAULT_SAME_GENRE_PKL = "data/labels/aistpp/keypoints2d_raw/gBR_sBM_c01_d04_mBR1_ch05.pkl"

# Genre preference order for the "different genre" curve.
DIFF_GENRE_DIRS = [
    "data/labels/aistpp/keypoints2d_raw",
    "data/keypoints2d",
]
DIFF_GENRE_PREFIXES = ["gHO_sBM_c01", "gPO_sBM_c01", "gKR_sBM_c01"]


# ----- helpers (self-contained) ---------------------------------------------

def _load_pkl_poses(path: str | Path) -> np.ndarray:
    with open(path, "rb") as f:
        d = pickle.load(f)
    kp = d["keypoints2d"] if isinstance(d, dict) and "keypoints2d" in d else d
    arr = np.asarray(kp, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (17, 3):
        raise ValueError(f"{path}: expected (T, 17, 3); got {arr.shape}")
    if not np.isfinite(arr).all():
        arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32, copy=False)
    return arr


def _aligned_cosine_curve(
    bench_pkl: str,
    user_pkl: str,
    bench_video: Optional[str],
    gnn_checkpoint: str,
    gnn_device: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return ``(aligned_seconds, cosine_per_step, fps)``."""
    bench_raw = _load_pkl_poses(bench_pkl)
    user_raw = _load_pkl_poses(user_pkl)

    bench_smooth = smooth_sequence(bench_raw, SmoothConfig())
    user_smooth = smooth_sequence(user_raw, SmoothConfig())

    norm_cfg = NormalizeConfig(min_visibility=0.2)
    bench_norm, _ = normalize_sequence(bench_smooth, norm_cfg)
    user_norm, _ = normalize_sequence(user_smooth, norm_cfg)

    model, device = load_pose_gnn_encoder(gnn_checkpoint, device=gnn_device)
    bench_emb = encode_pose_sequence(model, bench_norm, device=device)
    user_emb = encode_pose_sequence(model, user_norm, device=device)

    dtw = dtw_align(bench_emb, user_emb, DTWConfig(band_ratio=0.15, warp_penalty=0.05))
    a_idx = np.asarray(dtw.aligned_a_idx, dtype=np.int64)
    b_idx = np.asarray(dtw.aligned_b_idx, dtype=np.int64)
    L = int(min(a_idx.shape[0], b_idx.shape[0]))
    a_idx, b_idx = a_idx[:L], b_idx[:L]

    a = bench_emb[a_idx]
    b = user_emb[b_idx]
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    cos = np.sum(a * b, axis=1).astype(np.float32)

    fps = 30.0
    if bench_video and Path(bench_video).exists():
        meta = ffprobe_meta(str(bench_video))
        if getattr(meta, "fps", None):
            fps = float(meta.fps)
    aligned_seconds = a_idx.astype(np.float32) / max(fps, 1e-6)
    return aligned_seconds, cos, fps


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x.astype(np.float32, copy=True)
    w = int(min(window, x.size))
    kernel = np.ones(w, dtype=np.float32) / float(w)
    pad = w // 2
    padded = np.pad(x.astype(np.float32), (pad, w - pad - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _find_diff_genre_pkl() -> Optional[str]:
    for d in DIFF_GENRE_DIRS:
        p = Path(d)
        if not p.exists():
            continue
        for prefix in DIFF_GENRE_PREFIXES:
            for f in sorted(p.glob(f"{prefix}*.pkl")):
                # Skip 'cAll' multi-camera files -- those need slicing.
                if "_cAll_" in f.name:
                    continue
                return str(f)
    return None


# ----- plotting --------------------------------------------------------------

def _plot(
    curves: List[Tuple[str, np.ndarray, np.ndarray, dict]],
    out_stem: Path,
    *,
    title: str,
    threshold: float = 0.8,
    smooth_window: int = 9,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.set_facecolor("white")

    x_max = 0.0
    for label, times, cos, style in curves:
        smooth = _moving_average(cos, smooth_window)
        ax.plot(
            times,
            smooth,
            color=style["color"],
            linewidth=style.get("linewidth", 2.4),
            linestyle=style.get("linestyle", "-"),
            alpha=style.get("alpha", 1.0),
            label=label,
            zorder=style.get("zorder", 2),
        )
        if times.size:
            x_max = max(x_max, float(times[-1]))

    if threshold is not None:
        ax.axhline(threshold, color="#7f8c8d", linewidth=1.0,
                   linestyle="--", alpha=0.7)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.0, x_max if x_max > 0 else 1.0)
    ax.set_xlabel("Aligned time (s)")
    ax.set_ylabel("Embedding cosine similarity")
    ax.set_title(title, pad=14, weight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92,
              borderpad=0.6, handlelength=2.4)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=220)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


# ----- entry point -----------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Old-GNN: multi-pair embedding similarity over aligned time.")
    p.add_argument("--bench-pkl", default=DEFAULT_BENCH_PKL)
    p.add_argument("--bench-video", default=DEFAULT_BENCH_VIDEO)
    p.add_argument("--same-chore-pkl", default=DEFAULT_SAME_CHORE_PKL)
    p.add_argument("--same-genre-pkl", default=DEFAULT_SAME_GENRE_PKL)
    p.add_argument("--diff-genre-pkl", default=None,
                   help="Optional explicit different-genre PKL. If omitted, "
                        "auto-search gHO/gPO/gKR.")
    p.add_argument("--gnn-checkpoint", default=DEFAULT_GNN_CKPT)
    p.add_argument("--gnn-device", default="cpu",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--out", default="data/reports/poster_old_gnn_multi_similarity")
    p.add_argument("--title", default="Embedding Similarity Over Aligned Time")
    p.add_argument("--smooth-window", type=int, default=9)
    p.add_argument("--threshold", type=float, default=0.8)
    args = p.parse_args()

    diff_genre_pkl = args.diff_genre_pkl or _find_diff_genre_pkl()

    pairs: List[Tuple[str, str, dict]] = [
        (
            "Same choreography",
            args.same_chore_pkl,
            {"color": "#1f3b73", "linewidth": 3.0, "zorder": 4},
        ),
        (
            "Same genre, different choreography",
            args.same_genre_pkl,
            {"color": "#e69a00", "linewidth": 2.2, "zorder": 3},
        ),
    ]
    if diff_genre_pkl is not None:
        pairs.append((
            "Different genre",
            diff_genre_pkl,
            {"color": "#c0392b", "linewidth": 2.0, "linestyle": "--", "zorder": 2},
        ))
    else:
        print("[warn] no different-genre PKL found locally "
              "(searched gHO_sBM_c01*, gPO_sBM_c01*, gKR_sBM_c01* under "
              "data/labels/aistpp/keypoints2d_raw and data/keypoints2d). "
              "Producing the figure with same-choreography and same-genre "
              "curves only.")

    print(f"[pairs] benchmark: {args.bench_pkl}")
    for label, path, _style in pairs:
        print(f"  - {label}: {path}")

    curves: List[Tuple[str, np.ndarray, np.ndarray, dict]] = []
    for label, path, style in pairs:
        times, cos, fps = _aligned_cosine_curve(
            args.bench_pkl, path, args.bench_video,
            args.gnn_checkpoint, args.gnn_device,
        )
        print(f"  -> {label}: {len(cos)} aligned steps, "
              f"mean cos {float(cos.mean()):.3f}")
        curves.append((label, times, cos, style))

    _plot(
        curves, Path(args.out),
        title=args.title,
        threshold=(args.threshold if args.threshold >= 0 else None),
        smooth_window=args.smooth_window,
    )
    print(f"wrote {args.out}.png and {args.out}.pdf")


if __name__ == "__main__":
    main()
