"""Poster figure: cosine similarity over DTW-aligned time in the GNN
embedding space.

Loads two single-camera AIST++ keypoint PKLs, encodes each frame with the
specified GNN checkpoint, runs DTW in the learned embedding space, then
plots one clean curve of per-aligned-step cosine similarity. Saves PNG and
PDF to ``data/reports/`` by default.

Usage:
    python -m scripts.plot_embedding_similarity_curve \\
        --bench-pkl data/labels/aistpp/keypoints2d_raw/gBR_sBM_c01_d04_mBR0_ch04.pkl \\
        --user-pkl  data/labels/aistpp/keypoints2d_raw/gBR_sBM_c01_d04_mBR1_ch05.pkl \\
        --bench-video data/videos/gBR_sBM_c01_d04_mBR0_ch04.mp4 \\
        --user-video  data/videos/gBR_sBM_c01_d04_mBR1_ch05.mp4 \\
        --gnn-checkpoint checkpoints/pose_gnn_encoder_oscar.pt \\
        --out data/reports/poster_embedding_similarity_curve
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np

from src.compare.dtw_align import DTWConfig, dtw_align
from src.compare.embedding_features import (
    encode_pose_sequence,
    load_pose_gnn_encoder,
)
from src.compare.normalize_pose import NormalizeConfig, normalize_sequence
from src.infer.temporal_smooth import SmoothConfig, smooth_sequence
from src.utils.video import ffprobe_meta


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
    bench_video: str | None,
    gnn_checkpoint: str,
    gnn_device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return ``(aligned_seconds, cosine_per_step, fps)``.

    The DTW alignment runs in embedding space (the same path render_report
    takes when ``--alignment-method gnn_embedding``).
    """
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
    if bench_video:
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


def _plot(
    times: np.ndarray,
    cosine: np.ndarray,
    out_stem: Path,
    *,
    title: str,
    threshold: float = 0.8,
    smooth_window: int = 9,
    divergence_threshold: float = 0.7,
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
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "savefig.facecolor": "white",
    })

    smooth = _moving_average(cosine, smooth_window)

    fig, ax = plt.subplots(figsize=(11, 5.0), constrained_layout=True)
    ax.set_facecolor("white")

    # Shade divergence regions on the smoothed curve so they're stable
    # against single-frame spikes.
    below = smooth < divergence_threshold
    if np.any(below):
        # build contiguous-run spans
        diff = np.diff(below.astype(np.int8), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            ax.axvspan(times[s], times[min(e, len(times) - 1)],
                       color="#e74c3c", alpha=0.10, linewidth=0)

    # Raw similarity (thin, faded).
    ax.plot(times, cosine, color="#9aa6b2", linewidth=1.0, alpha=0.55)
    # Smoothed similarity (the main curve).
    ax.plot(times, smooth, color="#1f3b73", linewidth=2.6)

    if threshold is not None:
        ax.axhline(threshold, color="#7f8c8d", linewidth=1.1,
                   linestyle="--", alpha=0.85)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(times[0], times[-1] if times.size else 1.0)
    ax.set_xlabel("Aligned time (s)")
    ax.set_ylabel("Embedding cosine similarity")
    ax.set_title(title, pad=14, weight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=220)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Poster: embedding similarity over aligned time.")
    p.add_argument("--bench-pkl", required=True)
    p.add_argument("--user-pkl", required=True)
    p.add_argument("--bench-video", default=None,
                   help="Used only to read fps for the x-axis. Optional.")
    p.add_argument("--gnn-checkpoint", required=True)
    p.add_argument("--gnn-device", default="cpu", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--out", default="data/reports/poster_embedding_similarity_curve",
                   help="Output stem; .png and .pdf will be written.")
    p.add_argument("--title", default="Embedding Similarity Over Aligned Time")
    p.add_argument("--smooth-window", type=int, default=9)
    p.add_argument("--threshold", type=float, default=0.8,
                   help="Horizontal dashed reference line; pass <0 to disable.")
    p.add_argument("--divergence-threshold", type=float, default=0.7)
    args = p.parse_args()

    times, cos, fps = _aligned_cosine_curve(
        args.bench_pkl, args.user_pkl, args.bench_video,
        args.gnn_checkpoint, args.gnn_device,
    )
    print(f"aligned steps: {len(cos)}  fps: {fps:.2f}  "
          f"mean cos: {float(cos.mean()):.3f}  min: {float(cos.min()):.3f}")

    _plot(
        times, cos, Path(args.out),
        title=args.title,
        threshold=(args.threshold if args.threshold >= 0 else None),
        smooth_window=args.smooth_window,
        divergence_threshold=args.divergence_threshold,
    )
    print(f"wrote {args.out}.png and {args.out}.pdf")


if __name__ == "__main__":
    main()
