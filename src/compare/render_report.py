"""End-to-end CLI: benchmark_video + user_video -> artifacts in data/reports/.

Writes:
  report.json          aggregated numbers + feedback
  summary.png          bar-chart of per-body-part scores (matplotlib)
  aligned_side.mp4     synced side-by-side with pose overlays (optional)
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.compare.dtw_align import DTWConfig, build_feature_weights, dtw_align
from src.compare.feedback import (
    extract_feedback_intervals,
    generate_feedback,
    severity_sequence_per_part,
)
from src.compare.features import FeatureConfig, extract_features, framewise_distance_vector
from src.compare.normalize_pose import NormalizeConfig, normalize_sequence
from src.compare.score import ScoreConfig, compare_features, score_result_to_dict
from src.datasets.common import BODY_PART_GROUPS, NUM_JOINTS
from src.infer.temporal_smooth import SmoothConfig, smooth_sequence
from src.utils.config import load_yaml
from src.utils.io import ensure_dir
from src.utils.viz import (
    SEVERITY_BGR,
    draw_pose,
    draw_severity_legend,
    draw_skeleton_blank,
    draw_skeleton_overlay,
    side_by_side,
)
from src.utils.video import ffprobe_meta, write_video


def _run_pose_if_needed(
    video_path: str,
    model_config: str,
    ckpt: str,
    out_dir: Path,
) -> Path:
    """Run ``src.infer.run_pose_on_video.run`` and return its out_dir path."""
    if (out_dir / "poses.npy").exists():
        return out_dir
    from src.infer.run_pose_on_video import run as _run

    _run(video_path, model_config, ckpt, str(out_dir))
    return out_dir


def _load_pose_file(path: str | Path) -> np.ndarray:
    """Load a raw AIST-style keypoint file as ``(T, 17, 3)`` float32 poses."""
    path = Path(path)
    if path.suffix == ".npy":
        obj = np.load(path)
    else:
        with path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            if "keypoints2d" not in obj:
                raise ValueError(f"{path} is missing key 'keypoints2d'")
            obj = obj["keypoints2d"]

    poses = np.asarray(obj, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1] != NUM_JOINTS or poses.shape[2] not in (2, 3):
        raise ValueError(f"expected keypoints shape (T, 17, 2|3), got {poses.shape} in {path}")
    if poses.shape[2] == 2:
        conf = np.ones((*poses.shape[:2], 1), dtype=np.float32)
        poses = np.concatenate([poses, conf], axis=-1)
    return poses


def _pose_from_file(video_path: str, pose_path: str | Path, out_dir: Path) -> Path:
    """Materialize supplied keypoints into the same cache layout as pose inference."""
    out_dir = ensure_dir(out_dir)
    poses = _load_pose_file(pose_path)
    np.save(out_dir / "poses.npy", poses)

    video_meta = ffprobe_meta(video_path)
    fps = video_meta.fps if video_meta.ok and video_meta.fps > 0 else 30.0
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "video_path": str(video_path),
                "keypoints_path": str(pose_path),
                "source": "keypoints_file",
                "fps": fps,
                "num_frames": int(poses.shape[0]),
                "width": video_meta.width if video_meta.ok else 0,
                "height": video_meta.height if video_meta.ok else 0,
                "duration_sec": video_meta.duration_sec if video_meta.ok else 0.0,
            },
            indent=2,
        )
    )
    return out_dir


def _resolve_pose_input(
    video_path: str,
    model_config: Optional[str],
    ckpt: Optional[str],
    out_dir: Path,
    pose_file: Optional[str],
) -> Path:
    if pose_file:
        return _pose_from_file(video_path, pose_file, out_dir)
    if not model_config or not ckpt:
        raise ValueError("--model-config and --ckpt are required unless a pose file is supplied")
    return _run_pose_if_needed(video_path, model_config, ckpt, out_dir)


def _render_bar_chart(per_part: Dict[str, float], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(per_part.keys())
    scores = [per_part[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, scores, color="#4c8bf5")
    ax.set_ylim(0, 100)
    ax.set_ylabel("score (0-100)")
    ax.set_title("Per-body-part similarity to benchmark")
    for i, v in enumerate(scores):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _matrix_stats(matrix: np.ndarray) -> Dict[str, float | list]:
    """Small JSON-safe summary for a similarity matrix."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "shape": [int(arr.shape[0]), int(arr.shape[1])],
        }
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
    }


def _render_embedding_similarity_heatmap(
    similarity: np.ndarray,
    out_path: Path,
    *,
    aligned_a_idx: Optional[np.ndarray] = None,
    aligned_b_idx: Optional[np.ndarray] = None,
) -> None:
    """Render benchmark-vs-user embedding cosine similarities as a heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim = np.asarray(similarity, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    if sim.size == 0:
        shown = np.zeros((max(1, sim.shape[0]), max(1, sim.shape[1])), dtype=np.float32)
        image = ax.imshow(
            shown,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.text(
            0.5,
            0.5,
            "No embeddings available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=13,
        )
    else:
        image = ax.imshow(
            sim,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )

    if (
        aligned_a_idx is not None
        and aligned_b_idx is not None
        and len(aligned_a_idx) > 0
        and len(aligned_b_idx) > 0
    ):
        L = min(len(aligned_a_idx), len(aligned_b_idx))
        ax.plot(
            np.asarray(aligned_b_idx[:L], dtype=np.float32),
            np.asarray(aligned_a_idx[:L], dtype=np.float32),
            color="white",
            linewidth=1.0,
            alpha=0.9,
            label="DTW alignment",
        )
        ax.legend(loc="upper left", fontsize=10, framealpha=0.75)

    ax.set_xlabel("User Frames", fontsize=12)
    ax.set_ylabel("Benchmark Frames", fontsize=12)
    ax.set_title("Embedding Similarity Heatmap", fontsize=15)
    ax.tick_params(labelsize=10)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Cosine Similarity", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _render_side_by_side_skeleton(
    bench_poses: np.ndarray,
    user_poses: np.ndarray,
    aligned_a_idx: np.ndarray,
    aligned_b_idx: np.ndarray,
    severity_per_part_seq: Dict[str, list],
    out_path: Path,
    fps: float = 30.0,
    *,
    canvas_w: int = 480,
    canvas_h: int = 720,
) -> None:
    """Render the demo video: two blank panels with skeletons drawn on top.

    The benchmark panel uses a neutral color; the learner panel colors each
    body-part region by severity (green / yellow / red). One output frame
    per DTW step; output fps matches benchmark fps so the timeline aligns
    with the reference.
    """
    aligned_a_idx = np.asarray(aligned_a_idx, dtype=np.int64)
    aligned_b_idx = np.asarray(aligned_b_idx, dtype=np.int64)
    L = int(min(aligned_a_idx.shape[0], aligned_b_idx.shape[0]))
    if L == 0:
        return
    sev_lengths = {len(v) for v in severity_per_part_seq.values()}
    if sev_lengths and sev_lengths != {L}:
        raise ValueError(
            f"severity sequences length mismatch: expected {L}, got {sev_lengths}"
        )

    frames = []
    for k in range(L):
        ai = int(aligned_a_idx[k])
        bi = int(aligned_b_idx[k])
        bench_panel = draw_skeleton_blank(
            (canvas_h, canvas_w, 3), bench_poses[ai], severity_per_part=None
        )
        sev = {part: severity_per_part_seq[part][k] for part in severity_per_part_seq}
        user_panel = draw_skeleton_blank(
            (canvas_h, canvas_w, 3), user_poses[bi], severity_per_part=sev
        )
        cv2.putText(
            bench_panel, f"benchmark  t={ai / fps:.2f}s",
            (10, canvas_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (220, 220, 220), 1, cv2.LINE_AA,
        )
        cv2.putText(
            user_panel, f"learner    t={bi / fps:.2f}s",
            (10, canvas_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (220, 220, 220), 1, cv2.LINE_AA,
        )
        draw_severity_legend(user_panel, origin=(canvas_w - 130, 10))
        frames.append(np.hstack([bench_panel, user_panel]))

    if frames:
        h, w = frames[0].shape[:2]
        write_video(out_path, iter(frames), fps=fps, size=(w, h))


def _dominant_severity_part(severity: Dict[str, str]) -> Optional[str]:
    """Pick a label-worthy part: prefer red, then yellow; ignore green."""
    reds = [p for p, s in severity.items() if s == "red"]
    if reds:
        return reds[0]
    yellows = [p for p, s in severity.items() if s == "yellow"]
    if yellows:
        return yellows[0]
    return None


def _render_side_by_side_real(
    bench_video: str,
    user_video: str,
    bench_poses: np.ndarray,
    user_poses: np.ndarray,
    aligned_a_idx: np.ndarray,
    aligned_b_idx: np.ndarray,
    severity_per_part_seq: Dict[str, list],
    out_path: Path,
    fps: float,
    *,
    target_h: int = 540,
    bench_color: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    """Render the demo video onto the actual benchmark / learner footage.

    Both videos are walked sequentially (no per-frame seeking). Because the
    DTW path is monotonic non-decreasing in both indices, we can decode each
    video front-to-back exactly once and reuse the most recent decoded frame
    whenever the path holds an index constant. Frames are downscaled to
    ``target_h`` before drawing; the AIST 2D keypoints, which are in
    source-video pixel coordinates, are scaled by the same factor so the
    skeleton lines up.
    """
    aligned_a_idx = np.asarray(aligned_a_idx, dtype=np.int64)
    aligned_b_idx = np.asarray(aligned_b_idx, dtype=np.int64)
    L = int(min(aligned_a_idx.shape[0], aligned_b_idx.shape[0]))
    if L == 0:
        return

    cap_a = cv2.VideoCapture(str(bench_video))
    cap_b = cv2.VideoCapture(str(user_video))
    if not cap_a.isOpened() or not cap_b.isOpened():
        cap_a.release()
        cap_b.release()
        raise RuntimeError(f"cannot open video(s): {bench_video} / {user_video}")

    try:
        Wa = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
        Ha = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Wb = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))
        Hb = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if Ha <= 0 or Hb <= 0:
            raise RuntimeError("video reported zero height")

        panel_h = int(target_h)
        panel_wa = max(1, int(round(Wa * panel_h / Ha)))
        panel_wb = max(1, int(round(Wb * panel_h / Hb)))
        sa_x, sa_y = panel_wa / float(Wa), panel_h / float(Ha)
        sb_x, sb_y = panel_wb / float(Wb), panel_h / float(Hb)
        out_size = (panel_wa + panel_wb, panel_h)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            out_size,
        )
        try:
            prev_a = -1
            prev_b = -1
            last_a: Optional[np.ndarray] = None
            last_b: Optional[np.ndarray] = None

            T_b = int(bench_poses.shape[0])
            T_u = int(user_poses.shape[0])

            for k in range(L):
                ai = int(aligned_a_idx[k])
                bi = int(aligned_b_idx[k])

                # Sequential decode — uses grab() to skip to the target,
                # then read() once for the frame we actually render.
                while prev_a < ai - 1:
                    if not cap_a.grab():
                        break
                    prev_a += 1
                if prev_a < ai:
                    ok, frame = cap_a.read()
                    prev_a += 1
                    if ok:
                        last_a = frame
                while prev_b < bi - 1:
                    if not cap_b.grab():
                        break
                    prev_b += 1
                if prev_b < bi:
                    ok, frame = cap_b.read()
                    prev_b += 1
                    if ok:
                        last_b = frame
                if last_a is None or last_b is None:
                    continue

                a_disp = cv2.resize(last_a, (panel_wa, panel_h))
                b_disp = cv2.resize(last_b, (panel_wb, panel_h))

                if 0 <= ai < T_b:
                    bp = bench_poses[ai].astype(np.float32, copy=True)
                    bp[:, 0] *= sa_x
                    bp[:, 1] *= sa_y
                    a_disp = draw_skeleton_overlay(
                        a_disp, bp, severity_per_part=None, base_color=bench_color,
                    )
                if 0 <= bi < T_u:
                    up = user_poses[bi].astype(np.float32, copy=True)
                    up[:, 0] *= sb_x
                    up[:, 1] *= sb_y
                    sev = {part: severity_per_part_seq[part][k] for part in severity_per_part_seq}
                    b_disp = draw_skeleton_overlay(
                        b_disp, up, severity_per_part=sev,
                    )

                cv2.putText(
                    a_disp, f"benchmark  t={ai / fps:.2f}s",
                    (10, panel_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    b_disp, f"learner    t={bi / fps:.2f}s",
                    (10, panel_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
                draw_severity_legend(b_disp, origin=(panel_wb - 130, 10))

                worst = _dominant_severity_part(
                    {p: severity_per_part_seq[p][k] for p in severity_per_part_seq}
                )
                if worst is not None:
                    level = severity_per_part_seq[worst][k]
                    cv2.putText(
                        b_disp, f"{worst.replace('_', ' ')}: {level}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        SEVERITY_BGR[level], 2, cv2.LINE_AA,
                    )

                writer.write(np.hstack([a_disp, b_disp]))
        finally:
            writer.release()
    finally:
        cap_a.release()
        cap_b.release()


def _per_part_error_seq(
    aligned_joint_err: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Aggregate ``(L, 17)`` joint errors into per-body-part sequences ``(L,)``.

    Frames where every joint in a part is masked yield NaN from
    ``np.nanmean`` (with a ``RuntimeWarning``); we suppress that warning and
    treat such frames as zero error.
    """
    import warnings

    out: Dict[str, np.ndarray] = {}
    if aligned_joint_err.size == 0:
        for part in BODY_PART_GROUPS:
            out[part] = np.zeros((0,), dtype=np.float32)
        return out
    for part, idxs in BODY_PART_GROUPS.items():
        sub = aligned_joint_err[:, list(idxs)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(sub, axis=1)
        out[part] = np.nan_to_num(mean, nan=0.0).astype(np.float32, copy=False)
    return out


def _per_part_summary(
    per_part_err: Dict[str, np.ndarray],
    per_part_score: Dict[str, float],
    severities: Dict[str, list],
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for part, errs in per_part_err.items():
        sev = severities.get(part, [])
        n = max(len(sev), 1)
        pct_red = 100.0 * sum(1 for s in sev if s == "red") / n
        pct_yellow = 100.0 * sum(1 for s in sev if s == "yellow") / n
        if errs.size:
            mean_err = float(np.mean(errs))
            max_err = float(np.max(errs))
        else:
            mean_err = 0.0
            max_err = 0.0
        summary[part] = {
            "score": float(per_part_score.get(part, 0.0)),
            "mean_error": mean_err,
            "max_error": max_err,
            "pct_yellow": pct_yellow,
            "pct_red": pct_red,
        }
    return summary


def run(
    benchmark_video: str,
    user_video: str,
    model_config: Optional[str],
    ckpt: Optional[str],
    compare_config: str,
    out_root: str,
    render_video: bool = True,
    alignment_method: str = "raw_features",
    gnn_checkpoint: Optional[str] = None,
    gnn_device: str = "auto",
    gnn_batch_size: int = 256,
    bench_poses_pkl: Optional[str] = None,
    user_poses_pkl: Optional[str] = None,
    embedding_score_weight: float = 0.4,
) -> Path:
    out_root = ensure_dir(out_root)
    cfg = load_yaml(compare_config)
    if alignment_method not in {"raw_features", "gnn_embedding"}:
        raise ValueError(f"unknown alignment_method: {alignment_method!r}")
    if alignment_method == "gnn_embedding" and not gnn_checkpoint:
        raise ValueError("--gnn-checkpoint is required when --alignment-method gnn_embedding")

    bench_pred = _resolve_pose_input(
        benchmark_video, model_config, ckpt, out_root / "benchmark_pose", bench_poses_pkl
    )
    user_pred = _resolve_pose_input(
        user_video, model_config, ckpt, out_root / "user_pose", user_poses_pkl
    )

    bench_raw = np.load(bench_pred / "poses.npy")
    user_raw = np.load(user_pred / "poses.npy")

    bench_meta = json.loads((bench_pred / "meta.json").read_text())
    user_meta = json.loads((user_pred / "meta.json").read_text())
    fps = float(bench_meta.get("fps") or 30.0)

    bench_smooth = smooth_sequence(bench_raw, SmoothConfig())
    user_smooth = smooth_sequence(user_raw, SmoothConfig())

    norm_cfg = NormalizeConfig(
        scale_by=cfg.get("normalization", {}).get("scale_by", "torso"),
        min_visibility=float(cfg.get("normalization", {}).get("min_visibility", 0.2)),
        orient_torso=bool(cfg.get("normalization", {}).get("canonical_orient", False)),
    )
    bench_norm, bench_mask = normalize_sequence(bench_smooth, norm_cfg)
    user_norm, user_mask = normalize_sequence(user_smooth, norm_cfg)

    feat_cfg = FeatureConfig(smoothing_window=int(cfg.get("features", {}).get("smoothing_window", 5)))
    bench_feats = extract_features(bench_norm, bench_mask, feat_cfg)
    user_feats = extract_features(user_norm, user_mask, feat_cfg)

    embedding_dim: Optional[int] = None
    if alignment_method == "raw_features":
        A = framewise_distance_vector(bench_feats)
        B = framewise_distance_vector(user_feats)
        feature_weights = build_feature_weights(A.shape[1], 17)
    else:
        from src.compare.embedding_features import encode_pose_sequence, load_pose_gnn_encoder

        model, device_t = load_pose_gnn_encoder(gnn_checkpoint, device=gnn_device)
        A = encode_pose_sequence(model, bench_norm, device=device_t, batch_size=gnn_batch_size)
        B = encode_pose_sequence(model, user_norm, device=device_t, batch_size=gnn_batch_size)
        embedding_dim = int(A.shape[1]) if A.ndim == 2 else None
        feature_weights = None

    dtw_cfg = DTWConfig(
        band_ratio=float(cfg.get("dtw", {}).get("band_ratio", 0.15)),
        warp_penalty=float(cfg.get("dtw", {}).get("warp_penalty", 0.05)),
        feature_weights=feature_weights,
    )
    dtw = dtw_align(A, B, dtw_cfg, fps=fps)

    score_cfg = ScoreConfig(
        body_part_weights=cfg.get("body_part_weights", {}),
        score_weights=cfg.get("score_weights", {}),
        seconds_per_window=float(cfg.get("windowing", {}).get("seconds_per_window", 1.0)),
        top_k_worst_windows=int(cfg.get("windowing", {}).get("top_k_worst_windows", 3)),
        top_k_worst_parts=int(cfg.get("windowing", {}).get("top_k_worst_parts", 2)),
    )
    result = compare_features(bench_feats, user_feats, dtw, score_cfg, fps=fps)
    feedback_lines = generate_feedback(result)

    # Per-aligned-frame, per-body-part errors drive both severity coloring
    # in the video and the timestamped feedback intervals.
    per_part_err_seq = _per_part_error_seq(result.aligned_joint_err)
    aligned_t_sec = result.aligned_a_idx.astype(np.float32) / float(fps)
    severities, (low_t, high_t) = severity_sequence_per_part(per_part_err_seq)
    feedback_intervals = extract_feedback_intervals(per_part_err_seq, aligned_t_sec)
    per_part_summary = _per_part_summary(per_part_err_seq, result.per_body_part_score, severities)

    duration_seconds = float(bench_raw.shape[0]) / float(fps) if bench_raw.shape[0] else 0.0

    # ``compare_features`` returns the legacy raw overall score
    # (pose geometry + limb angles + timing) plus the literal body-part
    # geometry score. Keep both names explicit at the report top level.
    raw_overall_score = float(result.overall_score)
    pose_geometry_score = float(result.pose_geometry_score)

    # Embedding similarity is GNN-only: for raw_features we leave the
    # corresponding fields as None and the top-level overall_score stays equal
    # to the legacy raw overall score. Clamp the weight to ``[0, 1]`` so the
    # combined score is a true convex combination of literal geometry and
    # embedding similarity.
    embedding_similarity_score: Optional[float] = None
    embedding_stats: Optional[Dict[str, float]] = None
    embedding_similarity_heatmap: Optional[str] = None
    embedding_similarity_heatmap_stats: Optional[Dict[str, float | list]] = None
    combined_score: Optional[float] = None
    emb_weight = float(np.clip(embedding_score_weight, 0.0, 1.0))

    if alignment_method == "gnn_embedding":
        from src.compare.embedding_features import (
            compute_embedding_similarity,
            pairwise_cosine_similarity,
        )

        embedding_stats = compute_embedding_similarity(
            A, B, result.aligned_a_idx, result.aligned_b_idx
        )
        embedding_similarity_score = float(embedding_stats["score"])
        combined_score = float(
            (1.0 - emb_weight) * pose_geometry_score
            + emb_weight * embedding_similarity_score
        )
        overall_score = combined_score

        similarity_matrix = pairwise_cosine_similarity(A, B)
        embedding_similarity_heatmap = "embedding_similarity_heatmap.png"
        embedding_similarity_heatmap_stats = _matrix_stats(similarity_matrix)
        _render_embedding_similarity_heatmap(
            similarity_matrix,
            out_root / embedding_similarity_heatmap,
            aligned_a_idx=result.aligned_a_idx,
            aligned_b_idx=result.aligned_b_idx,
        )
    else:
        overall_score = raw_overall_score

    # Write artifacts. The schema keeps backward-compatible keys (``scores``,
    # ``feedback``, ``alignment``, ``dtw``) and adds the structured fields the
    # demo UI consumes (``overall_score``, ``feedback_intervals``,
    # ``per_part_summary``, plus the GNN-aware score breakdown).
    report = {
        "overall_score": float(overall_score),
        "raw_overall_score": raw_overall_score,
        "pose_geometry_score": pose_geometry_score,
        "embedding_similarity_score": embedding_similarity_score,
        "embedding_similarity_heatmap": embedding_similarity_heatmap,
        "embedding_similarity_heatmap_stats": embedding_similarity_heatmap_stats,
        "combined_score": combined_score,
        "embedding_score_weight": emb_weight if alignment_method == "gnn_embedding" else None,
        "fps": float(fps),
        "duration_seconds": duration_seconds,
        "alignment_method": alignment_method,
        "benchmark_video": benchmark_video,
        "user_video": user_video,
        "alignment": {
            "method": alignment_method,
            "feature_shape_benchmark": list(A.shape),
            "feature_shape_user": list(B.shape),
            "gnn_checkpoint": gnn_checkpoint if alignment_method == "gnn_embedding" else None,
            "embedding_dim": embedding_dim,
        },
        "fps_used_for_timing": fps,
        "dtw": {
            "cost": dtw.cost,
            "path_length": int(len(dtw.path)),
            "timing_skew_sec": dtw.timing_skew_sec,
        },
        "embedding_stats": embedding_stats,
        "severity_thresholds": {"yellow": float(low_t), "red": float(high_t)},
        "scores": score_result_to_dict(result),
        "per_part_summary": per_part_summary,
        "feedback": feedback_lines,
        "feedback_intervals": [iv.to_dict() for iv in feedback_intervals],
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    _render_bar_chart(result.per_body_part_score, out_root / "summary.png")

    if render_video:
        _render_side_by_side_skeleton(
            bench_smooth,
            user_smooth,
            result.aligned_a_idx,
            result.aligned_b_idx,
            severities,
            out_root / "aligned_side.mp4",
            fps=fps,
        )
        # The real-video overlay is best-effort: if either source video is
        # missing or unreadable (e.g. someone passed only the .pkl files
        # with placeholder paths), skip it but keep the blank-canvas mp4.
        if Path(benchmark_video).exists() and Path(user_video).exists():
            try:
                _render_side_by_side_real(
                    benchmark_video,
                    user_video,
                    bench_smooth,
                    user_smooth,
                    result.aligned_a_idx,
                    result.aligned_b_idx,
                    severities,
                    out_root / "aligned_side_real.mp4",
                    fps=fps,
                )
            except RuntimeError as e:
                print(f"warning: real-video render skipped ({e})")

    return out_root


def _main() -> None:
    p = argparse.ArgumentParser(description="End-to-end comparison report from two videos.")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--model-config", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--bench-poses-pkl", default=None, help="AIST-style benchmark keypoints .pkl/.npy with shape (T,17,2|3)")
    p.add_argument("--user-poses-pkl", default=None, help="AIST-style user keypoints .pkl/.npy with shape (T,17,2|3)")
    p.add_argument("--compare-config", default="configs/data/compare.yaml")
    p.add_argument("--out", default="data/reports/run_latest")
    p.add_argument("--no-video", action="store_true")
    p.add_argument(
        "--alignment-method",
        default="raw_features",
        choices=["raw_features", "gnn_embedding"],
        help="raw_features keeps the baseline DTW features; gnn_embedding uses PoseGNNEncoder embeddings for DTW",
    )
    p.add_argument(
        "--gnn-checkpoint",
        default=None,
        help="PoseGNNEncoder checkpoint required when --alignment-method gnn_embedding",
    )
    p.add_argument("--gnn-device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--gnn-batch-size", default=256, type=int)
    p.add_argument(
        "--embedding-score-weight",
        default=0.4,
        type=float,
        help=(
            "weight of the GNN embedding similarity in the combined score; "
            "the geometry score gets weight (1 - this). Only used when "
            "--alignment-method gnn_embedding. Default 0.4."
        ),
    )
    args = p.parse_args()
    if args.alignment_method == "gnn_embedding" and not args.gnn_checkpoint:
        p.error("--gnn-checkpoint is required when --alignment-method gnn_embedding")
    if (not args.bench_poses_pkl or not args.user_poses_pkl) and (not args.model_config or not args.ckpt):
        p.error("--model-config and --ckpt are required for any video without --*-poses-pkl")
    out = run(
        args.benchmark, args.user,
        args.model_config, args.ckpt,
        args.compare_config, args.out,
        render_video=not args.no_video,
        alignment_method=args.alignment_method,
        gnn_checkpoint=args.gnn_checkpoint,
        gnn_device=args.gnn_device,
        gnn_batch_size=args.gnn_batch_size,
        bench_poses_pkl=args.bench_poses_pkl,
        user_poses_pkl=args.user_poses_pkl,
        embedding_score_weight=args.embedding_score_weight,
    )
    print(f"report written to {out}")


if __name__ == "__main__":
    _main()
