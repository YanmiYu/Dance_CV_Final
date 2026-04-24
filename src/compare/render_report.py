"""End-to-end CLI: benchmark_video + user_video -> artifacts in data/reports/.

Writes:
  report.json          aggregated numbers + feedback
  summary.png          bar-chart of per-body-part scores (matplotlib)
  aligned_side.mp4     synced side-by-side with pose overlays (optional)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.compare.dtw_align import DTWConfig, build_feature_weights, dtw_align
from src.compare.feedback import generate_feedback
from src.compare.features import FeatureConfig, extract_features, framewise_distance_vector
from src.compare.normalize_pose import NormalizeConfig, normalize_sequence
from src.compare.score import ScoreConfig, compare_features, score_result_to_dict
from src.infer.temporal_smooth import SmoothConfig, smooth_sequence
from src.utils.config import load_yaml
from src.utils.io import ensure_dir
from src.utils.viz import draw_pose, side_by_side
from src.utils.video import ffprobe_meta, write_video

# #region agent log
import json as _dbg_json
import time as _dbg_time

_DBG_LOG_PATH = "/Users/mohanwang/Desktop/Projects/CV_Tool_for_Dance_Choreography_Practice/.cursor/debug-a418b2.log"


def _dbg_log(location: str, message: str, data: dict, hypothesis: str = "") -> None:
    try:
        payload = {
            "sessionId": "a418b2",
            "id": f"log_{int(_dbg_time.time() * 1000)}_{location}",
            "timestamp": int(_dbg_time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "runId": "render-debug",
            "hypothesisId": hypothesis,
        }
        with open(_DBG_LOG_PATH, "a") as _f:
            _f.write(_dbg_json.dumps(payload, default=str) + "\n")
    except Exception:
        pass
# #endregion


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


def _render_side_by_side(
    bench_video: str, user_video: str,
    bench_poses: np.ndarray, user_poses: np.ndarray,
    dtw_path: np.ndarray, out_path: Path, fps: float = 30.0,
    bench_fps: Optional[float] = None,
    user_fps: Optional[float] = None,
) -> None:
    """Render side-by-side using uniform time-based sampling.

    ``dtw_path`` is kept in the signature (DTW is used for scoring) but is
    intentionally NOT used for rendering: using it causes the right panel to
    freeze whenever DTW assigns many benchmark frames to a single user frame
    (common when videos have leading/trailing idle time or differing fps).
    Instead, both videos are played back at their natural fps on a shared
    timeline so each side moves at natural speed.
    """
    cap_a = cv2.VideoCapture(bench_video)
    cap_b = cv2.VideoCapture(user_video)
    Ta = int(len(bench_poses))
    Tb = int(len(user_poses))
    if bench_fps is None or bench_fps <= 0:
        bench_fps = float(cap_a.get(cv2.CAP_PROP_FPS)) or fps
    if user_fps is None or user_fps <= 0:
        user_fps = float(cap_b.get(cv2.CAP_PROP_FPS)) or fps
    out_fps = float(fps)

    bench_duration = Ta / max(bench_fps, 1e-6)
    user_duration = Tb / max(user_fps, 1e-6)
    out_duration = min(bench_duration, user_duration)
    out_len = max(1, int(round(out_duration * out_fps)))

    # #region agent log
    _dbg_log(
        "render_report.py:render_side_by_side:entry",
        "side-by-side entry (post-fix): uniform-time sampling",
        {
            "bench_video": bench_video,
            "user_video": user_video,
            "bench_fps": float(bench_fps),
            "user_fps": float(user_fps),
            "out_fps": float(out_fps),
            "Ta": Ta,
            "Tb": Tb,
            "bench_duration_sec": float(bench_duration),
            "user_duration_sec": float(user_duration),
            "out_len_frames": int(out_len),
            "dtw_path_len_unused": int(len(dtw_path)),
            "user_poses_xy_min": [float(user_poses[..., 0].min()), float(user_poses[..., 1].min())],
            "user_poses_xy_max": [float(user_poses[..., 0].max()), float(user_poses[..., 1].max())],
        },
        hypothesis="post-fix",
    )
    _iter_samples = []
    # #endregion

    a_next = 0
    b_next = 0
    last_a = None
    last_b = None
    frames: list[np.ndarray] = []
    for t in range(out_len):
        sec = t / out_fps
        ai = min(Ta - 1, int(round(sec * bench_fps)))
        bi = min(Tb - 1, int(round(sec * user_fps)))

        while a_next <= ai:
            ok, fr = cap_a.read()
            if not ok:
                break
            last_a = fr
            a_next += 1
        while b_next <= bi:
            ok, fr = cap_b.read()
            if not ok:
                break
            last_b = fr
            b_next += 1

        # #region agent log
        if t < 5 or t % max(1, out_len // 20) == 0:
            _iter_samples.append({
                "t": int(t),
                "ai": int(ai),
                "bi": int(bi),
                "a_next": int(a_next),
                "b_next": int(b_next),
                "last_a_none": last_a is None,
                "last_b_none": last_b is None,
            })
        # #endregion

        if last_a is None or last_b is None:
            continue
        overlay_a = draw_pose(last_a, bench_poses[ai])
        overlay_b = draw_pose(last_b, user_poses[bi])
        frame = side_by_side(overlay_a, overlay_b, label_a="benchmark", label_b="you")
        frames.append(frame)

    cap_a.release()
    cap_b.release()

    # #region agent log
    _dbg_log(
        "render_report.py:render_side_by_side:iter_samples",
        "per-iter sample captures (post-fix)",
        {
            "samples": _iter_samples,
            "total_iters": int(out_len),
            "frames_written": int(len(frames)),
        },
        hypothesis="post-fix",
    )
    # #endregion

    if frames:
        h, w = frames[0].shape[:2]
        write_video(out_path, iter(frames), fps=out_fps, size=(w, h))


def run(
    benchmark_video: str,
    user_video: str,
    model_config: str,
    ckpt: str,
    compare_config: str,
    out_root: str,
    render_video: bool = True,
) -> Path:
    out_root = ensure_dir(out_root)
    cfg = load_yaml(compare_config)

    bench_pred = _run_pose_if_needed(
        benchmark_video, model_config, ckpt, out_root / "benchmark_pose"
    )
    user_pred = _run_pose_if_needed(
        user_video, model_config, ckpt, out_root / "user_pose"
    )

    bench_raw = np.load(bench_pred / "poses.npy")
    user_raw = np.load(user_pred / "poses.npy")

    bench_meta = json.loads((bench_pred / "meta.json").read_text())
    user_meta = json.loads((user_pred / "meta.json").read_text())
    fps = float(bench_meta.get("fps") or 30.0)
    # #region agent log
    _dbg_log(
        "render_report.py:run:fps_mismatch",
        "fps and frame counts for both videos",
        {
            "bench_fps": float(bench_meta.get("fps") or 0),
            "user_fps": float(user_meta.get("fps") or 0),
            "bench_frames": int(bench_meta.get("num_frames") or 0),
            "user_frames": int(user_meta.get("num_frames") or 0),
            "output_fps_used": float(fps),
        },
        hypothesis="H1,H3",
    )
    # #endregion

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

    A = framewise_distance_vector(bench_feats)
    B = framewise_distance_vector(user_feats)
    dtw_cfg = DTWConfig(
        band_ratio=float(cfg.get("dtw", {}).get("band_ratio", 0.15)),
        warp_penalty=float(cfg.get("dtw", {}).get("warp_penalty", 0.05)),
        feature_weights=build_feature_weights(A.shape[1], 17),
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

    # Write artifacts.
    report = {
        "benchmark_video": benchmark_video,
        "user_video": user_video,
        "fps_used_for_timing": fps,
        "dtw": {
            "cost": dtw.cost,
            "path_length": int(len(dtw.path)),
            "timing_skew_sec": dtw.timing_skew_sec,
        },
        "scores": score_result_to_dict(result),
        "feedback": feedback_lines,
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    _render_bar_chart(result.per_body_part_score, out_root / "summary.png")

    if render_video:
        _render_side_by_side(
            benchmark_video, user_video,
            bench_smooth, user_smooth,
            dtw.path, out_root / "aligned_side.mp4", fps=fps,
            bench_fps=float(bench_meta.get("fps") or fps),
            user_fps=float(user_meta.get("fps") or fps),
        )

    return out_root


def _main() -> None:
    p = argparse.ArgumentParser(description="End-to-end comparison report from two videos.")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--compare-config", default="configs/data/compare.yaml")
    p.add_argument("--out", default="data/reports/run_latest")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    out = run(
        args.benchmark, args.user,
        args.model_config, args.ckpt,
        args.compare_config, args.out,
        render_video=not args.no_video,
    )
    print(f"report written to {out}")


if __name__ == "__main__":
    _main()
