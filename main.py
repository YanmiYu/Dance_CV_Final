"""
main.py — CLI entry point for the Dance Choreography Practice Tool.

Used by the SLURM job script (slurm_run.sh) on Oscar.

Tasks
-----
extract      Extract and save keypoints for a single video.
extract_all  Extract keypoints for every video in data/.
train        Train the BiGRU DeviationClassifier on data/train/.
test         Evaluate the trained model on data/test/.
analyze      Run the full inference pipeline on one benchmark/learner pair.
batch        Run analyze on every phrase_XX/ folder found in data/.

Examples
--------
  python main.py extract --video data/phrase_01/benchmark.mp4 --out data/phrase_01/keypoints/benchmark_kp.npy
  python main.py train   --train-dir data/train/ --val-dir data/val/ \
                         --checkpoint checkpoints/best_model.pt --epochs 30
  python main.py test    --test-dir data/test/ --checkpoint checkpoints/best_model.pt \
                         --out results/test_metrics.json
  python main.py analyze --benchmark data/phrase_01/keypoints/benchmark_kp.npy \
                         --learner   data/phrase_01/keypoints/learner_kp.npy \
                         --checkpoint checkpoints/best_model.pt --out results/phrase_01/
  python main.py batch   --data data/ --out results/ --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

from video_io import load_video, standardize
from pose_extraction import (
    PoseEstimator,
    smooth,
    interpolate_missing,
    save_keypoints,
    load_keypoints,
)
from normalization import normalize
from alignment import dtw_align, warping_path_to_timestamps
from scoring import (
    compute_joint_errors,
    per_part_error_over_time,
    find_off_moments,
    overall_score,
)
from visualization import render_comparison_video, plot_error_timeline
from feedback import generate_feedback


# ---------------------------------------------------------------------------
# Task: extract
# ---------------------------------------------------------------------------

def task_extract(args: argparse.Namespace) -> None:
    """Extract keypoints from a single video and save to .npy."""
    print(f"[extract] Loading video: {args.video}")
    frames_raw, src_fps = load_video(args.video)
    frames, fps = standardize(frames_raw, src_fps, target_fps=args.fps)
    print(f"[extract] {len(frames)} frames at {fps} fps")

    print(f"[extract] Running pose estimator ({args.backend})...")
    estimator = PoseEstimator(backend=args.backend)
    kp_raw = estimator.extract_sequence(frames)
    estimator.close()

    kp = interpolate_missing(smooth(kp_raw))
    save_keypoints(kp, args.out)
    print(f"[extract] Saved keypoints → {args.out}  shape={kp.shape}")


# ---------------------------------------------------------------------------
# Task: analyze
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Task: train
# ---------------------------------------------------------------------------

def task_train(args: argparse.Namespace) -> None:
    """Train the BiGRU DeviationClassifier."""
    from train import train
    train(
        train_dir       = args.train_dir,
        val_dir         = args.val_dir,
        checkpoint_path = args.checkpoint,
        epochs          = args.epochs,
        lr              = args.lr,
        hidden_size     = args.hidden,
        num_layers      = args.layers,
        dropout         = args.dropout,
        batch_size      = args.batch_size,
        log_path        = args.log,
    )


# ---------------------------------------------------------------------------
# Task: test
# ---------------------------------------------------------------------------

def task_test(args: argparse.Namespace) -> None:
    """Evaluate the trained model on the held-out test split."""
    from test import evaluate
    evaluate(
        test_dir        = args.test_dir,
        checkpoint_path = args.checkpoint,
        out_path        = args.out,
    )


# ---------------------------------------------------------------------------
# Task: analyze (inference with trained model)
# ---------------------------------------------------------------------------

def task_analyze(args: argparse.Namespace) -> None:
    """Run the full analysis pipeline on one benchmark/learner pair."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] Loading keypoints...")
    bench_kp = load_keypoints(args.benchmark)
    user_kp  = load_keypoints(args.learner)
    print(f"  benchmark: {bench_kp.shape}")
    print(f"  learner:   {user_kp.shape}")

    print("[analyze] Normalizing...")
    bench_norm = normalize(bench_kp)
    user_norm  = normalize(user_kp)

    print("[analyze] DTW alignment...")
    bench_al, user_al, path = dtw_align(bench_norm, user_norm)
    timestamps = warping_path_to_timestamps(path, args.fps)
    print(f"  aligned length T' = {len(path)}")

    print("[analyze] Building diff features...")
    from dataset import build_diff_features
    diff_feats = build_diff_features(bench_al, user_al)  # (T', 24)

    # Compute joint errors and part errors (used for score, intervals, and video)
    joint_errors = compute_joint_errors(bench_al, user_al)
    part_errors  = per_part_error_over_time(joint_errors)
    intervals    = find_off_moments(
        part_errors, threshold=args.threshold,
        fps=args.fps, min_duration_s=args.min_duration, timestamps=timestamps,
    )
    score    = overall_score(part_errors)
    feedback = generate_feedback(intervals)

    print(f"\n  Overall score: {score:.1f} / 100")
    print(f"  Off intervals: {len(intervals)}")
    for line in feedback:
        print(f"    {line}")

    # Save JSON report
    report_path = out_dir / "report.json"
    report_data = {
        "overall_score": round(score, 2),
        "intervals": [
            {
                "start_s":    round(iv.start_s, 2),
                "end_s":      round(iv.end_s, 2),
                "part":       iv.part,
                "mean_error": round(iv.mean_error, 4),
            }
            for iv in intervals
        ],
        "feedback": feedback,
    }
    report_path.write_text(json.dumps(report_data, indent=2))
    print(f"\n[analyze] Report saved → {report_path}")

    # Save per-part error arrays
    np_out = out_dir / "part_errors.npz"
    np.savez(str(np_out), **{k: v for k, v in part_errors.items()})
    print(f"[analyze] Part errors saved → {np_out}")

    # Render comparison video if raw videos are provided
    if args.bench_video and args.learner_video:
        print("[analyze] Rendering comparison video (this may take a minute)...")
        bench_frames_raw, src_fps = load_video(args.bench_video)
        user_frames_raw, _        = load_video(args.learner_video)
        bench_frames, _ = standardize(bench_frames_raw, src_fps, args.fps)
        user_frames, _  = standardize(user_frames_raw,  src_fps, args.fps)

        bench_idx = [p[0] for p in path]
        user_idx  = [p[1] for p in path]
        bench_frames_al = [bench_frames[min(i, len(bench_frames) - 1)] for i in bench_idx]
        user_frames_al  = [user_frames[min(i,  len(user_frames)  - 1)] for i in user_idx]
        bench_kp_px = bench_kp[bench_idx]
        user_kp_px  = user_kp[user_idx]

        video_out = str(out_dir / "comparison.mp4")
        render_comparison_video(
            bench_frames_al, user_frames_al,
            bench_kp_px, user_kp_px,
            joint_errors, intervals,
            video_out, args.fps,
            timestamps=timestamps,
        )
        print(f"[analyze] Comparison video saved → {video_out}")

    print("[analyze] Done.")


# ---------------------------------------------------------------------------
# Task: batch
# ---------------------------------------------------------------------------

def task_batch(args: argparse.Namespace) -> None:
    """Run analyze on every phrase_XX/ directory found in --data."""
    data_dir = Path(args.data)
    phrase_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())

    if not phrase_dirs:
        print(f"[batch] No subdirectories found in {data_dir}")
        return

    print(f"[batch] Found {len(phrase_dirs)} phrase directories.")
    for phrase_dir in phrase_dirs:
        kp_dir = phrase_dir / "keypoints"
        bench_kp_path   = kp_dir / "benchmark_kp.npy"
        learner_kp_path = kp_dir / "learner_kp.npy"
        bench_vid  = phrase_dir / "benchmark.mp4"
        learner_vid = phrase_dir / "learner.mp4"

        if not bench_kp_path.exists() or not learner_kp_path.exists():
            print(f"[batch] Skipping {phrase_dir.name}: keypoints not found. "
                  "Run 'extract' first.")
            continue

        print(f"\n{'='*50}")
        print(f"[batch] Processing: {phrase_dir.name}")
        print(f"{'='*50}")

        sub_args = argparse.Namespace(
            benchmark     = str(bench_kp_path),
            learner       = str(learner_kp_path),
            bench_video   = str(bench_vid)    if bench_vid.exists()    else None,
            learner_video = str(learner_vid)  if learner_vid.exists()  else None,
            fps           = args.fps,
            threshold     = args.threshold,
            min_duration  = args.min_duration,
            checkpoint    = args.checkpoint,
            out           = str(Path(args.out) / phrase_dir.name),
        )
        try:
            task_analyze(sub_args)
        except Exception as exc:
            print(f"[batch] ERROR on {phrase_dir.name}: {exc}")

    print(f"\n[batch] All done. Results in: {args.out}")


# ---------------------------------------------------------------------------
# Task: extract_all  (convenience: extract keypoints for every video in data/)
# ---------------------------------------------------------------------------

def task_extract_all(args: argparse.Namespace) -> None:
    """Extract keypoints for every benchmark.mp4 and learner.mp4 in data/."""
    data_dir = Path(args.data)
    phrase_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())

    if not phrase_dirs:
        print(f"[extract_all] No subdirectories found in {data_dir}")
        return

    for phrase_dir in phrase_dirs:
        for role in ("benchmark", "learner"):
            vid = phrase_dir / f"{role}.mp4"
            if not vid.exists():
                print(f"[extract_all] {vid} not found, skipping.")
                continue
            out = phrase_dir / "keypoints" / f"{role}_kp.npy"
            sub_args = argparse.Namespace(
                video   = str(vid),
                out     = str(out),
                fps     = args.fps,
                backend = args.backend,
            )
            print(f"\n[extract_all] {phrase_dir.name}/{role}.mp4")
            task_extract(sub_args)

    print("\n[extract_all] All extractions complete.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dance Choreography Practice Tool — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="task", required=True)

    # --- extract ---
    p_ext = sub.add_parser("extract", help="Extract keypoints from one video")
    p_ext.add_argument("--video",   required=True, help="Path to input .mp4")
    p_ext.add_argument("--out",     required=True, help="Output .npy path")
    p_ext.add_argument("--fps",     type=float, default=15.0)
    p_ext.add_argument("--backend", default="mediapipe", choices=["mediapipe", "mmpose"])

    # --- extract_all ---
    p_ea = sub.add_parser("extract_all", help="Extract keypoints for all phrases in data/")
    p_ea.add_argument("--data",    default="data/", help="Root data directory")
    p_ea.add_argument("--fps",     type=float, default=15.0)
    p_ea.add_argument("--backend", default="mediapipe", choices=["mediapipe", "mmpose"])

    # --- analyze ---
    p_an = sub.add_parser("analyze", help="Run full analysis on one benchmark/learner pair")
    p_an.add_argument("--benchmark",     required=True, help="Benchmark keypoints .npy")
    p_an.add_argument("--learner",       required=True, help="Learner keypoints .npy")
    p_an.add_argument("--bench-video",   default=None,  dest="bench_video",
                      help="Benchmark .mp4 (optional, for comparison video output)")
    p_an.add_argument("--learner-video", default=None,  dest="learner_video",
                      help="Learner .mp4 (optional, for comparison video output)")
    p_an.add_argument("--fps",           type=float, default=15.0)
    p_an.add_argument("--threshold",     type=float, default=0.25)
    p_an.add_argument("--min-duration",  type=float, default=0.5, dest="min_duration")
    p_an.add_argument("--checkpoint",    default=None,
                      help="Trained model checkpoint .pt (optional; uses threshold if absent)")
    p_an.add_argument("--out",           default="results/", help="Output directory")

    # --- train ---
    p_tr = sub.add_parser("train", help="Train the BiGRU DeviationClassifier")
    p_tr.add_argument("--train-dir",  required=True, dest="train_dir")
    p_tr.add_argument("--val-dir",    required=True, dest="val_dir")
    p_tr.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p_tr.add_argument("--epochs",     type=int,   default=30)
    p_tr.add_argument("--lr",         type=float, default=1e-3)
    p_tr.add_argument("--hidden",     type=int,   default=64)
    p_tr.add_argument("--layers",     type=int,   default=2)
    p_tr.add_argument("--dropout",    type=float, default=0.3)
    p_tr.add_argument("--batch-size", type=int,   default=16, dest="batch_size")
    p_tr.add_argument("--log",        default="results/training_log.csv")

    # --- test ---
    p_te = sub.add_parser("test", help="Evaluate the trained model on the test split")
    p_te.add_argument("--test-dir",   required=True, dest="test_dir")
    p_te.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p_te.add_argument("--out",        default="results/test_metrics.json")

    # --- batch ---
    p_ba = sub.add_parser("batch", help="Run analyze on all phrase dirs in data/")
    p_ba.add_argument("--data",         default="data/",    help="Root data directory")
    p_ba.add_argument("--out",          default="results/", help="Root results directory")
    p_ba.add_argument("--fps",          type=float, default=15.0)
    p_ba.add_argument("--threshold",    type=float, default=0.25)
    p_ba.add_argument("--checkpoint",   default=None)
    p_ba.add_argument("--min-duration", type=float, default=0.5, dest="min_duration")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "extract":     task_extract,
        "extract_all": task_extract_all,
        "train":       task_train,
        "test":        task_test,
        "analyze":     task_analyze,
        "batch":       task_batch,
    }
    dispatch[args.task](args)


if __name__ == "__main__":
    main()
