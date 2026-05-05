"""End-to-end orchestrator for the integrated dance-CV pipeline.

Wires the pose-extraction adapters, the per-stream error/similarity heads,
and the fusion layer into a single benchmark-vs-learner call. Outputs a
markdown report, a JSON report, and an npz of all per-model curves.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from src.error import embedding_stream as emb_stream
from src.error import keypoint_stream as kp_stream
from src.fusion.fuse import FusionResult, fuse
from src.pose import gnn_adapter, hrnet_adapter, simple_baseline_adapter
from src.pose.base import PoseRunResult


def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_lstm_section(
    cfg: dict,
    *,
    lstm_checkpoint: Optional[str] = None,
    use_lstm: Optional[bool] = None,
    require_lstm: bool = False,
) -> tuple[Optional[str], dict]:
    """Resolve LSTM config and optional CLI overrides.

    The integrated pipeline can run without a Mia checkpoint by falling back to
    geometric threshold probabilities. When ``require_lstm`` is true, fail early
    instead of silently falling back.
    """
    section = dict(cfg.get("lstm") or {})
    if lstm_checkpoint is not None:
        section["checkpoint"] = lstm_checkpoint
        section["enabled"] = True
    if use_lstm is not None:
        section["enabled"] = bool(use_lstm)

    enabled = bool(section.get("enabled", False))
    checkpoint = section.get("checkpoint")
    checkpoint_exists = bool(checkpoint and Path(checkpoint).exists())

    if require_lstm:
        if not enabled:
            raise RuntimeError("LSTM was required, but the LSTM section is disabled.")
        if not checkpoint:
            raise RuntimeError("LSTM was required, but no checkpoint path was configured.")
        if not checkpoint_exists:
            raise FileNotFoundError(f"LSTM checkpoint not found: {checkpoint}")

    cfg["lstm"] = section
    status = {
        "enabled": enabled,
        "checkpoint": checkpoint,
        "checkpoint_exists": checkpoint_exists,
        "required": require_lstm,
    }
    return (checkpoint if enabled else None), status


def _run_pose_for_model(
    model_name: str,
    model_cfg: dict,
    inference_cfg: dict,
    bench_video: str,
    user_video: str,
    device: Optional[str],
    upstream_results: dict[str, tuple[PoseRunResult, PoseRunResult]],
) -> tuple[PoseRunResult, PoseRunResult]:
    if model_name == "hrnet":
        bench = hrnet_adapter.run_video(
            bench_video,
            model_config=model_cfg["config"],
            checkpoint=model_cfg["checkpoint"],
            device=device,
            inference=inference_cfg,
        )
        user = hrnet_adapter.run_video(
            user_video,
            model_config=model_cfg["config"],
            checkpoint=model_cfg["checkpoint"],
            device=device,
            inference=inference_cfg,
        )
    elif model_name == "simple_baseline":
        bench = simple_baseline_adapter.run_video(
            bench_video,
            model_config=model_cfg["config"],
            checkpoint=model_cfg["checkpoint"],
            device=device,
            inference=inference_cfg,
        )
        user = simple_baseline_adapter.run_video(
            user_video,
            model_config=model_cfg["config"],
            checkpoint=model_cfg["checkpoint"],
            device=device,
            inference=inference_cfg,
        )
    elif model_name == "gnn":
        upstream_name = model_cfg.get("upstream")
        if upstream_name:
            if upstream_name not in upstream_results:
                raise RuntimeError(
                    f"GNN adapter requested upstream {upstream_name!r}, but it has not run."
                )
            u_bench, u_user = upstream_results[upstream_name]
        else:
            try:
                u_bench, u_user = next(
                    pair for pair in upstream_results.values()
                    if pair[0].poses is not None and pair[1].poses is not None
                )
            except StopIteration as e:
                raise RuntimeError("GNN adapter requires an upstream keypoint model.") from e
        bench = gnn_adapter.encode_from_pose_result(
            u_bench, checkpoint=model_cfg["checkpoint"], device=device or "auto"
        )
        user = gnn_adapter.encode_from_pose_result(
            u_user, checkpoint=model_cfg["checkpoint"], device=device or "auto"
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return bench, user


def run(
    benchmark_video: str,
    learner_video: str,
    output_dir: str,
    *,
    config_path: str = "configs/integrate/pipeline.yaml",
    device: Optional[str] = None,
    lstm_checkpoint: Optional[str] = None,
    use_lstm: Optional[bool] = None,
    require_lstm: bool = False,
) -> dict:
    cfg = _load_cfg(config_path)
    out = _ensure_dir(output_dir)
    preprocessing_cfg = cfg.get("preprocessing", {})
    lstm_ckpt, lstm_status = _resolve_lstm_section(
        cfg,
        lstm_checkpoint=lstm_checkpoint,
        use_lstm=use_lstm,
        require_lstm=require_lstm,
    )

    # 1. Pose extraction. HRNet runs first so GNN can borrow its keypoints.
    upstream: dict[str, tuple[PoseRunResult, PoseRunResult]] = {}
    order = ["hrnet", "simple_baseline", "gnn"]
    for name in order:
        section = cfg["models"].get(name, {})
        if not section.get("enabled", False):
            continue
        bench_pr, user_pr = _run_pose_for_model(
            name, section, preprocessing_cfg, benchmark_video, learner_video, device, upstream
        )
        upstream[name] = (bench_pr, user_pr)

    if not upstream:
        raise RuntimeError("No pose model was enabled in config; nothing to fuse.")

    # 2. Error / similarity streams.
    keypoint_streams = []
    embedding_streams = []
    canonical_path = None
    canonical_timestamps = None

    for name, (bench_pr, user_pr) in upstream.items():
        if bench_pr.poses is not None and user_pr.poses is not None and bench_pr.embeddings is None:
            lstm_device = "cpu" if device is None or str(device).lower() == "auto" else device
            ks = kp_stream.build(
                name=name,
                bench_kp=bench_pr.poses,
                user_kp=user_pr.poses,
                fps=bench_pr.fps,
                lstm_ckpt=lstm_ckpt,
                device=lstm_device,
            )
            keypoint_streams.append(ks)
            if canonical_path is None:
                canonical_path = ks.path
                canonical_timestamps = ks.timestamps
        if bench_pr.embeddings is not None and user_pr.embeddings is not None:
            if canonical_path is None or canonical_timestamps is None:
                raise RuntimeError(
                    "Embedding stream needs a keypoint stream first to define the alignment path."
                )
            es = emb_stream.build(
                name=name,
                emb_bench=bench_pr.embeddings,
                emb_user=user_pr.embeddings,
                path=canonical_path,
                timestamps=canonical_timestamps,
            )
            embedding_streams.append(es)

    # 3. Fuse.
    fusion_cfg = cfg.get("fusion", {})
    err_cfg = cfg.get("error_detection", {})
    result: FusionResult = fuse(
        keypoint_streams,
        embedding_streams,
        geom_threshold=float(err_cfg.get("threshold", 0.25)),
        off_threshold=float(fusion_cfg.get("off_threshold", 0.5)),
        similarity_weight=float(fusion_cfg.get("similarity_weight", 0.4)),
        min_duration_s=float(err_cfg.get("min_duration_s", 0.5)),
        body_part_weights=cfg.get("scoring", {}).get("body_part_weights"),
    )

    # 4. Persist.
    (out / "report.md").write_text(result.markdown_report)
    lstm_streams = sorted(result.per_model_part_probs.keys())
    lstm_status = {
        **lstm_status,
        "used": bool(lstm_streams),
        "streams": lstm_streams,
    }
    report_json = {
        "overall_score": result.overall_score,
        "intervals": [asdict(iv) for iv in result.intervals],
        "feedback": result.feedback,
        "fps": result.fps,
        "canonical_stream": result.extra.get("canonical_stream"),
        "curve_png": "report_curves.png",
        "fusion_params": {
            k: result.extra.get(k)
            for k in ("geom_threshold", "off_threshold", "similarity_weight", "min_duration_s")
        },
        "models_enabled": list(upstream.keys()),
        "lstm_used": lstm_status["used"],
        "lstm": lstm_status,
    }
    (out / "report.json").write_text(json.dumps(report_json, indent=2))

    npz_payload: dict[str, np.ndarray] = {
        "time_axis": result.time_axis,
        "final_off": result.final_off,
        "confidence_curve": 1.0 - result.final_off.mean(axis=1),
        "sim_avg": result.sim_avg,
        "part_probs_avg": result.part_probs_avg,
    }
    for name, sig in result.per_model_part_signal.items():
        npz_payload[f"{name}_part_signal"] = sig
    for name, probs in result.per_model_part_probs.items():
        npz_payload[f"{name}_part_probs"] = probs
    for name, cos in result.per_model_cosine.items():
        npz_payload[f"{name}_cosine"] = cos
    np.savez(out / "streams.npz", **npz_payload)
    _write_curve_plot(out / "report_curves.png", result)

    return report_json


def _write_curve_plot(path: Path, result: FusionResult) -> None:
    """Write the output similarity/confidence curve from the project diagram."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    similarity = ((result.sim_avg + 1.0) * 0.5).clip(0.0, 1.0)
    confidence = (1.0 - result.final_off.mean(axis=1)).clip(0.0, 1.0)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(result.time_axis, similarity, label="Similarity", color="#2563eb", linewidth=2.0)
    ax.plot(result.time_axis, confidence, label="Confidence", color="#16a34a", linewidth=2.0)
    for interval in result.intervals:
        ax.axvspan(interval.start_s, interval.end_s, color="#f97316", alpha=0.16)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
