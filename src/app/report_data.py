"""Pure data helpers for the Streamlit dance dashboard."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.mia.dataset import PART_ORDER


REPORT_ROOTS = (Path("data/reports"), Path("results"))

PART_LABELS = {
    "LEFT_ARM": "Left Arm",
    "RIGHT_ARM": "Right Arm",
    "LEFT_LEG": "Left Leg",
    "RIGHT_LEG": "Right Leg",
    "TORSO": "Torso",
    "HEAD": "Head",
}

LEGACY_PARTS = {
    "left_arm": "LEFT_ARM",
    "right_arm": "RIGHT_ARM",
    "left_leg": "LEFT_LEG",
    "right_leg": "RIGHT_LEG",
    "torso": "TORSO",
    "head": "HEAD",
    "head/face": "HEAD",
}


@dataclass(frozen=True)
class ReportRun:
    label: str
    path: Path
    modified: float


def discover_report_runs(roots: Iterable[Path] = REPORT_ROOTS) -> list[ReportRun]:
    runs: list[ReportRun] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.iterdir():
            report_path = candidate / "report.json"
            if not candidate.is_dir() or not report_path.exists():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            runs.append(
                ReportRun(
                    label=f"{root.as_posix()}/{candidate.name}",
                    path=candidate,
                    modified=report_path.stat().st_mtime,
                )
            )
    return sorted(runs, key=lambda item: (item.modified, item.label), reverse=True)


def load_json_report(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "report.json").read_text())


def load_streams(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "streams.npz"
    if not path.exists():
        return {}
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def load_dashboard_data(run_dir: Path) -> dict[str, Any]:
    return build_dashboard_data(load_json_report(run_dir), load_streams(run_dir), run_dir=run_dir)


def build_dashboard_data(
    report: dict[str, Any],
    streams: dict[str, np.ndarray] | None = None,
    *,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    streams = streams or {}
    if _is_legacy(report):
        data = _build_legacy_dashboard(report, streams)
    else:
        data = _build_integrated_dashboard(report, streams)

    artifacts = {
        "report_json": str(run_dir / "report.json") if run_dir else "",
        "report_md": str(run_dir / "report.md") if run_dir and (run_dir / "report.md").exists() else "",
        "aligned_video": str(run_dir / "aligned_side.mp4") if run_dir and (run_dir / "aligned_side.mp4").exists() else "",
        "summary_png": str(run_dir / "summary.png") if run_dir and (run_dir / "summary.png").exists() else "",
        "curve_png": "",
        "streams_npz": str(run_dir / "streams.npz") if run_dir and (run_dir / "streams.npz").exists() else "",
    }
    if run_dir:
        curve_name = str(report.get("curve_png") or "report_curves.png")
        curve_path = run_dir / curve_name
        if curve_path.exists():
            artifacts["curve_png"] = str(curve_path)
    data["artifacts"] = artifacts
    return data


def _build_integrated_dashboard(report: dict[str, Any], streams: dict[str, np.ndarray]) -> dict[str, Any]:
    overall = _safe_float(report.get("overall_score"), 0.0)
    intervals = _normalize_intervals(report.get("intervals") or [])
    time_axis = _array(streams.get("time_axis"))
    final_off = _array(streams.get("final_off"))
    sim_avg = _array(streams.get("sim_avg"))
    confidence_curve = _array(streams.get("confidence_curve"))

    per_part_score = _normalize_part_dict(report.get("per_body_part_score") or {})
    per_part_off = _normalize_part_dict(report.get("per_body_part_off_mean") or {})
    if not per_part_score and final_off.ndim == 2 and final_off.shape[1] == len(PART_ORDER):
        means = final_off.mean(axis=0)
        per_part_off = {part: float(means[idx]) for idx, part in enumerate(PART_ORDER)}
        per_part_score = {part: _score_from_off(per_part_off[part]) for part in PART_ORDER}
    elif not per_part_score:
        per_part_score, per_part_off = _part_scores_from_intervals(intervals)

    model_similarity = {
        str(key): _safe_float(value)
        for key, value in (report.get("model_similarity_score") or {}).items()
    }
    if not model_similarity:
        model_similarity = _model_scores_from_streams(streams)

    timeline_windows = list(report.get("timeline_windows") or [])
    if not timeline_windows:
        timeline_windows = _timeline_from_streams(time_axis, final_off, sim_avg, confidence_curve)

    score_breakdown = dict(report.get("score_breakdown") or {})
    if not score_breakdown:
        score_breakdown = _score_breakdown_from_curves(
            overall=overall,
            intervals=intervals,
            final_off=final_off,
            sim_avg=sim_avg,
            confidence_curve=confidence_curve,
            per_part_score=per_part_score,
        )

    coaching = dict(report.get("coaching_report") or {})
    if not coaching:
        coaching = _fallback_coaching(overall, intervals, per_part_score, report.get("feedback") or [])

    return {
        "report_type": "integrated_v2" if int(report.get("report_version") or 1) >= 2 else "integrated",
        "overall_score": overall,
        "score_breakdown": score_breakdown,
        "body_part_rows": _body_part_rows(per_part_score, per_part_off),
        "model_rows": _model_rows(model_similarity),
        "timeline_windows": timeline_windows,
        "interval_rows": intervals,
        "coaching_report": coaching,
        "models_enabled": list(report.get("models_enabled") or []),
        "lstm_used": bool(report.get("lstm_used")),
        "lstm": dict(report.get("lstm") or {}),
        "fps": _safe_float(report.get("fps"), 0.0),
        "curves": _curves_payload(time_axis, final_off, sim_avg, confidence_curve),
        "raw_report": report,
    }


def _build_legacy_dashboard(report: dict[str, Any], streams: dict[str, np.ndarray]) -> dict[str, Any]:
    scores = dict(report.get("scores") or {})
    overall = _safe_float(scores.get("overall_score"), 0.0)
    per_part_score = _normalize_part_dict(scores.get("per_body_part_score") or {})
    per_part_off = {
        part: max(0.0, min(1.0, 1.0 - (_safe_float(score) / 100.0)))
        for part, score in per_part_score.items()
    }
    worst_windows = scores.get("worst_windows") or []
    timeline_windows = []
    window_times = scores.get("per_window_time_sec") or []
    window_scores = scores.get("per_window_score") or []
    for idx, score in enumerate(window_scores):
        start, end = (window_times[idx] if idx < len(window_times) else [float(idx), float(idx + 1)])
        timeline_windows.append(
            {
                "start_s": _safe_float(start),
                "end_s": _safe_float(end),
                "score": _safe_float(score),
                "confidence": _safe_float(score),
                "similarity": _safe_float(score),
                "off_pose_mean": max(0.0, min(1.0, 1.0 - _safe_float(score) / 100.0)),
                "weakest_part": "",
                "weakest_part_score": _safe_float(score),
                "mean_part_probability": 0.0,
            }
        )
    interval_rows = [
        {
            "start_s": _safe_float(item.get("start_sec")),
            "end_s": _safe_float(item.get("end_sec")),
            "part": "",
            "label": "Weak Window",
            "mean_error": max(0.0, min(1.0, 1.0 - _safe_float(item.get("score")) / 100.0)),
            "score": _safe_float(item.get("score")),
        }
        for item in worst_windows
    ]
    movement = _safe_float(scores.get("pose_geometry_score"), overall)
    confidence = _safe_float(scores.get("timing_score"), overall)
    breakdown = {
        "overall": overall,
        "movement_similarity": movement,
        "pose_control": _safe_float(scores.get("limb_angle_score"), overall),
        "confidence": confidence,
        "total_off_pose_time_s": _interval_union_duration(interval_rows),
        "interval_count": len(interval_rows),
    }
    coaching = _fallback_coaching(overall, interval_rows, per_part_score, report.get("feedback") or [])
    return {
        "report_type": "legacy",
        "overall_score": overall,
        "score_breakdown": breakdown,
        "body_part_rows": _body_part_rows(per_part_score, per_part_off),
        "model_rows": [],
        "timeline_windows": timeline_windows,
        "interval_rows": interval_rows,
        "coaching_report": coaching,
        "models_enabled": [],
        "lstm_used": False,
        "lstm": {},
        "fps": _safe_float(report.get("fps_used_for_timing"), 0.0),
        "curves": _curves_payload(_array(streams.get("time_axis")), _array(streams.get("final_off")), _array(streams.get("sim_avg")), _array(streams.get("confidence_curve"))),
        "raw_report": report,
    }


def _is_legacy(report: dict[str, Any]) -> bool:
    return isinstance(report.get("scores"), dict) and "overall_score" in report["scores"]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(out) or np.isinf(out):
        return default
    return out


def _array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _normalize_part_name(name: str) -> str:
    key = str(name).strip()
    upper = key.upper()
    if upper in PART_ORDER:
        return upper
    return LEGACY_PARTS.get(key.lower(), upper)


def _normalize_part_dict(values: dict[str, Any]) -> dict[str, float]:
    normalized = {
        _normalize_part_name(part): _safe_float(value)
        for part, value in values.items()
    }
    return {part: normalized[part] for part in PART_ORDER if part in normalized}


def _normalize_intervals(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in intervals:
        part = _normalize_part_name(str(item.get("part") or ""))
        mean_error = _safe_float(item.get("mean_error"))
        out.append(
            {
                "start_s": _safe_float(item.get("start_s", item.get("start_sec"))),
                "end_s": _safe_float(item.get("end_s", item.get("end_sec"))),
                "part": part if part in PART_ORDER else "",
                "label": PART_LABELS.get(part, part.replace("_", " ").title() if part else "Window"),
                "mean_error": mean_error,
                "score": _score_from_off(mean_error),
            }
        )
    return sorted(out, key=lambda item: (item["start_s"], item["part"]))


def _score_from_off(value: float) -> float:
    return float(max(0.0, min(100.0, (1.0 - _safe_float(value)) * 100.0)))


def _score_from_cosine(values: np.ndarray) -> float:
    arr = _array(values)
    if arr.size == 0:
        return 0.0
    return float(max(0.0, min(100.0, float((((arr.clip(-1.0, 1.0) + 1.0) * 0.5).mean()) * 100.0))))


def _model_scores_from_streams(streams: dict[str, np.ndarray]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, values in streams.items():
        if key.endswith("_cosine"):
            scores[key.removesuffix("_cosine")] = _score_from_cosine(values)
    return scores


def _part_scores_from_intervals(intervals: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    for item in intervals:
        part = item.get("part")
        if part in PART_ORDER:
            grouped.setdefault(part, []).append(_safe_float(item.get("mean_error")))
    off_mean = {part: float(np.mean(grouped[part])) for part in grouped}
    scores = {part: _score_from_off(off_mean[part]) for part in grouped}
    return scores, off_mean


def _score_breakdown_from_curves(
    *,
    overall: float,
    intervals: list[dict[str, Any]],
    final_off: np.ndarray,
    sim_avg: np.ndarray,
    confidence_curve: np.ndarray,
    per_part_score: dict[str, float],
) -> dict[str, float]:
    confidence = _safe_float(np.mean(confidence_curve) * 100.0) if confidence_curve.size else overall
    if final_off.size:
        confidence = _safe_float((1.0 - final_off.mean()) * 100.0)
    return {
        "overall": overall,
        "movement_similarity": _score_from_cosine(sim_avg) if sim_avg.size else overall,
        "pose_control": _safe_float(np.mean(list(per_part_score.values()))) if per_part_score else overall,
        "confidence": confidence,
        "total_off_pose_time_s": _interval_union_duration(intervals),
        "interval_count": len(intervals),
    }


def _timeline_from_streams(
    time_axis: np.ndarray,
    final_off: np.ndarray,
    sim_avg: np.ndarray,
    confidence_curve: np.ndarray,
) -> list[dict[str, Any]]:
    time = _array(time_axis)
    off = _array(final_off)
    if time.size == 0 or off.ndim != 2:
        return []
    sim = _array(sim_avg)
    conf = _array(confidence_curve)
    end = float(time[-1]) if time.size else 0.0
    windows: list[dict[str, Any]] = []
    start = 0.0
    while start <= end + 1e-6:
        stop = start + 1.0
        upper_mask = time < stop if stop <= end else time <= stop
        mask = (time >= start) & upper_mask
        if mask.any():
            segment = off[mask]
            part_mean = segment.mean(axis=0)
            weakest_idx = int(np.argmax(part_mean))
            score = _score_from_off(float(segment.mean()))
            similarity = _score_from_cosine(sim[mask]) if sim.size else score
            confidence = _safe_float(conf[mask].mean() * 100.0, score) if conf.size else score
            windows.append(
                {
                    "start_s": start,
                    "end_s": min(stop, end),
                    "score": score,
                    "confidence": confidence,
                    "similarity": similarity,
                    "off_pose_mean": float(segment.mean()),
                    "weakest_part": PART_ORDER[weakest_idx],
                    "weakest_part_score": _score_from_off(float(part_mean[weakest_idx])),
                    "mean_part_probability": float(segment.mean()),
                }
            )
        start = stop
    return windows


def _body_part_rows(per_part_score: dict[str, float], per_part_off: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for part in PART_ORDER:
        if part not in per_part_score:
            continue
        rows.append(
            {
                "part": part,
                "label": PART_LABELS[part],
                "score": _safe_float(per_part_score.get(part)),
                "off_pose_mean": _safe_float(per_part_off.get(part), max(0.0, 1.0 - _safe_float(per_part_score.get(part)) / 100.0)),
            }
        )
    return sorted(rows, key=lambda row: row["score"])


def _model_rows(model_similarity: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {"model": name, "score": _safe_float(score)}
        for name, score in sorted(model_similarity.items())
    ]


def _curves_payload(
    time_axis: np.ndarray,
    final_off: np.ndarray,
    sim_avg: np.ndarray,
    confidence_curve: np.ndarray,
) -> dict[str, Any]:
    time = _array(time_axis)
    off = _array(final_off)
    sim = _array(sim_avg)
    confidence = _array(confidence_curve)
    if confidence.size == 0 and off.size:
        confidence = 1.0 - off.mean(axis=1)
    similarity = ((sim.clip(-1.0, 1.0) + 1.0) * 0.5) if sim.size else np.asarray([], dtype=np.float32)
    return {
        "time_axis": time.tolist(),
        "confidence": (confidence * 100.0).tolist() if confidence.size else [],
        "similarity": (similarity * 100.0).tolist() if similarity.size else [],
        "final_off": off.tolist() if off.size else [],
        "part_order": list(PART_ORDER),
        "part_labels": [PART_LABELS[p] for p in PART_ORDER],
    }


def _fallback_coaching(
    overall: float,
    intervals: list[dict[str, Any]],
    per_part_score: dict[str, float],
    feedback: list[str],
) -> dict[str, Any]:
    ordered = sorted(per_part_score.items(), key=lambda item: item[1], reverse=True)
    strongest = ordered[:2]
    weakest = sorted(per_part_score.items(), key=lambda item: item[1])[:3]
    strengths = [
        f"{PART_LABELS.get(part, part)} was one of your cleaner areas ({score:.1f}/100)."
        for part, score in strongest
    ] or ["You have a complete run ready for detailed comparison."]
    priorities = [
        {
            "part": part,
            "label": PART_LABELS.get(part, part),
            "score": score,
            "off_pose_mean": max(0.0, 1.0 - score / 100.0),
            "flagged_time_s": 0.0,
            "main_window": "review the timeline",
            "cue": "Practice this section slowly, then return to tempo.",
        }
        for part, score in weakest
    ] or [
        {
            "part": "",
            "label": "Timing",
            "score": overall,
            "off_pose_mean": max(0.0, 1.0 - overall / 100.0),
            "flagged_time_s": _interval_union_duration(intervals),
            "main_window": "review the flagged windows",
            "cue": "Use the benchmark as a timing anchor and repeat the weakest window.",
        }
    ]
    timestamped = feedback or [
        "No sustained off-pose moments were detected. Keep the same control at full speed."
    ]
    return {
        "headline": f"{overall:.1f}/100 performance review",
        "summary": timestamped[0] if timestamped else f"Overall score: {overall:.1f}/100.",
        "strengths": strengths,
        "improvement_priorities": priorities,
        "practice_plan": [
            "Review the weakest window at half speed.",
            "Repeat the same phrase with the benchmark visible.",
            "Record one full-speed pass and compare whether the score improves.",
        ],
        "timestamped_feedback": timestamped,
        "status_notes": [],
    }


def _interval_union_duration(intervals: list[dict[str, Any]]) -> float:
    spans = sorted(
        (_safe_float(item.get("start_s")), _safe_float(item.get("end_s")))
        for item in intervals
        if _safe_float(item.get("end_s")) >= _safe_float(item.get("start_s"))
    )
    if not spans:
        return 0.0
    total = 0.0
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        total += cur_end - cur_start
        cur_start, cur_end = start, end
    total += cur_end - cur_start
    return float(max(0.0, total))
