from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.app.report_data import build_dashboard_data, discover_report_runs, load_dashboard_data


def _streams() -> dict[str, np.ndarray]:
    time_axis = np.linspace(0.0, 2.0, 6, dtype=np.float32)
    final_off = np.array(
        [
            [0.10, 0.20, 0.10, 0.20, 0.10, 0.05],
            [0.15, 0.25, 0.10, 0.20, 0.10, 0.05],
            [0.20, 0.60, 0.15, 0.25, 0.20, 0.10],
            [0.20, 0.70, 0.20, 0.30, 0.20, 0.10],
            [0.10, 0.25, 0.10, 0.20, 0.10, 0.05],
            [0.10, 0.20, 0.10, 0.20, 0.10, 0.05],
        ],
        dtype=np.float32,
    )
    sim_avg = np.array([0.95, 0.92, 0.75, 0.70, 0.90, 0.94], dtype=np.float32)
    return {
        "time_axis": time_axis,
        "final_off": final_off,
        "confidence_curve": 1.0 - final_off.mean(axis=1),
        "sim_avg": sim_avg,
        "hrnet_cosine": sim_avg,
        "gnn_cosine": np.ones_like(sim_avg) * 0.96,
    }


def test_dashboard_data_uses_integrated_v2_fields() -> None:
    report = {
        "report_version": 2,
        "overall_score": 82.0,
        "intervals": [{"start_s": 0.8, "end_s": 1.3, "part": "RIGHT_ARM", "mean_error": 0.62}],
        "score_breakdown": {
            "overall": 82.0,
            "movement_similarity": 91.0,
            "pose_control": 80.0,
            "confidence": 84.0,
            "total_off_pose_time_s": 0.5,
            "interval_count": 1,
        },
        "per_body_part_score": {"RIGHT_ARM": 55.0, "LEFT_ARM": 88.0},
        "per_body_part_off_mean": {"RIGHT_ARM": 0.45, "LEFT_ARM": 0.12},
        "model_similarity_score": {"hrnet": 87.0},
        "timeline_windows": [{"start_s": 0.0, "end_s": 1.0, "score": 80.0, "weakest_part": "RIGHT_ARM"}],
        "coaching_report": {
            "headline": "82.0/100 - Strong match",
            "summary": "Good run.",
            "strengths": ["Left arm was strong."],
            "improvement_priorities": [{"label": "Right Arm", "score": 55.0, "cue": "Clean the pathway."}],
            "practice_plan": ["Repeat the phrase."],
            "timestamped_feedback": ["0.8s to 1.3s: right arm drifted."],
            "status_notes": [],
        },
        "models_enabled": ["hrnet", "gnn"],
        "lstm_used": True,
    }

    data = build_dashboard_data(report, _streams())

    assert data["report_type"] == "integrated_v2"
    assert data["score_breakdown"]["overall"] == 82.0
    assert data["body_part_rows"][0]["part"] == "RIGHT_ARM"
    assert data["model_rows"] == [{"model": "hrnet", "score": 87.0}]
    assert data["model_score_summary"]["weighted_model_score"] == 87.0
    assert data["model_curves"]["hrnet"]["mean_score"] > 90.0
    assert np.isclose(data["model_curves"]["gnn"]["mean_cosine"], 0.96)
    assert data["timeline_windows"][0]["weakest_part"] == "RIGHT_ARM"
    assert data["coaching_report"]["summary"] == "Good run."


def test_dashboard_data_builds_current_integrated_fallback_from_streams() -> None:
    report = {
        "overall_score": 76.0,
        "intervals": [{"start_s": 0.8, "end_s": 1.3, "part": "RIGHT_ARM", "mean_error": 0.62}],
        "feedback": ["At 0.8 s to 1.3 s your right arm is off."],
        "models_enabled": ["hrnet", "gnn"],
        "lstm_used": False,
        "fps": 30.0,
    }

    data = build_dashboard_data(report, _streams())

    assert data["report_type"] == "integrated"
    assert data["score_breakdown"]["interval_count"] == 1
    assert len(data["body_part_rows"]) == 6
    assert data["body_part_rows"][0]["part"] == "RIGHT_ARM"
    assert {row["model"] for row in data["model_rows"]} == {"hrnet", "gnn"}
    assert {row["model"] for row in data["model_score_summary"]["rows"]} == {"hrnet", "gnn"}
    assert set(data["model_curves"]) == {"hrnet", "gnn"}
    assert data["timeline_windows"]
    assert data["coaching_report"]["practice_plan"]


def test_dashboard_data_handles_legacy_nested_scores() -> None:
    report = {
        "fps_used_for_timing": 30.0,
        "scores": {
            "overall_score": 61.0,
            "pose_geometry_score": 70.0,
            "limb_angle_score": 52.0,
            "timing_score": 66.0,
            "per_body_part_score": {"left_arm": 72.0, "right_arm": 45.0},
            "per_window_score": [72.0, 48.0],
            "per_window_time_sec": [[0.0, 1.0], [1.0, 2.0]],
            "worst_windows": [{"start_sec": 1.0, "end_sec": 2.0, "score": 48.0}],
        },
        "feedback": ["Overall similarity vs the benchmark: 61.0/100."],
    }

    data = build_dashboard_data(report, {})

    assert data["report_type"] == "legacy"
    assert data["overall_score"] == 61.0
    assert data["body_part_rows"][0]["part"] == "RIGHT_ARM"
    assert data["model_score_summary"]["rows"] == []
    assert data["timeline_windows"][1]["score"] == 48.0
    assert data["interval_rows"][0]["label"] == "Weak Window"


def test_load_dashboard_data_degrades_when_streams_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    report = {
        "overall_score": 90.0,
        "intervals": [],
        "feedback": ["Great job."],
        "models_enabled": ["hrnet"],
        "lstm_used": False,
    }
    (run_dir / "report.json").write_text(json.dumps(report))

    data = load_dashboard_data(run_dir)

    assert data["report_type"] == "integrated"
    assert data["curves"]["time_axis"] == []
    assert data["coaching_report"]["timestamped_feedback"] == ["Great job."]


def test_discover_report_runs_sorts_by_newest(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "report.json").write_text("{}")
    (new / "report.json").write_text("{}")
    (old / "report.json").touch()
    (new / "report.json").touch()

    runs = discover_report_runs([tmp_path])

    assert {run.path.name for run in runs} == {"old", "new"}
    assert runs[0].modified >= runs[1].modified
