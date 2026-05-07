"""Fresh scenario comparison driver.

Runs the integrated pipeline (HRNet + SimpleBaseline + GNN) on a curated set
of new pairs covering 5 invariance scenarios, then aggregates per-model
similarity scores and writes CSVs / charts / a summary.md to
``data/reports/fresh_scenario_comparison/``.

Scenarios:
    A. same_choreography_different_dancer  -- not achievable in the local
       AIST++ subset (each genre has a single dancer); kept as a documented
       gap rather than silently substituted.
    B. same_choreography_different_music
    C. same_genre_different_choreography
    D. same_dancer_different_choreography
    E. different_genre

Per-model similarity score is taken from ``report.json`` ->
``model_similarity_score.{hrnet,simple_baseline,gnn}`` (0-100).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
VIDEO_DIR = REPO / "data" / "videos"
OUT_ROOT = REPO / "data" / "reports" / "fresh_scenario_comparison"
RUNS_DIR = OUT_ROOT / "raw_runs"
CONFIG = REPO / "configs" / "integrate" / "pipeline.yaml"


def clip_meta(stem: str) -> dict:
    # gBR_sBM_c01_d04_mBR0_ch02
    parts = stem.split("_")
    return {
        "clip": stem,
        "genre": parts[0],
        "situation": parts[1],
        "camera": parts[2],
        "dancer": parts[3],
        "music": parts[4],
        "choreography": parts[5],
    }


PAIRS: list[tuple[str, str, str]] = [
    # B. same_choreography_different_music (same genre, same dancer, same chXX, different mXX#)
    ("B_same_choreography_different_music", "gBR_sBM_c01_d04_mBR0_ch02", "gBR_sBM_c01_d04_mBR1_ch02"),
    ("B_same_choreography_different_music", "gBR_sBM_c01_d04_mBR0_ch03", "gBR_sBM_c01_d04_mBR1_ch03"),
    ("B_same_choreography_different_music", "gBR_sBM_c01_d04_mBR0_ch07", "gBR_sBM_c01_d04_mBR1_ch07"),
    ("B_same_choreography_different_music", "gBR_sBM_c01_d04_mBR0_ch10", "gBR_sBM_c01_d04_mBR1_ch10"),
    ("B_same_choreography_different_music", "gHO_sBM_c01_d19_mHO0_ch02", "gHO_sBM_c01_d19_mHO1_ch02"),
    ("B_same_choreography_different_music", "gHO_sBM_c01_d19_mHO0_ch07", "gHO_sBM_c01_d19_mHO1_ch07"),
    ("B_same_choreography_different_music", "gHO_sBM_c01_d19_mHO0_ch09", "gHO_sBM_c01_d19_mHO1_ch09"),
    # C. same_genre_different_choreography (same dancer, same music, different choreography)
    ("C_same_genre_different_choreography", "gBR_sBM_c01_d04_mBR0_ch02", "gBR_sBM_c01_d04_mBR0_ch07"),
    ("C_same_genre_different_choreography", "gBR_sBM_c01_d04_mBR0_ch08", "gBR_sBM_c01_d04_mBR0_ch10"),
    ("C_same_genre_different_choreography", "gBR_sBM_c01_d04_mBR1_ch03", "gBR_sBM_c01_d04_mBR1_ch09"),
    ("C_same_genre_different_choreography", "gBR_sBM_c01_d04_mBR1_ch02", "gBR_sBM_c01_d04_mBR1_ch08"),
    ("C_same_genre_different_choreography", "gHO_sBM_c01_d19_mHO0_ch01", "gHO_sBM_c01_d19_mHO0_ch08"),
    ("C_same_genre_different_choreography", "gHO_sBM_c01_d19_mHO1_ch04", "gHO_sBM_c01_d19_mHO1_ch10"),
    # D. same_dancer_different_choreography (same dancer; allow different music + different choreography)
    ("D_same_dancer_different_choreography", "gBR_sBM_c01_d04_mBR0_ch02", "gBR_sBM_c01_d04_mBR1_ch07"),
    ("D_same_dancer_different_choreography", "gBR_sBM_c01_d04_mBR1_ch03", "gBR_sBM_c01_d04_mBR0_ch10"),
    ("D_same_dancer_different_choreography", "gBR_sBM_c01_d04_mBR0_ch06", "gBR_sBM_c01_d04_mBR1_ch08"),
    ("D_same_dancer_different_choreography", "gHO_sBM_c01_d19_mHO0_ch02", "gHO_sBM_c01_d19_mHO1_ch09"),
    ("D_same_dancer_different_choreography", "gHO_sBM_c01_d19_mHO1_ch03", "gHO_sBM_c01_d19_mHO0_ch10"),
    # E. different_genre (gBR d04 vs gHO d19; sBM + c01 fixed)
    ("E_different_genre", "gBR_sBM_c01_d04_mBR0_ch02", "gHO_sBM_c01_d19_mHO0_ch02"),
    ("E_different_genre", "gBR_sBM_c01_d04_mBR0_ch07", "gHO_sBM_c01_d19_mHO0_ch07"),
    ("E_different_genre", "gBR_sBM_c01_d04_mBR1_ch03", "gHO_sBM_c01_d19_mHO1_ch03"),
    ("E_different_genre", "gBR_sBM_c01_d04_mBR1_ch08", "gHO_sBM_c01_d19_mHO1_ch08"),
    ("E_different_genre", "gBR_sBM_c01_d04_mBR0_ch10", "gHO_sBM_c01_d19_mHO1_ch06"),
]


def pair_id(scenario: str, bench_stem: str, learner_stem: str) -> str:
    short = scenario.split("_", 1)[0]
    return f"{short}__{bench_stem}__VS__{learner_stem}"


def run_one(scenario: str, bench_stem: str, learner_stem: str, *, device: str = "mps") -> dict:
    bench = VIDEO_DIR / f"{bench_stem}.mp4"
    learner = VIDEO_DIR / f"{learner_stem}.mp4"
    assert bench.exists() and learner.exists(), f"missing video: {bench} / {learner}"

    out_dir = RUNS_DIR / pair_id(scenario, bench_stem, learner_stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"

    if not report_path.exists():
        cmd = [
            sys.executable,
            "run.py",
            "--benchmark", str(bench),
            "--learner",   str(learner),
            "--out",       str(out_dir),
            "--config",    str(CONFIG),
            "--device",    device,
            "--no-lstm",
        ]
        env = {"PYTHONPATH": str(REPO)}
        t0 = time.time()
        proc = subprocess.run(
            cmd, cwd=REPO, env={**env, **__import__("os").environ},
            capture_output=True, text=True
        )
        dt = time.time() - t0
        if proc.returncode != 0:
            (out_dir / "stderr.log").write_text(proc.stderr)
            (out_dir / "stdout.log").write_text(proc.stdout)
            raise RuntimeError(f"pipeline failed for {bench_stem} vs {learner_stem}: see {out_dir}")
        print(f"  done in {dt:.1f}s")

    report = json.loads(report_path.read_text())
    sims = report["model_similarity_score"]
    return {
        "scenario": scenario,
        "benchmark": bench_stem,
        "learner": learner_stem,
        **{f"benchmark_{k}": v for k, v in clip_meta(bench_stem).items() if k != "clip"},
        **{f"learner_{k}":   v for k, v in clip_meta(learner_stem).items() if k != "clip"},
        "simple_baseline_similarity": float(sims["simple_baseline"]),
        "hrnet_similarity":           float(sims["hrnet"]),
        "gnn_similarity":             float(sims["gnn"]),
        "overall_score":              float(report.get("overall_score", float("nan"))),
        "report_dir": str(out_dir.relative_to(REPO)),
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n = len(PAIRS)
    for i, (scenario, b, l) in enumerate(PAIRS, 1):
        print(f"[{i}/{n}] {scenario}  {b}  vs  {l}")
        rows.append(run_one(scenario, b, l))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "selected_pairs.csv", index=False)
    print(f"wrote {OUT_ROOT/'selected_pairs.csv'} ({len(df)} rows)")

    g = df.groupby("scenario")
    summary = pd.DataFrame({
        "n_pairs": g.size(),
        "simple_baseline_mean": g["simple_baseline_similarity"].mean(),
        "simple_baseline_std":  g["simple_baseline_similarity"].std(),
        "hrnet_mean":           g["hrnet_similarity"].mean(),
        "hrnet_std":            g["hrnet_similarity"].std(),
        "gnn_mean":             g["gnn_similarity"].mean(),
        "gnn_std":              g["gnn_similarity"].std(),
    }).reset_index()
    summary.to_csv(OUT_ROOT / "scenario_summary.csv", index=False)
    print(f"wrote {OUT_ROOT/'scenario_summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
