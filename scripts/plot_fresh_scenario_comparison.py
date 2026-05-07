"""Aggregate selected_pairs.csv into scenario-level summary + charts.

Reads ``data/reports/fresh_scenario_comparison/selected_pairs.csv`` and
writes:
  - scenario_summary.csv        (mean / std per model per scenario)
  - scenario_comparison_chart.{png,pdf}  (grouped bar chart, error bars)
  - score_distribution.{png,pdf}         (per-pair strip + box per model)
  - summary.md                  (selection notes + main findings)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "reports" / "fresh_scenario_comparison"

MODELS = [
    ("simple_baseline_similarity", "SimpleBaseline", "#7c8aa6"),
    ("hrnet_similarity",           "HRNet",          "#1f77b4"),
    ("gnn_similarity",             "GNN",            "#d6336c"),
]

SCENARIO_LABELS = {
    "B_same_choreography_different_music":  "B. same choreo\ndifferent music",
    "C_same_genre_different_choreography":  "C. same genre\ndifferent choreo",
    "D_same_dancer_different_choreography": "D. same dancer\ndifferent choreo",
    "E_different_genre":                    "E. different\ngenre",
}


def main() -> None:
    df = pd.read_csv(OUT / "selected_pairs.csv")
    scenarios = [s for s in SCENARIO_LABELS if s in df["scenario"].unique()]
    if not scenarios:
        raise SystemExit("no scenarios found in selected_pairs.csv")

    # Summary
    g = df.groupby("scenario")
    summary = pd.DataFrame({
        "n_pairs": g.size(),
        "simple_baseline_mean": g["simple_baseline_similarity"].mean(),
        "simple_baseline_std":  g["simple_baseline_similarity"].std(ddof=1),
        "hrnet_mean":           g["hrnet_similarity"].mean(),
        "hrnet_std":            g["hrnet_similarity"].std(ddof=1),
        "gnn_mean":             g["gnn_similarity"].mean(),
        "gnn_std":              g["gnn_similarity"].std(ddof=1),
    }).reindex(scenarios).reset_index()
    summary.to_csv(OUT / "scenario_summary.csv", index=False)

    # ---- Grouped bar chart (poster-friendly) ----
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    n_scen = len(scenarios)
    n_models = len(MODELS)
    width = 0.78 / n_models
    x = np.arange(n_scen)
    for i, (col, label, color) in enumerate(MODELS):
        means = [df[df.scenario == s][col].mean() for s in scenarios]
        stds  = [df[df.scenario == s][col].std(ddof=1) for s in scenarios]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, means, width=width, label=label, color=color,
               yerr=stds, capsize=4, edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], fontsize=10)
    ax.set_ylabel("Similarity score (0–100)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Per-model similarity across invariance scenarios (fresh AIST++ sBM/c01 pairs)",
                 fontsize=12, pad=12)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "scenario_comparison_chart.png", dpi=200)
    fig.savefig(OUT / "scenario_comparison_chart.pdf")
    plt.close(fig)

    # ---- Distribution chart (per-pair strip + box) ----
    fig, ax = plt.subplots(figsize=(11, 5.6))
    box_positions = []
    box_data = []
    box_colors = []
    rng = np.random.default_rng(42)
    for si, s in enumerate(scenarios):
        sub = df[df.scenario == s]
        for mi, (col, _label, color) in enumerate(MODELS):
            pos = si * (n_models + 1.2) + mi
            vals = sub[col].values
            box_positions.append(pos)
            box_data.append(vals)
            box_colors.append(color)
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter([pos] * len(vals) + jitter, vals,
                       color=color, alpha=0.75, s=22, zorder=3, edgecolor="white", linewidth=0.4)

    bp = ax.boxplot(box_data, positions=box_positions, widths=0.55, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(color="#444", linewidth=0.9),
                    capprops=dict(color="#444", linewidth=0.9),
                    flierprops=dict(marker=""))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.22)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.0)

    centers = [si * (n_models + 1.2) + (n_models - 1) / 2 for si, _ in enumerate(scenarios)]
    ax.set_xticks(centers)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], fontsize=10)
    ax.set_ylabel("Similarity score (0–100)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Per-pair score distributions by scenario and model", fontsize=12, pad=12)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6, label=lbl) for _, lbl, c in MODELS]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "score_distribution.png", dpi=200)
    fig.savefig(OUT / "score_distribution.pdf")
    plt.close(fig)

    # ---- summary.md ----
    lines: list[str] = []
    lines.append("# Fresh scenario comparison — summary\n")
    lines.append("Per-model similarity scores (0–100) on freshly selected video pairs from the local AIST++ subset.\n")
    lines.append("Basic Dance only (sBM), camera c01, no LSTM head. Score field used for every model: ")
    lines.append("`report.json -> model_similarity_score.{simple_baseline, hrnet, gnn}` from the integrated pipeline.\n\n")
    lines.append("## Pair selection\n")
    lines.append("Local data has only two genres (gBR with dancer d04, gHO with dancer d19). ")
    lines.append("**Scenario A (same choreography, different dancer) is not achievable** with the current local dataset, ")
    lines.append("so it is intentionally omitted rather than substituted. ")
    lines.append("Pairs avoid the previously overused `ch04`/`mBR0`–`mBR1` combinations seen in `data/reports/ch04_*` and ")
    lines.append("`scripts/plot_old_gnn_multi_similarity.py`, and lean on chXX values 02/03/06/07/08/09/10 plus the gHO clips.\n\n")
    lines.append(f"- Total pairs: **{len(df)}**\n")
    for s in scenarios:
        n = int((df.scenario == s).sum())
        lines.append(f"- {SCENARIO_LABELS[s].replace(chr(10),' ')}: {n} pairs\n")

    lines.append("\n## Scenario summary\n\n")
    lines.append(summary.round(2).to_markdown(index=False))
    lines.append("\n\n## Main findings\n")

    # Compute discriminative power: gap from the easiest scenario (B) to the hardest (E).
    if {"B_same_choreography_different_music", "E_different_genre"}.issubset(set(scenarios)):
        sep_rows = []
        for col, label, _ in MODELS:
            top = df[df.scenario == "B_same_choreography_different_music"][col].mean()
            bot = df[df.scenario == "E_different_genre"][col].mean()
            sep_rows.append((label, top, bot, top - bot))
        sep_rows.sort(key=lambda r: r[3], reverse=True)
        lines.append("Separation between the easiest (B: same-choreography, different-music) and hardest ")
        lines.append("(E: different-genre) scenarios — larger gap = better discrimination:\n\n")
        lines.append("| Model | mean(B) | mean(E) | gap (B-E) |\n|---|---:|---:|---:|\n")
        for label, top, bot, gap in sep_rows:
            lines.append(f"| {label} | {top:.2f} | {bot:.2f} | **{gap:.2f}** |\n")
        lines.append(f"\nMost discriminative model: **{sep_rows[0][0]}**.\n")

    lines.append("\n## Files\n")
    lines.append("- `selected_pairs.csv` — every pair, with scenario / metadata / per-model similarity\n")
    lines.append("- `scenario_summary.csv` — n_pairs, per-model mean/std per scenario\n")
    lines.append("- `scenario_comparison_chart.png` / `.pdf` — grouped bar chart with error bars (main figure)\n")
    lines.append("- `score_distribution.png` / `.pdf` — per-pair boxplot + strip plot\n")
    lines.append("- `raw_runs/<pair_id>/report.json` — pipeline output for each pair\n")

    (OUT / "summary.md").write_text("".join(lines))
    print("wrote", OUT / "summary.md")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
