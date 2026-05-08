"""Generate paper-quality figures for the final report.

Outputs saved to figures/ (sibling of the .tex file):
  fig1_training.png       LSTM training curves (both runs)
  fig2_per_part.png       Per-part test metrics grouped bar
  fig3_intervals.png      Interval count reduction (threshold vs model)
  fig4_gantt.png          phrase_05 Gantt timeline
  fig5_pipeline_score.png Full-pipeline score with vs without LSTM
"""

import csv, json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG  = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 200,
    "lines.linewidth": 1.6,
})

# Okabe–Ito colour-blind-safe palette (IEEE/Nature recommended)
BLUE   = "#4878CF"   # steel blue
ORANGE = "#D65F00"   # burnt sienna
GREEN  = "#6ACC65"   # sage green
RED    = "#B47CC7"   # muted lavender (replaces harsh red)
GREY   = "#777777"
PURPLE = "#C4AD66"   # warm tan
BROWN  = "#77BEDB"   # sky blue

# Sequential accent used for bar pairs
BAR_A  = "#4878CF"   # threshold bars
BAR_B  = "#D65F00"   # model bars

PART_COLOURS = {
    "LEFT_ARM":  "#4878CF",   # steel blue
    "RIGHT_ARM": "#D65F00",   # burnt sienna
    "LEFT_LEG":  "#6ACC65",   # sage green
    "RIGHT_LEG": "#B47CC7",   # muted lavender
    "HEAD":      "#C4AD66",   # warm tan
    "TORSO":     "#77BEDB",   # sky blue
}

def read_csv(path):
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]

def save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → figures/{name}")


# ── Fig 1: Training curve (integrate LSTM run) ───────────────────────────────
def fig1_training():
    integ = read_csv(ROOT / "results" / "lstm_training_log.csv")
    ep_i  = [r["epoch"] for r in integ]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
    fig.suptitle("LSTM Temporal Error Detector — Training Curves", fontsize=9, y=1.01)

    ax = axes[0]
    ax.plot(ep_i, [r["train_loss"] for r in integ], color="#4878CF", label="Train loss",  linestyle="-")
    ax.plot(ep_i, [r["val_loss"]   for r in integ], color="#D65F00", label="Val loss",     linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.set_title("(a) Training / Validation Loss")
    ax.legend(loc="upper right"); ax.grid(alpha=0.25)

    ax = axes[1]
    vf1 = [r["val_f1"] for r in integ]
    ax.plot(ep_i, vf1, color="#6ACC65", label="Val F1")
    best_ep = int(integ[int(np.argmax(vf1))]["epoch"])
    best_f1 = max(vf1)
    ax.axvline(best_ep, color=GREY, linewidth=0.9, linestyle=":")
    ax.scatter([best_ep], [best_f1], color="#6ACC65", zorder=5, s=25)
    ax.annotate(f" {best_f1:.3f}\n ep {best_ep}",
                xy=(best_ep, best_f1), fontsize=7.5, color=GREY)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation F1")
    ax.set_title("(b) Validation F1")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    save(fig, "fig1_training.png")


# ── Fig 2: Per-part test metrics (from Mia Oscar evaluation) ─────────────────
def fig2_per_part():
    # Metrics from Mia branch Oscar evaluation (test set = ch10 choreographies)
    data = {
        "LEFT_ARM":  {"precision": 0.9386, "f1": 0.9683, "auc": 0.8808},
        "RIGHT_ARM": {"precision": 0.9781, "f1": 0.9889, "auc": 0.8273},
        "LEFT_LEG":  {"precision": 0.9430, "f1": 0.9707, "auc": 0.8728},
        "RIGHT_LEG": {"precision": 0.9693, "f1": 0.9844, "auc": 0.8440},
        "HEAD":      {"precision": 0.8787, "f1": 0.9354, "auc": 0.7282},
        "TORSO":     {"precision": 0.8068, "f1": 0.8902, "auc": 0.7946},
    }
    parts = list(data.keys())
    prec  = [data[p]["precision"] for p in parts]
    f1    = [data[p]["f1"]        for p in parts]
    auc   = [data[p]["auc"]       for p in parts]
    short = [p.replace("_", "\n") for p in parts]

    x = np.arange(len(parts))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    b1 = ax.bar(x - w, prec, w, label="Precision", color="#4878CF", alpha=0.85)
    b2 = ax.bar(x,     f1,   w, label="F1",        color="#6ACC65", alpha=0.85)
    b3 = ax.bar(x + w, auc,  w, label="ROC AUC",   color="#C4AD66", alpha=0.85)

    for bars in (b1, b2, b3):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                    f"{bar.get_height():.2f}", ha="center", fontsize=6.5)

    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylim(0.65, 1.07)
    ax.set_ylabel("Score")
    ax.set_title("Per-Body-Part LSTM Test Metrics on Held-Out Choreographies (Recall $\\approx$ 1.0 everywhere)")
    ax.legend(loc="lower left"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig2_per_part.png")


# ── Fig 3: Interval reduction ─────────────────────────────────────────────────
def fig3_intervals():
    phrases = [f"P{i:02d}" for i in range(1, 6)]
    thr = [4, 12, 4, 18, 36]
    mdl = [2,  9, 1,  7, 18]
    x = np.arange(5); w = 0.35

    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    bt = ax.bar(x - w/2, thr, w, label="Fixed threshold", color="#B47CC7", alpha=0.85)
    bm = ax.bar(x + w/2, mdl, w, label="LSTM model",      color="#4878CF", alpha=0.85)
    for bar in list(bt) + list(bm):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(int(bar.get_height())), ha="center", fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(phrases)
    ax.set_ylabel("Flagged intervals")
    ax.set_title(f"Interval Reduction: Threshold (74 total) $\\rightarrow$ LSTM (37 total, $-$50\\%)")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig3_intervals.png")


# ── Fig 4: phrase_05 Gantt (hardcoded from Mia branch results) ───────────────
def fig4_gantt():
    # From results_threshold/phrase_05/report.json (36 intervals)
    thr_data = [
        {"start_s":0.0,"end_s":0.53,"part":"HEAD"},{"start_s":0.13,"end_s":0.6,"part":"LEFT_ARM"},
        {"start_s":0.67,"end_s":1.2,"part":"RIGHT_LEG"},{"start_s":1.8,"end_s":3.13,"part":"LEFT_ARM"},
        {"start_s":1.8,"end_s":4.07,"part":"RIGHT_ARM"},{"start_s":1.87,"end_s":2.6,"part":"LEFT_LEG"},
        {"start_s":2.13,"end_s":2.8,"part":"HEAD"},{"start_s":2.27,"end_s":3.13,"part":"RIGHT_LEG"},
        {"start_s":3.33,"end_s":3.87,"part":"RIGHT_LEG"},{"start_s":4.2,"end_s":6.67,"part":"LEFT_ARM"},
        {"start_s":4.2,"end_s":7.0,"part":"RIGHT_ARM"},{"start_s":4.27,"end_s":5.13,"part":"HEAD"},
        {"start_s":5.2,"end_s":5.87,"part":"LEFT_LEG"},{"start_s":5.2,"end_s":5.87,"part":"RIGHT_LEG"},
        {"start_s":5.4,"end_s":6.07,"part":"TORSO"},{"start_s":6.73,"end_s":7.87,"part":"HEAD"},
        {"start_s":7.07,"end_s":8.4,"part":"RIGHT_ARM"},{"start_s":7.07,"end_s":8.4,"part":"LEFT_ARM"},
        {"start_s":7.47,"end_s":8.0,"part":"LEFT_LEG"},{"start_s":7.47,"end_s":8.0,"part":"RIGHT_LEG"},
        {"start_s":7.6,"end_s":8.27,"part":"TORSO"},{"start_s":8.53,"end_s":9.6,"part":"HEAD"},
        {"start_s":8.53,"end_s":10.13,"part":"LEFT_ARM"},{"start_s":8.53,"end_s":10.13,"part":"RIGHT_ARM"},
        {"start_s":8.67,"end_s":9.33,"part":"LEFT_LEG"},{"start_s":8.67,"end_s":9.33,"part":"RIGHT_LEG"},
        {"start_s":8.8,"end_s":9.47,"part":"TORSO"},{"start_s":9.8,"end_s":10.8,"part":"HEAD"},
        {"start_s":10.2,"end_s":11.07,"part":"LEFT_ARM"},{"start_s":10.2,"end_s":11.07,"part":"RIGHT_ARM"},
        {"start_s":10.27,"end_s":11.07,"part":"LEFT_LEG"},{"start_s":10.27,"end_s":11.07,"part":"RIGHT_LEG"},
        {"start_s":10.4,"end_s":11.07,"part":"TORSO"},{"start_s":11.2,"end_s":11.87,"part":"LEFT_ARM"},
        {"start_s":11.2,"end_s":11.87,"part":"RIGHT_ARM"},{"start_s":11.33,"end_s":11.87,"part":"HEAD"},
    ]
    # From results/phrase_05/report.json (18 intervals — LSTM)
    mdl_data = [
        {"start_s":0.0,"end_s":1.6,"part":"LEFT_ARM"},{"start_s":0.07,"end_s":1.6,"part":"RIGHT_ARM"},
        {"start_s":2.0,"end_s":11.87,"part":"RIGHT_ARM"},{"start_s":2.07,"end_s":11.87,"part":"LEFT_ARM"},
        {"start_s":2.2,"end_s":3.6,"part":"LEFT_LEG"},{"start_s":2.2,"end_s":3.53,"part":"RIGHT_LEG"},
        {"start_s":2.2,"end_s":4.47,"part":"HEAD"},{"start_s":2.27,"end_s":3.2,"part":"TORSO"},
        {"start_s":5.13,"end_s":11.87,"part":"HEAD"},{"start_s":5.2,"end_s":7.27,"part":"LEFT_LEG"},
        {"start_s":5.2,"end_s":7.27,"part":"RIGHT_LEG"},{"start_s":5.2,"end_s":5.87,"part":"TORSO"},
        {"start_s":7.93,"end_s":10.53,"part":"LEFT_LEG"},{"start_s":7.93,"end_s":10.53,"part":"RIGHT_LEG"},
        {"start_s":8.0,"end_s":9.07,"part":"TORSO"},{"start_s":10.87,"end_s":11.87,"part":"LEFT_LEG"},
        {"start_s":10.87,"end_s":11.87,"part":"RIGHT_LEG"},{"start_s":11.0,"end_s":11.87,"part":"TORSO"},
    ]

    parts = ["LEFT_ARM", "RIGHT_ARM", "LEFT_LEG", "RIGHT_LEG", "HEAD", "TORSO"]
    ymap  = {p: i for i, p in enumerate(parts)}

    fig, (ax_t, ax_m) = plt.subplots(2, 1, figsize=(6.5, 3.8), sharex=True)
    fig.suptitle("phrase\_05 Flagged Intervals by Body Part", fontsize=9)

    def draw(ax, ivs, title):
        ax.set_title(title, fontsize=8)
        ax.set_yticks(range(len(parts)))
        ax.set_yticklabels([p.replace("_", " ").title() for p in parts], fontsize=7)
        ax.set_xlim(0, 12); ax.grid(axis="x", alpha=0.2)
        ax.set_xlabel("Time (s)", fontsize=7.5)
        for iv in ivs:
            y = ymap[iv["part"]]
            ax.barh(y, iv["end_s"] - iv["start_s"], left=iv["start_s"],
                    height=0.55, color=PART_COLOURS[iv["part"]], alpha=0.78,
                    edgecolor="white", linewidth=0.4)

    draw(ax_t, thr_data, f"(a) Fixed Threshold — {len(thr_data)} intervals")
    draw(ax_m, mdl_data, f"(b) LSTM Model — {len(mdl_data)} intervals")

    handles = [mpatches.Patch(color=PART_COLOURS[p], label=p.replace("_", " ").title()) for p in parts]
    fig.legend(handles=handles, loc="upper right", fontsize=7, ncol=2,
               bbox_to_anchor=(0.99, 0.98))
    fig.tight_layout()
    save(fig, "fig4_gantt.png")


# ── Fig 5: Full-pipeline score with vs without LSTM ──────────────────────────
def fig5_pipeline_score():
    runs = [
        ("No LSTM\n(threshold)", 56.76, 7,  "#B47CC7"),
        ("With LSTM",            93.89, 0,  "#4878CF"),
    ]
    labels = [r[0] for r in runs]
    scores = [r[1] for r in runs]
    n_ivs  = [r[2] for r in runs]
    colors = [r[3] for r in runs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.0, 2.6))
    fig.suptitle("Integrated Pipeline: LSTM vs Threshold on Same Video Pair", fontsize=8.5)

    bars = ax1.bar(labels, scores, color=colors, alpha=0.85, width=0.45)
    for bar, s in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{s:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 110); ax1.set_ylabel("Overall Score (0–100)")
    ax1.set_title("(a) Score"); ax1.grid(axis="y", alpha=0.25)

    bars2 = ax2.bar(labels, n_ivs, color=colors, alpha=0.85, width=0.45)
    for bar, n in zip(bars2, n_ivs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 str(n), ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Flagged Intervals"); ax2.set_ylim(0, 12)
    ax2.set_title("(b) Flagged Intervals"); ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save(fig, "fig5_pipeline_score.png")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_training()
    fig2_per_part()
    fig3_intervals()
    fig4_gantt()
    fig5_pipeline_score()
    print(f"\nAll figures saved to figures/")
