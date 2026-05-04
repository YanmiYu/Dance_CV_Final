"""
Generate result visualizations:
  1. Training curves (loss + val F1)
  2. Per-part test metrics (Precision / F1 / AUC)
  3. Interval count comparison: threshold vs model
  4. Interval timeline (Gantt) for phrase_05
"""

import json
import csv
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT  = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── colour palette ───────────────────────────────────────────────────────────
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
PURPLE = "#8172B2"
BROWN  = "#937860"
PART_COLOURS = {
    "LEFT_ARM":  BLUE,
    "RIGHT_ARM": ORANGE,
    "LEFT_LEG":  GREEN,
    "RIGHT_LEG": RED,
    "HEAD":      PURPLE,
    "TORSO":     BROWN,
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows

def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p.relative_to(ROOT)}")

# ── 1. Training curves ────────────────────────────────────────────────────────

def plot_training():
    rows  = load_csv(ROOT / "results" / "training_log.csv")
    epochs = [r["epoch"] for r in rows]
    t_loss = [r["train_loss"] for r in rows]
    v_loss = [r["val_loss"]   for r in rows]
    v_f1   = [r["val_f1"]     for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("LSTM Temporal Error Detector — Training Curves", fontsize=13, y=1.01)

    ax1.plot(epochs, t_loss, color=BLUE,   label="Train loss", linewidth=2)
    ax1.plot(epochs, v_loss, color=ORANGE, label="Val loss",   linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, v_f1, color=GREEN, linewidth=2)
    best_ep = int(rows[np.argmax(v_f1)]["epoch"])
    best_f1 = max(v_f1)
    ax2.axvline(best_ep, color="grey", linestyle=":", linewidth=1.2)
    ax2.scatter([best_ep], [best_f1], color=GREEN, zorder=5)
    ax2.annotate(f" best {best_f1:.4f}\n epoch {best_ep}",
                 xy=(best_ep, best_f1), fontsize=8.5, color="grey")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val F1")
    ax2.set_title("Validation F1"); ax2.grid(alpha=0.3)

    fig.tight_layout()
    save(fig, "1_training_curves.png")

# ── 2. Per-part test metrics ──────────────────────────────────────────────────

def plot_per_part():
    data   = load_json(ROOT / "results" / "test_metrics.json")["per_part"]
    parts  = list(data.keys())
    prec   = [data[p]["precision"] for p in parts]
    f1     = [data[p]["f1"]        for p in parts]
    auc    = [data[p]["auc"]       for p in parts]

    x = np.arange(len(parts))
    w = 0.25
    short = [p.replace("_", "\n") for p in parts]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w, prec, w, label="Precision", color=BLUE,   alpha=0.85)
    ax.bar(x,     f1,   w, label="F1",        color=GREEN,  alpha=0.85)
    ax.bar(x + w, auc,  w, label="ROC AUC",   color=ORANGE, alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
    ax.set_ylim(0.65, 1.05)
    ax.set_ylabel("Score"); ax.set_title("Per-Part Test Metrics (Recall ≈ 1.0 for all parts)")
    ax.legend(loc="lower left"); ax.grid(axis="y", alpha=0.3)

    for i, (p, f, a) in enumerate(zip(prec, f1, auc)):
        ax.text(i - w, p + 0.005, f"{p:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(i,     f + 0.005, f"{f:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w, a + 0.005, f"{a:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    save(fig, "2_per_part_metrics.png")

# ── 3. Interval count: threshold vs model ────────────────────────────────────

def plot_interval_counts():
    threshold_counts = [4, 12, 4, 18, 36]   # phrase 01-05
    model_counts     = [2,  9, 1,  7, 18]
    phrases = [f"phrase_0{i}" for i in range(1, 6)]

    x = np.arange(len(phrases))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    bars_t = ax.bar(x - w/2, threshold_counts, w, label="Fixed threshold", color=ORANGE, alpha=0.85)
    bars_m = ax.bar(x + w/2, model_counts,     w, label="LSTM model",      color=BLUE,   alpha=0.85)

    for bar in bars_t:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(int(bar.get_height())), ha="center", fontsize=9)
    for bar in bars_m:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(int(bar.get_height())), ha="center", fontsize=9)

    total_t = sum(threshold_counts)
    total_m = sum(model_counts)
    ax.set_title(f"Flagged Intervals per Phrase  (total: {total_t} threshold → {total_m} model, "
                 f"{100*(total_t-total_m)//total_t}% reduction)")
    ax.set_xticks(x); ax.set_xticklabels(phrases)
    ax.set_ylabel("# intervals"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "3_interval_counts.png")

# ── 4. Gantt timeline for phrase_05 ──────────────────────────────────────────

def plot_gantt_phrase05():
    thresh_path = ROOT / "results_threshold" / "phrase_05" / "report.json"
    model_path  = ROOT / "results"           / "phrase_05" / "report.json"

    thresh_data = load_json(thresh_path)["intervals"]
    model_data  = load_json(model_path)["intervals"]

    fig, (ax_t, ax_m) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("phrase_05 — Flagged Intervals by Body Part", fontsize=12)

    parts_order = ["LEFT_ARM", "RIGHT_ARM", "LEFT_LEG", "RIGHT_LEG", "HEAD", "TORSO"]
    y_ticks = {p: i for i, p in enumerate(parts_order)}

    def draw_intervals(ax, intervals, title):
        ax.set_title(title, fontsize=10)
        ax.set_yticks(range(len(parts_order)))
        ax.set_yticklabels([p.replace("_", " ").title() for p in parts_order], fontsize=8)
        ax.set_xlim(0, 12); ax.grid(axis="x", alpha=0.3)
        ax.set_xlabel("Time (s)")
        for iv in intervals:
            part = iv["part"]
            y    = y_ticks[part]
            color = PART_COLOURS[part]
            ax.barh(y, iv["end_s"] - iv["start_s"], left=iv["start_s"],
                    height=0.6, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)

    draw_intervals(ax_t, thresh_data, f"Fixed Threshold — {len(thresh_data)} intervals")
    draw_intervals(ax_m, model_data,  f"LSTM Model     — {len(model_data)} intervals")

    handles = [mpatches.Patch(color=PART_COLOURS[p], label=p.replace("_", " ").title())
               for p in parts_order]
    fig.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "4_phrase05_gantt.png")

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...")
    plot_training()
    plot_per_part()
    plot_interval_counts()
    plot_gantt_phrase05()
    print(f"\nAll figures saved to results/figures/")
