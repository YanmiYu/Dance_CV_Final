"""Generate model comparison figure: main-branch BiGRU vs integrate-branch LSTM.

Outputs:
  figures/fig6_model_comparison.png  — architecture + design table side-by-side
  figures/fig7_label_design.png      — 3-class vs binary label space illustration
"""

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG  = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "figure.dpi": 200,
})

SLATE  = "#334155"
BLUE   = "#4878CF"
ORANGE = "#D65F00"
GREEN  = "#6ACC65"
TAN    = "#C4AD66"
LAVEND = "#B47CC7"
SKY    = "#77BEDB"
LIGHT  = "#F8FAFC"
GREY   = "#94A3B8"


# ── Fig 6: side-by-side design comparison table ──────────────────────────────
def fig6_comparison():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.axis("off")
    fig.patch.set_facecolor(LIGHT)

    cols  = ["Design choice", "main  (BiGRU baseline)", "integrate  (LSTM improved)"]
    rows  = [
        ["Recurrent unit",     "Bidirectional GRU",           "Unidirectional LSTM"],
        ["Effective hidden",   "128  (64 × 2 directions)",    "64"],
        ["Output head",        "Linear(128 → 6×3) logits",    "Linear(64 → 6) logits"],
        ["Classes per part",   "3  (good / moderate / off)",  "2  (correct / off)"],
        ["Loss function",      "CrossEntropy  wt=[1,2,3]",    "BCEWithLogitsLoss  pos_w=3"],
        ["Label threshold",    "good <0.15 / mod 0.15–0.35 / off >0.35",
                               "off if error ≥ 0.35 (binary)"],
        ["Interval detection", "argmax class==2  for ≥0.5 s", "P(off)>0.5  for ≥0.5 s"],
        ["Val F1 (reported)",  "—  (no committed run)",       "0.580  (integrate run)"],
        ["Test F1 (overall)",  "—  (no committed run)",       "0.958  (held-out ch10)"],
        ["Interval reduction", "baseline (74 intervals)",     "37 intervals  (−50%)"],
    ]

    col_w = [0.32, 0.34, 0.34]
    x_starts = [0.0, 0.32, 0.66]
    row_h = 0.082
    y_top = 0.96

    # Header
    for ci, (label, xs, w) in enumerate(zip(cols, x_starts, col_w)):
        bg = SLATE if ci == 0 else (BLUE if ci == 1 else ORANGE)
        ax.add_patch(FancyBboxPatch((xs + 0.003, y_top - row_h + 0.005),
                                    w - 0.006, row_h - 0.008,
                                    boxstyle="round,pad=0.005",
                                    facecolor=bg, edgecolor="none"))
        ax.text(xs + w / 2, y_top - row_h / 2, label,
                ha="center", va="center", fontsize=8,
                color="white", fontweight="bold")

    # Rows
    for ri, row in enumerate(rows):
        y = y_top - (ri + 1) * row_h
        bg_row = "#EFF6FF" if ri % 2 == 0 else LIGHT
        for ci, (cell, xs, w) in enumerate(zip(row, x_starts, col_w)):
            ax.add_patch(FancyBboxPatch((xs + 0.003, y + 0.005),
                                        w - 0.006, row_h - 0.008,
                                        boxstyle="round,pad=0.005",
                                        facecolor=bg_row, edgecolor=GREY,
                                        linewidth=0.3))
            color = SLATE
            if ci == 2 and ri >= 7:   # highlight improvements
                color = "#16803C"
            ax.text(xs + w / 2, y + row_h / 2, cell,
                    ha="center", va="center", fontsize=6.8,
                    color=color, wrap=True)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Model Design: main Branch (BiGRU) vs integrate Branch (LSTM)",
                 fontsize=9, pad=8, color=SLATE)
    fig.tight_layout()
    p = FIG / "fig6_model_comparison.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=LIGHT)
    plt.close(fig)
    print(f"  saved → figures/fig6_model_comparison.png")


# ── Fig 7: 3-class vs binary label illustration ───────────────────────────────
def fig7_label_design():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.6))
    fig.suptitle("Label Design: 3-class (main) vs Binary (integrate)", fontsize=9)

    t = np.linspace(0, 12, 300)
    # Simulated error signal
    np.random.seed(42)
    signal = (0.18 + 0.14 * np.sin(1.8 * t)
              + 0.22 * np.sin(0.5 * t + 1)
              + 0.06 * np.random.randn(len(t)))
    signal = np.clip(signal, 0, None)

    # --- 3-class axis ---
    ax1.plot(t, signal, color=SLATE, linewidth=1.2, label="error signal")
    ax1.axhline(0.15, color=GREEN,  linewidth=1.0, linestyle="--", label="good / mod (0.15)")
    ax1.axhline(0.35, color=ORANGE, linewidth=1.0, linestyle="--", label="mod / off (0.35)")
    ax1.fill_between(t, 0,    0.15, alpha=0.15, color=GREEN,  label="good")
    ax1.fill_between(t, 0.15, 0.35, alpha=0.15, color=TAN,    label="moderate")
    ax1.fill_between(t, 0.35, 0.65, alpha=0.15, color=ORANGE, label="off")
    ax1.set_ylim(0, 0.65); ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Error (torso units)")
    ax1.set_title("(a) main — 3 classes", fontsize=8)
    ax1.legend(fontsize=6.5, loc="upper right", ncol=2)
    ax1.grid(alpha=0.2)

    # --- binary axis ---
    ax2.plot(t, signal, color=SLATE, linewidth=1.2)
    ax2.axhline(0.35, color=LAVEND, linewidth=1.0, linestyle="--", label="threshold (0.35)")
    ax2.fill_between(t, 0,    0.35, alpha=0.18, color=BLUE,   label="correct")
    ax2.fill_between(t, 0.35, 0.65, alpha=0.18, color=LAVEND, label="off")
    off_mask = signal >= 0.35
    ax2.fill_between(t, 0, signal, where=off_mask, alpha=0.35, color=LAVEND)
    ax2.set_ylim(0, 0.65); ax2.set_xlabel("Time (s)")
    ax2.set_title("(b) integrate — binary (P(off))", fontsize=8)
    ax2.legend(fontsize=6.5, loc="upper right")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    p = FIG / "fig7_label_design.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → figures/fig7_label_design.png")


if __name__ == "__main__":
    print("Generating comparison figures...")
    fig6_comparison()
    fig7_label_design()
    print("Done.")
