"""Regenerates panel_emergence.png/pdf ("Morphology Emergence", Fig 4a) with:
(a) y-axis as a percentage of the 9 runs per (model, budget) cell instead of a raw count,
(b) nonparametric bootstrap 95% confidence intervals on each point,
(c) a color/marker legend identifying each model line,
(d) a distinct marker shape per model (on top of color) for colorblind readers.

Reads results/csv/emergence_runs.csv; writes results/figures/panel_emergence.{png,pdf}.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import morphology.common as C

# Short labels (no version numbers) match the x-tick labels in
# plot_panel_decode_production.py, and keep the legend narrow enough to sit
# clear of the x=800 tick.
MODEL_COLOR = C.by_short(C.MODEL_COLOR)
MODEL_MARKER = {
    "Opus": "o",
    "Sonnet": "s",
    "GPT": "^",
}
MODEL_LABEL = dict(C.MODEL_SHORT)
MODEL_ORDER = [C.MODEL_SHORT[m] for m in C.MODEL_IDS]
N_PER_CELL = 9
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 42


def bootstrap_interval(n_success: int, n_total: int) -> tuple[float, float]:
    """Percentile bootstrap 95% interval for a cell's non-flat proportion.

    Resamples the cell's ``n_total`` run-level binary outcomes with replacement
    ``N_BOOTSTRAP`` times and takes the 2.5th/97.5th percentiles of the resampled
    proportions. Seeded per cell so the rendered figure is reproducible.
    """
    outcomes = np.concatenate([np.ones(n_success), np.zeros(n_total - n_success)])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(outcomes, size=(N_BOOTSTRAP, n_total), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)

FONT = C.PAPER_FONT
plt.rcParams["font.family"] = FONT
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["mathtext.fontset"] = "custom"

BATTERY_DIR = C.CSV
OUT_DIR = C.FIGURES

TITLE_FS = 11
LABEL_FS = 11
TICK_FS = 10
LEGEND_FS = 8.0


def _style_axis(ax, title):
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", fontfamily=FONT, pad=4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontfamily(FONT)
        label.set_fontsize(TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)


# ---------------------------------------------------------------------------
# Morphology emergence by time budget (single panel), as % of 9 runs/cell.
# ---------------------------------------------------------------------------
fig_a, ax0 = plt.subplots(1, 1, figsize=(2.25, 2.5), dpi=300)

induction_rows = [
    json.loads(line) for line in (C.DATA / "induction_results.jsonl").read_text().splitlines()
    if line.strip()
]
ind_df = pd.DataFrame(induction_rows)
ind_df = ind_df[ind_df["ok"]]
ind_df["model_label"] = ind_df["model"].map(MODEL_LABEL)
counts = (
    ind_df[~ind_df["below_gate"]]
    .groupby(["model_label", "budget"], observed=True)
    .size()
    .reset_index(name="n_non_flat")
)
budgets = sorted(int(b) for b in ind_df["budget"].unique())
full_grid = pd.MultiIndex.from_product([MODEL_ORDER, budgets], names=["model_label", "budget"]).to_frame(index=False)
counts = full_grid.merge(counts, on=["model_label", "budget"], how="left").fillna({"n_non_flat": 0})

counts["p"] = counts["n_non_flat"] / N_PER_CELL
counts["pct"] = counts["p"] * 100
_bounds = counts["n_non_flat"].apply(lambda k: bootstrap_interval(n_success=int(k), n_total=N_PER_CELL))
counts["ci_lo_pct"] = [lo * 100 for lo, _ in _bounds]
counts["ci_hi_pct"] = [hi * 100 for _, hi in _bounds]

for model in MODEL_ORDER:
    sub = counts[counts["model_label"] == model].sort_values("budget")
    # Clamp to zero: the percentile bounds can fall a hair the wrong side of the
    # point estimate when the bootstrap distribution is heavily skewed.
    yerr = [
        (sub["pct"] - sub["ci_lo_pct"]).clip(lower=0).to_numpy(),
        (sub["ci_hi_pct"] - sub["pct"]).clip(lower=0).to_numpy(),
    ]
    ax0.errorbar(
        sub["budget"], sub["pct"], yerr=yerr,
        marker=MODEL_MARKER[model], color=MODEL_COLOR[model],
        linewidth=1.6, markersize=4.5, markerfacecolor="white", markeredgewidth=1.3,
        capsize=2, elinewidth=0.9, label=model,
    )
ax0.set_xscale("log")
ax0.set_xticks(budgets)
ax0.set_xticklabels([str(b) for b in budgets], rotation=0)
ax0.set_xlabel("Time Budget (s)", fontsize=LABEL_FS, fontweight="bold", fontfamily=FONT)
# Drop the x-axis label so it sits at the same on-page height as the decode
# figure's bottom legend when the two subfigures are placed side by side.
ax0.xaxis.set_label_coords(0.5, -0.165)
ax0.set_ylabel(
    "Percent of Runs with\nCompositional Morphology",
    fontsize=LABEL_FS, fontweight="bold", fontfamily=FONT,
)
ax0.set_ylim(-5, 100)
ax0.set_yticks([0, 25, 50, 75, 100])
_style_axis(ax0, "Morphology Emergence")
# The x=800 tick sits at 0.646 of the axes width and the legend block is 0.339
# wide, so anchoring at 0.655 puts the whole legend right of that tick while
# staying inside the axes (anything past 0.654 clips). The short handlelength
# is what pays for the wider marker-to-text gap at 8pt; markerscale stays 1.0.
ax0.legend(
    loc="upper left", bbox_to_anchor=(0.692, 1.005), ncol=1, frameon=False,
    prop={"family": FONT, "size": LEGEND_FS, "weight": "bold"},
    handletextpad=0.45, handlelength=0.5, labelspacing=0.5, borderaxespad=0.0,
    markerscale=1.0,
)

fig_a.subplots_adjust(top=0.91, bottom=0.20, left=0.30, right=0.97)
out_a_png = OUT_DIR / "panel_emergence.png"
out_a_pdf = OUT_DIR / "panel_emergence.pdf"
fig_a.savefig(out_a_png, dpi=300)
fig_a.savefig(out_a_pdf)
plt.close(fig_a)
print(f"wrote {out_a_png}")
print(f"wrote {out_a_pdf}")
