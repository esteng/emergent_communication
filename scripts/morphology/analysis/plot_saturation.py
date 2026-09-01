"""Paradigm saturation per construction, with and without semantic-role pooling.

Saturation is the share of the filler combinations a construction's paradigm licenses
that the agents actually lexicalized. Computed two ways from
`results/csv/construction_table.csv`:

  without pooling  n_attested_codebook / native_capacity   (the construction's own slots)
  with pooling     n_attested_codebook / pooled_capacity   (slots cross-filled with
                                                            same-role morphemes)

Each construction is one jittered point, sized by the size of its native grid, joined to
the model's mean by a slope line.

  python -m morphology.analysis.plot_saturation
    -> results/figures/saturation_by_model.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import morphology.common as C


FONT = C.PAPER_FONT
plt.rcParams["font.family"] = FONT

MODEL_LABEL = C.MODEL_LABEL
MODEL_COLOR = C.by_label(C.MODEL_COLOR)
MODEL_ORDER = [C.MODEL_LABEL[m] for m in C.MODEL_IDS]
CONDITIONS = ["Without Pooling", "With Pooling"]

JITTER = 0.135
POINT_ALPHA = 0.45
SIZE_BASE = 18
SIZE_PER_CELL = 3.2
TITLE_FS = 34
LABEL_FS = 30
TICK_FS = 28
LEGEND_FS = 25


def main() -> None:
    table = pd.read_csv(C.CSV / "construction_table.csv")
    table = table[(table.native_capacity > 0) & (table.pooled_capacity > 0)].copy()
    table["model_label"] = table.model.map(MODEL_LABEL)
    table["without"] = table.n_attested_codebook / table.native_capacity
    table["with"] = table.n_attested_codebook / table.pooled_capacity

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(10.4, 9.2), dpi=150)

    for model in MODEL_ORDER:
        rows = table[table.model_label == model]
        if rows.empty:
            continue
        color = MODEL_COLOR[model]
        sizes = SIZE_BASE + SIZE_PER_CELL * rows.native_capacity
        for position, column in enumerate(["without", "with"]):
            ax.scatter(position + rng.uniform(-JITTER, JITTER, len(rows)), rows[column],
                       s=sizes, color=color, alpha=POINT_ALPHA, linewidths=0, zorder=2)
        ax.plot([0, 1], [rows["without"].mean(), rows["with"].mean()],
                color=color, linewidth=2.4, marker="o", markersize=15,
                markerfacecolor="none", markeredgewidth=2.4, zorder=3)

    ax.set_title("Paradigm Saturation", fontsize=TITLE_FS, pad=16)
    ax.set_ylabel("Grid Saturation Per Construction", fontsize=LABEL_FS)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(CONDITIONS, fontsize=TICK_FS, color="0.3")
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(-0.03, 1.06)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=TICK_FS, color="0.3")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color("0.55")

    handles = [plt.Line2D([], [], color=MODEL_COLOR[m], marker="o", markersize=13,
                          markerfacecolor="none", markeredgewidth=2.2, linewidth=2.2, label=m)
               for m in MODEL_ORDER]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=3, frameon=False, fontsize=LEGEND_FS, handletextpad=0.4, columnspacing=1.4)

    fig.tight_layout()
    C.save(fig, "saturation_by_model.png")
    summary = table.groupby("model_label")[["without", "with"]].mean().round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
