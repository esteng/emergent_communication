"""Regenerates panel_decode_production.png/pdf (Fig 4b): decode accuracy and
production accuracy by construction, as two side-by-side panels with a shared
legend. Styling matches plot_panel_emergence.py so the two subfigures read as a
pair: identical title fontsize, despined axes, and a fixed 180pt figure height
(via subplots_adjust rather than a variable bbox_inches="tight" crop) so both
subfigures render to the same on-page height in LaTeX.
"""


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import morphology.common as C
from morphology.analysis.item_table_view import MODEL_ORDER, load_items

MODEL_COLOR = C.by_label(C.MODEL_COLOR)
MODEL_SHORT = C.by_label(C.MODEL_SHORT)
RNG = np.random.default_rng(42)

FONT = C.PAPER_FONT
plt.rcParams["font.family"] = FONT
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["mathtext.fontset"] = "custom"

BATTERY_DIR = C.CSV
OUT_DIR = C.FIGURES

# Title fontsize matches plot_panel_emergence.py. The emergence panel renders at
# 0.32*textwidth from a 162pt-wide figure; this panel renders at 0.65*textwidth
# from a 324pt-wide figure. Those two scale factors are within ~1.5%, so equal
# point sizes here render at the same on-page size as the emergence panel.
TITLE_FS = 11
LABEL_FS = 11
TICK_FS = 10
LEGEND_FS = 8


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
# Shared item-level data for the decode/production panels.
# ---------------------------------------------------------------------------
item_df = load_items()

MODEL_X = {m: i for i, m in enumerate(MODEL_ORDER)}
DODGE = {"Control": -0.19, "Novel": 0.19}
BAR_W = 0.34


def _grouped_by_construction(df, value_cols):
    keys = ["model_label", "construction_id", "cell_type"]
    per_group = df.groupby(keys, observed=True)[value_cols].mean().reset_index()
    per_group["n"] = df.groupby(keys, observed=True).size().reset_index(name="n")["n"]
    per_model = per_group.groupby(["model_label", "cell_type"], observed=True)[value_cols].mean().reset_index()
    return per_group, per_model


def _jitter_x(per_group):
    x = per_group["model_label"].map(MODEL_X).astype(float) + per_group["cell_type"].map(DODGE).astype(float)
    return x + RNG.uniform(-0.1, 0.1, size=len(per_group))


# ---------------------------------------------------------------------------
# Decode accuracy + production accuracy by construction (two panels).
# Height is chosen so the figure renders to the SAME on-page height as
# panel_emergence.pdf. The emergence panel is 162pt wide, placed at 0.32*tw;
# this figure is 324pt wide, placed at 0.65*tw. Equal rendered height requires
# H = 180 * (0.32*324)/(0.65*162) = 177.23pt = 2.4616in. No tight-bbox crop.
# ---------------------------------------------------------------------------
fig_bc, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.5, 2.4616), dpi=300)

# --- Panel (b): decode accuracy by construction ----------------------------
valid_decode = item_df.dropna(subset=["decodability"])
per_group_b, per_model_b = _grouped_by_construction(valid_decode, ["decodability"])

for model in MODEL_ORDER:
    for cell_type in ("Control", "Novel"):
        row = per_model_b[(per_model_b["model_label"] == model) & (per_model_b["cell_type"] == cell_type)]
        if row.empty:
            continue
        x = MODEL_X[model] + DODGE[cell_type]
        ax1.bar(
            x, row["decodability"].iloc[0], width=BAR_W,
            color=MODEL_COLOR[model], alpha=(0.85 if cell_type == "Control" else 0.35),
            edgecolor="black", linewidth=0.6,
        )
per_group_b = per_group_b.copy()
per_group_b["x"] = _jitter_x(per_group_b)
sizes_b = 6 + 30 * (per_group_b["n"] - per_group_b["n"].min()) / max(1, per_group_b["n"].max() - per_group_b["n"].min())
ax1.scatter(
    per_group_b["x"], per_group_b["decodability"], s=sizes_b,
    c=[MODEL_COLOR[m] for m in per_group_b["model_label"]],
    alpha=0.55, edgecolors="black", linewidths=0.3,
)
ax1.set_xticks(list(MODEL_X.values()))
ax1.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
ax1.set_ylabel("Decode Rate", fontsize=LABEL_FS, fontweight="bold", fontfamily=FONT)
ax1.set_ylim(-0.02, 1.08)
ax1.set_xlim(-0.55, 2.55)
_style_axis(ax1, "Decode Accuracy")

# --- Panel (c): production accuracy by construction ------------------------
valid_prod = item_df.dropna(subset=["exact_match", "bag_match"])
per_group_c, per_model_c = _grouped_by_construction(valid_prod, ["exact_match", "bag_match"])

for model in MODEL_ORDER:
    for cell_type in ("Control", "Novel"):
        row = per_model_c[(per_model_c["model_label"] == model) & (per_model_c["cell_type"] == cell_type)]
        if row.empty:
            continue
        x = MODEL_X[model] + DODGE[cell_type]
        exact = row["exact_match"].iloc[0]
        bag = row["bag_match"].iloc[0]
        alpha = 0.85 if cell_type == "Control" else 0.35
        ax2.bar(x, exact, width=BAR_W, color=MODEL_COLOR[model], alpha=alpha, edgecolor="black", linewidth=0.6)
        ax2.bar(
            x, bag - exact, width=BAR_W, bottom=exact, color=MODEL_COLOR[model], alpha=alpha,
            edgecolor="black", linewidth=0.6, hatch="///",
        )
per_group_c = per_group_c.copy()
per_group_c["x"] = _jitter_x(per_group_c)
sizes_c = 6 + 30 * (per_group_c["n"] - per_group_c["n"].min()) / max(1, per_group_c["n"].max() - per_group_c["n"].min())
ax2.scatter(
    per_group_c["x"], per_group_c["bag_match"], s=sizes_c,
    c=[MODEL_COLOR[m] for m in per_group_c["model_label"]],
    alpha=0.55, edgecolors="black", linewidths=0.3,
)
ax2.set_xticks(list(MODEL_X.values()))
ax2.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
ax2.set_ylabel("Production Rate", fontsize=LABEL_FS, fontweight="bold", fontfamily=FONT)
ax2.set_ylim(-0.02, 1.08)
ax2.set_xlim(-0.55, 2.55)
_style_axis(ax2, "Production Accuracy")

# ---------------------------------------------------------------------------
# Shared legend under both panels. Fixed margins (no tight bbox) keep the
# figure exactly 180pt tall so it matches panel_emergence.pdf on the page.
# ---------------------------------------------------------------------------
fig_bc.subplots_adjust(top=0.90, bottom=0.20, left=0.115, right=0.985, wspace=0.42)

cell_handles = [
    mpatches.Patch(facecolor="gray", edgecolor="black", alpha=0.85, label="Control"),
    mpatches.Patch(facecolor="gray", edgecolor="black", alpha=0.35, label="Novel"),
]
match_handles = [
    mpatches.Patch(facecolor="white", edgecolor="black", label="Exact"),
    mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Bag (Any Order)"),
]
all_handles = cell_handles + match_handles
fig_bc.legend(
    handles=all_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
    ncol=4, frameon=False, prop={"family": FONT, "size": LEGEND_FS, "weight": "bold"},
    columnspacing=0.8, handletextpad=0.35, handlelength=1.4, handleheight=0.9,
)

out_bc_png = OUT_DIR / "panel_decode_production.png"
out_bc_pdf = OUT_DIR / "panel_decode_production.pdf"
fig_bc.savefig(out_bc_png, dpi=300)
fig_bc.savefig(out_bc_pdf)
plt.close(fig_bc)
print(f"wrote {out_bc_png}")
print(f"wrote {out_bc_pdf}")
