"""Shared paths, plot styling and aggregation helpers.

Every path is relative to the release root, so scripts run from any working directory.
"""
from pathlib import Path

import matplotlib
import matplotlib.ticker
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
CSV = RESULTS / "csv"

for _d in (FIGURES, TABLES, CSV):
    _d.mkdir(parents=True, exist_ok=True)

CI_Z = 1.96  # 95% CI


def use_paper_style() -> None:
    """Serif text, editable PDF fonts, no grid — matches the figures in the paper."""
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["axes.grid"] = False


def save(fig, name: str) -> Path:
    out = FIGURES / name
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out.relative_to(ROOT)}")
    return out


# --------------------------------------------------------------------------
# Perplexity and success across rounds, by time budget and postmortem
# --------------------------------------------------------------------------

BUDGETS = [150, 2000]                        # the two extremes the figures plot
COLOR = {150: "#af8dc3", 2000: "#7fbf7b"}    # colorblind-safe purple / green
MARKER = {150: "o", 2000: "s"}
PM_STYLE = {True: "-", False: "--"}
PM_LABEL = {True: "On", False: "Off"}
LINEWIDTH = 1.2


def agg_ci(d, value, clip01=False):
    """Mean and 95% CI of `value` per (budget, postmortem, round), across runs."""
    s = (d.groupby(["budget", "postmortem", "round_number"])[value]
         .agg(["mean", "std", "count"]).reset_index())
    s["sem"] = s["std"] / np.sqrt(s["count"])
    s["lo"] = s["mean"] - CI_Z * s["sem"]
    s["hi"] = s["mean"] + CI_Z * s["sem"]
    if clip01:
        s["lo"], s["hi"] = s["lo"].clip(lower=0.0), s["hi"].clip(upper=1.0)
    return s


def draw(ax, stat) -> None:
    """One line per (budget, postmortem) cell, with a shaded CI band."""
    for b in BUDGETS:
        for pm in (True, False):
            sub = stat[(stat.budget == b) & (stat.postmortem == pm)].sort_values("round_number")
            if sub.empty:
                continue
            ax.fill_between(sub.round_number, sub.lo, sub.hi,
                            color=COLOR[b], alpha=0.15, linewidth=0)
            ax.plot(sub.round_number, sub["mean"], color=COLOR[b], linestyle=PM_STYLE[pm],
                    linewidth=LINEWIDTH, marker=MARKER[b], markersize=5, markeredgewidth=0)


def budget_pm_handles():
    """Legend handles: one per time budget, one per postmortem setting."""
    budget = [mlines.Line2D([], [], color=COLOR[b], marker=MARKER[b], linestyle="-",
                            linewidth=LINEWIDTH, markersize=5, label=f"{b}s") for b in BUDGETS]
    pm = [mlines.Line2D([], [], color="0.3", linestyle=PM_STYLE[p], linewidth=LINEWIDTH,
                        label=PM_LABEL[p]) for p in (True, False)]
    return budget, pm


def sci_yaxis(ax, rotation=70) -> None:
    """Scientific notation on the perplexity axis (values run to ~1e4)."""
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0e"))
    for label in ax.get_yticklabels():
        label.set_rotation(rotation)


# --------------------------------------------------------------------------
# Post-swap success vs. amount of imported history
# --------------------------------------------------------------------------

HISTORY_LEVELS = [0, 1, 5, 10]
MODELS = ["gpt-5.4", "claude-sonnet-4-6", "claude-opus-4-7"]   # panel order
MODEL_LUT = {"gpt-5.4": "GPT 5.4",
             "claude-sonnet-4-6": "Sonnet 4.6",
             "claude-opus-4-7": "Opus 4.7"}
MODEL_COLOR = {"gpt-5.4": "#2166ac",
               "claude-sonnet-4-6": "#b2182b",
               "claude-opus-4-7": "#1a9850"}
C_RESUME = "#762a83"        # same team continuing WITH postmortem
C_RESUME_NOPM = "#e08214"   # same team continuing WITHOUT postmortem
ROOT_ALPHA = 0.32           # faint per-root trajectories

# --------------------------------------------------------------------------
# MDL figures
# --------------------------------------------------------------------------

MDL_PALETTE = {"gpt54": "#dd8452", "opus47": "#4c72b0", "sonnet": "#55a868"}
MDL_ORDER = ["gpt54", "opus47", "sonnet"]


def load_transmission(swap_phase: str):
    """Run-level table for the swap experiments, keyed by the root's agent model.

    swap_phase is `replace_learned` (a same-family newcomer) or `replace_llama`
    (a Llama-3.3-70B newcomer). Returns (df, per_root) where per_root is the mean over
    each root's 3 seeds at each history level — the independent unit.
    """
    import pandas as pd

    df = pd.read_csv(DATA / "learnability_run_level.csv")
    base = df[df.phase == "baseline"][["run_id", "field_observer_model"]]
    model_of = dict(zip(base.run_id, base.field_observer_model))

    df = df[df.phase.isin([swap_phase, "resume_expected",
                           "resume_expected_no_postmortem"])].copy()
    df["model"] = df.src_id.map(model_of)
    per_root = (df[df.phase == swap_phase]
                .groupby(["src_id", "model", "history"])["round_success_after_resume"]
                .mean().reset_index())
    return df, per_root


def history_curve(per_root, model):
    """Mean over roots and 95% CI half-width at each history level (n = #roots)."""
    means, cis = [], []
    for h in HISTORY_LEVELS:
        s = per_root[(per_root.model == model) & (per_root.history == h)]["round_success_after_resume"]
        means.append(s.mean())
        cis.append(CI_Z * s.std(ddof=1) / np.sqrt(s.count()) if s.count() > 1 else 0.0)
    return np.array(means), np.array(cis)


def history_panel(ax, df, per_root, model, title):
    """One transmission panel: faint per-root lines, bold mean+CI, two reference lines."""
    color = MODEL_COLOR[model]
    x = np.arange(len(HISTORY_LEVELS))

    for _, g in per_root[per_root.model == model].groupby("src_id"):
        g = g.set_index("history").reindex(HISTORY_LEVELS)
        ax.plot(x, g["round_success_after_resume"].values,
                color=color, alpha=ROOT_ALPHA, linewidth=0.7, zorder=1)

    m, ci = history_curve(per_root, model)
    ax.fill_between(x, m - ci, m + ci, color=color, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(x, m, color=color, marker="o", markersize=5, markeredgewidth=0,
            linewidth=1.8, zorder=4)

    # what the ORIGINAL team achieved over the same rounds, with and without postmortem
    for phase, c in [("resume_expected", C_RESUME),
                     ("resume_expected_no_postmortem", C_RESUME_NOPM)]:
        ref = df[(df.phase == phase) & (df.model == model)]["round_success_after_resume"].mean()
        ax.axhline(ref, color=c, linestyle="--", linewidth=1.2, zorder=3)

    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in HISTORY_LEVELS])
    ax.set_xlim(-0.3, len(HISTORY_LEVELS) - 0.7)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("History (# Rounds)", fontsize=14)
    ax.set_ylabel("Post-Resume Success Rate", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)


def history_legend_handles(short=False):
    pm_on = "Same Team, w/ Postmortem (Avg.)" if short else "Same Team, with Postmortem (Avg.)"
    pm_off = "Same Team, w/o Postmortem (Avg.)" if short else "Same Team, without Postmortem (Avg.)"
    return [
        mlines.Line2D([], [], color="0.3", marker="o", markersize=5, linewidth=1.8,
                      label="Mean over Roots (95% CI)"),
        mlines.Line2D([], [], color="0.3", alpha=0.4, linewidth=0.7, label="Individual Runs"),
        mlines.Line2D([], [], color=C_RESUME, linestyle="--", linewidth=1.2, label=pm_on),
        mlines.Line2D([], [], color=C_RESUME_NOPM, linestyle="--", linewidth=1.2, label=pm_off),
    ]
