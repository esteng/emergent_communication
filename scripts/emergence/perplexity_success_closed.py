"""Perplexity and success rate across rounds, proprietary models.

Two stacked panels over the 120 runs at the extreme budgets (3 models x 2 budgets
x postmortem on/off x n=10). Color = time budget, linestyle = postmortem, bands
are 95% CIs across runs.

  python scripts/emergence/perplexity_success_closed.py
    -> results/figures/perplexity_success_stacked.pdf
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402


def main() -> None:
    C.use_paper_style()
    df = pd.read_csv(C.CSV / "per_round_closed.csv")
    plotted = df[df.budget.isin(C.BUDGETS)]
    print(f"{plotted.run_id.nunique()} runs at budgets {C.BUDGETS}")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(4.2, 4.8), sharex=True)

    C.draw(ax0, C.agg_ci(df, "perplexity"))
    ax0.set_ylabel("Perplexity")
    C.sci_yaxis(ax0)

    C.draw(ax1, C.agg_ci(df, "success", clip01=True))
    ax1.set_ylabel("Success Rate")
    ax1.set_xlabel("Round")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlim(1, df.round_number.max())

    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)

    budget_h, pm_h = C.budget_pm_handles()
    leg = ax1.legend(handles=budget_h, title="Time Budget", frameon=False, loc="lower center",
                     ncol=2, fontsize=8, title_fontsize=8, bbox_to_anchor=(0.8, -0.5))
    ax1.add_artist(leg)
    ax1.legend(handles=pm_h, title="Postmortem", frameon=False, loc="lower center",
               ncol=2, fontsize=8, title_fontsize=8, bbox_to_anchor=(0.2, -0.5))

    fig.tight_layout()
    C.save(fig, "perplexity_success_stacked.pdf")


if __name__ == "__main__":
    main()
