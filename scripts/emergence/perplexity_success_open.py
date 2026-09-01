"""Perplexity and success rate across rounds, open-weight models.

Same layout as the proprietary-model figure but 2x2: columns are Qwen3-32B and
Llama-3.3-70B-Instruct, rows are perplexity and success rate. 32 runs (2 models x
2 budgets x postmortem on/off x n=4).

  python scripts/emergence/perplexity_success_open.py
    -> results/figures/perplexity_success_2x2.pdf
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

PANELS = [("Qwen", "Qwen/Qwen3-32B"), ("Llama", "meta-llama/Llama-3.3-70B-Instruct")]
TITLE_SZ = LABEL_SZ = TICK_SZ = LEG_SZ = 16


def main() -> None:
    C.use_paper_style()
    df = pd.read_csv(C.CSV / "per_round_open.csv")
    print(f"{df.run_id.nunique()} runs")

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 4.8), sharex=True, sharey="row")
    for col, (name, model) in enumerate(PANELS):
        d = df[df.model == model]
        axp, axs = axes[0, col], axes[1, col]
        C.draw(axp, C.agg_ci(d, "perplexity"))
        C.draw(axs, C.agg_ci(d, "success", clip01=True))
        axp.set_title(name, fontsize=TITLE_SZ)
        axs.set_ylim(-0.02, 1.02)
        axs.set_xlim(1, d.round_number.max())
        axs.set_xlabel("Round", fontsize=LABEL_SZ)
        for a in (axp, axs):
            a.spines[["top", "right"]].set_visible(False)
            a.tick_params(labelsize=TICK_SZ)
        if col == 0:
            axp.set_ylabel("Perplexity", fontsize=LABEL_SZ)
            axs.set_ylabel("Success Rate", fontsize=LABEL_SZ)

    C.sci_yaxis(axes[0, 0])
    axes[0, 0].set_ylim(0, 3e4)
    axes[0, 0].set_yticks([0.0, 3e4])
    axes[0, 0].set_yticklabels(["0.00", "3.00e+4"])
    axes[1, 0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axes[1, 0].set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])

    budget_h, pm_h = C.budget_pm_handles()
    fig.legend(handles=pm_h, title="Postmortem", frameon=False, loc="upper center",
               ncol=2, bbox_to_anchor=(0.30, 0.10), fontsize=LEG_SZ, title_fontsize=LEG_SZ)
    fig.legend(handles=budget_h, title="Time Budget", frameon=False, loc="upper center",
               ncol=2, bbox_to_anchor=(0.72, 0.10), fontsize=LEG_SZ, title_fontsize=LEG_SZ)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    C.save(fig, "perplexity_success_2x2.pdf")


if __name__ == "__main__":
    main()
