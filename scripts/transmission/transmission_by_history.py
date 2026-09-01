"""Post-swap success vs. amount of imported history (same-family newcomer).

A team plays 14 rounds with a postmortem stage, then one agent is replaced by a fresh
agent of the same model that has seen 0, 1, 5 or 10 rounds of the transcript but none
of the postmortems, and the team plays 11 more rounds without postmortem. One panel
per source model; 15 roots each, 3 seeds per root.

  python scripts/transmission/transmission_by_history.py
    -> results/figures/replace_learned_vs_history_row3.pdf
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402


def main() -> None:
    C.use_paper_style()
    df, per_root = C.load_transmission("replace_learned")
    n_roots = {m: int((per_root.model == m).sum() // len(C.HISTORY_LEVELS)) for m in C.MODELS}
    print(f"roots per model: {n_roots}")

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), sharey=True)
    for ax, model in zip(axes, C.MODELS):
        C.history_panel(ax, df, per_root, model, C.MODEL_LUT[model])
    for ax in axes[1:]:
        ax.set_ylabel("")

    fig.legend(handles=C.history_legend_handles(), loc="lower center",
               bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    C.save(fig, "replace_learned_vs_history_row3.pdf")


if __name__ == "__main__":
    main()
