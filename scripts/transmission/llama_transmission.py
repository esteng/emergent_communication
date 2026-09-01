"""The same swap experiment with a Llama-3.3-70B-Instruct newcomer.

Llama never develops a language of its own, so this asks whether it can still
acquire one. Produces the Sonnet-source panel on its own, and the full three-panel
version across all source models.

  python scripts/transmission/llama_transmission.py
    -> results/figures/replace_llama_vs_history_sonnet.pdf   Sonnet source only
    -> results/figures/replace_llama_vs_history_row3.pdf     all three source models
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

MAIN_TEXT_MODEL = "claude-sonnet-4-6"


def single_panel(df, per_root) -> None:
    fig, ax = plt.subplots(figsize=(4, 4.0))
    C.history_panel(ax, df, per_root, MAIN_TEXT_MODEL,
                    f"Source={C.MODEL_LUT[MAIN_TEXT_MODEL]}")
    fig.legend(handles=C.history_legend_handles(short=True), loc="upper right",
               bbox_to_anchor=(1, 0.8), ncol=1, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    C.save(fig, "replace_llama_vs_history_sonnet.pdf")


def three_panels(df, per_root) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), sharey=True)
    for ax, model in zip(axes, C.MODELS):
        C.history_panel(ax, df, per_root, model, f"Source={C.MODEL_LUT[model]}")
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.legend(handles=C.history_legend_handles(), loc="lower center",
               bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    C.save(fig, "replace_llama_vs_history_row3.pdf")


def main() -> None:
    C.use_paper_style()
    df, per_root = C.load_transmission("replace_llama")
    n_roots = {m: int((per_root.model == m).sum() // len(C.HISTORY_LEVELS)) for m in C.MODELS}
    print(f"roots per model: {n_roots}")
    single_panel(df, per_root)
    three_panels(df, per_root)


if __name__ == "__main__":
    main()
