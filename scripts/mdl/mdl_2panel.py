"""Grammar and data description length against post-postmortem success.

One point per baseline root (mean over its 3 resume seeds, n=45), colored by the
agent model and sized by how much of the held-out window that root's grammar parses.
The regression line and the reported correlations are fit on these root means.

  python scripts/mdl/mdl_2panel.py
    -> results/figures/mdl_components_2panel.pdf
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

PANELS = [("DL_G", "DL(G) (absolute)", "DL(G) (bits)"),
          ("pcfg_DgG_pm", "PCFG DL(D|G) per msg", "PCFG DL(D|G) per msg (bits)")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", type=Path, default=C.CSV / "mdl_resume_components.csv")
    ap.add_argument("--train", default="10-15")
    ap.add_argument("--test", default="16-25")
    args = ap.parse_args()

    agg = (pd.read_csv(args.components).groupby("root_id")
           .agg(model=("model", "first"), DL_G=("DL_G", "mean"),
                pcfg_DgG_pm=("pcfg_DgG_pm", "mean"), coverage=("test_coverage", "mean"),
                succ_test=("succ_test", "mean")).reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (col, name, xlab) in zip(axes, PANELS):
        sns.regplot(data=agg, x=col, y="succ_test", scatter=False, color="gray",
                    line_kws={"linewidth": 1}, ax=ax)
        sns.scatterplot(data=agg, x=col, y="succ_test", hue="model", hue_order=C.MDL_ORDER,
                        palette=C.MDL_PALETTE, size="coverage", sizes=(30, 220),
                        edgecolor="black", linewidth=0.4, legend=(ax is axes[0]), ax=ax)
        r, _ = pearsonr(agg[col], agg.succ_test)
        rho, _ = spearmanr(agg[col], agg.succ_test)
        ax.set_title(f"{name}\n(root-avg n={len(agg)}: Pearson r={r:.2f}, Spearman={rho:.2f})",
                     fontsize=10)
        ax.set_xlabel(xlab)
        ax.set_ylabel(f"resume success (rounds {args.test})")
        print(f"{name:24s} Pearson r={r:+.3f}  Spearman={rho:+.3f}")
    axes[1].set_ylabel(f"success ({args.test})")
    if axes[0].get_legend():
        axes[0].legend(fontsize=7, loc="upper right")

    fig.suptitle(f"Same-team resume, no postmortem (rounds {args.test} held out)  |  "
                 "points = per-root mean over 3 seeds (n=45)")
    fig.tight_layout()
    C.save(fig, "mdl_components_2panel.pdf")


if __name__ == "__main__":
    main()
