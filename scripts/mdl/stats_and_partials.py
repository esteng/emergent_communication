"""Does description length predict post-postmortem success?

Correlations are computed over the 45 ROOTS, averaging each root's 3 resume seeds
first: the seeds share a language and a grammar, so they are not independent.

Coverage (the fraction of held-out messages the grammar parses) drives part of
DL(D|G) and also correlates with success, so each association with success is also
reported partialled on coverage.

  python scripts/mdl/stats_and_partials.py
    -> results/tables/mdl_correlations.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t as tdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CSV, TABLES  # noqa: E402


def partial_pearson(x, y, z):
    """Pearson r between x and y after regressing both on z, with df = n-3."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.c_[np.ones(len(x)), np.asarray(z, float)]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r = pearsonr(rx, ry)[0]
    df = len(x) - 3
    return r, 2 * tdist.sf(abs(r * np.sqrt(df / (1 - r**2))), df)


def root_level(components: pd.DataFrame) -> pd.DataFrame:
    return (components.groupby("root_id")
            .agg(DL_G=("DL_G", "mean"), DgG=("pcfg_DgG_pm", "mean"),
                 coverage=("test_coverage", "mean"), succ=("succ_test", "mean"),
                 model=("model", "first")).reset_index())


def correlations(a: pd.DataFrame) -> pd.DataFrame:
    G, D, C, S = a.DL_G.values, a.DgG.values, a.coverage.values, a.succ.values
    n = len(a)
    rows = []
    for label, x, y in [("DL(G) ~ success", G, S),
                        ("DL(D|G) ~ success", D, S),
                        ("DL(G) ~ DL(D|G)", G, D),
                        ("coverage ~ DL(D|G)", C, D),
                        ("coverage ~ success", C, S)]:
        r, p = pearsonr(x, y)
        rho, ps = spearmanr(x, y)
        rows.append(dict(stat=label, pearson_r=r, pearson_p=p,
                         spearman_r=rho, spearman_p=ps, n=n))
    for label, x in [("partial DL(G)|cov ~ success", G), ("partial DL(D|G)|cov ~ success", D)]:
        r, p = partial_pearson(x, S, C)
        rows.append(dict(stat=label, pearson_r=r, pearson_p=p,
                         spearman_r=np.nan, spearman_p=np.nan, n=n))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", type=Path, default=CSV / "mdl_resume_components.csv")
    ap.add_argument("--out", type=Path, default=TABLES / "mdl_correlations.csv")
    args = ap.parse_args()

    a = root_level(pd.read_csv(args.components))
    stats = correlations(a)
    stats.to_csv(args.out, index=False)

    def sig(p):
        return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "." if p < .1 else "ns"

    print(f"root-level correlations (n={len(a)} roots)\n")
    print(f"{'stat':<32}{'pearson r':>10}{'p':>9}     {'spearman':>9}{'p':>9}")
    for r in stats.itertuples():
        rho = f"{r.spearman_r:+.3f}" if r.spearman_r == r.spearman_r else "    --"
        rp = f"{r.spearman_p:.3f}" if r.spearman_p == r.spearman_p else "   --"
        print(f"{r.stat:<32}{r.pearson_r:>+10.3f}{r.pearson_p:>9.4f} {sig(r.pearson_p):<4}{rho:>9}{rp:>9}")
    print("\n*** p<.001  ** p<.01  * p<.05  . p<.1  ns p>=.1")
    print(f"wrote {args.out.name}")


if __name__ == "__main__":
    main()
