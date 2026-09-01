"""Do newcomers ask fewer questions about compositional terms?

Two claims, both over the 37 protocols whose swapped-in agent asked at least one
metalinguistic question during the 11 post-swap rounds:

  1. Compositional targets are queried at a LOWER rate than atomic ones.
     Wilcoxon signed-rank over per-protocol rates (paired, one-sided).
  2. Queries about compositional targets fall off as the newcomer sees more history,
     while queries about atomic targets do not.
     Poisson GEE on counts, clustered by protocol, offset = log(exposure), with
     history as an ordinal rank 0<1<5<10 so the coefficient is a per-level log-rate.

  python scripts/transmission/stats_metalinguistic.py
    -> results/tables/metalinguistic_stats.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, TABLES  # noqa: E402

HISTORY = [0, 1, 5, 10]
HRANK = {0: 0, 1: 1, 5: 2, 10: 3}
NEWCOMER_ROUNDS = 11   # rounds 15-25, after the swap


def build():
    """One count per (protocol, history, compositional/atomic), with its exposure."""
    rl = pd.read_csv(DATA / "learnability_run_level.csv")
    inst = pd.read_csv(DATA / "annotations/metaling_instances.csv")

    q = inst[inst.has_question == True]                     # noqa: E712
    q = q[q.compositional.isin([True, False])].copy()       # drop unidentifiable targets
    q["comp"] = q.compositional.map({True: "compositional", False: "atomic"})

    runs = (rl[rl.phase == "replace_learned"]
            .groupby(["src_id", "history"]).run_id.nunique().reset_index(name="n_runs"))
    exposure = {(r.src_id, r.history): r.n_runs * NEWCOMER_ROUNDS for r in runs.itertuples()}

    roots = sorted(q.root.unique())
    grid = pd.MultiIndex.from_product(
        [roots, [float(h) for h in HISTORY], ["compositional", "atomic"]],
        names=["root", "history", "comp"]).to_frame(index=False)
    d = grid.merge(q.groupby(["root", "history", "comp"]).size().reset_index(name="n"),
                   how="left").fillna({"n": 0})
    d["exposure"] = [exposure.get((r.root, r.history), np.nan) for r in d.itertuples()]
    d = d.dropna(subset=["exposure"])
    d["hrank"] = d.history.map(HRANK)
    return d, roots


def compositional_vs_atomic(d, rows) -> None:
    """Claim 1: paired comparison of per-protocol query rates."""
    piv = (d.groupby(["root", "comp"]).agg(n=("n", "sum"), exposure=("exposure", "sum"))
           .assign(rate=lambda x: x.n / x.exposure).reset_index()
           .pivot(index="root", columns="comp", values="rate").dropna())
    w = stats.wilcoxon(piv.compositional, piv.atomic, alternative="less")
    lower = int((piv.compositional < piv.atomic).sum())
    print(f"\ncompositional rate {piv.compositional.mean():.3f} "
          f"vs atomic {piv.atomic.mean():.3f}  (n={len(piv)} protocols)")
    print(f"  Wilcoxon signed-rank, compositional < atomic: "
          f"W={w.statistic:.0f}, p={w.pvalue:.2e}, lower in {lower}/{len(piv)}")
    rows.append({"test": "Wilcoxon compositional<atomic", "statistic": w.statistic,
                 "p": w.pvalue, "n": len(piv), "detail": f"lower in {lower}/{len(piv)}"})


def rate_vs_history(d, rows) -> None:
    """Claim 2: Poisson GEE of query counts on history rank, per target type."""
    print()
    for label in ["compositional", "atomic"]:
        sub = d[d.comp == label]
        m = smf.gee("n ~ hrank", "root", data=sub, family=sm.families.Poisson(),
                    offset=np.log(sub.exposure)).fit()
        b, p = m.params["hrank"], m.pvalues["hrank"]
        print(f"  [{label:13}] GEE coef={b:+.3f}  rate ratio/level={np.exp(b):.2f}  p={p:.4f}")
        rows.append({"test": f"Poisson GEE rate~history ({label})", "statistic": np.exp(b),
                     "p": p, "n": sub.root.nunique(), "detail": f"coef={b:+.3f}"})


def main() -> None:
    d, roots = build()
    print(f"protocols with >=1 metalinguistic question: {len(roots)}")
    rows = []
    compositional_vs_atomic(d, rows)
    rate_vs_history(d, rows)
    out = TABLES / "metalinguistic_stats.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
