"""Average no-postmortem success rate, proprietary vs. open-weight.

The paper reports 92.1% (proprietary) vs. 30.8% (open-weight) at the 2000s budget.
Both are postmortem-OFF runs only, and success is strongly budget-dependent, so the
table is split by budget rather than pooled.

Each (model x budget) no-postmortem cell is subsampled to a common n so every model
is weighted equally; a class score is then the mean over rounds of the per-round mean.

  python scripts/emergence/stat_success_by_budget.py
    -> results/tables/success_by_budget_nopm.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, TABLES  # noqa: E402
from prep_per_round import OPEN_MODELS, SEED, per_round  # noqa: E402


def avg_across_rounds(d: pd.DataFrame) -> float:
    """Mean success per round, then averaged over rounds (rounds weighted equally)."""
    return d.groupby("round_number").success.mean().mean()


def main() -> None:
    m = pd.read_csv(DATA / "baseline_message_level.csv")
    groups = {
        "closed-source": per_round(m[m.model_class == "closed"]),
        "open-source": pd.concat(
            [per_round(m[(m.field_observer_model == mo) & (m.engineer_model == mo)])
             for mo in OPEN_MODELS], ignore_index=True),
    }

    # a common no-postmortem sample size per (model x budget) cell, across ALL models
    nopm_all = pd.concat(groups.values(), ignore_index=True)
    nopm_all = nopm_all[~nopm_all.postmortem.astype(bool)]
    cells = nopm_all[["run_id", "model", "budget"]].drop_duplicates()
    n = int(cells.groupby(["model", "budget"]).run_id.nunique().min())
    print(f"subsampling each (model x budget) no-postmortem cell to n = {n}\n")

    rows = []
    for name, pr in groups.items():
        nopm = pr[~pr.postmortem.astype(bool)]
        keep = []
        for _, g in nopm[["run_id", "model", "budget"]].drop_duplicates().groupby(["model", "budget"]):
            keep += g.sort_values("run_id").sample(n=n, random_state=SEED).run_id.tolist()
        sub = nopm[nopm.run_id.isin(keep)]
        for b in sorted(sub.budget.unique()):
            d = sub[sub.budget == b]
            rows.append({"group": name, "budget": b,
                         "avg_success": round(avg_across_rounds(d), 3),
                         "n_runs": d.run_id.nunique()})
        rows.append({"group": name, "budget": "ALL (pooled)",
                     "avg_success": round(avg_across_rounds(sub), 3),
                     "n_runs": sub.run_id.nunique()})

    summary = pd.DataFrame(rows)
    out = TABLES / "success_by_budget_nopm.csv"
    summary.to_csv(out, index=False)

    print("No-postmortem average success across the 15 rounds, by budget:\n")
    print(summary.pivot_table(index="budget", columns="group",
                              values="avg_success", sort=False).to_string())
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
