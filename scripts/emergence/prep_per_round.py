"""Build the per-(run, round) tables behind the perplexity and success-rate figures.

For each round of each run:
  perplexity  exp of the mean GPT-2 surprisal over that round's messages. The exported
              column named `perplexity` is actually mean surprisal in nats (see
              REPRODUCIBILITY.md); exponentiating is what makes it a perplexity.
  success     1 iff every message in the round succeeded, matching round_success.

Runs are subsampled deterministically to an equal number per
(model x budget x postmortem) cell, so every plotted point rests on the same n.

  python scripts/emergence/prep_per_round.py
    -> results/csv/per_round_closed.csv   proprietary models, all five budgets
    -> results/csv/per_round_open.csv     open-weight models, budgets {150, 2000}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CSV, DATA  # noqa: E402

SEED = 0
N_ROUNDS = 15
OPEN_MODELS = ["Qwen/Qwen3-32B", "meta-llama/Llama-3.3-70B-Instruct"]
OPEN_BUDGETS = [150, 2000]   # the two budgets the open-weight figure plots


def per_round(sub: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per (run, round), over runs that completed all 15 rounds."""
    maxr = sub.groupby("run_id").round_number.max()
    sub = sub[sub.run_id.isin(maxr[maxr == N_ROUNDS].index)]
    return (sub.groupby(["run_id", "round_number"])
            .agg(perplexity=("perplexity", "mean"),   # still surprisal; exp'd by caller
                 success=("success", "min"),
                 budget=("round_time_budget_seconds", "first"),
                 postmortem=("postmortem", "first"),
                 model=("field_observer_model", "first"))
            .reset_index())


def min_cell(pr: pd.DataFrame, budgets) -> int:
    """Smallest (model x budget x postmortem) run count over the given budgets."""
    rm = pr[pr.budget.isin(budgets)][["run_id", "model", "budget", "postmortem"]].drop_duplicates()
    return int(rm.groupby(["model", "budget", "postmortem"]).run_id.nunique().min())


def balance(pr: pd.DataFrame, budgets, n: int) -> pd.DataFrame:
    """Subsample to exactly n runs per (model x budget x postmortem) cell."""
    pr = pr[pr.budget.isin(budgets)]
    cells = pr[["run_id", "model", "budget", "postmortem"]].drop_duplicates()
    keep = []
    for _, g in cells.groupby(["model", "budget", "postmortem"]):
        keep += g.sort_values("run_id").sample(n=n, random_state=SEED).run_id.tolist()
    out = pr[pr.run_id.isin(keep)].copy()
    out["postmortem"] = out["postmortem"].astype(bool)
    return out


def build_closed(m: pd.DataFrame) -> pd.DataFrame:
    """Proprietary models. Note: exponentiates at MESSAGE level, then averages."""
    mc = m[m.model_class == "closed"].copy()
    mc["perplexity"] = np.exp(mc["perplexity"])
    pr = per_round(mc)
    return balance(pr, sorted(pr.budget.unique()), min_cell(pr, sorted(pr.budget.unique())))


def build_open(m: pd.DataFrame) -> pd.DataFrame:
    """Open-weight self-play pairs. Note: averages FIRST, then exponentiates — the
    opposite order to build_closed. Both orders are kept because the two figures in the
    paper were produced that way; see REPRODUCIBILITY.md."""
    raw = {mo: per_round(m[(m.field_observer_model == mo) & (m.engineer_model == mo)])
           for mo in OPEN_MODELS}
    n = min(min_cell(pr, OPEN_BUDGETS) for pr in raw.values())
    parts = []
    for mo in OPEN_MODELS:
        d = balance(raw[mo], OPEN_BUDGETS, n)
        d["perplexity"] = np.exp(d["perplexity"])
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def report(df: pd.DataFrame, out: Path) -> None:
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows ({df.run_id.nunique()} runs) -> {out.name}")
    print(df.groupby(["model", "budget", "postmortem"]).run_id.nunique()
          .rename("n_runs").to_string(), "\n")


def main() -> None:
    m = pd.read_csv(DATA / "baseline_message_level.csv")
    report(build_closed(m), CSV / "per_round_closed.csv")
    report(build_open(m), CSV / "per_round_open.csv")


if __name__ == "__main__":
    main()
