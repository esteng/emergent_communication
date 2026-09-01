"""Description-length components on the held-out language-use rounds.

Design. Each of 45 baseline teams plays 14 rounds WITH a postmortem stage, developing
a language; the same team is then resumed for rounds 15-25 with the postmortem stage
removed. Round 15 is the first post-swap round. Each root has 3 resume seeds.

For every seed we reuse that root's baseline grammar (never re-inducing it), turn it
into a PCFG by estimating rule-expansion probabilities on the TRAIN window, and score
the held-out TEST window:

  TRAIN = rounds 10-15   the postmortem tail plus the first no-postmortem round
  TEST  = rounds 16-25   held-out, no postmortem

  python scripts/mdl/compute_components.py
    -> results/csv/mdl_resume_components.csv   one row per resume seed (135)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mdl  # noqa: E402
import scfg  # noqa: E402
from common import CSV, DATA  # noqa: E402

MDL_DATA = DATA / "mdl"


def rounds_arg(spec: str) -> set:
    lo, hi = (int(x) for x in spec.split("-"))
    return set(range(lo, hi + 1))


def message_forms(rounds_data, rounds) -> list:
    """Both agents' coded messages from the given rounds."""
    out = []
    for r in rounds_data:
        if r["round"] not in rounds:
            continue
        for a in r["attempts"]:
            out += [a[k] for k in ("engineer_msg", "observer_msg") if a.get(k)]
    return out


def round_success(seed: str, rounds) -> float | None:
    series = json.loads((MDL_DATA / f"round_success/{seed}.json").read_text())["per_round"]
    vals = [x["value"] for x in series if x["round_number"] in rounds]
    return sum(vals) / len(vals) if vals else None


def one_seed(root, seed, idx, grammars, dlg, model, train, test, k) -> dict:
    g = scfg.Grammar(json.loads((grammars / f"grammar_joint_{root}.json").read_text()))
    rounds_data = json.loads((MDL_DATA / f"grounded/veyru_{seed}.json").read_text())["rounds"]
    train_msgs = message_forms(rounds_data, train)
    test_msgs = message_forms(rounds_data, test)

    ccost = mdl._char_cost(train_msgs or test_msgs)
    dld, info = mdl.dl_data_given_grammar_pcfg(g, train_msgs, test_msgs, ccost, k=k)
    n = info["n_msgs"]
    s = round_success(seed, test)
    unseen = info["frac_rules_unseen"]
    return {"root_id": root, "seed_id": seed, "seed_index": idx, "model": model[root],
            "DL_G": round(dlg[root], 1),
            "pcfg_DgG_total": round(dld, 1),
            "pcfg_DgG_pm": round(dld / n, 2) if n else None,
            "MDL_abs": round(dlg[root] + dld, 1),
            "test_coverage": round(info["coverage"], 3),
            "n_test": n,
            "frac_rules_unseen": round(unseen, 3) if unseen is not None else None,
            "succ_test": round(s, 4) if s is not None else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="10-15", help="inclusive round range for rule probabilities")
    ap.add_argument("--test", default="16-25", help="inclusive round range scored")
    ap.add_argument("--k", type=float, default=0.5, help="add-k Laplace smoothing")
    ap.add_argument("--grammars", type=Path, default=MDL_DATA / "grammars")
    ap.add_argument("--table", type=Path, default=MDL_DATA / "mdl_table_joint.csv")
    ap.add_argument("--out", type=Path, default=CSV / "mdl_resume_components.csv")
    args = ap.parse_args()

    train, test = rounds_arg(args.train), rounds_arg(args.test)
    tbl = pd.read_csv(args.table, dtype={"run_id": str}).set_index("run_id")
    dlg, model = tbl.DL_G.to_dict(), tbl.model.to_dict()
    resume = json.loads((MDL_DATA / "resume_nopm_runs.json").read_text())

    rows = []
    for root, seeds in resume.items():
        for idx, seed in enumerate(seeds):
            try:
                rows.append(one_seed(root, seed, idx, args.grammars, dlg, model,
                                     train, test, args.k))
            except Exception as e:
                print(f"  FAILED {root}/{seed}: {type(e).__name__}: {e}")
    rows = [r for r in rows if r["succ_test"] is not None and r["n_test"]]

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"train={args.train} test={args.test} k={args.k}: "
          f"{len(df)} seed-runs over {df.root_id.nunique()} roots")
    print(f"  median test messages {int(np.median(df.n_test))}, "
          f"mean coverage {df.test_coverage.mean():.2f}, "
          f"mean success {df.succ_test.mean():.2f}")
    print(f"wrote {args.out.name}")


if __name__ == "__main__":
    main()
