"""Derive the slimmed inputs in `data/` from the full working repo.

Provenance only — you do NOT need to run this to reproduce the paper. It records
exactly how each slimmed file in `data/` was cut down from the (much larger) source
exports, so the reduction is auditable.

  python scripts/make_slim_data.py --source /path/to/schmidt_multiagent_2026

What it writes:
  data/learnability_message_level_slim.csv  115MB export -> only the columns and the
        baseline / replace_learned runs that the paper's analyses read.
  data/mdl/grounded/veyru_<id>.json         per-round agent message forms, for the 45
        baseline roots and the 135 same-team resume seeds.
  data/mdl/round_success/<id>.json          the round_success series per resume seed,
        pulled out of each run's full evaluation report.
  data/mdl/labels/<id>.json                 agent model per baseline root.
  data/mdl/postmortems/<id>.txt             the language-design discussion the grammar
        is induced from, pulled out of each root's 50MB event log.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"

# columns the release's analyses actually read from the learnability message export
ML_COLS = ["run_id", "phase", "src_id", "round_number", "substage",
           "message_index_in_substage", "message_agent", "message_text", "success",
           "symptoms", "actions"]
ML_PHASES = ["baseline", "replace_learned"]


def slim_messages(src: Path) -> None:
    ml = src / "results/data/Veyru Learnability Runs (15 rounds, fork, +10 rounds) - message_level.csv"
    out = DATA / "learnability_message_level_slim.csv"
    keep = []
    for chunk in pd.read_csv(ml, usecols=ML_COLS, chunksize=200_000):
        keep.append(chunk[chunk.phase.isin(ML_PHASES)])
    df = pd.concat(keep, ignore_index=True)
    # the environment denotations are long and only read for the baseline example rows,
    # so blank them elsewhere rather than repeating them on all 70k swap-run messages
    df.loc[df.phase != "baseline", ["symptoms", "actions"]] = ""
    df.to_csv(out, index=False)
    print(f"{out.name}: {len(df):,} rows, {out.stat().st_size / 1e6:.1f} MB")


def slim_grounded(src: Path, run_ids: list[str]) -> None:
    """Keep each attempt's two message forms and the meaning they should denote."""
    outdir = DATA / "mdl/grounded"
    outdir.mkdir(parents=True, exist_ok=True)
    for rid in run_ids:
        full = json.loads((src / f"results/grounded_v2/veyru_{rid}.json").read_text())
        rounds = [{"round": r["round"],
                   "attempts": [{"engineer_msg": a.get("engineer_msg"),
                                 "observer_msg": a.get("observer_msg"),
                                 "motif": a.get("motif"),
                                 "correct": a.get("correct")}
                                for a in r["attempts"]]}
                  for r in full["rounds"]]
        (outdir / f"veyru_{rid}.json").write_text(json.dumps({"rounds": rounds}))
    print(f"mdl/grounded: {len(run_ids)} runs")


def extract_postmortems(src: Path, root_ids: list[str], train_max: int = 14) -> None:
    """The agents' out-of-band language-design discussion, which the grammar is induced
    from. Pulled out of each root's event log so the 50MB of raw logs need not ship."""
    sys.path.insert(0, str(src / "scripts/ground_v2"))
    from scfg_prototype import postmortem_by_round

    outdir = DATA / "mdl/postmortems"
    outdir.mkdir(parents=True, exist_ok=True)
    rounds = set(range(1, train_max + 1))
    for rid in root_ids:
        text = postmortem_by_round(src / f"results/bundles/extracted/{rid}/veyru.jsonl", rounds)
        (outdir / f"{rid}.txt").write_text("\n".join(text) if isinstance(text, list) else str(text))
    print(f"mdl/postmortems: {len(root_ids)} roots (rounds 1-{train_max})")


def slim_reports(src: Path, seed_ids: list[str]) -> None:
    """Pull the round_success series out of each seed's full evaluation report."""
    outdir = DATA / "mdl/round_success"
    outdir.mkdir(parents=True, exist_ok=True)
    for rid in seed_ids:
        rep = json.loads((src / f"results/bundles/extracted/{rid}/veyru_report.json").read_text())
        rs = next(m for m in rep["measurements"] if m["metric_name"] == "round_success")
        series = [{"round_number": x["round_number"], "value": x["value"]}
                  for x in rs["per_round"]]
        (outdir / f"{rid}.json").write_text(json.dumps({"run_id": rid, "per_round": series}))
    print(f"mdl/round_success: {len(seed_ids)} runs")


def copy_labels(src: Path, root_ids: list[str]) -> None:
    outdir = DATA / "mdl/labels"
    outdir.mkdir(parents=True, exist_ok=True)
    for rid in root_ids:
        p = src / f"results/bundles/extracted/{rid}/labels.json"
        if p.exists():
            (outdir / f"{rid}.json").write_text(p.read_text())
    print(f"mdl/labels: {len(list(outdir.glob('*.json')))} runs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=HERE.parent,
                    help="the full working repo these files were cut from")
    args = ap.parse_args()

    roots = (DATA / "mdl/protocol_roots.txt").read_text().split()
    resume = json.loads((DATA / "mdl/resume_nopm_runs.json").read_text())
    seeds = [s for v in resume.values() for s in v]

    slim_messages(args.source)
    slim_grounded(args.source, roots + seeds)
    slim_reports(args.source, seeds)
    copy_labels(args.source, roots)
    extract_postmortems(args.source, roots)


if __name__ == "__main__":
    main()
