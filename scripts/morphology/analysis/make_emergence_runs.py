"""Regenerates emergence_runs.csv from the induction results JSONL.

The JSONL carries one row per run for the full 3 models x 5 budgets x 9 seeds
design (135 rows). run_table.csv is the below_gate == False subset (43 rows),
so it cannot be used as the source here: the emergence model's outcome is
whether a run cleared the productivity gate, and the 92 gate-failing runs are
the zeros in that regression.
"""

import json
import pathlib

import pandas as pd

import morphology.common as C

SOURCE_JSONL = C.DATA / "induction_results.jsonl"
OUTPUT_CSV = C.CSV / "emergence_runs.csv"

COLUMNS = [
    "run_key",
    "model",
    "budget",
    "seed",
    "below_gate",
    "nonflat",
    "n_multimorpheme",
    "n_codes",
    "n_constructions",
]


def build_emergence_runs(source_jsonl: pathlib.Path) -> pd.DataFrame:
    """Reads the induction JSONL and returns the per-run emergence table."""
    rows = [
        json.loads(line)
        for line in source_jsonl.read_text().splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(rows)
    frame = frame[frame["ok"]].copy()
    frame["below_gate"] = frame["below_gate"].astype(int)
    frame["nonflat"] = 1 - frame["below_gate"]
    frame = frame.sort_values(["model", "budget", "run_key"])
    return frame[COLUMNS].reset_index(drop=True)


def main() -> None:
    """Writes emergence_runs.csv next to the statistics scripts."""
    frame = build_emergence_runs(source_jsonl=SOURCE_JSONL)
    # The committed CSV was written by the stdlib csv module (excel dialect),
    # so it uses CRLF terminators; match it for a byte-identical rewrite.
    frame.to_csv(OUTPUT_CSV, index=False, lineterminator="\r\n")
    print(f"wrote {OUTPUT_CSV} ({len(frame)} runs, {int(frame['nonflat'].sum())} non-flat)")


if __name__ == "__main__":
    main()
