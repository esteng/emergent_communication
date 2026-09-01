"""Export `data/morphology/` from the working tree, and record how.

Provenance only -- you do NOT need to run this to reproduce the paper, and a release
recipient cannot: it reads the working `runs/` corpus and `runs_extracted/`, neither of
which ships. What it is for is making the export auditable, since two of the steps below
make choices that are invisible in the result.

  python scripts/make_slim_morphology_data.py --runs ../../runs --extracted ../../runs_extracted

Most of the export is a straight copy; only the caches and the judge validation are
actually reduced.

  transcripts/         159 MB, 270 files. Each of the 135 induced roots' event log and run
        summary, copied VERBATIM -- no filtering, no reshaping, no re-serialisation. These
        are the transcripts and postmortems everything else is derived from, and they are
        the bulk of the release. Only `veyru_debug.jsonl` is left behind, being a duplicate
        logging sink rather than simulation state.

  cache/               2.8 MB, 313 files. REDUCED. The working tree keeps the full
        induction history (`joint_paradigm_v1..v4`, `role_expansion_v1`); the release ships
        one generation, under unversioned names. `PRODUCTION_JOINT_GENERATION` below is the
        only record of which generation the paper's grammars came from.

  encode_decode/       1.9 MB, 130 files. Per tested root: the report, the scored items and
        the raw model responses. One root also carries `decode_items.jsonl`, from the decode
        battery -- a small minority of the item table, but it enters the reported
        decode-accuracy error analysis, so it ships.

  robustness/          604 KB, 33 files. The cached replica outputs behind the pooled
        agreement statistics, plus the per-run report those statistics are weighted by.

  judge_validation/    44 KB, 2 files. REDUCED. The labeling session ran as three sittings
        of 50 blind items, split across a blind CSV and a `_key.csv` each, with human scores
        appended to a JSONL -- eleven files. None of that structure means anything
        downstream, so the release ships one flat 150-row table plus the rubric both raters
        scored against. An earlier sitting of 100 items is dropped entirely: it was labeled
        with the judge's verdict revealed after each commit, so it is not blind.

  induction_results.jsonl   32 KB. One row per induced root: the gate outcome the emergence
        figure is computed from, for all 135 roots rather than the 43 that were tested.

  run_manifest.json    The 135 induced and 43 tested run ids.

`prompts/` is not written here. Those files are the prompts themselves, loaded at call time
by the pipeline -- source, not a reduction of anything in the working tree.

It reads the working tree and writes only inside `data/morphology/`; nothing outside it is
created, modified or removed.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

RELEASE = Path(__file__).resolve().parents[1]
DATA = RELEASE / "data" / "morphology"

# The production induction generation behind every number in the paper. The working tree
# also holds v1-v3 (and a v5 constant with no files); those are superseded and never ship.
PRODUCTION_JOINT_GENERATION = "joint_paradigm_v4_"
PRODUCTION_ROLE_GENERATION = "role_expansion_v1_"
PRODUCTION_MODEL_SUFFIX = "gpt-5-4"

ENCODE_DECODE_FILES = {
    "encode_decode_report.json": "report.json",
    "encode_decode_items.jsonl": "items.jsonl",
    "encode_decode_responses.jsonl": "responses.jsonl",
    # One run also carries decode-battery items. They are a small minority of the item
    # table but they enter the reported decode-accuracy error analysis, so they ship.
    "morphology_pipeline_items.jsonl": "decode_items.jsonl",
}


def induced_run_ids(battery: Path) -> list[str]:
    """The 135 roots the emergence figure is computed over."""
    rows = [json.loads(line) for line in
            (battery / "induction_results_135.jsonl").read_text().splitlines() if line.strip()]
    return [r["run_key"].split("/")[-1] for r in rows]


def tested_run_ids(battery: Path) -> list[str]:
    """The roots that cleared the gate and went through the encode->decode battery."""
    items = pd.read_csv(battery / "item_table.csv")
    keys = items[items["pipeline"] == "encode_decode"]["run_key"].unique()
    return sorted({k.split("/")[-1] for k in keys})


def copy_caches(cache_src: Path, induced: list[str]) -> int:
    """Copy the production induction generation, stripping the version from each name."""
    out = DATA / "cache"
    n = 0
    for run_id in induced:
        pairs = [
            (f"{PRODUCTION_JOINT_GENERATION}{run_id}_{PRODUCTION_MODEL_SUFFIX}.json",
             f"joint_paradigm_{run_id}_{PRODUCTION_MODEL_SUFFIX}.json"),
            (f"{PRODUCTION_ROLE_GENERATION}{run_id}_{PRODUCTION_MODEL_SUFFIX}.json",
             f"role_expansion_{run_id}_{PRODUCTION_MODEL_SUFFIX}.json"),
            (f"cb_{run_id}_{PRODUCTION_MODEL_SUFFIX}.parquet",
             f"cb_{run_id}_{PRODUCTION_MODEL_SUFFIX}.parquet"),
        ]
        for src_name, dst_name in pairs:
            src = cache_src / src_name
            if src.exists():
                shutil.copy2(src, out / dst_name)
                n += 1
    return n


def copy_transcripts(runs_root: Path, scenario: str, run_ids: list[str]) -> tuple[int, int]:
    """Copy each root's event log and run summary verbatim into the release.

    The event log ships untouched: no filtering, no reshaping, no re-serialisation. Only
    ~7% of it is `message_sent` -- the rest is tool calls, tool results and raw LLM
    responses that no release script reads -- but shipping it whole means the transcripts
    are exactly the artifact the simulation produced, byte for byte, rather than anything
    this script decided to keep.

    `veyru_debug.jsonl` is the one run-dir file left behind: it is a duplicate logging
    sink, not simulation state.
    """
    out = DATA / "transcripts" / scenario
    n_files = 0
    n_bytes = 0
    for run_id in run_ids:
        src_dir = runs_root / scenario / run_id
        for name in ("veyru.jsonl", "run_summary_cache.json"):
            src = src_dir / name
            if not src.exists():
                continue
            dst = out / run_id / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n_files += 1
            n_bytes += src.stat().st_size
    return n_files, n_bytes


def copy_encode_decode(runs_root: Path, scenario: str, tested: list[str]) -> int:
    """Copy the encode->decode report, items and responses for every tested run."""
    n = 0
    for run_id in tested:
        run_dir = runs_root / scenario / run_id
        dst = DATA / "encode_decode" / run_id
        for src_name, dst_name in ENCODE_DECODE_FILES.items():
            src = run_dir / src_name
            if src.exists():
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst / dst_name)
                n += 1
    return n



# The judge-validation session was run as three sittings of 50 blind items, each split
# across a blind CSV (the two meanings) and a `_key.csv` (the judge's verdict), with the
# human scores appended to a JSONL. None of that structure means anything downstream --
# the reported statistic pools all 150 -- so the release ships one flat table instead of
# the eleven working files. An earlier sitting of 100 items is dropped entirely: it was
# labeled with the judge's verdict revealed after each commit, so it is not blind.
BLIND_BATCHES = ["set1", "set2", "set3"]
JUDGE_VALIDATION_COLUMNS = [
    "item", "run", "source_item_id", "purpose", "n_morphemes",
    "expected_form", "produced_form", "intended_meaning", "decoder_reading",
    "judge_score", "human_score",
]


def build_judge_validation(source: Path) -> int:
    """Flatten the blind judge-validation session into one 150-row human-vs-judge table."""
    human: dict[tuple[str, str], float] = {}
    for line in (source / "human_labels.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["set_id"] in BLIND_BATCHES:
            human[(row["set_id"], row["item"])] = float(row["score"])

    out_rows = []
    for batch in BLIND_BATCHES:
        blind = {
            row["item"]: row
            for row in csv.DictReader((source / f"judge_validation_{batch}.csv").open(encoding="utf-8"))
        }
        key_path = source / f"judge_validation_{batch}_key.csv"
        for row in csv.DictReader(key_path.open(encoding="utf-8")):
            score = human.get((batch, row["item"]))
            if score is None:
                continue
            shown = blind.get(row["item"], {})
            out_rows.append({
                "run": row["run"],
                "source_item_id": row["source_item_id"],
                "purpose": row["purpose"],
                "n_morphemes": row["n_morphemes"],
                "expected_form": row["expected_form"],
                "produced_form": row["produced_form"],
                "intended_meaning": shown.get("intended_meaning", ""),
                "decoder_reading": shown.get("decoder_reading", ""),
                "judge_score": row["judge_score"],
                "human_score": score,
            })

    out_dir = DATA / "judge_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "judge_validation.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=JUDGE_VALIDATION_COLUMNS)
        writer.writeheader()
        for index, row in enumerate(out_rows, start=1):
            writer.writerow({"item": index, **row})
    shutil.copy2(source / "judge_rubric_verbatim.txt", out_dir / "judge_rubric.txt")
    return len(out_rows), 2  # rows in the table, files written


def copy_tree(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(src.rglob("*")):
        if path.is_file():
            target = dst / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True, help="the working runs/ corpus")
    ap.add_argument("--extracted", type=Path, required=True, help="the working runs_extracted/")
    ap.add_argument("--scenario", default="veyru")
    args = ap.parse_args()

    battery = args.extracted / "encode_decode_battery"
    if not battery.exists():
        sys.exit(f"no encode_decode_battery under {args.extracted}")

    induced = induced_run_ids(battery)
    tested = tested_run_ids(battery)

    for sub in ("cache", "transcripts", "encode_decode", "robustness", "judge_validation"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)

    n_cache = copy_caches(args.extracted / "morphology_pipeline_cache", induced)
    n_tx, tx_bytes = copy_transcripts(args.runs, args.scenario, induced)
    n_ed = copy_encode_decode(args.runs, args.scenario, tested)
    n_rob = copy_tree(args.extracted / "morphology_pipeline_cache" / "robustness_cache",
                      DATA / "robustness")
    shutil.copy2(args.extracted / "pipeline_robustness_report_final.json",
                 DATA / "robustness" / "per_run_report.json")
    n_rob = sum(1 for p in (DATA / "robustness").rglob("*") if p.is_file())
    n_judge_rows, n_judge = build_judge_validation(args.extracted / "judge_validation")

    shutil.copy2(battery / "induction_results_135.jsonl", DATA / "induction_results.jsonl")

    manifest = {"scenario": args.scenario,
                "induced_run_ids": sorted(induced),
                "tested_run_ids": sorted(tested),
                "induction_model": "gpt-5.4"}
    (DATA / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print(f"cache            {n_cache:5d} files ({len(induced)} induced roots)")
    print(f"transcripts      {n_tx:5d} files ({tx_bytes / 1048576:.1f} MB, verbatim event logs)")
    print(f"encode_decode    {n_ed:5d} files ({len(tested)} tested roots)")
    print(f"robustness       {n_rob:5d} files")
    print(f"judge_validation {n_judge:5d} files ({n_judge_rows} pooled judgments)")


if __name__ == "__main__":
    main()
