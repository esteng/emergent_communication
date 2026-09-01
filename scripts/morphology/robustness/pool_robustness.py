"""Pooled replica-agreement statistics for the induction and encode->decode stages.

Five runs were re-run three times; `pipeline_robustness.py` writes one replica per file
and a per-run report. This module pools those replicas across all five runs before
computing each statistic once, rather than averaging per-run values: every replica pair
contributes its item-level observations to a single confusion matrix.

Induction is pooled over three units. Segmentation compares the morpheme list each
replica assigns to a symbol. Construction co-clustering compares, for each pair of
symbols in a run, whether both replicas group them together; symbols in no construction
share one bin. Gloss agreement compares the meaning each replica assigns to a morpheme.

Encode->decode is pooled per item, keyed by target meaning: whether the coined form
matches the canonical one exactly, and whether it uses the expected morphemes in any
order. Semantic equivalence of the decoder's readings is weighted by the number of
readings each run sent to the judge.

Judge agreement reruns a fixed judge call three times per run and compares the verdicts.

  python -m morphology.robustness.pool_robustness
    -> results/tables/pipeline_robustness_pooled.csv
"""
import csv
import glob
import itertools
import json

import morphology.common as C
from morphology.robustness.agreement import (
    agreement_rate,
    kappa,
    ordinal_labels,
)


RUNS = ["1778877857", "1778877932", "1778879414", "1778881541", "1780417820"]
REPLICAS = (0, 1, 2)
SCENARIO = "veyru"
INDUCTION_MODEL_SUFFIX = "gpt-5-4"
# Symbols the induction assigned to no construction are compared as one bin.
UNGROUPED = "__ungrouped__"


def _induction_replica(run_id: str, replica: int) -> dict | None:
    """One induction replica; replica 0 is the run's production paradigm."""
    matches = glob.glob(str(C.ROBUSTNESS / f"induction_r{replica}_{SCENARIO}_{run_id}_*.json"))
    if not matches and replica == 0:
        matches = glob.glob(str(C.CACHE / f"joint_paradigm_{run_id}_{INDUCTION_MODEL_SUFFIX}.json"))
    if not matches:
        return None
    return json.loads(open(matches[0], encoding="utf-8").read())


def _encode_decode_replica(run_id: str, replica: int) -> dict | None:
    """One encode->decode replica keyed by target meaning; replica 0 is production."""
    if replica == 0:
        path = C.ENCODE_DECODE / run_id / "items.jsonl"
        if not path.exists():
            return None
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {
            row["target_meaning"]: {"exact": row.get("exact_match"), "bag": row.get("bag_match")}
            for row in rows
            if row.get("pipeline") == "encode_decode" and row.get("target_meaning")
        }
    path = C.ROBUSTNESS / f"encode_decode_r{replica}_{SCENARIO}_{run_id}.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["results"]
    return {
        row["target_meaning"]: {"exact": row.get("match_expected"), "bag": row.get("match_bag")}
        for row in rows
        if row.get("target_meaning")
    }


def _replicas(run_id: str, loader) -> dict[int, dict]:
    loaded = {replica: loader(run_id, replica) for replica in REPLICAS}
    return {replica: value for replica, value in loaded.items() if value}


def pooled_segmentation() -> tuple[int, float]:
    """Fraction of symbol comparisons where both replicas split the symbol identically."""
    total = matched = 0
    for run_id in RUNS:
        replicas = _replicas(run_id, _induction_replica)
        segmentations = {r: v["grammar"]["segmentation"] for r, v in replicas.items()}
        for left, right in itertools.combinations(sorted(segmentations), 2):
            for symbol in set(segmentations[left]) & set(segmentations[right]):
                total += 1
                matched += segmentations[left][symbol] == segmentations[right][symbol]
    return total, matched / total


def pooled_co_clustering() -> tuple[int, float, float]:
    """Agreement and kappa on whether two symbols share a construction."""
    left_labels: list[bool] = []
    right_labels: list[bool] = []
    for run_id in RUNS:
        replicas = _replicas(run_id, _induction_replica)
        symbols = sorted(replicas[1]["grammar"]["segmentation"])
        groups = {
            replica: {
                symbol: (value["analysis"].get("construction_by_symbol") or {}).get(symbol) or UNGROUPED
                for symbol in symbols
            }
            for replica, value in replicas.items()
        }
        for left, right in itertools.combinations(sorted(groups), 2):
            for first, second in itertools.combinations(symbols, 2):
                left_labels.append(groups[left][first] == groups[left][second])
                right_labels.append(groups[right][first] == groups[right][second])
    return (len(left_labels), agreement_rate(left_labels, right_labels),
            kappa(left_labels, right_labels))


def pooled_gloss_agreement() -> tuple[int, float]:
    """Fraction of morpheme comparisons where both replicas give the same gloss."""
    total = matched = 0
    for run_id in RUNS:
        replicas = _replicas(run_id, _induction_replica)
        glosses = {}
        for replica, value in replicas.items():
            forms = {}
            for slot in value["grammar"].get("slots") or []:
                for filler in slot.get("fillers") or []:
                    forms.setdefault(filler.get("form"), filler.get("gloss"))
            glosses[replica] = forms
        for left, right in itertools.combinations(sorted(glosses), 2):
            for form in set(glosses[left]) & set(glosses[right]):
                first, second = glosses[left][form], glosses[right][form]
                if first is None or second is None:
                    continue
                total += 1
                matched += first == second
    return total, matched / total


def pooled_production() -> tuple[int, float, float]:
    """Kappa on exact-match and any-order-match of the coined form, pooled per item."""
    exact: tuple[list, list] = ([], [])
    bag: tuple[list, list] = ([], [])
    for run_id in RUNS:
        replicas = _replicas(run_id, _encode_decode_replica)
        for left, right in itertools.combinations(sorted(replicas), 2):
            for meaning in set(replicas[left]) & set(replicas[right]):
                for key, store in (("exact", exact), ("bag", bag)):
                    first, second = replicas[left][meaning][key], replicas[right][meaning][key]
                    if first is None or second is None:
                        continue
                    store[0].append(first)
                    store[1].append(second)
    return len(exact[0]), kappa(*exact), kappa(*bag)


def pooled_reading_equivalence() -> tuple[int, float]:
    """Semantic-equivalence rate of decoder readings, weighted by readings judged."""
    report = json.loads((C.ROBUSTNESS / "per_run_report.json").read_text(encoding="utf-8"))
    judged = equivalent = 0.0
    for value in report.values():
        stage = value["encode_decode"]
        count = stage["n_reading_disagreements_sent_to_judge"]
        judged += count
        equivalent += count * stage["reading_semantic_equivalence_rate"]
    return int(judged), equivalent / judged


def pooled_judge_agreement() -> tuple[int, float]:
    """Linear-weighted kappa on the judge's decode verdicts across reruns of a fixed call."""
    left_labels: list[float] = []
    right_labels: list[float] = []
    for path in sorted(C.ROBUSTNESS.glob("judge_consistency_encode_decode_*.json")):
        reruns = json.loads(path.read_text(encoding="utf-8"))["decode_scores"]
        for left, right in itertools.combinations(range(len(reruns)), 2):
            left_labels += list(reruns[left])
            right_labels += list(reruns[right])
    return len(left_labels), kappa(ordinal_labels(left_labels),
                                   ordinal_labels(right_labels), weights="linear")


def main() -> None:
    rows = []
    n, rate = pooled_segmentation()
    rows.append(("segmentation_exact_match", n, rate))
    n, agreement, kappa = pooled_co_clustering()
    rows.append(("construction_co_clustering_agreement", n, agreement))
    rows.append(("construction_co_clustering_kappa", n, kappa))
    n, rate = pooled_gloss_agreement()
    rows.append(("gloss_literal_agreement", n, rate))
    n, exact_kappa, bag_kappa = pooled_production()
    rows.append(("production_exact_match_kappa", n, exact_kappa))
    rows.append(("production_any_order_match_kappa", n, bag_kappa))
    n, rate = pooled_reading_equivalence()
    rows.append(("reading_semantic_equivalence", n, rate))
    n, kappa = pooled_judge_agreement()
    rows.append(("judge_decode_verdict_kappa", n, kappa))

    for name, n, value in rows:
        print(f"{name:38s} n={n:6d}  {value:.4f}")

    out = C.TABLES / "pipeline_robustness_pooled.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "n", "value"])
        for name, n, value in rows:
            writer.writerow([name, n, f"{value:.4f}"])
    print(f"wrote {out.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
