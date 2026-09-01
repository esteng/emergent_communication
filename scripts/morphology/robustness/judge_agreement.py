"""Agreement between the semantic decode judge and a blind human annotator.

The judge scores whether the decoder agent's reading matches the meaning the encoder was
given, on a three-point ordinal (0 = none, 0.5 = partial, 1.0 = full). One author
re-scored the same items against the same rubric without seeing the judge's verdict.

All 150 blind judgments are pooled into one human-vs-judge comparison. There is no
per-batch or per-round structure: `data/morphology/judge_validation/judge_validation.csv`
is a flat table of the two raters' scores, one row per item, with the item's provenance
and the two meanings that were actually compared. `judge_rubric.txt` is the rubric both
raters worked from.

  python -m morphology.robustness.judge_agreement
    -> results/tables/judge_agreement.csv
"""
import csv

import morphology.common as C
from morphology.robustness.agreement import kappa, ordinal_labels


# The three-point ordinal both raters score on.
SCORE_VALUES = [0.0, 0.5, 1.0]


def main() -> None:
    path = C.JUDGE_VALIDATION / "judge_validation.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    human = [float(row["human_score"]) for row in rows]
    judge = [float(row["judge_score"]) for row in rows]

    n = len(rows)
    n_agree = sum(1 for a, b in zip(human, judge) if a == b)
    harsher = sum(1 for a, b in zip(human, judge) if b < a)
    human_labels, judge_labels = ordinal_labels(human), ordinal_labels(judge)
    unweighted = kappa(human_labels, judge_labels)
    weighted = kappa(human_labels, judge_labels, weights="linear")

    print(f"n                        {n}")
    print(f"exact agreement          {n_agree}/{n} ({n_agree / n:.1%})")
    print(f"unweighted kappa         {unweighted:.3f}")
    print(f"linear-weighted kappa    {weighted:.3f}")
    print(f"judge harsher than human {harsher}/{n - n_agree} of the mismatches")

    out = C.TABLES / "semantic_judge_agreement.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["n", n])
        writer.writerow(["exact_agreement", f"{n_agree / n:.4f}"])
        writer.writerow(["n_exact_agree", n_agree])
        writer.writerow(["unweighted_kappa", f"{unweighted:.4f}"])
        writer.writerow(["linear_weighted_kappa", f"{weighted:.4f}"])
        writer.writerow(["n_mismatch", n - n_agree])
        writer.writerow(["n_judge_harsher", harsher])
    print(f"wrote {out.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
