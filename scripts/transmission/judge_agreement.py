"""How well does the GPT-5.5 judge agree with a human annotator?

50 items were annotated blind by one author, with no sight of the judge's labels.
Two agreements are reported:

  has_question    over all 50 items — is this a metalinguistic query at all?
  compositional   over the items the HUMAN marked as questions. Compositionality is
                  only defined for questions, so scoring it on non-questions would
                  inflate agreement with trivially-matching nulls.

  python scripts/transmission/judge_agreement.py
    -> results/tables/judge_agreement.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, TABLES  # noqa: E402

VAL = DATA / "annotations/judge_validation"


def kappa(a, b) -> float:
    """Cohen's kappa for two aligned label sequences."""
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b), key=str)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def score(field, ids, judge, human, note) -> dict:
    j = [judge[i].get(field) for i in ids]
    h = [human[i].get(field) for i in ids]
    agree = sum(x == y for x, y in zip(j, h))
    print(f"  {field:14} {agree}/{len(ids)} = {agree / len(ids):.0%}   "
          f"kappa={kappa(j, h):+.2f}   ({note})")
    return {"field": field, "scope": note, "n": len(ids),
            "agree": agree, "kappa": round(kappa(j, h), 3)}


def main() -> None:
    judge = json.loads((VAL / "model_labels.json").read_text())
    human = json.loads((VAL / "human_labels.json").read_text())
    ids = [i for i in human if i in judge]
    print(f"{len(ids)} blind-annotated items\n")

    rows = [score("has_question", ids, judge, human, "all items")]
    q_ids = [i for i in ids if human[i].get("has_question")]
    rows.append(score("compositional", q_ids, judge, human, "human-marked questions"))

    out = TABLES / "judge_agreement.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
