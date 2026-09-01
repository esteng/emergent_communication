"""Wilson intervals on the novel-cell decode and production rates, per model.

Novel cells are the paradigm-licensed forms the agents never coined. Decode counts a
success when the judge scored the receiver's reading a full match; production counts one
when the coined form equals the canonical form exactly. Both are binomial over items.

These are item-level rates. The bars in `plot_panel_decode_production.py` average within a
construction and then across constructions, so the point estimates here differ from the
bar heights; both read the same rows via `item_table_view.load_items`.

  python -m morphology.analysis.stat_decode_production
    -> results/tables/decode_production_wilson.csv
"""
import csv

from statsmodels.stats.proportion import proportion_confint

import morphology.common as C
from morphology.analysis.item_table_view import FULL_MATCH, MODEL_ORDER, load_items

METRICS = (("decode", "decodability"), ("production", "exact_match"))


def main() -> None:
    novel = load_items()
    novel = novel[novel["cell_type"] == "Novel"]

    rows = []
    for metric, column in METRICS:
        scored = novel.dropna(subset=[column])
        for model in MODEL_ORDER:
            subset = scored[scored["model_label"] == model]
            if subset.empty:
                continue
            n_items = len(subset)
            n_success = int((subset[column] >= FULL_MATCH).sum())
            low, high = proportion_confint(n_success, n_items, alpha=0.05, method="wilson")
            rows.append((metric, model, n_items, n_success, n_success / n_items, low, high))

    print(f"{'metric':11s} {'model':19s} {'k/n':>10s} {'rate':>7s}   95% Wilson CI")
    for metric, model, n_items, n_success, rate, low, high in rows:
        print(f"{metric:11s} {model:19s} {n_success:5d}/{n_items:<4d} "
              f"{rate:7.3f}   [{low:.3f}, {high:.3f}]")
    print(f"all lower bounds > 0: {all(row[5] > 0 for row in rows)}")

    out = C.TABLES / "decode_production_wilson.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "model", "n_items", "n_success", "rate", "ci_low", "ci_high"])
        for metric, model, n_items, n_success, rate, low, high in rows:
            writer.writerow([metric, model, n_items, n_success,
                             f"{rate:.4f}", f"{low:.4f}", f"{high:.4f}"])
    print(f"wrote {out.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
