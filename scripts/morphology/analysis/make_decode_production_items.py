"""Regenerates the decode/production item CSVs from the tidy tables.

decode_production_items_full.csv is a row-wise projection of item_table.csv
(preserving its file order) joined to construction_table.csv for the per-
construction native_capacity. decode_production_items.csv is that table
filtered to the novel items whose construction has native_capacity >= 2 --
the same filter decode_production.R applies internally at line 21, so only
the _full table is actually read by the R script.
"""

import pathlib

import pandas as pd

import morphology.common as C

ITEM_TABLE = C.CSV / "item_table.csv"
CONSTRUCTION_TABLE = C.CSV / "construction_table.csv"
FULL_CSV = C.CSV / "decode_production_items_full.csv"
FILTERED_CSV = C.CSV / "decode_production_items.csv"

FULL_COLUMNS = ["run", "model", "cx", "attested", "native_cap", "dec_succ", "exact"]
FILTERED_COLUMNS = [
    "run",
    "model",
    "cx",
    "budget",
    "native_cap",
    "dec_succ",
    "dec_frac",
    "exact",
]


def _add_join_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds the run-timestamp and construction key columns used by both tables."""
    frame = frame.copy()
    frame["run"] = frame["run_key"].str.split("/").str[1].astype(int)
    frame["cx"] = frame["run"].astype(str) + "_" + frame["construction_id"]
    return frame


def build_full_items(
    item_table: pathlib.Path, construction_table: pathlib.Path
) -> pd.DataFrame:
    """Returns the per-item decode/production table in item_table.csv row order."""
    items = _add_join_keys(pd.read_csv(item_table))
    constructions = _add_join_keys(pd.read_csv(construction_table))
    native_cap = constructions.set_index("cx")["native_capacity"]

    items["attested"] = items["is_attested"].map({True: "attested", False: "novel"})
    # Nullable Int64 keeps the 70 unjudged items blank rather than rendering
    # the whole column as floats.
    items["native_cap"] = items["cx"].map(native_cap).astype("Int64")
    # decodability is the mean of two decode judgments per item (0, 0.5, 1.0);
    # dec_succ is the raw success count out of 2.
    items["dec_succ"] = (items["decodability"] * 2).astype(int)
    items["exact"] = items["exact_match"].astype("boolean").astype("Int64")
    return items[FULL_COLUMNS]


def build_filtered_items(full_items: pd.DataFrame, item_table: pathlib.Path) -> pd.DataFrame:
    """Returns the novel, native_cap >= 2 subset carrying budget and dec_frac."""
    items = _add_join_keys(pd.read_csv(item_table))
    frame = full_items.copy()
    frame["budget"] = items["round_time_budget_seconds"].values
    frame["dec_frac"] = frame["dec_succ"] / 2
    frame = frame[(frame["attested"] == "novel") & (frame["native_cap"] >= 2)]
    return frame[FILTERED_COLUMNS]


def main() -> None:
    """Writes both decode/production CSVs next to the statistics scripts."""
    full_items = build_full_items(
        item_table=ITEM_TABLE, construction_table=CONSTRUCTION_TABLE
    )
    filtered_items = build_filtered_items(full_items=full_items, item_table=ITEM_TABLE)
    # Both committed CSVs use the stdlib csv module's CRLF terminators.
    full_items.to_csv(FULL_CSV, index=False, lineterminator="\r\n")
    filtered_items.to_csv(FILTERED_CSV, index=False, lineterminator="\r\n")
    print(f"wrote {FULL_CSV} ({len(full_items)} items)")
    print(f"wrote {FILTERED_CSV} ({len(filtered_items)} items)")


if __name__ == "__main__":
    main()
