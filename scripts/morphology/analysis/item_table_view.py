"""Shared view over the per-item encode->decode table.

The decode/production figure and the Wilson intervals reported alongside it must be
computed over the same rows. This module owns that definition -- the pipeline filter, the
numeric coercion, and the control/novel split -- so the two cannot drift apart.
"""
import pandas as pd

import morphology.common as C

MODEL_LABEL = C.MODEL_LABEL
MODEL_ORDER = [C.MODEL_LABEL[m] for m in C.MODEL_IDS]
SCORE_COLUMNS = ("exact_match", "bag_match", "decodability")
# The judge's full-match score; the point at which an item counts as a success.
FULL_MATCH = 1.0


def load_items() -> pd.DataFrame:
    """Encode->decode items with numeric scores, model labels, and the cell split.

    ``cell_type`` is ``Control`` for cells already attested in the run's codebook and
    ``Novel`` for the paradigm-licensed cells the agents never coined.
    """
    items = pd.read_csv(C.CSV / "item_table.csv")
    items = items[items["pipeline"] == "encode_decode"].copy()
    for column in SCORE_COLUMNS:
        items[column] = pd.to_numeric(items[column], errors="coerce").astype(float)
    items["cell_type"] = items["is_attested"].map({True: "Control", False: "Novel"})
    items["model_label"] = items["model"].map(MODEL_LABEL)
    return items
