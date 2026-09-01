"""Tidy per-item score rows shared by the decode and encode->decode batteries.

Both pipelines report aggregate rates (per role, per depth, per construction), which is
enough to plot but not to model: a per-slot mean discards how many items it averaged and
which item scored what, so the corpus cannot be fit or tested item-wise. This module
defines the one-row-per-tested-item table both pipelines write alongside their reports.

One row is one stimulus presented to one agent. ``pipeline`` says which battery produced
it, and the production columns (``exact_match`` / ``bag_match`` / ``swapped_order``) are
None for decode rows, where the code was composed by the pipeline rather than coined by an
agent -- there is no production to score. ``decodability`` is populated for every row, so
it is the column both batteries can be compared on.
"""

import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

DECODE_PIPELINE = "decode"
ENCODE_DECODE_PIPELINE = "encode_decode"


class ItemScore(BaseModel):
    """One tested paradigm cell's scores, with the factors needed to model it downstream.

    Grouping factors: ``run_key``, ``pipeline``, ``model``, ``construction_id``,
    ``purpose`` (novel vs control vs depth), ``varied_slot`` (the semantic role under
    test), and ``n_morphemes`` (agglutinative depth). Outcomes: the three production
    booleans (encode->decode only) and ``decodability``.
    """

    run_key: str
    pipeline: str
    item_id: str  # stable within a run: pipeline + agent + target meaning
    encoder_agent: str  # empty for decode rows -- no agent coined the form
    decoder_agent: str
    model: str
    construction_id: str
    purpose: str
    varied_slot: str
    varied_slot_label: str
    expected_slots: list[str] = []  # every slot_id this item's form spans, in order
    pooled_slots: list[str] = []  # subset of expected_slots that drew a pooled filler
    n_morphemes: int
    target_meaning: str
    expected_form: str
    produced_form: str  # empty for decode rows
    reading: str
    is_valid: bool
    exact_match: bool | None
    bag_match: bool | None
    swapped_order: bool | None
    decodability: float


def write_item_scores(path: Path, rows: list[ItemScore]) -> None:
    """Write the per-item table as JSONL, one row per tested item."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")
    logger.info("wrote %d per-item score rows to %s", len(rows), path)


def read_item_scores(path: Path) -> list[ItemScore]:
    """Load a per-item table written by ``write_item_scores``."""
    return [
        ItemScore.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
