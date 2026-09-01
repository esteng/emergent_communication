"""Paradigm saturation: how much of each construction's licensed grid the system covers.

Grid saturation is a property of the codebook alone -- the share of a construction's
licensed filler combinations that the agents actually lexicalized. It says nothing about
the cells they never wrote, which is precisely what the wug batteries test.

``effective_saturation`` closes that gap by crediting the untested remainder at the rate
the wug sample succeeded::

    effective = grid + success * (1 - grid)

The attested cells are covered by definition; of the unattested remainder, the battery
estimates the fraction the agents handle, and that fraction is extrapolated to the rest of
the grid. It is bounded below by ``grid`` (a system that generalizes to nothing covers only
what it wrote) and above by 1.0 (a system that generalizes perfectly covers the whole
grid), so the lift ``effective - grid`` is the productivity the codebook understates.

The same construction is scored under both batteries, which is the point of comparing them:
the decode battery credits a cell the receiver can READ, while the encode->decode battery
credits one the sender can also COIN (under ``exact``, ``bag``, or ``decodability``). Coining
is the stricter test, so its saturation is the more conservative estimate of the grid the
agents genuinely command.
"""

import logging
import math
from typing import NamedTuple

from morphology.pipeline.item_scores import (
    DECODE_PIPELINE,
    ENCODE_DECODE_PIPELINE,
    ItemScore,
)
from morphology.pipeline.joint_paradigm_induction import JointParadigmResult

logger = logging.getLogger(__name__)

# Purposes whose cells are attested codes rather than held-out wugs. They calibrate the
# battery (the comprehension ceiling) and are excluded from the generalization estimate,
# which must be measured on unattested cells only.
CONTROL_PURPOSES = frozenset({"control_attested", "grounded_control"})
# The success columns each battery can define. Decode has no production, so only the
# shared decodability column applies to it.
DECODE_METRICS = ("decodability",)
ENCODE_DECODE_METRICS = ("exact", "bag", "decodability")


class ConstructionSaturation(NamedTuple):
    """One construction's grid saturation: attested codes over the licensed filler product."""

    run_key: str
    construction_id: str
    attested: int
    possible: int
    saturation: float


class EffectiveSaturation(NamedTuple):
    """One construction's saturation under one battery/metric, before and after wug credit.

    ``novel_success`` is the battery's mean score on that construction's held-out cells
    (a rate for the boolean production metrics, a mean 0/0.5/1.0 score for decodability);
    ``effective`` extrapolates it over the unattested remainder of the grid.
    """

    run_key: str
    pipeline: str
    metric: str
    construction_id: str
    attested: int
    possible: int
    grid: float
    n_novel: int
    novel_success: float
    effective: float


def construction_saturations(
    result: JointParadigmResult, run_key: str
) -> list[ConstructionSaturation]:
    """Grid saturation of every construction: attested codes over the licensed product.

    Per construction, each template slot's filler count is the distinct fillers used in
    that construction's own codes, plus one "absent" value when any code omits the slot
    (an optional slot). ``possible`` is the product of those counts and ``attested`` the
    distinct codes; every code maps to a unique slot-value tuple, so saturation <= 1.
    """
    grammar = result.grammar
    analysis = result.analysis
    codes_by_construction: dict[str, list[str]] = {}
    for symbol, construction_id in analysis.construction_by_symbol.items():
        codes_by_construction.setdefault(construction_id, []).append(symbol)
    rows: list[ConstructionSaturation] = []
    for construction_id, template in analysis.construction_templates.items():
        symbols = codes_by_construction.get(construction_id, [])
        if not symbols:
            continue
        fillers: dict[str, set[str]] = {slot_id: set() for slot_id in template}
        optional: dict[str, bool] = {slot_id: False for slot_id in template}
        for symbol in symbols:
            path = analysis.slots_by_symbol.get(symbol, [])
            parts = grammar.segmentation.get(symbol, [])
            present = set()
            for slot_id, form in zip(path, parts):
                if slot_id in fillers:
                    fillers[slot_id].add(form)
                    present.add(slot_id)
            for slot_id in template:
                if slot_id not in present:
                    optional[slot_id] = True
        sizes = [
            len(fillers[slot_id]) + (1 if optional[slot_id] else 0) for slot_id in template
        ]
        possible = math.prod(sizes) if all(sizes) else 0
        if possible <= 0:
            continue
        attested = len(set(symbols))
        rows.append(
            ConstructionSaturation(
                run_key=run_key,
                construction_id=construction_id,
                attested=attested,
                possible=possible,
                saturation=attested / possible,
            )
        )
    return rows


def _item_success(item: ItemScore, metric: str) -> float | None:
    """This item's score under one metric, or None when the metric does not apply to it."""
    if metric == "decodability":
        return item.decodability
    if metric == "exact":
        return None if item.exact_match is None else float(item.exact_match)
    if metric == "bag":
        return None if item.bag_match is None else float(item.bag_match)
    raise ValueError(f"unknown saturation metric {metric!r}")


def effective_saturations(
    grids: list[ConstructionSaturation], items: list[ItemScore]
) -> list[EffectiveSaturation]:
    """Credit each construction's unattested grid at the rate its held-out cells succeeded.

    Emits one row per (construction, pipeline, metric) for which the run tested at least
    one novel cell of that construction. Constructions whose grid is fully saturated, or
    that the battery never sampled, contribute no row -- their generalization is untested,
    not zero.
    """
    grid_by_key = {(row.run_key, row.construction_id): row for row in grids}
    novel = [item for item in items if item.purpose not in CONTROL_PURPOSES]
    metrics_by_pipeline = {
        DECODE_PIPELINE: DECODE_METRICS,
        ENCODE_DECODE_PIPELINE: ENCODE_DECODE_METRICS,
    }
    rows: list[EffectiveSaturation] = []
    grouped: dict[tuple[str, str, str], list[ItemScore]] = {}
    for item in novel:
        grouped.setdefault((item.run_key, item.pipeline, item.construction_id), []).append(item)
    for (run_key, pipeline, construction_id), cells in sorted(grouped.items()):
        grid = grid_by_key.get((run_key, construction_id))
        if grid is None:
            logger.info(
                "no grid for construction %s in %s; skipping its effective saturation",
                construction_id, run_key,
            )
            continue
        for metric in metrics_by_pipeline.get(pipeline, ()):
            scores = [
                score
                for score in (_item_success(item, metric) for item in cells)
                if score is not None
            ]
            if not scores:
                continue
            success = sum(scores) / len(scores)
            rows.append(
                EffectiveSaturation(
                    run_key=run_key,
                    pipeline=pipeline,
                    metric=metric,
                    construction_id=construction_id,
                    attested=grid.attested,
                    possible=grid.possible,
                    grid=grid.saturation,
                    n_novel=len(scores),
                    novel_success=success,
                    effective=grid.saturation + success * (1.0 - grid.saturation),
                )
            )
    return rows
