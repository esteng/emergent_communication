"""End-to-end encode->decode morphology pipeline driver: one run in, a report out.

Reuses the shared paradigm loader (``paradigm_cache``) and the novel-cell battery's
novel-cell generator, then runs the encode->decode chain (``paradigm_encode_decode_batch``)
for one encoder->decoder pair and reduces the per-cell results into comparable rates:

* ``exact_match_rate`` -- coined codes equal to the canonical compositional form;
* ``swapped_order_rate`` -- coined codes using the right morphemes in the wrong order;
* ``bag_match_rate`` -- coined codes using exactly the expected morphemes in any order
  (the bag-of-morphemes signal; exact match is its ordered special case);
* ``decode_rate`` -- the receiver's mean judge decodability of the coined codes (headline);
* ``decoder_valid_rate`` -- fraction the receiver ratified as valid negotiated codes;
* ``decode_negative_control_rate`` / ``decode_positive_control_rate`` -- two-sided judge
  trust: unrelated pairs scored 'none', identical pairs scored 'full'.

Breakdowns are reported per semantic role (which slot the cell varies) and per morpheme
length (so depth cells separate from single-slot substitutions). A run below the
degradation gate reports ``flat_lexicon`` and spends nothing.
"""

import logging
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from morphology.pipeline.grounded_encode_stimuli import (
    build_construction_wugs,
    build_length_gen_wugs,
    mine_argument_inventory,
)
from morphology.pipeline.item_scores import (
    ENCODE_DECODE_PIPELINE,
    ItemScore,
    write_item_scores,
)
from morphology.pipeline.codebook import run_agent_ids
from morphology.pipeline.paradigm_encode_decode_batch import (
    EncodeDecodeResult,
    build_encode_stimuli,
    run_encode_decode_batch,
)
from morphology.pipeline.paradigm_cache import load_run_paradigm, read_run_metadata
from morphology.pipeline.semantic_equivalence_judge import JUDGE_MODEL

logger = logging.getLogger(__name__)

_RESPONSES_FILE = "encode_decode_responses.jsonl"
_REPORT_FILE = "encode_decode_report.json"
_ITEMS_FILE = "encode_decode_items.jsonl"


class RoleEncodeProductivity(BaseModel):
    """Encode->decode rates broken out by the semantic role the cell varies."""

    slot_id: str
    slot_label: str
    n: int
    exact_match_rate: float
    swapped_order_rate: float
    bag_match_rate: float
    decode_rate: float


class DepthEncodeProfile(BaseModel):
    """Encode->decode rates at one morpheme length, to separate depth from substitution."""

    length: int
    n: int
    exact_match_rate: float
    decode_rate: float


class PurposeEncodeBreakdown(BaseModel):
    """Encode->decode rates for one cell kind (e.g. grounded_control vs grounded_novel)."""

    purpose: str
    n: int
    exact_match_rate: float
    swapped_order_rate: float
    bag_match_rate: float
    decode_rate: float


class SlotGeneralizationBreakdown(BaseModel):
    """Novel-cell (non-control) rates for one paradigm slot, split by filler source.

    ``pooled=True`` rows are cells whose filler at this slot came from cross-construction
    role pooling; ``pooled=False`` rows are cells whose filler is native to this slot but
    was still unattested (a within-construction novel value). Comparing the two for the
    same slot -- and comparing ``stem``-role slots to ``suffix``-role slots -- is the
    direct empirical answer to whether pooling into a given slot kind produces valid
    negotiated codes or category errors, rather than assuming it either way.
    """

    slot_id: str
    slot_label: str
    slot_role: str
    pooled: bool
    n: int
    valid_rate: float
    exact_match_rate: float
    decode_rate: float


class ConstructionEncodeBreakdown(BaseModel):
    """Encode->decode rates for one construction, split into control and novel counts."""

    construction_id: str
    n: int
    n_control: int
    n_novel: int
    exact_match_rate: float
    swapped_order_rate: float
    bag_match_rate: float
    decode_rate: float
    novel_exact_match_rate: float
    novel_decode_rate: float


class EncodeDecodeAgentReport(BaseModel):
    """The reduced metrics for one encoder->decoder pair on one run-global paradigm."""

    encoder_agent: str
    decoder_agent: str
    encoder_model: str
    decoder_model: str
    context_budget: int | None
    n_cells: int
    exact_match_rate: float
    swapped_order_rate: float
    bag_match_rate: float
    decode_rate: float
    decoder_valid_rate: float
    decode_negative_control_rate: float
    decode_positive_control_rate: float
    per_role: list[RoleEncodeProductivity]
    per_depth: list[DepthEncodeProfile]
    per_purpose: list[PurposeEncodeBreakdown]
    per_construction: list[ConstructionEncodeBreakdown]
    per_slot_generalization: list[SlotGeneralizationBreakdown]


class EncodeDecodePipelineReport(BaseModel):
    """The whole-run encode->decode result: gate status plus the per-pair report."""

    run_key: str
    status: str  # "analyzed" | "flat_lexicon"
    n_codes: int
    n_multimorpheme: int
    report: EncodeDecodeAgentReport | None


def _mean(values: list[float]) -> float:
    """Arithmetic mean, or 0.0 for an empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _per_role(results: list[EncodeDecodeResult]) -> list[RoleEncodeProductivity]:
    """Group cells by the varied slot into per-role encode->decode rates."""
    by_role: dict[tuple[str, str], list[EncodeDecodeResult]] = defaultdict(list)
    for result in results:
        if result.varied_slot:
            by_role[(result.varied_slot, result.varied_slot_label)].append(result)
    return [
        RoleEncodeProductivity(
            slot_id=slot_id,
            slot_label=slot_label or slot_id,
            n=len(rows),
            exact_match_rate=_mean([1.0 if row.match_expected else 0.0 for row in rows]),
            swapped_order_rate=_mean([1.0 if row.match_swapped_order else 0.0 for row in rows]),
            bag_match_rate=_mean([1.0 if row.match_bag else 0.0 for row in rows]),
            decode_rate=_mean([row.decodability for row in rows]),
        )
        for (slot_id, slot_label), rows in sorted(by_role.items())
    ]


def _per_depth(results: list[EncodeDecodeResult]) -> list[DepthEncodeProfile]:
    """Group cells by morpheme length into an exact-match / decode profile over depth."""
    by_length: dict[int, list[EncodeDecodeResult]] = defaultdict(list)
    for result in results:
        by_length[result.n_morphemes].append(result)
    return [
        DepthEncodeProfile(
            length=length,
            n=len(rows),
            exact_match_rate=_mean([1.0 if row.match_expected else 0.0 for row in rows]),
            decode_rate=_mean([row.decodability for row in rows]),
        )
        for length, rows in sorted(by_length.items())
    ]


def _per_purpose(results: list[EncodeDecodeResult]) -> list[PurposeEncodeBreakdown]:
    """Group cells by their purpose tag (e.g. grounded_control vs grounded_novel)."""
    by_purpose: dict[str, list[EncodeDecodeResult]] = defaultdict(list)
    for result in results:
        by_purpose[result.purpose].append(result)
    return [
        PurposeEncodeBreakdown(
            purpose=purpose,
            n=len(rows),
            exact_match_rate=_mean([1.0 if row.match_expected else 0.0 for row in rows]),
            swapped_order_rate=_mean([1.0 if row.match_swapped_order else 0.0 for row in rows]),
            bag_match_rate=_mean([1.0 if row.match_bag else 0.0 for row in rows]),
            decode_rate=_mean([row.decodability for row in rows]),
        )
        for purpose, rows in sorted(by_purpose.items())
    ]


def _per_slot_generalization(
    results: list[EncodeDecodeResult],
    slot_meta: dict[str, tuple[str, str]],
) -> list[SlotGeneralizationBreakdown]:
    """Explode novel cells across each slot they fill, split into pooled vs. native fillers.

    ``slot_meta`` maps slot_id -> (label, role). A cell contributes to a slot's ``pooled``
    bucket if that slot_id is in the cell's ``pooled_slots``, else to the ``native`` bucket
    for that slot (its own within-construction filler, still unattested). Control cells are
    excluded -- this reports generalization, not reproduction.
    """
    buckets: dict[tuple[str, bool], list[EncodeDecodeResult]] = defaultdict(list)
    for result in results:
        if "control" in result.purpose:
            continue
        pooled_here = set(result.pooled_slots)
        for slot_id in result.expected_slots:
            buckets[(slot_id, slot_id in pooled_here)].append(result)
    breakdowns: list[SlotGeneralizationBreakdown] = []
    for (slot_id, pooled), rows in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        label, role = slot_meta.get(slot_id, (slot_id, ""))
        breakdowns.append(
            SlotGeneralizationBreakdown(
                slot_id=slot_id,
                slot_label=label,
                slot_role=role,
                pooled=pooled,
                n=len(rows),
                valid_rate=_mean([1.0 if row.is_valid else 0.0 for row in rows]),
                exact_match_rate=_mean([1.0 if row.match_expected else 0.0 for row in rows]),
                decode_rate=_mean([row.decodability for row in rows]),
            )
        )
    return breakdowns


def _per_construction(
    results: list[EncodeDecodeResult],
) -> list[ConstructionEncodeBreakdown]:
    """Group cells by construction, reporting overall plus novel-only exact/decode rates."""
    by_construction: dict[str, list[EncodeDecodeResult]] = defaultdict(list)
    for result in results:
        if result.construction_id:
            by_construction[result.construction_id].append(result)
    breakdowns: list[ConstructionEncodeBreakdown] = []
    for construction_id, rows in sorted(by_construction.items()):
        novel = [row for row in rows if "control" not in row.purpose]
        controls = [row for row in rows if "control" in row.purpose]
        breakdowns.append(
            ConstructionEncodeBreakdown(
                construction_id=construction_id,
                n=len(rows),
                n_control=len(controls),
                n_novel=len(novel),
                exact_match_rate=_mean([1.0 if row.match_expected else 0.0 for row in rows]),
                swapped_order_rate=_mean([1.0 if row.match_swapped_order else 0.0 for row in rows]),
                bag_match_rate=_mean([1.0 if row.match_bag else 0.0 for row in rows]),
                decode_rate=_mean([row.decodability for row in rows]),
                novel_exact_match_rate=_mean([1.0 if row.match_expected else 0.0 for row in novel]),
                novel_decode_rate=_mean([row.decodability for row in novel]),
            )
        )
    return breakdowns


def _item_rows(
    results: list[EncodeDecodeResult],
    run_key: str,
    encoder_agent: str,
    decoder_agent: str,
    model: str,
) -> list[ItemScore]:
    """Project every encode->decode cell into a tidy per-item row for downstream stats."""
    return [
        ItemScore(
            run_key=run_key,
            pipeline=ENCODE_DECODE_PIPELINE,
            item_id=f"{ENCODE_DECODE_PIPELINE}:{encoder_agent}:{result.target_meaning}",
            encoder_agent=encoder_agent,
            decoder_agent=decoder_agent,
            model=model,
            construction_id=result.construction_id,
            purpose=result.purpose,
            varied_slot=result.varied_slot,
            varied_slot_label=result.varied_slot_label,
            expected_slots=result.expected_slots,
            pooled_slots=result.pooled_slots,
            n_morphemes=result.n_morphemes,
            target_meaning=result.target_meaning,
            expected_form=result.expected_form,
            produced_form=result.produced_form,
            reading=result.produced_reading,
            is_valid=result.is_valid,
            exact_match=result.match_expected,
            bag_match=result.match_bag,
            swapped_order=result.match_swapped_order,
            decodability=result.decodability,
        )
        for result in results
    ]


def _reduce(
    results: list[EncodeDecodeResult],
    decode_negative_control_rate: float,
    decode_positive_control_rate: float,
    encoder_agent: str,
    decoder_agent: str,
    encoder_model: str,
    decoder_model: str,
    context_budget: int | None,
    slot_meta: dict[str, tuple[str, str]],
) -> EncodeDecodeAgentReport:
    """Reduce per-cell encode->decode results into the comparable per-pair report."""
    return EncodeDecodeAgentReport(
        encoder_agent=encoder_agent,
        decoder_agent=decoder_agent,
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        context_budget=context_budget,
        n_cells=len(results),
        exact_match_rate=_mean([1.0 if row.match_expected else 0.0 for row in results]),
        swapped_order_rate=_mean([1.0 if row.match_swapped_order else 0.0 for row in results]),
        bag_match_rate=_mean([1.0 if row.match_bag else 0.0 for row in results]),
        decode_rate=_mean([row.decodability for row in results]),
        decoder_valid_rate=_mean([1.0 if row.is_valid else 0.0 for row in results]),
        decode_negative_control_rate=decode_negative_control_rate,
        decode_positive_control_rate=decode_positive_control_rate,
        per_role=_per_role(results),
        per_depth=_per_depth(results),
        per_purpose=_per_purpose(results),
        per_construction=_per_construction(results),
        per_slot_generalization=_per_slot_generalization(results, slot_meta),
    )


def run_encode_decode_pipeline(
    run_key: str,
    runs_root: Path,
    cache_dir: Path,
    model: str,
    encoder_agent: str,
    decoder_agent: str,
    cutoff_round: int | None,
    max_cells: int | None,
    grounded: bool,
    wugs_per_construction: int,
    controls_per_construction: int,
    length_gen: bool,
    length_gen_count: int,
    joint_model: str,
    role_pool_model: str | None,
    judge_model: str,
) -> EncodeDecodePipelineReport:
    """Run the encode->decode chain on one run for one encoder->decoder pair.

    Synchronous like ``run_morphology_pipeline``: the batch call manages its own event
    loop, so it is invoked as a single ``asyncio.run`` inside the batch module.
    """
    import asyncio

    scenario, run_dir_name = run_key.split("/", 1)
    run_dir = runs_root / scenario / run_dir_name
    loaded = load_run_paradigm(
        run_key=run_key,
        runs_root=runs_root,
        cache_dir=cache_dir,
        model=model,
        joint_model=joint_model,
        role_pool_model=role_pool_model,
    )
    if loaded.below_gate:
        return EncodeDecodePipelineReport(
            run_key=run_key,
            status="flat_lexicon",
            n_codes=len(loaded.grammar.segmentation),
            n_multimorpheme=loaded.n_multimorpheme,
            report=None,
        )
    assert loaded.analysis is not None and loaded.role_expansion is not None

    available_agents = run_agent_ids(run_dir) or ["agent_0"]
    for role_name, agent in (("encoder", encoder_agent), ("decoder", decoder_agent)):
        if agent not in available_agents:
            raise ValueError(
                f"{role_name}_agent {agent!r} not in run agents {available_agents}"
            )

    if grounded:
        inventory = mine_argument_inventory(run_dir=run_dir, grammar=loaded.grammar)
        stimuli = build_construction_wugs(
            analysis=loaded.analysis,
            inventory=inventory,
            wugs_per_construction=wugs_per_construction,
            controls_per_construction=controls_per_construction,
            seed=42,
            role_pooled_fillers=loaded.role_expansion.per_slot,
        )
        if length_gen:
            stimuli += build_length_gen_wugs(
                single_cells=stimuli, inventory=inventory, n_wugs=length_gen_count, seed=42
            )
        if max_cells is not None:
            stimuli = stimuli[:max_cells]
    else:
        stimuli = build_encode_stimuli(
            grammar=loaded.grammar,
            analysis=loaded.analysis,
            role_pooled_fillers=loaded.role_expansion.per_slot,
            role_depth_fillers=loaded.role_expansion.stackable,
            max_cells=max_cells,
        )
    outcome = asyncio.run(
        run_encode_decode_batch(
            stimuli=stimuli,
            run_dir=run_dir,
            scenario_name=scenario,
            encoder_agent_id=encoder_agent,
            decoder_agent_id=decoder_agent,
            cutoff_round=cutoff_round,
            judge_model=judge_model,
        )
    )
    encoder_meta = read_run_metadata(run_dir, encoder_agent)
    decoder_meta = read_run_metadata(run_dir, decoder_agent)
    report = _reduce(
        results=outcome.results,
        decode_negative_control_rate=outcome.decode_negative_control_rate,
        decode_positive_control_rate=outcome.decode_positive_control_rate,
        encoder_agent=encoder_agent,
        decoder_agent=decoder_agent,
        encoder_model=encoder_meta.model,
        decoder_model=decoder_meta.model,
        context_budget=decoder_meta.context_budget,
        slot_meta={slot.slot_id: (slot.label, slot.role) for slot in loaded.analysis.slots},
    )
    responses_path = run_dir / _RESPONSES_FILE
    with responses_path.open("w", encoding="utf-8") as handle:
        for result in outcome.results:
            handle.write(result.model_dump_json() + "\n")
    logger.info("wrote %d encode->decode rows to %s", len(outcome.results), responses_path)
    write_item_scores(
        path=run_dir / _ITEMS_FILE,
        rows=_item_rows(
            results=outcome.results,
            run_key=run_key,
            encoder_agent=encoder_agent,
            decoder_agent=decoder_agent,
            model=decoder_meta.model,
        ),
    )
    return EncodeDecodePipelineReport(
        run_key=run_key,
        status="analyzed",
        n_codes=len(loaded.grammar.segmentation),
        n_multimorpheme=loaded.n_multimorpheme,
        report=report,
    )


def main() -> None:
    """CLI: run the encode->decode pipeline on one run and write its report."""
    import argparse

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--joint-model", default="gpt-5.4",
                        help="Single-step codebook-to-paradigm model.")
    parser.add_argument(
        "--role-pool-model",
        default="gpt-5.4",
        help="Within-run same-role filler pooling model. Pass 'none' to disable.",
    )
    parser.add_argument("--encoder-agent", default="stabilization_engineer")
    parser.add_argument("--decoder-agent", default="field_observer")
    parser.add_argument("--cutoff-round", type=int, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument(
        "--grounded",
        action="store_true",
        help="Instantiate construction templates with concrete faces/numbers (e.g. F[N]t6g), "
        "sampling per-construction wugs, instead of encoding template-level meanings.",
    )
    parser.add_argument(
        "--wugs-per-construction",
        type=int,
        default=30,
        help="Grounded mode: novel concrete cells sampled per construction.",
    )
    parser.add_argument(
        "--controls-per-construction",
        type=int,
        default=10,
        help=(
            "Grounded mode: attested concrete cells kept per construction as controls. "
            "These set the ceiling that novel decodability is read against, so they need "
            "enough n to be a usable baseline (at 5 the ceiling's own SE is ~0.2). Cells "
            "are effectively free -- the frozen thread dominates cost -- so this is "
            "cheap precision."
        ),
    )
    parser.add_argument(
        "--length-gen",
        action="store_true",
        help="Grounded mode: also coin novel two-routine compound (A+B) length-gen wugs.",
    )
    parser.add_argument("--length-gen-count", type=int, default=30)
    parser.add_argument(
        "--judge-model",
        default=JUDGE_MODEL,
        help="Semantic-equivalence judge model (decodability + trust controls). Default is "
        "the Claude judge routed via Portkey; pass a gpt-* model to route via Azure instead.",
    )
    args = parser.parse_args()

    report = run_encode_decode_pipeline(
        run_key=args.run_key,
        runs_root=Path(args.runs_root),
        cache_dir=Path(args.cache_dir),
        model=args.model,
        encoder_agent=args.encoder_agent,
        decoder_agent=args.decoder_agent,
        cutoff_round=args.cutoff_round,
        max_cells=args.max_cells,
        grounded=args.grounded,
        wugs_per_construction=args.wugs_per_construction,
        controls_per_construction=args.controls_per_construction,
        length_gen=args.length_gen,
        length_gen_count=args.length_gen_count,
        joint_model=args.joint_model,
        role_pool_model=None if args.role_pool_model == "none" else args.role_pool_model,
        judge_model=args.judge_model,
    )
    print(f"\n=== {report.run_key} :: {report.status} ===")
    print(f"codes={report.n_codes} multi-morpheme={report.n_multimorpheme}")
    if report.report is not None:
        row = report.report
        print(
            f"  {row.encoder_agent} -> {row.decoder_agent} "
            f"({row.encoder_model} -> {row.decoder_model}) n={row.n_cells}\n"
            f"    exact={row.exact_match_rate:.2f} swapped={row.swapped_order_rate:.2f} "
            f"bag={row.bag_match_rate:.2f} decode={row.decode_rate:.2f} "
            f"valid={row.decoder_valid_rate:.2f}\n"
            f"    judge trust: neg->none={row.decode_negative_control_rate:.2f} "
            f"pos->full={row.decode_positive_control_rate:.2f} "
            "(trust rates only if BOTH are high)"
        )
        if row.per_role:
            per_role = "  ".join(
                f"{r.slot_label}[exact={r.exact_match_rate:.2f} dec={r.decode_rate:.2f} n{r.n}]"
                for r in row.per_role
            )
            print(f"    by role: {per_role}")
        if row.per_depth:
            per_depth = " ".join(
                f"L{d.length}[exact={d.exact_match_rate:.2f} dec={d.decode_rate:.2f} n{d.n}]"
                for d in row.per_depth
            )
            print(f"    by depth: {per_depth}")
        if len(row.per_purpose) > 1:
            for p in row.per_purpose:
                print(
                    f"    {p.purpose}: exact={p.exact_match_rate:.2f} "
                    f"swapped={p.swapped_order_rate:.2f} bag={p.bag_match_rate:.2f} "
                    f"decode={p.decode_rate:.2f} n={p.n}"
                )
        if row.per_construction:
            print("    by construction (novel exact / novel decode):")
            for c in row.per_construction:
                print(
                    f"      {c.construction_id:22} novel_exact={c.novel_exact_match_rate:.2f} "
                    f"novel_decode={c.novel_decode_rate:.2f} swapped={c.swapped_order_rate:.2f} "
                    f"(novel n={c.n_novel}, control n={c.n_control})"
                )
        if row.per_slot_generalization:
            print("    by slot, novel cells only (pooled = cross-construction substitution):")
            for s in row.per_slot_generalization:
                source = "pooled" if s.pooled else "native"
                print(
                    f"      {s.slot_id:6} {s.slot_label:22} role={s.slot_role:7} {source:6} "
                    f"valid={s.valid_rate:.2f} exact={s.exact_match_rate:.2f} "
                    f"decode={s.decode_rate:.2f} n={s.n}"
                )
    out = Path(args.runs_root) / args.run_key / _REPORT_FILE
    out.write_text(report.model_dump_json(indent=2))
    logger.info("wrote report to %s", out)


if __name__ == "__main__":
    main()
