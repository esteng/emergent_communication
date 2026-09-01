"""Batched encode->decode paradigm probe: the sender coins codes, the receiver reads them.

The decode-only battery (``paradigm_decode_batch``) invents a novel code deterministically
and asks whether the receiver can read it. This module instead asks the ENCODER (sender)
to coin a code for each held-out meaning, then measures the coined code four independent
ways:

* ``match_expected`` -- the coined code equals the canonical compositional form we predict
  from the paradigm (right morphemes, right order);
* ``match_bag`` -- the coined code is exactly the expected morphemes in SOME order (the
  bag-of-morphemes signal; ``match_expected`` is its ordered special case);
* ``match_swapped_order`` -- the bag minus the canonical order: right pieces, wrong order;
* ``decodability`` -- the DECODER (receiver), reading the coined code cold, recovers the
  intended meaning (nano semantic judge, 0/0.5/1.0). This is the headline.

Both frozen agent threads (~47k tokens each, identical across stimuli) are exported once:
the encoder is asked for every meaning in bounded structured calls, then the receiver
decodes every distinct coined form via the same bounded decode path the decode battery
uses. The nano judge scores decodability plus two-sided trust controls -- unrelated
(meaning, reading) pairs it must score 'none', and identical pairs it must score 'full'
-- which together bracket a yes-bias and a stopped-discriminating judge.
"""

import asyncio
import logging
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from morphology.pipeline.analysis_mode import export_agent_thread, call_structured
from morphology.pipeline.morphological_grammar import MorphologicalGrammar
from morphology.pipeline.paradigm_network import SlotAnalysis
from morphology.pipeline.novel_cell_stimuli import (
    decode_forms,
    build_stimuli,
)
from morphology.pipeline.semantic_equivalence_judge import (
    JUDGE_MODEL,
    DecodeItem,
    build_negative_control_items,
    build_positive_control_items,
    run_semantic_judge,
)

import morphology.common as C

logger = logging.getLogger(__name__)

_RESPONSES_FILE = "encode_decode_responses.jsonl"
# The frozen encoder thread dominates cost and is identical across meanings, so ask for
# many codes per bounded structured reply. Matches the decode battery's per-call budget.
_ENCODE_MEANINGS_PER_CALL = 200
_BATCH_MAX_OUTPUT_TOKENS = 16384
# Cap negative-control judge items so their cost does not scale with the whole battery.
_MAX_DECODE_NEGATIVE_CONTROLS = 20
# The positive control is a cheap regression floor, so a handful is enough.
_MAX_DECODE_POSITIVE_CONTROLS = 10
# Fixed seed for negative-control partner choice, so judge trust is reproducible.
_NEGATIVE_CONTROL_SEED = 42
# Characters an agent may insert between morphemes; stripped before comparing surfaces so
# "Lf-12" and "Lf 12" both match the canonical "Lf12". Case is preserved (codes are
# case-sensitive symbols).
_SEPARATOR_STRIP = str.maketrans("", "", " \t\n\r-._·|/")

_ENCODE_HEADER = C.load_prompt("encode_batch")


class EncodeStimulus(BaseModel):
    """One held-out meaning to encode, with the code the paradigm predicts for it."""

    target_meaning: str
    expected_form: str  # canonical concatenation of the expected morphemes
    expected_morphemes: list[str]
    expected_slots: list[str]
    purpose: str
    varied_slot: str = ""
    varied_slot_label: str = ""
    n_morphemes: int
    construction_id: str = ""
    pooled_slots: list[str] = []  # slot_ids (parallel-indexed into expected_slots) whose
    # filler came from cross-construction role pooling rather than the slot's own fillers


class EncodeDecodeResult(BaseModel):
    """One encode->decode cell: the coined code, its reading, and the four match signals."""

    target_meaning: str
    expected_form: str
    expected_morphemes: list[str]
    expected_slots: list[str] = []
    produced_form: str
    produced_reading: str
    purpose: str
    varied_slot: str = ""
    varied_slot_label: str = ""
    n_morphemes: int
    construction_id: str = ""
    pooled_slots: list[str] = []
    is_valid: bool  # the receiver ratified the coined code as a valid negotiated code
    # PRODUCTION -- match_expected is the ordered form; match_bag is the same morphemes in
    # any order (match_expected implies it); match_swapped_order is the bag minus the
    # canonical order, i.e. right pieces / wrong order.
    match_expected: bool
    match_bag: bool
    match_swapped_order: bool
    decodability: float  # nano-judge 0/0.5/1.0 of the receiver's reading vs the target


class EncodeDecodeOutcome(BaseModel):
    """The per-cell results plus the two-sided judge-trust control rates for the batch.

    ``decode_negative_control_rate`` is the share of unrelated (meaning, reading) pairs
    correctly scored 'none' -- it catches a yes/'decodable' bias. ``decode_positive_
    control_rate`` is the share of identical pairs correctly scored 'full' -- it catches
    the opposite failure, a judge that has stopped discriminating and answers 'none'
    throughout. Both must be high for the batch's decodability scores to mean anything.
    """

    results: list[EncodeDecodeResult]
    decode_negative_control_rate: float
    n_decode_negative_controls: int
    decode_positive_control_rate: float
    n_decode_positive_controls: int


class _EncodeItem(BaseModel):
    """The encoder's coined code for one echoed target meaning."""

    target_meaning: str
    produced_form: str


class _BatchEncode(BaseModel):
    """The encoder's per-meaning coined codes, in one bounded structured reply."""

    productions: list[_EncodeItem]


def _normalize(form: str) -> str:
    """Strip inter-morpheme separators/whitespace, preserving case, for surface matching."""
    return form.translate(_SEPARATOR_STRIP)


def _match_bag(produced_form: str, morphemes: list[str]) -> bool:
    """True when the coined code is exactly the expected morphemes in SOME order.

    The bag-of-morphemes signal: every expected morpheme is present, order free. Rather
    than enumerating the ``n!`` orderings and comparing surfaces (which forces an
    approximation at the depths where compounds live), this consumes the expected
    multiset from the front of the surface, backtracking when a prefix choice dead-ends.
    Exact match is the ordered special case, so ``match_expected`` implies this.
    """
    if not morphemes:
        return False
    remaining = Counter(_normalize(morpheme) for morpheme in morphemes)

    def consume(text: str) -> bool:
        if not text:
            return not +remaining  # every expected morpheme was used exactly as often
        # Longest first: a short morpheme that prefixes a longer one is still reachable
        # by backtracking, so no valid parse is missed.
        for morpheme in sorted(remaining, key=len, reverse=True):
            if remaining[morpheme] and text.startswith(morpheme):
                remaining[morpheme] -= 1
                if consume(text[len(morpheme):]):
                    return True
                remaining[morpheme] += 1
        return False

    return consume(_normalize(produced_form))


def _match_expected(produced_form: str, expected_form: str) -> bool:
    """True when the coined code is the canonical compositional form (order included)."""
    return _normalize(produced_form) == _normalize(expected_form)


def _match_swapped_order(produced_form: str, expected_form: str, morphemes: list[str]) -> bool:
    """True when the coined code uses exactly the expected morphemes in a NON-canonical order."""
    return _match_bag(produced_form, morphemes) and not _match_expected(
        produced_form, expected_form
    )


def build_encode_stimuli(
    grammar: MorphologicalGrammar,
    analysis: SlotAnalysis,
    role_pooled_fillers: dict[str, list[tuple[str, str]]] | None,
    role_depth_fillers: list[tuple[str, str]] | None,
    max_cells: int | None,
) -> list[EncodeStimulus]:
    """Turn the decode battery's novel cells into encode targets (each meaning + its code).

    Reuses ``build_stimuli`` so the encode and decode probes cover the same paradigm cells,
    then drops attested controls and any stimulus without a recorded morpheme decomposition
    (only novel/depth cells carry one). ``max_cells`` deterministically caps the total.
    """
    raw = build_stimuli(
        grammar,
        analysis,
        role_pooled_fillers=role_pooled_fillers,
        role_depth_fillers=role_depth_fillers,
    )
    cells: list[EncodeStimulus] = []
    seen_meanings: set[str] = set()
    for stimulus in raw:
        if stimulus.purpose == "control_attested" or not stimulus.expected_morphemes:
            continue
        if not stimulus.expected_meaning or stimulus.expected_meaning in seen_meanings:
            continue
        seen_meanings.add(stimulus.expected_meaning)
        cells.append(
            EncodeStimulus(
                target_meaning=stimulus.expected_meaning,
                expected_form="".join(stimulus.expected_morphemes),
                expected_morphemes=stimulus.expected_morphemes,
                expected_slots=stimulus.expected_slots,
                purpose=stimulus.purpose,
                varied_slot=stimulus.varied_slot,
                varied_slot_label=stimulus.varied_slot_label,
                n_morphemes=stimulus.n_morphemes,
            )
        )
    if max_cells is not None:
        cells = cells[:max_cells]
    return cells


async def _encode_meanings(
    base_messages: list[dict[str, str]], meanings: list[str], model: str
) -> dict[str, str]:
    """Ask the frozen encoder to coin a code for every meaning in bounded chunked calls."""
    chunks = [
        meanings[start:start + _ENCODE_MEANINGS_PER_CALL]
        for start in range(0, len(meanings), _ENCODE_MEANINGS_PER_CALL)
    ]
    calls = await asyncio.gather(
        *(
            call_structured(
                messages=base_messages + [
                    {
                        "role": "user",
                        "content": _ENCODE_HEADER + "\n".join(
                            f"{index + 1}. {meaning}" for index, meaning in enumerate(chunk)
                        ),
                    }
                ],
                output_schema=_BatchEncode,
                model=model,
                caller="paradigm_encode_decode_batch",
                max_output_tokens=_BATCH_MAX_OUTPUT_TOKENS,
            )
            for chunk in chunks
        )
    )
    produced_by_meaning: dict[str, str] = {}
    for chunk, call in zip(chunks, calls):
        reply: _BatchEncode = call.result
        by_meaning = {item.target_meaning: item for item in reply.productions}
        for position, meaning in enumerate(chunk):
            item = by_meaning.get(meaning)
            if item is None and position < len(reply.productions):
                item = reply.productions[position]
            produced_by_meaning[meaning] = item.produced_form if item else ""
    return produced_by_meaning


def _negative_control_items(
    stimuli: list[EncodeStimulus], reading_of: dict[str, str]
) -> list[DecodeItem]:
    """Mismatched (unrelated meaning, this reading) items a trustworthy judge scores 'none'.

    Delegates the pairing to ``build_negative_control_items``, which requires the wrong
    meaning to share no content word with the reading's true meaning. A low pass rate
    reveals a yes/'decodable' bias for the batch.
    """
    return build_negative_control_items(
        meaning_readings=[
            (stimulus.target_meaning, reading_of.get(stimulus.target_meaning, ""))
            for stimulus in stimuli
        ],
        max_items=_MAX_DECODE_NEGATIVE_CONTROLS,
        seed=_NEGATIVE_CONTROL_SEED,
    )


async def run_encode_decode_batch(
    stimuli: list[EncodeStimulus],
    run_dir: Path,
    scenario_name: str,
    encoder_agent_id: str,
    decoder_agent_id: str,
    cutoff_round: int | None,
    judge_model: str = JUDGE_MODEL,
) -> EncodeDecodeOutcome:
    """Coin a code for every meaning with the encoder, decode each with the receiver, score.

    One bounded send to the frozen encoder produces every code; one bounded send to the
    frozen decoder reads back every distinct coined form; the nano judge then scores the
    receiver's reading against the intended meaning (plus mismatched negative controls).
    """
    encoder_export = await export_agent_thread(
        run_dir=run_dir,
        scenario_name=scenario_name,
        agent_id=encoder_agent_id,
        cutoff_round=cutoff_round,
        output_format="openai_chat",
        include_thinking=False,
        flatten_tools=True,
    )
    encoder_messages = [
        {"role": message.role, "content": message.content or ""}
        for message in encoder_export.request.messages
    ]
    meanings = list(dict.fromkeys(stimulus.target_meaning for stimulus in stimuli))
    produced_by_meaning = await _encode_meanings(
        base_messages=encoder_messages, meanings=meanings, model=encoder_export.meta.model
    )

    decoder_export = await export_agent_thread(
        run_dir=run_dir,
        scenario_name=scenario_name,
        agent_id=decoder_agent_id,
        cutoff_round=cutoff_round,
        output_format="openai_chat",
        include_thinking=False,
        flatten_tools=True,
    )
    decoder_messages = [
        {"role": message.role, "content": message.content or ""}
        for message in decoder_export.request.messages
    ]
    coined_forms = [
        form
        for form in dict.fromkeys(
            produced_by_meaning[stimulus.target_meaning] for stimulus in stimuli
        )
        if form
    ]
    reading_by_form, valid_by_form = await decode_forms(
        base_messages=decoder_messages, forms=coined_forms, model=decoder_export.meta.model
    )

    reading_by_meaning = {
        stimulus.target_meaning: reading_by_form.get(
            produced_by_meaning[stimulus.target_meaning], ""
        )
        for stimulus in stimuli
    }
    decode_items = [
        DecodeItem(
            intended=stimulus.target_meaning,
            reading=reading_by_meaning[stimulus.target_meaning],
        )
        for stimulus in stimuli
    ]
    negative_items = _negative_control_items(stimuli=stimuli, reading_of=reading_by_meaning)
    positive_items = build_positive_control_items(
        meanings=[stimulus.target_meaning for stimulus in stimuli],
        max_items=_MAX_DECODE_POSITIVE_CONTROLS,
    )
    judge = await run_semantic_judge(
        decode_items=decode_items + negative_items + positive_items,
        equiv_items=[],
        judge_model=judge_model,
    )
    n_stimuli = len(stimuli)
    stimulus_scores = judge.decode_scores[:n_stimuli]
    negative_scores = judge.decode_scores[n_stimuli:n_stimuli + len(negative_items)]
    positive_scores = judge.decode_scores[n_stimuli + len(negative_items):]

    results: list[EncodeDecodeResult] = []
    for stimulus, score in zip(stimuli, stimulus_scores):
        produced_form = produced_by_meaning[stimulus.target_meaning]
        results.append(
            EncodeDecodeResult(
                target_meaning=stimulus.target_meaning,
                expected_form=stimulus.expected_form,
                expected_morphemes=stimulus.expected_morphemes,
                expected_slots=stimulus.expected_slots,
                produced_form=produced_form,
                produced_reading=reading_by_meaning[stimulus.target_meaning],
                purpose=stimulus.purpose,
                varied_slot=stimulus.varied_slot,
                varied_slot_label=stimulus.varied_slot_label,
                n_morphemes=stimulus.n_morphemes,
                construction_id=stimulus.construction_id,
                pooled_slots=stimulus.pooled_slots,
                is_valid=valid_by_form.get(produced_form, False),
                match_expected=_match_expected(produced_form, stimulus.expected_form),
                match_bag=_match_bag(produced_form, stimulus.expected_morphemes),
                match_swapped_order=_match_swapped_order(
                    produced_form, stimulus.expected_form, stimulus.expected_morphemes
                ),
                decodability=score,
            )
        )

    decode_negative_control_rate = 0.0
    if negative_scores:
        decode_negative_control_rate = sum(
            1 for score in negative_scores if score == 0.0
        ) / len(negative_scores)
    decode_positive_control_rate = 0.0
    if positive_scores:
        decode_positive_control_rate = sum(
            1 for score in positive_scores if score == 1.0
        ) / len(positive_scores)
    return EncodeDecodeOutcome(
        results=results,
        decode_negative_control_rate=decode_negative_control_rate,
        n_decode_negative_controls=len(negative_items),
        decode_positive_control_rate=decode_positive_control_rate,
        n_decode_positive_controls=len(positive_items),
    )


def main() -> None:
    """CLI: build the encode->decode battery for one run; ``--dry-run`` prints it offline."""
    import argparse

    from dotenv import load_dotenv

    from morphology.pipeline.paradigm_cache import load_run_paradigm

    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--joint-model", default="gpt-5.4")
    parser.add_argument(
        "--role-pool-model",
        default="none",
        help="Same-role filler pooling model. Default 'none' keeps --dry-run fully offline.",
    )
    parser.add_argument("--encoder-agent", default="stabilization_engineer")
    parser.add_argument("--decoder-agent", default="field_observer")
    parser.add_argument("--cutoff-round", type=int, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print stimuli and exit (no API call)."
    )
    args = parser.parse_args()

    loaded = load_run_paradigm(
        run_key=args.run_key,
        runs_root=Path(args.runs_root),
        cache_dir=Path(args.cache_dir),
        model=args.model,
        joint_model=args.joint_model,
        role_pool_model=None if args.role_pool_model == "none" else args.role_pool_model,
    )
    if loaded.below_gate:
        print(
            f"{args.run_key}: flat_lexicon "
            f"(n_multimorpheme={loaded.n_multimorpheme}); nothing to encode"
        )
        return
    assert loaded.analysis is not None and loaded.role_expansion is not None
    stimuli = build_encode_stimuli(
        grammar=loaded.grammar,
        analysis=loaded.analysis,
        role_pooled_fillers=loaded.role_expansion.per_slot,
        role_depth_fillers=loaded.role_expansion.stackable,
        max_cells=args.max_cells,
    )
    logger.info("built %d encode cells", len(stimuli))
    for stimulus in stimuli:
        print(
            f"  [{stimulus.purpose}] say {stimulus.target_meaning[:44]!r} "
            f"-> expect {stimulus.expected_form} "
            f"(bag: {'+'.join(stimulus.expected_morphemes) or '-'})"
        )
    if args.dry_run:
        return

    scenario, run_dir_name = args.run_key.split("/", 1)
    run_dir = Path(args.runs_root) / scenario / run_dir_name
    outcome = asyncio.run(
        run_encode_decode_batch(
            stimuli=stimuli,
            run_dir=run_dir,
            scenario_name=scenario,
            encoder_agent_id=args.encoder_agent,
            decoder_agent_id=args.decoder_agent,
            cutoff_round=args.cutoff_round,
        )
    )
    out_path = run_dir / _RESPONSES_FILE
    with out_path.open("w", encoding="utf-8") as handle:
        for result in outcome.results:
            handle.write(result.model_dump_json() + "\n")
    print(f"\n=== ENCODE->DECODE ({args.encoder_agent} -> {args.decoder_agent}) ===")
    for result in outcome.results:
        marks = "".join(
            flag
            for flag, present in (
                ("E", result.match_expected),
                ("B", result.match_bag),
                ("S", result.match_swapped_order),
            )
            if present
        ) or "-"
        print(
            f"  say {result.target_meaning[:36]!r} exp={result.expected_form} "
            f"got={result.produced_form or '(blank)'} [{marks}] "
            f"d={result.decodability:.2f} :: {result.produced_reading[:44]}"
        )
    logger.info("wrote %d encode->decode rows to %s", len(outcome.results), out_path)


if __name__ == "__main__":
    main()
