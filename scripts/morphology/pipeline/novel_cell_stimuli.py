"""Stimulus construction and batched decoding shared by the encode->decode battery.

``build_stimuli`` edits each attested code against its slot template to generate the
novel-cell battery -- within-slot substitutions, same-role cross-fills for closed slots,
licensed optional-slot insertions, and agglutination-depth sweeps. ``decode_forms`` asks
the frozen decoder to read a batch of those forms back. Both are used by the
encode->decode battery in ``paradigm_encode_decode_batch``.
"""

import asyncio
from collections.abc import Sequence
from collections import Counter, defaultdict
from itertools import product

from pydantic import BaseModel

from morphology.pipeline.analysis_mode import call_structured
from morphology.pipeline.morphological_grammar import MorphologicalGrammar
from morphology.pipeline.paradigm_network import SlotAnalysis

import morphology.common as C

# Distinct words sampled at each agglutinative length in the depth sweep, so the
# per-length decode/accept estimate averages over lexical choice. Raise for a
# smoother curve at the cost of a larger single batch.
_DEPTH_SAMPLES_PER_LENGTH = 6
# A large run-global grammar yields substantially more unique decode forms than the
# old per-construction batches.  The shared 4k structured-call default truncated the
# 64-entry live run mid-JSON, so this caller needs its own larger response allowance.
_BATCH_MAX_OUTPUT_TOKENS = 16384
# The combined order paradigms can still contain hundreds of distinct surfaces. Keep
# each frozen-agent structured response bounded, while decoding each unique form once.
_DECODE_FORMS_PER_CALL = 200
_PROMPT_HEADER = C.load_prompt("decode_batch")
class BatchStimulus(BaseModel):
    """One decode stimulus: an edited code, what edit produced it, and its expected reading."""

    purpose: str
    form: str
    expected_meaning: str
    detail: str
    n_morphemes: int  # morpheme count -- lets the depth sweep profile by agglutinative length
    construction_id: str = ""
    # The slot (semantic role) whose filler this stimulus varies, so novel productivity
    # can be reported per role. Empty for controls and depth-sweep stimuli.
    varied_slot: str = ""
    varied_slot_label: str = ""
    # For role_depth stacks: the attested maximal code the stack extends, used to gate a
    # stacked cell on whether its base code itself decodes. Empty otherwise.
    base_form: str = ""
    # The ordered morphemes composing this form and their slot ids (parallel lists). The
    # decode path ignores these; the encode->decode probe uses them to compare an
    # agent-produced code against the canonical form and its swapped-order permutations.
    expected_morphemes: list[str] = []
    expected_slots: list[str] = []
class _Judgment(BaseModel):
    """The agent's structured judgment of one candidate code."""

    form: str
    is_valid: bool
    interpreted_meaning: str
class _BatchDecode(BaseModel):
    """The agent's per-candidate decode + validity judgments, in one call.

    The agent only DECODES forms; it never judges pairs. Order-invariance is decided
    downstream by the nano judge comparing the agent's readings of a pair's members.
    """

    judgments: list[_Judgment]
def _code_slots(
    grammar: MorphologicalGrammar,
    analysis: SlotAnalysis,
    symbol: str,
    parts: list[str],
    form_slot: dict[str, str],
) -> list[str | None]:
    """Return network assignments, then legacy semantic positions/form slots."""
    network_slots = analysis.slots_by_symbol.get(symbol)
    if network_slots is not None and len(network_slots) == len(parts):
        return list(network_slots)
    positions = grammar.positions_by_symbol.get(symbol)
    if positions is not None and len(positions) == len(parts):
        return [f"pos{position}" for position in positions]
    return [form_slot.get(morpheme) for morpheme in parts]
def build_stimuli(
    grammar: MorphologicalGrammar,
    analysis: SlotAnalysis,
    role_pooled_fillers: dict[str, list[tuple[str, str]]] | None = None,
    role_depth_fillers: list[tuple[str, str]] | None = None,
) -> list[BatchStimulus]:
    """Generate the decode battery by editing attested codes against the slot template.

    ``role_pooled_fillers`` maps slot_id to extra same-role (form, gloss) fillers pooled
    from elsewhere in the run (see ``role_pooled_stimuli``). They cross-fill each slot --
    including closed/saturated ones -- with ``role_substitution`` stimuli, turning an
    otherwise-untestable slot into a real wug test. ``role_depth_fillers`` are (form,
    gloss) morphemes that legally lengthen a maximal code by one role dimension; they
    produce ``role_depth`` stimuli that probe agglutination past the attested ceiling
    even when the grammar caps at its slot count.
    """
    role_pooled_fillers = role_pooled_fillers or {}
    role_depth_fillers = role_depth_fillers or []
    form_slot = {f.form: s.slot_id for s in analysis.slots for f in s.fillers}
    gloss = {(s.slot_id, f.form): f.gloss for s in analysis.slots for f in s.fillers}
    for slot_id, extras in role_pooled_fillers.items():
        for form, form_gloss in extras:
            gloss.setdefault((slot_id, form), form_gloss)
    slot_fillers = {s.slot_id: [f.form for f in s.fillers] for s in analysis.slots}
    slot_label = {s.slot_id: (s.label.strip() or s.slot_id) for s in analysis.slots}
    codes = {sym: parts for sym, parts in grammar.segmentation.items() if len(parts) >= 2}
    attested = set(codes)
    max_depth = max((len(parts) for parts in codes.values()), default=0)
    licensed_templates = (
        list(analysis.construction_templates.values())
        if analysis.construction_templates
        else analysis.orderings
    )

    stimuli: list[BatchStimulus] = []
    seen: set[str] = set()

    def gloss_of(parts: list[str], slots: Sequence[str | None]) -> str:
        return " ".join(
            gloss.get((slot_id, morpheme), morpheme) if slot_id is not None else morpheme
            for morpheme, slot_id in zip(parts, slots)
        )

    def add(
        purpose: str,
        parts: list[str],
        slots: Sequence[str | None],
        detail: str,
        varied_slot: str = "",
    ) -> bool:
        form = "".join(parts)
        if form in attested or form in seen:
            return False
        seen.add(form)
        stimuli.append(
            BatchStimulus(
                purpose=purpose, form=form, expected_meaning=gloss_of(parts, slots),
                detail=detail, n_morphemes=len(parts),
                varied_slot=varied_slot,
                varied_slot_label=slot_label.get(varied_slot, "") if varied_slot else "",
                expected_morphemes=list(parts),
                expected_slots=[slot_id or "" for slot_id in slots],
            )
        )
        return True

    for symbol, parts in codes.items():
        slots = _code_slots(grammar, analysis, symbol, parts, form_slot)
        for index, morpheme in enumerate(parts):
            slot_id = slots[index]
            if slot_id is None:
                continue
            for alternate in slot_fillers[slot_id]:
                if alternate != morpheme:
                    add(
                        "substitution",
                        parts[:index] + [alternate] + parts[index + 1:],
                        slots,
                        f"{morpheme}->{alternate} in {slot_id}",
                        varied_slot=slot_id,
                    )
            # Cross-fill the slot with same-role morphemes pooled from elsewhere in the
            # run. This is the wug test for closed/saturated slots: the within-slot loop
            # above produced nothing new, but a different filler of the same role does.
            for alternate, _ in role_pooled_fillers.get(slot_id, []):
                if alternate != morpheme:
                    add(
                        "role_substitution",
                        parts[:index] + [alternate] + parts[index + 1:],
                        slots,
                        f"{morpheme}->{alternate} in {slot_id} (role-pooled)",
                        varied_slot=slot_id,
                    )
        # Only add a slot when a licensed template extends this exact class path by
        # one position.  A paradigm network must not insert classes from an unrelated
        # construction merely because they occur somewhere in the run.
        concrete_slots = [slot for slot in slots if slot is not None]
        for template in licensed_templates:
            if len(template) != len(concrete_slots) + 1:
                continue
            for insert_at, slot_id in enumerate(template):
                if template[:insert_at] + template[insert_at + 1:] != concrete_slots:
                    continue
                fillers = slot_fillers.get(slot_id, [])
                if not fillers:
                    continue
                extended = parts[:insert_at] + [fillers[0]] + parts[insert_at:]
                extended_slots = concrete_slots[:insert_at] + [slot_id] + concrete_slots[insert_at:]
                add(
                    "optional_slot", extended, extended_slots,
                    f"insert {fillers[0]}({slot_id}) via licensed template",
                    varied_slot=slot_id,
                )

    # Agglutination depth SWEEP: for each morpheme length from the longest attested
    # code up to the number of slots (the theoretical maximum -- one filler per slot),
    # build a few words of that length by filling the most-obligatory slots (in
    # template order), plus filler variants. Tagging each with its length lets the
    # report profile decoding/acceptance as a function of agglutinative depth, rather
    # than testing only the single maximal word.
    # Generate depth variants only inside licensed templates.  The old universal-row
    # logic concatenated unrelated construction branches and produced false words.
    slot_by_id = {slot.slot_id: slot for slot in analysis.slots if slot.fillers}
    added_by_length: Counter[int] = Counter()
    for template in sorted(licensed_templates, key=lambda value: (len(value), value)):
        length = len(template)
        if length < max(2, max_depth) or any(slot not in slot_by_id for slot in template):
            continue
        filler_options = [
            [filler.form for filler in slot_by_id[slot_id].fillers[:3]]
            for slot_id in template
        ]
        for combo in product(*filler_options):
            if added_by_length[length] >= _DEPTH_SAMPLES_PER_LENGTH:
                break
            if add(
                "agglutination_depth", list(combo), template,
                f"length {length} in licensed template (attested max {max_depth})",
            ):
                added_by_length[length] += 1

    # Role-stacked depth: append a same-run role the code lacks onto a maximal attested
    # code, producing a form deeper than the attested ceiling from real run roles (not a
    # repeated slot). Fires even when the grammar caps at its slot count. base_form pins
    # the attested code so the cell can be gated on whether its base itself decodes.
    if role_depth_fillers:
        maximal = [
            (symbol, parts)
            for symbol, parts in codes.items()
            if len(parts) == max_depth
        ]
        for symbol, parts in maximal:
            slots = _code_slots(grammar, analysis, symbol, parts, form_slot)
            base_meaning = grammar.meaning_by_symbol.get(symbol) or gloss_of(parts, slots)
            for stack_form, stack_gloss in role_depth_fillers:
                if stack_form in parts:
                    continue
                form = "".join(parts) + stack_form
                if form in attested or form in seen:
                    continue
                seen.add(form)
                stimuli.append(
                    BatchStimulus(
                        purpose="role_depth",
                        form=form,
                        expected_meaning=f"{base_meaning} {stack_gloss}".strip(),
                        detail=f"stack {stack_form} onto {symbol} (role-pooled depth)",
                        n_morphemes=len(parts) + 1,
                        varied_slot_label="stacked role",
                        base_form=symbol,
                        expected_morphemes=list(parts) + [stack_form],
                        expected_slots=[slot_id or "" for slot_id in slots] + [""],
                    )
                )

    caps = {
        "substitution": 8,
        "role_substitution": 10,
        "optional_slot": 5,
        "agglutination_depth": 60,
        "role_depth": 30,
    }
    by_purpose: dict[str, list[BatchStimulus]] = defaultdict(list)
    for stimulus in stimuli:
        by_purpose[stimulus.purpose].append(stimulus)
    scoped: list[BatchStimulus] = []
    for purpose, cap in caps.items():
        scoped.extend(by_purpose[purpose][:cap])

    # Attested-code controls: the agent SHOULD accept these. If it rejects them too,
    # the all-invalid result is blanket conservatism, not a non-productive grammar.
    for symbol, parts in sorted(codes.items(), key=lambda item: -len(item[1]))[:5]:
        scoped.append(
            BatchStimulus(
                purpose="control_attested",
                form=symbol,
                expected_meaning=grammar.meaning_by_symbol.get(symbol)
                or gloss_of(parts, _code_slots(grammar, analysis, symbol, parts, form_slot)),
                detail="attested code -- expected VALID",
                n_morphemes=len(parts),
            )
        )
    return scoped
async def decode_forms(
    base_messages: list[dict[str, str]], forms: list[str], model: str
) -> tuple[dict[str, str], dict[str, bool]]:
    """Decode unique forms in bounded concurrent calls and merge by exact surface."""
    chunks = [
        forms[start:start + _DECODE_FORMS_PER_CALL]
        for start in range(0, len(forms), _DECODE_FORMS_PER_CALL)
    ]
    calls = await asyncio.gather(
        *(
            call_structured(
                messages=base_messages + [
                    {
                        "role": "user",
                        "content": _PROMPT_HEADER + "\n".join(
                            f"{index + 1}. {form}" for index, form in enumerate(chunk)
                        ),
                    }
                ],
                output_schema=_BatchDecode,
                model=model,
                caller="paradigm_decode_batch",
                max_output_tokens=_BATCH_MAX_OUTPUT_TOKENS,
            )
            for chunk in chunks
        )
    )
    reading_by_form: dict[str, str] = {}
    valid_by_form: dict[str, bool] = {}
    for chunk, call in zip(chunks, calls):
        by_form = {judgment.form: judgment for judgment in call.result.judgments}
        for position, form in enumerate(chunk):
            judgment = by_form.get(form)
            if judgment is None and position < len(call.result.judgments):
                judgment = call.result.judgments[position]
            reading_by_form[form] = judgment.interpreted_meaning if judgment else ""
            valid_by_form[form] = bool(judgment.is_valid) if judgment else False
    return reading_by_form, valid_by_form
