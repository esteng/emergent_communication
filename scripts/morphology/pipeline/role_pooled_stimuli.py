"""Role-pooled filler expansion for closed slots.

A slot whose attested fillers are exhausted (single-filler or saturated grid) yields
no held-out cell, so the substitution wug test cannot fire. But the SAME semantic role
is usually filled by other morphemes elsewhere in the run's codebook -- other faces,
locations, durations -- sitting in other constructions or as standalone lexemes. This
module runs one meaning-grounded LLM pass, STRICTLY within a single run, that pools
those same-role morphemes per slot so the battery can cross-fill closed slots and test
whether the agent generalizes. Nothing is pooled across runs.
"""

import asyncio
import json
from typing import NamedTuple

from pydantic import BaseModel

from morphology.pipeline.analysis_mode import call_structured
from morphology.pipeline.morphological_grammar import MorphologicalGrammar
from morphology.pipeline.paradigm_network import SlotAnalysis

import morphology.common as C

class _SlotExpansion(BaseModel):
    """Extra same-role filler forms proposed for one slot, drawn from the run pool."""

    slot_id: str
    extra_filler_forms: list[str]

class _RoleExpansionReply(BaseModel):
    """Per-slot same-role filler expansions plus appendable stacking roles for one run."""

    expansions: list[_SlotExpansion]
    # Morphemes from the pool that add a NEW role dimension when appended to a maximal
    # code (an extra face like opposite-face, a duration, a modifier), so a code can be
    # agglutinated one level deeper than any attested form. Required (no default): Azure
    # strict structured output rejects optional fields.
    stackable_role_forms: list[str]

class RoleExpansion(NamedTuple):
    """The two products of one within-run role-pooling pass.

    ``per_slot`` maps slot_id to extra same-role (form, gloss) fillers for that slot.
    ``stackable`` is (form, gloss) morphemes that legally lengthen a maximal code by one,
    used by the depth probe to test agglutination past the attested ceiling.
    """

    per_slot: dict[str, list[tuple[str, str]]]
    stackable: list[tuple[str, str]]

_ROLE_SYSTEM = C.load_prompt("role_pooling")

def expand_slot_fillers_by_role(
    grammar: MorphologicalGrammar,
    analysis: SlotAnalysis,
    model: str = "gpt-5.4",
) -> RoleExpansion:
    """Pool same-role slot fillers and appendable stacking roles, within one run.

    Empty expansion when the run has no slots. Every returned form is validated to be an
    attested run morpheme/lexeme; slot fillers must not already fill that slot.
    """
    if not analysis.slots:
        return RoleExpansion(per_slot={}, stackable=[])
    attested_fillers = {
        slot.slot_id: {filler.form for filler in slot.fillers} for slot in analysis.slots
    }
    pool_gloss: dict[str, str] = {}
    for slot in analysis.slots:
        for filler in slot.fillers:
            pool_gloss.setdefault(filler.form, filler.gloss or filler.form)
    for symbol, meaning in grammar.meaning_by_symbol.items():
        if len(grammar.segmentation.get(symbol, [symbol])) == 1:
            pool_gloss.setdefault(symbol, meaning)
    slots_payload = [
        {
            "slot_id": slot.slot_id,
            "role_label": slot.label,
            "attested_fillers": [
                {"form": filler.form, "gloss": filler.gloss} for filler in slot.fillers
            ],
        }
        for slot in analysis.slots
    ]
    candidate_pool = [
        {"form": form, "gloss": gloss} for form, gloss in sorted(pool_gloss.items())
    ]
    messages = [
        {"role": "system", "content": _ROLE_SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                {"slots": slots_payload, "candidate_pool": candidate_pool},
                ensure_ascii=False,
            ),
        },
    ]
    call = asyncio.run(
        call_structured(
            messages=messages,
            output_schema=_RoleExpansionReply,
            model=model,
            caller="role_filler_expansion",
            max_output_tokens=16384,
            reasoning_effort="medium" if model.startswith("gpt-5") else None,
        )
    )
    reply = _RoleExpansionReply.model_validate(call.result)
    expanded: dict[str, list[tuple[str, str]]] = {}
    for expansion in reply.expansions:
        if expansion.slot_id not in attested_fillers:
            continue
        extras: list[tuple[str, str]] = []
        seen: set[str] = set()
        for form in expansion.extra_filler_forms:
            if (
                form in pool_gloss
                and form not in attested_fillers[expansion.slot_id]
                and form not in seen
            ):
                extras.append((form, pool_gloss[form]))
                seen.add(form)
        if extras:
            expanded[expansion.slot_id] = extras
    stackable: list[tuple[str, str]] = []
    stackable_seen: set[str] = set()
    for form in reply.stackable_role_forms:
        if form in pool_gloss and form not in stackable_seen:
            stackable.append((form, pool_gloss[form]))
            stackable_seen.add(form)
    return RoleExpansion(per_slot=expanded, stackable=stackable)
