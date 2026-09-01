"""Joint codebook-to-paradigm induction.

Segmentation and slot induction are one inference problem here: a boundary is
licensed only when it participates in a meaning-consistent, multi-entry paradigm.
Opaque codebook labels remain lexemes and never enter downstream morphology plots.
"""

import asyncio
import json
import logging
from collections import Counter, defaultdict
from typing import Literal, NamedTuple

import pandas as pd
from pydantic import BaseModel, Field

from morphology.pipeline.analysis_mode import call_structured
from morphology.pipeline.morphological_grammar import MorphologicalGrammar, assemble_grammar
from morphology.pipeline.paradigm_network import (
    SlotAnalysis,
    is_subsequence,
    ParadigmClass,
    ParadigmConstruction,
    ParadigmEntry,
    ParadigmMorpheme,
    ParadigmNetwork,
    network_to_slot_analysis,
)

import morphology.common as C

logger = logging.getLogger(__name__)


class JointParadigmEntry(BaseModel):
    """One compositionally segmented code and its paradigm assignment."""

    symbol: str
    construction_id: str
    head_morpheme_index: int = Field(ge=0)
    morphemes: list[ParadigmMorpheme]


class JointParadigmReply(BaseModel):
    """A complete partition of a codebook into lexemes and paradigm members."""

    analysis_type: Literal["flat_lexicon", "paradigm_network"]
    rationale: str
    lexical_symbols: list[str]
    classes: list[ParadigmClass]
    constructions: list[ParadigmConstruction]
    entries: list[JointParadigmEntry]


class JointParadigmResult(BaseModel):
    """The two downstream views derived from one joint LM decision."""

    grammar: MorphologicalGrammar
    analysis: SlotAnalysis
    reply: JointParadigmReply


_JOINT_SYSTEM = C.load_prompt("joint_paradigm_induction")


def _validate_joint_reply(
    symbols: set[str], reply: JointParadigmReply
) -> ParadigmNetwork:
    """Reject surface changes, incomplete partitions, and unsupported structures."""
    lexical = reply.lexical_symbols
    entry_symbols = [entry.symbol for entry in reply.entries]
    if len(lexical) != len(set(lexical)) or len(entry_symbols) != len(set(entry_symbols)):
        raise ValueError("symbols must not be duplicated within either partition")
    overlap = set(lexical) & set(entry_symbols)
    covered = set(lexical) | set(entry_symbols)
    if overlap or covered != symbols:
        raise ValueError(
            f"lexical/structured partition is not exact: overlap={sorted(overlap)}, "
            f"missing={sorted(symbols - covered)}, extra={sorted(covered - symbols)}"
        )
    if reply.analysis_type == "flat_lexicon":
        if reply.entries or reply.classes or reply.constructions:
            raise ValueError("flat_lexicon must have no entries, classes, or constructions")
        if set(lexical) != symbols:
            raise ValueError("flat_lexicon must list every input as lexical")
        return ParadigmNetwork(classes=[], constructions=[], entries=[])
    if not reply.entries or not reply.classes or not reply.constructions:
        raise ValueError("paradigm_network must contain entries, classes, and constructions")

    class_ids = [item.class_id for item in reply.classes]
    construction_ids = [item.construction_id for item in reply.constructions]
    if any(not value.strip() for value in class_ids + construction_ids):
        raise ValueError("class and construction ids must be nonempty")
    if len(class_ids) != len(set(class_ids)):
        raise ValueError("class ids must be unique")
    if len(construction_ids) != len(set(construction_ids)):
        raise ValueError("construction ids must be unique")
    class_by_id = {item.class_id: item for item in reply.classes}
    construction_by_id = {
        item.construction_id: item for item in reply.constructions
    }
    entries_by_construction: Counter[str] = Counter()
    used_classes: set[str] = set()
    network_entries: list[ParadigmEntry] = []
    for entry in reply.entries:
        if len(entry.morphemes) < 2:
            raise ValueError(f"{entry.symbol!r} is structured but has fewer than two morphemes")
        forms = [item.form for item in entry.morphemes]
        if any(not form for form in forms) or "".join(forms) != entry.symbol:
            raise ValueError(
                f"{entry.symbol!r} morphemes do not reconstruct its exact surface: {forms!r}"
            )
        if entry.head_morpheme_index >= len(forms):
            raise ValueError(f"{entry.symbol!r} has an out-of-range head index")
        if entry.construction_id not in construction_by_id:
            raise ValueError(
                f"{entry.symbol!r} uses undeclared construction {entry.construction_id!r}"
            )
        path = [item.class_id for item in entry.morphemes]
        unknown = set(path) - set(class_by_id)
        if unknown:
            raise ValueError(f"{entry.symbol!r} uses undeclared classes {sorted(unknown)}")
        template = construction_by_id[entry.construction_id].class_sequence
        cursor = iter(template)
        if not all(any(candidate == item for candidate in cursor) for item in path):
            raise ValueError(f"{entry.symbol!r} is not a subsequence of its template")
        for class_id, count in Counter(path).items():
            if count > 1 and not class_by_id[class_id].repeatable:
                raise ValueError(f"{entry.symbol!r} repeats non-repeatable class {class_id!r}")
        entries_by_construction[entry.construction_id] += 1
        used_classes.update(path)
        network_entries.append(
            ParadigmEntry(
                symbol=entry.symbol,
                construction_id=entry.construction_id,
                morphemes=entry.morphemes,
            )
        )
    unused_constructions = set(construction_by_id) - set(entries_by_construction)
    unused_classes = set(class_by_id) - used_classes
    if unused_constructions or unused_classes:
        raise ValueError(
            f"unused declarations: constructions={sorted(unused_constructions)}, "
            f"classes={sorted(unused_classes)}"
        )
    return ParadigmNetwork(
        classes=reply.classes,
        constructions=reply.constructions,
        entries=network_entries,
    )


def _merge_nested_constructions(network: ParadigmNetwork) -> ParadigmNetwork:
    """Collapse depth-variant constructions into one template with optional slots.

    Two constructions belong to the same family when one's class sequence is an
    order-preserving subsequence of the other's (they share the global class ids, so
    the test is a plain subsequence check). Each linearly nested family with a unique
    longest template is merged into that template; every shorter entry's class path
    is already a subsequence of it and is repointed to the canonical construction. A
    family whose longest template is ambiguous (two equal-length maxima) is left
    untouched, so genuinely distinct constructions are never fused.
    """
    if not network.entries:
        return network
    templates = {
        construction.construction_id: list(construction.class_sequence)
        for construction in network.constructions
    }
    construction_by_id = {
        construction.construction_id: construction
        for construction in network.constructions
    }
    parent = {construction_id: construction_id for construction_id in templates}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    ids = list(templates)
    for left_index in range(len(ids)):
        for right_index in range(left_index + 1, len(ids)):
            left, right = ids[left_index], ids[right_index]
            if is_subsequence(templates[left], templates[right]) or is_subsequence(
                templates[right], templates[left]
            ):
                parent[find(left)] = find(right)

    groups: dict[str, list[str]] = defaultdict(list)
    for construction_id in ids:
        groups[find(construction_id)].append(construction_id)

    remap: dict[str, str] = {}
    for members in groups.values():
        longest_len = max(len(templates[member]) for member in members)
        maxima = [member for member in members if len(templates[member]) == longest_len]
        canonical = maxima[0]
        linearly_nested = (
            len(maxima) == 1
            and all(
                is_subsequence(templates[member], templates[canonical])
                for member in members
            )
        )
        target = canonical if linearly_nested else None
        for member in members:
            remap[member] = target if target is not None else member

    merged_entries = [
        entry.model_copy(update={"construction_id": remap[entry.construction_id]})
        for entry in network.entries
    ]
    referenced = {entry.construction_id for entry in merged_entries}
    merged_constructions = [
        construction_by_id[construction_id]
        for construction_id in ids
        if construction_id in referenced
    ]
    return ParadigmNetwork(
        classes=network.classes,
        constructions=merged_constructions,
        entries=merged_entries,
    )


def _require_paradigm_contrast(network: ParadigmNetwork) -> None:
    """Require each surviving construction to hold a real substitution contrast.

    A construction earns its status only when at least one of its classes is filled
    by two or more distinct forms across the construction's own entries. This is the
    slot-level support that replaces the former "two full entries per construction"
    rule: a deep template attested by a single maximal code survives as long as one of
    its slots demonstrably substitutes, and hapax optional slots ride along without
    inflating the requirement.
    """
    if not network.entries:
        return
    fillers_by_construction_class: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entry in network.entries:
        for morpheme in entry.morphemes:
            fillers_by_construction_class[
                (entry.construction_id, morpheme.class_id)
            ].add(morpheme.form)
    contrastive = {
        construction_id
        for (construction_id, _), forms in fillers_by_construction_class.items()
        if len(forms) >= 2
    }
    uncontrasted = sorted(
        {entry.construction_id for entry in network.entries} - contrastive
    )
    if uncontrasted:
        raise ValueError(
            "every construction needs a slot with two or more contrasting fillers; "
            f"no substitution contrast in: {uncontrasted}"
        )


def to_result(
    run_key: str, codebook: pd.DataFrame, reply: JointParadigmReply
) -> JointParadigmResult:
    """Validate one joint decision and derive compatible grammar/network objects."""
    meaning_by_symbol = {
        str(row.symbol): str(row.meaning) for row in codebook.itertuples()
    }
    network = _merge_nested_constructions(
        _validate_joint_reply(set(meaning_by_symbol), reply)
    )
    _require_paradigm_contrast(network)
    entry_by_symbol = {entry.symbol: entry for entry in reply.entries}
    segmentation = {
        symbol: [item.form for item in entry_by_symbol[symbol].morphemes]
        if symbol in entry_by_symbol
        else [symbol]
        for symbol in meaning_by_symbol
    }
    head_index = {
        symbol: entry_by_symbol[symbol].head_morpheme_index
        if symbol in entry_by_symbol
        else 0
        for symbol in meaning_by_symbol
    }
    class_index = {item.class_id: index for index, item in enumerate(reply.classes)}
    positions_by_symbol = {
        entry.symbol: [class_index[item.class_id] for item in entry.morphemes]
        for entry in reply.entries
    }
    gloss_by_form: dict[str, Counter[str]] = {}
    position_glosses: dict[int, dict[str, str]] = {}
    for entry in reply.entries:
        for item in entry.morphemes:
            gloss_by_form.setdefault(item.form, Counter())[item.gloss] += 1
            position_glosses.setdefault(class_index[item.class_id], {})[
                item.form
            ] = item.gloss
    grammar = assemble_grammar(
        run_key=run_key,
        defined=frozenset(meaning_by_symbol),
        segmentation=segmentation,
        head_index=head_index,
        gloss_by_form=gloss_by_form,
        form_positions={
            form: position
            for position, forms in position_glosses.items()
            for form in forms
        },
        position_labels={
            index: item.label for index, item in enumerate(reply.classes)
        },
        positions_by_symbol=positions_by_symbol,
        position_glosses=position_glosses,
        meaning_by_symbol=meaning_by_symbol,
    )
    if not network.entries:
        analysis = SlotAnalysis(
            run_key=run_key,
            n_codes=len(segmentation),
            n_multimorpheme=0,
            slots=[],
            cooccurrence=[],
            orderings=[],
            unslotted=[],
        )
    else:
        analysis = network_to_slot_analysis(grammar=grammar, reply=network)
    return JointParadigmResult(grammar=grammar, analysis=analysis, reply=reply)


class _CandidateFamily(NamedTuple):
    """One deterministic shared-affix family offered to the inducer as a hint."""

    kind: Literal["suffix", "prefix"]
    affix: str
    members: list[tuple[str, str, str]]


_MIN_AFFIX_LEN = 2
_MAX_CANDIDATE_FAMILIES = 12


def _candidate_families(
    records: list[dict[str, str]]
) -> list[_CandidateFamily]:
    """Cluster codebook symbols by a shared prefix/suffix with contrasting residues.

    A family is a shared affix (length >= 2) attested by at least two symbols whose
    complementary residues are not all identical, i.e. a surface substitution frame.
    Families whose member set is contained in a longer-affix family are dropped, so
    ``R10`` is kept over the coarser ``10`` when they cover the same symbols. These are
    non-binding hints; the inducer accepts, refines, or rejects each by meaning.
    """
    meaning_by_symbol = {record["symbol"]: record["meaning"] for record in records}
    symbols = list(meaning_by_symbol)
    families: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for kind in ("suffix", "prefix"):
        buckets: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
        for symbol in symbols:
            for length in range(_MIN_AFFIX_LEN, len(symbol)):
                if kind == "suffix":
                    affix, residue = symbol[-length:], symbol[:-length]
                else:
                    affix, residue = symbol[:length], symbol[length:]
                if residue:
                    buckets[(kind, affix)].append(
                        (symbol, residue, meaning_by_symbol[symbol])
                    )
        for key, members in buckets.items():
            residues = {residue for _, residue, _ in members}
            if len(members) >= 2 and len(residues) >= 2:
                families[key] = members

    ordered = sorted(
        families.items(),
        key=lambda item: (len(item[1]), len(item[0][1])),
        reverse=True,
    )
    kept: list[_CandidateFamily] = []
    kept_member_sets: list[set[str]] = []
    for (kind, affix), members in ordered:
        member_symbols = {symbol for symbol, _, _ in members}
        if any(member_symbols <= existing for existing in kept_member_sets):
            continue
        kept.append(_CandidateFamily(kind=kind, affix=affix, members=members))
        kept_member_sets.append(member_symbols)
        if len(kept) >= _MAX_CANDIDATE_FAMILIES:
            break
    return kept


def _render_candidate_families(families: list[_CandidateFamily]) -> str:
    """Render candidate families as a compact, non-binding hint block."""
    lines = [
        "CANDIDATE PARADIGM FAMILIES (deterministic surface hints — NON-BINDING). "
        "Each is a shared affix with contrasting residues. Accept, refine, split, or "
        "REJECT each purely on whether the residues make a coherent meaning contrast "
        "in a shared frame; a shared affix alone is not evidence. Missing families may "
        "still exist; listed families may be spurious.",
    ]
    for family in families:
        rendered_members = ", ".join(
            f"{symbol} (={residue}|{family.affix} → {meaning!r})"
            for symbol, residue, meaning in family.members
        )
        lines.append(
            f"- shared {family.kind} {family.affix!r}: {rendered_members}"
        )
    return "\n".join(lines)


def induce_joint_paradigm_llm(
    run_key: str,
    codebook: pd.DataFrame,
    model: str = "gpt-5-nano",
    max_attempts: int = 3,
    scaffold: bool = True,
    negotiation_evidence: list[dict[str, object]] | None = None,
) -> JointParadigmResult:
    """Make one structured LM call from full codebook meanings to paradigms."""
    records = [
        {"symbol": str(row.symbol), "meaning": str(row.meaning)}
        for row in codebook.itertuples()
    ]
    user_sections = [
        json.dumps(
            {
                "codebook": records,
                "negotiation_evidence": negotiation_evidence or [],
            },
            ensure_ascii=False,
        )
    ]
    if scaffold:
        families = _candidate_families(records)
        if families:
            user_sections.append(_render_candidate_families(families))
    messages = [
        {"role": "system", "content": _JOINT_SYSTEM},
        {"role": "user", "content": "\n\n".join(user_sections)},
    ]
    for attempt in range(max_attempts):
        call = asyncio.run(
            call_structured(
                messages=messages,
                output_schema=JointParadigmReply,
                model=model,
                caller="joint_paradigm_induction",
                max_output_tokens=24576,
                reasoning_effort="low" if model.startswith("gpt-5") else None,
            )
        )
        reply = JointParadigmReply.model_validate(call.result)
        try:
            return to_result(run_key=run_key, codebook=codebook, reply=reply)
        except ValueError as error:
            if attempt + 1 == max_attempts:
                raise
            logger.warning(
                "%s joint %s reply failed validation (%s); retrying with feedback",
                run_key,
                model,
                error,
            )
            messages.extend(
                [
                    {"role": "assistant", "content": reply.model_dump_json()},
                    {
                        "role": "user",
                        "content": (
                            f"The deterministic validator rejected that analysis: {error}. "
                            "Correct the complete partition and paradigm. Do not invent "
                            "morphology merely to avoid a flat result."
                        ),
                    },
                ]
            )
    raise AssertionError("unreachable joint induction retry state")
