"""The paradigm network: the inducer's reply schema, its validation, and its projection
onto ``SlotAnalysis``.

``SlotAnalysis`` is the package-wide description of a run's substitution classes -- which
morphemes fill which slot, which slots co-occur, and the licensed template of every
construction. The joint inducer returns a paradigm network; ``network_to_slot_analysis``
validates it and projects it onto that shape.
"""

import logging
from collections import Counter, defaultdict

from pydantic import BaseModel, Field, model_validator

from morphology.pipeline.morphological_grammar import MorphologicalGrammar


logger = logging.getLogger(__name__)

# Stopwords never used to name a slot from its fillers' glosses.


class SlotFiller(BaseModel):
    """One morpheme filling a slot, with its gloss and attested occurrence count."""

    form: str
    gloss: str
    count: int


class CategorialSlot(BaseModel):
    """A substitution class: morphemes mutually substitutable at one position.

    ``role`` is the positional role its members share (stem / prefix / suffix).
    ``label`` is a human tag inferred from the fillers' glosses (the category name,
    e.g. ``location``); ``slot_id`` is the stable identifier used elsewhere.
    ``obligatoriness`` is the share of multi-morpheme codes that carry a filler of
    this slot; ``multifill_codes`` counts codes that stack two of its fillers (a
    rigid single-fill slot has zero).
    """

    slot_id: str
    role: str
    label: str
    fillers: list[SlotFiller]
    paradigm_size: int
    obligatoriness: float
    multifill_codes: int


class SlotCooccurrence(BaseModel):
    """Grid saturation between two slots: how much of their product is attested."""

    slot_a: str
    slot_b: str
    attested_combos: int
    possible_combos: int
    saturation: float


class SlotAnalysis(BaseModel):
    """The observational slot grammar induced for one run."""

    run_key: str
    n_codes: int
    n_multimorpheme: int
    slots: list[CategorialSlot]
    cooccurrence: list[SlotCooccurrence]
    orderings: list[list[str]]
    unslotted: list[str]
    # Authoritative occurrence-level class sequence for each segmented code.  A
    # paradigm network can license several templates, so there need not be one
    # universal linear order containing every class.
    slots_by_symbol: dict[str, list[str]] = Field(default_factory=dict)
    construction_by_symbol: dict[str, str] = Field(default_factory=dict)
    construction_labels: dict[str, str] = Field(default_factory=dict)
    # Full licensed class sequence for each construction.  Attested codes may omit
    # optional classes but their paths must remain subsequences of this template.
    construction_templates: dict[str, list[str]] = Field(default_factory=dict)


class _DeclaredFiller(BaseModel):
    """A class member declared by a standalone codebook key entry, not observed in a code.

    ``form`` is the surface the value takes inside codes (bare ``g`` for a suffix,
    bracketed ``[N]`` for a bracketed class); ``gloss`` is its meaning.
    """

    form: str
    gloss: str


class ParadigmClass(BaseModel):
    """One run-global substitution class proposed by the paradigm inducer.

    ``declared_fillers`` carries members the codebook declares as standalone key entries
    (e.g. ``m``/``f`` intensities, the concrete faces) even when they never appear inside a
    multi-morpheme code; they are unioned into the slot's observed fillers downstream.
    """

    class_id: str
    label: str
    description: str
    repeatable: bool
    declared_fillers: list[_DeclaredFiller]

    @model_validator(mode="before")
    @classmethod
    def _tolerate_missing_declared_fillers(cls, data: object) -> object:
        """Default ``declared_fillers`` to empty for pre-change cached replies on load.

        The field is required in the emitted JSON schema (strict structured output makes
        every no-default field required), but caches written before it existed omit it.
        """
        if isinstance(data, dict) and "declared_fillers" not in data:
            return {**data, "declared_fillers": []}
        return data


class ParadigmMorpheme(BaseModel):
    """One already-segmented morpheme assigned to a substitution class."""

    form: str
    class_id: str
    gloss: str


class ParadigmEntry(BaseModel):
    """Occurrence-level class assignment for one multimorphemic code."""

    symbol: str
    construction_id: str
    morphemes: list[ParadigmMorpheme]


class ParadigmConstruction(BaseModel):
    """One licensed morphotactic template over the global class inventory."""

    construction_id: str
    label: str
    description: str
    class_sequence: list[str]


class ParadigmNetwork(BaseModel):
    """Global class inventory plus the licensed class path of every code."""

    classes: list[ParadigmClass]
    constructions: list[ParadigmConstruction]
    entries: list[ParadigmEntry]


def is_subsequence(shorter: list[str], longer: list[str]) -> bool:
    """Return whether ``shorter`` occurs in order inside ``longer``."""
    cursor = iter(longer)
    return all(any(candidate == item for candidate in cursor) for item in shorter)


def _normalize_paradigm_reply(
    grammar: MorphologicalGrammar, reply: ParadigmNetwork
) -> ParadigmNetwork:
    """Derive licensed templates from occurrence assignments, the primary evidence.

    Models sometimes echo a stale or misspelled class id only in the redundant full
    template declaration.  We therefore retain their construction grouping but split
    incompatible paths and use the longest attested path in each subsequence family as
    its full template.  No adjacency is invented by this normalization.
    """
    repaired_entries_by_symbol: dict[str, ParadigmEntry] = {}
    for entry in reply.entries:
        expected = grammar.segmentation.get(entry.symbol)
        morphemes = entry.morphemes
        filtered = [
            morpheme
            for morpheme in morphemes
            if morpheme.form.strip() not in {"", "|"}
            and not morpheme.form.strip().startswith("::")
        ]
        if expected is not None and len(filtered) == len(expected):
            # Segmentation is fixed upstream.  The slot model supplies only the
            # occurrence-level class/gloss; normalize its echoed surface back to the
            # authoritative slices rather than paying for retries over punctuation.
            morphemes = [
                morpheme.model_copy(update={"form": form})
                for morpheme, form in zip(filtered, expected)
            ]
        repaired_entries_by_symbol.setdefault(
            entry.symbol, entry.model_copy(update={"morphemes": morphemes})
        )
    reply = reply.model_copy(update={"entries": list(repaired_entries_by_symbol.values())})

    used_classes = {
        morpheme.class_id for entry in reply.entries for morpheme in entry.morphemes
    }
    repeated_classes = {
        class_id
        for entry in reply.entries
        for class_id, count in Counter(
            morpheme.class_id for morpheme in entry.morphemes
        ).items()
        if count > 1
    }
    classes = [
        item.model_copy(
            update={"repeatable": item.repeatable or item.class_id in repeated_classes}
        )
        for item in reply.classes
        if item.class_id in used_classes
    ]
    declared = {item.construction_id: item for item in reply.constructions}
    entries_by_construction: dict[str, list[ParadigmEntry]] = defaultdict(list)
    for entry in reply.entries:
        entries_by_construction[entry.construction_id].append(entry)

    constructions: list[ParadigmConstruction] = []
    entries: list[ParadigmEntry] = []
    for construction_id, grouped_entries in entries_by_construction.items():
        source = declared.get(construction_id)
        label = source.label if source is not None else construction_id
        description = source.description if source is not None else label
        clusters: list[tuple[list[str], list[ParadigmEntry]]] = []
        for entry in sorted(grouped_entries, key=lambda item: -len(item.morphemes)):
            path = [morpheme.class_id for morpheme in entry.morphemes]
            target = next(
                (
                    index
                    for index, (template, _) in enumerate(clusters)
                    if is_subsequence(path, template) or is_subsequence(template, path)
                ),
                None,
            )
            if target is None:
                clusters.append((path, [entry]))
            else:
                template, members = clusters[target]
                clusters[target] = (
                    path if len(path) > len(template) else template,
                    members + [entry],
                )
        for index, (template, members) in enumerate(clusters, start=1):
            normalized_id = (
                construction_id if len(clusters) == 1 else f"{construction_id}_{index}"
            )
            normalized_label = label if len(clusters) == 1 else f"{label} {index}"
            constructions.append(
                ParadigmConstruction(
                    construction_id=normalized_id,
                    label=normalized_label,
                    description=description,
                    class_sequence=template,
                )
            )
            entries.extend(
                member.model_copy(update={"construction_id": normalized_id})
                for member in members
            )
    return ParadigmNetwork(
        classes=classes, constructions=constructions, entries=entries
    )


def _validate_paradigm_reply(
    grammar: MorphologicalGrammar, reply: ParadigmNetwork
) -> None:
    """Reject incomplete, surface-changing, or internally contradictory networks."""
    expected = {
        symbol: parts
        for symbol, parts in grammar.segmentation.items()
        if len(parts) >= 2
    }
    declared = [item.class_id.strip() for item in reply.classes]
    if not declared or any(not class_id for class_id in declared):
        raise ValueError("paradigm reply contains an empty class inventory or class id")
    if len(declared) != len(set(declared)):
        raise ValueError("paradigm reply contains duplicate class ids")
    class_by_id = {item.class_id: item for item in reply.classes}
    construction_ids = [item.construction_id.strip() for item in reply.constructions]
    if not construction_ids or any(not value for value in construction_ids):
        raise ValueError("paradigm reply contains an empty construction inventory or id")
    if len(construction_ids) != len(set(construction_ids)):
        raise ValueError("paradigm reply contains duplicate construction ids")
    construction_by_id = {
        item.construction_id: item for item in reply.constructions
    }
    for construction in reply.constructions:
        if not construction.class_sequence:
            raise ValueError(
                f"construction {construction.construction_id!r} has an empty template"
            )
        unknown = set(construction.class_sequence) - set(class_by_id)
        if unknown:
            raise ValueError(
                f"construction {construction.construction_id!r} uses undeclared "
                f"classes: {sorted(unknown)}"
            )
        for class_id, count in Counter(construction.class_sequence).items():
            if count > 1 and not class_by_id[class_id].repeatable:
                raise ValueError(
                    f"construction {construction.construction_id!r} repeats "
                    f"non-repeatable class {class_id!r}"
                )
    entries = {entry.symbol: entry for entry in reply.entries}
    if len(entries) != len(reply.entries) or set(entries) != set(expected):
        missing = sorted(set(expected) - set(entries))
        extra = sorted(set(entries) - set(expected))
        raise ValueError(
            "paradigm reply must contain every multimorphemic symbol exactly once; "
            f"missing={missing}, extra={extra}, duplicates="
            f"{len(reply.entries) - len(entries)}"
        )
    used: set[str] = set()
    used_constructions: set[str] = set()
    for symbol, parts in expected.items():
        entry = entries[symbol]
        if entry.construction_id not in construction_by_id:
            raise ValueError(
                f"{symbol!r} uses undeclared construction {entry.construction_id!r}"
            )
        forms = [morpheme.form for morpheme in entry.morphemes]
        if forms != parts:
            raise ValueError(
                f"paradigm reply changed segmentation for {symbol!r}: {forms!r} != {parts!r}"
            )
        class_ids = [morpheme.class_id for morpheme in entry.morphemes]
        unknown = set(class_ids) - set(class_by_id)
        if unknown:
            raise ValueError(f"paradigm reply uses undeclared classes: {sorted(unknown)}")
        for class_id, count in Counter(class_ids).items():
            if count > 1 and not class_by_id[class_id].repeatable:
                raise ValueError(
                    f"{symbol!r} fills non-repeatable class {class_id!r} {count} times"
                )
        used.update(class_ids)
        used_constructions.add(entry.construction_id)
        template = construction_by_id[entry.construction_id].class_sequence
        template_cursor = iter(template)
        if not all(any(candidate == class_id for candidate in template_cursor) for class_id in class_ids):
            raise ValueError(
                f"{symbol!r} class path is not a subsequence of construction "
                f"{entry.construction_id!r}"
            )
    unused = set(class_by_id) - used
    if unused:
        raise ValueError(f"paradigm reply declares unused classes: {sorted(unused)}")
    unused_constructions = set(construction_by_id) - used_constructions
    if unused_constructions:
        raise ValueError(
            f"paradigm reply declares unused constructions: {sorted(unused_constructions)}"
        )
    paths_by_construction: dict[str, list[list[str]]] = defaultdict(list)
    for entry in reply.entries:
        paths_by_construction[entry.construction_id].append(
            [morpheme.class_id for morpheme in entry.morphemes]
        )
    for construction_id, construction in construction_by_id.items():
        for left, right in zip(
            construction.class_sequence, construction.class_sequence[1:]
        ):
            if not any(
                any(a == left and b == right for a, b in zip(path, path[1:]))
                for path in paths_by_construction[construction_id]
            ):
                raise ValueError(
                    f"construction {construction_id!r} invents unsupported adjacent "
                    f"classes {left!r}->{right!r}"
                )


def network_to_slot_analysis(
    grammar: MorphologicalGrammar, reply: ParadigmNetwork
) -> SlotAnalysis:
    """Convert a validated LM network into the downstream observational model."""
    reply = _normalize_paradigm_reply(grammar=grammar, reply=reply)
    _validate_paradigm_reply(grammar=grammar, reply=reply)
    class_index = {item.class_id: index for index, item in enumerate(reply.classes)}
    slot_id = {class_id: f"pos{index}" for class_id, index in class_index.items()}
    entries = {entry.symbol: entry for entry in reply.entries}
    slots_by_symbol = {
        symbol: [slot_id[morpheme.class_id] for morpheme in entries[symbol].morphemes]
        for symbol in entries
    }
    n_multi = len(entries)
    slots: list[CategorialSlot] = []
    for item in reply.classes:
        occurrences: list[tuple[str, str, str]] = []
        roles: Counter[str] = Counter()
        for symbol, entry in entries.items():
            head = grammar.head_index[symbol]
            symbol_roles = _roles_of(len(entry.morphemes), head)
            for morpheme, role in zip(entry.morphemes, symbol_roles):
                if morpheme.class_id == item.class_id:
                    occurrences.append((morpheme.form, morpheme.gloss, symbol))
                    roles[role] += 1
        by_form: dict[str, list[str]] = defaultdict(list)
        for form, gloss, _ in occurrences:
            by_form[form].append(gloss)
        fillers = [
            SlotFiller(
                form=form,
                gloss=Counter(gloss for gloss in glosses if gloss).most_common(1)[0][0]
                if any(glosses)
                else "",
                count=len(glosses),
            )
            for form, glosses in sorted(by_form.items())
        ]
        # Union the codebook's standalone declarations (e.g. the m/f intensities, the
        # concrete faces) with count=0 -- declared paradigm members that never surfaced
        # inside a multi-morpheme code, so pure co-occurrence missed them.
        present_forms = {filler.form for filler in fillers}
        for declared in item.declared_fillers:
            if declared.form and declared.form not in present_forms:
                present_forms.add(declared.form)
                fillers.append(SlotFiller(form=declared.form, gloss=declared.gloss, count=0))
        fillers.sort(key=lambda filler: filler.form)
        symbols_with_class = {symbol for _, _, symbol in occurrences}
        multifill = sum(
            assignments.count(slot_id[item.class_id]) >= 2
            for assignments in slots_by_symbol.values()
        )
        slots.append(
            CategorialSlot(
                slot_id=slot_id[item.class_id],
                role=roles.most_common(1)[0][0],
                label=item.label.strip() or item.class_id,
                fillers=fillers,
                paradigm_size=len(fillers),
                obligatoriness=len(symbols_with_class) / n_multi if n_multi else 0.0,
                multifill_codes=multifill,
            )
        )

    size = {slot.slot_id: slot.paradigm_size for slot in slots}
    combos: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for symbol, assignments in slots_by_symbol.items():
        forms = grammar.segmentation[symbol]
        for left in range(len(forms)):
            for right in range(left + 1, len(forms)):
                slot_a, slot_b = assignments[left], assignments[right]
                if slot_a == slot_b:
                    continue
                key = tuple(sorted((slot_a, slot_b), key=lambda value: int(value[3:])))
                pair = (forms[left], forms[right])
                combos[key].add(pair if slot_a == key[0] else pair[::-1])
    cooccurrence = [
        SlotCooccurrence(
            slot_a=left,
            slot_b=right,
            attested_combos=len(attested),
            possible_combos=size[left] * size[right],
            saturation=len(attested) / (size[left] * size[right]),
        )
        for (left, right), attested in sorted(combos.items())
    ]
    orderings = sorted({tuple(path) for path in slots_by_symbol.values()})
    construction_by_symbol = {
        symbol: entry.construction_id for symbol, entry in entries.items()
    }
    construction_labels = {
        item.construction_id: item.label.strip() or item.construction_id
        for item in reply.constructions
    }
    construction_templates = {
        item.construction_id: [slot_id[class_id] for class_id in item.class_sequence]
        for item in reply.constructions
    }
    unslotted = sorted(
        f"{slot.fillers[0].form}@{slot.slot_id}"
        for slot in slots
        if slot.paradigm_size == 1 and sum(filler.count for filler in slot.fillers) == 1
    )
    return SlotAnalysis(
        run_key=grammar.run_key,
        n_codes=len(grammar.segmentation),
        n_multimorpheme=n_multi,
        slots=slots,
        cooccurrence=cooccurrence,
        orderings=[list(ordering) for ordering in orderings],
        unslotted=unslotted,
        slots_by_symbol=slots_by_symbol,
        construction_by_symbol=construction_by_symbol,
        construction_labels=construction_labels,
        construction_templates=construction_templates,
    )


def _roles_of(n_parts: int, head: int) -> list[str]:
    """Return the positional role of each morpheme given the head (stem) index."""
    roles: list[str] = []
    for index in range(n_parts):
        if index < head:
            roles.append("prefix")
        elif index == head:
            roles.append("stem")
        else:
            roles.append("suffix")
    return roles


