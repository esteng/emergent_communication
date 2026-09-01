"""Morphological grammar for one run: slots, fillers, and the segmentation of each code.

``MorphologicalGrammar`` is the package-wide description of a run's morphology --
which codes decompose, into which morphemes, filling which slots. It is built by
``assemble_grammar`` from the joint inducer's reply, and ``_notation_held_out_cells``
seeds wug cells from the agents' own bound-affix notation.
"""

import logging
import re
from collections import Counter, defaultdict

from pydantic import BaseModel, Field



logger = logging.getLogger(__name__)

# A boundary needs this many distinct complementary pieces to count a non-defined
# piece as a real (distributionally supported) morpheme. Tuned on the running
# example: the location suffixes recur across 3-6 action stems.

# Nano spends a substantial fraction of its completion budget reasoning over the
# whole-run inventory before emitting the (potentially large) structured segmentation.
# 16k truncated even a 27-entry live codebook before JSON was produced, and the
# successful replies used 16k-22k.  The corpus contains substantially larger runs,
# so 64k leaves room for their structured entries while remaining cheap on nano.

# Function words never treated as an affix's contributed meaning when glossing.

# A defined symbol whose notation marks it as a bound suffix affix: a single joiner
# delimiter on the leading edge followed by an alphabetic base (``+opp`` -> a suffix
# joined by ``+`` that attaches after a stem). The agents declare the affix with
# this notation, so it is a real morpheme even when it never co-occurs with a stem
# inside the negotiated codebook (it may only be combined on the link channel, e.g.
# ``L+opp``). Trailing-delimiter symbols (``C+``, ``h-``) are NOT seeded: they are a
# content code carrying a sign, not a bound affix, and crossing them with every stem
# only over-generates. Genuine prefixes are recovered by the distributional inducer.


# A defined symbol whose notation marks it as a bound affix (``+opp``), and a defined
# content stem (``L``, ``ch``) it can attach to.
_LEADING_AFFIX = re.compile(r"^([/:@+\-])([A-Za-z][A-Za-z0-9]*)$")
_CONTENT_STEM = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


class Filler(BaseModel):
    """One morpheme occupying a slot position, with its inferred gloss."""

    form: str
    gloss: str
    support: int  # distinct complementary morphemes it combined with
    is_defined: bool  # whether the bare morpheme is itself a codebook symbol


class Slot(BaseModel):
    """One morphological role and the fillers attested in it.

    ``role`` is ``stem`` (the content head), ``suffix`` (affix after the head), or
    ``prefix`` (affix before the head).
    """

    role: str
    fillers: list[Filler]


class HeldOutCell(BaseModel):
    """A stem x affix combination the grammar can express but never coined.

    ``role`` is the affix's direction; ``form`` is the surface string (``stem`` +
    ``affix`` for a suffix, ``affix`` + ``stem`` for a prefix). ``delimiter`` is the
    joiner the surface form uses (``+`` for ``L+opp``, ``""`` for concatenative
    cells like ``shC``); it is recorded so stimulus generation can render the form.
    """

    stem: str
    affix: str
    role: str  # "suffix" | "prefix"
    form: str
    delimiter: str


class MorphologicalGrammar(BaseModel):
    """The induced grammar for one run.

    ``segmentation`` maps every attested code to its morpheme sequence;
    ``head_index`` gives the position of the stem (content head) in that sequence,
    so morphemes before it are prefixes and after it are suffixes.
    ``attested_orderings`` are the distinct role sequences produced (e.g.
    ``["stem", "suffix"]``). ``depth_ceiling`` is the longest morpheme sequence.
    ``notation_cells`` are wug cells seeded from the agents' own affix notation
    (a defined ``+``-marked bound affix crossed with the defined content stems),
    which catches combinations the distributional inventory misses because the
    affix never co-occurs with a stem inside the codebook (e.g. ``L+opp``).
    LLM grammars additionally carry ``positions_by_symbol`` (the authoritative
    occurrence-level global syntax), ``position_labels``, and position-specific
    glosses. ``form_positions`` is only a compatibility projection for older callers.
    """

    run_key: str
    slots: list[Slot]
    segmentation: dict[str, list[str]]
    head_index: dict[str, int]
    attested_orderings: list[list[str]]
    depth_ceiling: int
    notation_cells: list[HeldOutCell]
    # Optional run-global syntax supplied by the meaning-grounded LLM backend.
    # ``form_positions`` maps a morpheme surface form to one position in the single
    # template shared by every construction in the run.  The defaults keep old
    # heuristic grammars and pre-global JSON caches readable.
    form_positions: dict[str, int] = Field(default_factory=dict)
    position_labels: dict[int, str] = Field(default_factory=dict)
    # Exact LLM assignment for each morpheme occurrence.  This is authoritative for
    # global induction because a short emergent code may be genuinely homographic
    # (e.g. ``R`` used as a target in one form and an anchor in another).
    positions_by_symbol: dict[str, list[int]] = Field(default_factory=dict)
    position_glosses: dict[int, dict[str, str]] = Field(default_factory=dict)
    # Full attested meanings are retained separately from composed morpheme glosses.
    # Controls must be scored against these meanings; gloss concatenation can omit
    # quantification, relations, and sequence structure.
    meaning_by_symbol: dict[str, str] = Field(default_factory=dict)


# Longest bound affix the right-edge peeler will consider (chars after the stem).


def _roles_of(parts: list[str], head: int) -> list[str]:
    """Return the role (prefix / stem / suffix) of each morpheme in a segmentation."""
    roles: list[str] = []
    for index in range(len(parts)):
        if index < head:
            roles.append("prefix")
        elif index == head:
            roles.append("stem")
        else:
            roles.append("suffix")
    return roles


def _notation_held_out_cells(defined: frozenset[str]) -> list[HeldOutCell]:
    """Seed wug cells from the agents' affix notation (``+``-marked bound affixes).

    A defined symbol whose notation marks it as a bound suffix affix (``+opp``) is
    crossed with every defined content stem (``L``, ``ch``, ...), keeping the joiner
    so the surface renders ``L+opp`` and not ``Lopp``. Combinations already defined
    as a code, and the degenerate pairing of an affix with its own base stem
    (``opp`` x ``+opp``), are dropped. This admits an affix with zero attested stem
    co-occurrences inside the codebook -- the case the distributional grid misses
    because the combination only ever appears on the link channel.
    """
    affixes: list[tuple[str, str, str]] = []  # form, delimiter, base
    stems: list[str] = []
    for symbol in sorted(defined):
        lead = _LEADING_AFFIX.match(symbol)
        if lead:
            affixes.append((symbol, lead.group(1), lead.group(2)))
        elif _CONTENT_STEM.match(symbol):
            stems.append(symbol)
    cells: list[HeldOutCell] = []
    seen: set[str] = set()
    for affix, delimiter, base in affixes:
        for stem in stems:
            if stem == base:
                continue
            form = stem + affix
            if form in defined or form in seen:
                continue
            seen.add(form)
            cells.append(
                HeldOutCell(
                    stem=stem, affix=affix, role="suffix", form=form, delimiter=delimiter
                )
            )
    return cells


# The segmenter is meaning-grounded: it decides boundaries from the glosses, not
# from cross-code frequency, so it recovers morphemes coined only once (the
# ``-2adj``/``pair`` count markers a distributional support threshold prunes) while
# refusing phantom splits (an ``S`` shared by ``BS``/``ES``/``HS`` whose meanings
# carry no common ``S`` sub-meaning). Frequency is orthogonal to morphemehood in a
# ~20-code emergent lexicon, so no ``min_support`` gate can separate the two cases.


def assemble_grammar(
    run_key: str,
    defined: frozenset[str],
    segmentation: dict[str, list[str]],
    head_index: dict[str, int],
    gloss_by_form: dict[str, Counter[str]],
    form_positions: dict[str, int] | None = None,
    position_labels: dict[int, str] | None = None,
    positions_by_symbol: dict[str, list[int]] | None = None,
    position_glosses: dict[int, dict[str, str]] | None = None,
    meaning_by_symbol: dict[str, str] | None = None,
) -> MorphologicalGrammar:
    """Build a ``MorphologicalGrammar`` from a resolved segmentation + head map.

    The legacy ``slots`` field is grouped by positional role (``_roles_of``); each
    filler's gloss
    is the most common gloss the segmenter assigned that form, and its support is
    the count of distinct complementary morpheme tuples it combined with. Orderings,
    depth ceiling, and notation-seeded cells follow the same rules as the
    distributional path so both grammars are drop-in for the downstream probes. The
    occurrence-level global positions are persisted alongside it for the newer slot
    inducer and stimulus builders.
    """
    role_partners: dict[str, dict[str, set[tuple[str, ...]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    role_forms: dict[str, set[str]] = defaultdict(set)
    for symbol, parts in segmentation.items():
        if len(parts) < 2:
            continue
        roles = _roles_of(parts, head_index[symbol])
        for index, (morpheme, role) in enumerate(zip(parts, roles)):
            role_forms[role].add(morpheme)
            others = tuple(m for i, m in enumerate(parts) if i != index)
            if others:
                role_partners[role][morpheme].add(others)

    slots: list[Slot] = []
    for role in ("prefix", "stem", "suffix"):
        if role not in role_forms:
            continue
        fillers: list[Filler] = []
        for morpheme in sorted(role_forms[role]):
            glosses = gloss_by_form.get(morpheme)
            gloss = glosses.most_common(1)[0][0] if glosses else ""
            fillers.append(
                Filler(
                    form=morpheme,
                    gloss=gloss,
                    support=len(role_partners[role].get(morpheme, set())),
                    is_defined=morpheme in defined,
                )
            )
        slots.append(Slot(role=role, fillers=fillers))

    attested_orderings = sorted(
        {
            tuple(_roles_of(parts, head_index[symbol]))
            for symbol, parts in segmentation.items()
            if len(parts) >= 2
        },
        key=lambda o: (len(o), o),
    )
    depth_ceiling = max((len(m) for m in segmentation.values()), default=1)

    return MorphologicalGrammar(
        run_key=run_key,
        slots=slots,
        segmentation=segmentation,
        head_index=head_index,
        attested_orderings=[list(o) for o in attested_orderings],
        depth_ceiling=depth_ceiling,
        notation_cells=_notation_held_out_cells(defined=defined),
        form_positions=form_positions or {},
        position_labels=position_labels or {},
        positions_by_symbol=positions_by_symbol or {},
        position_glosses=position_glosses or {},
        meaning_by_symbol=meaning_by_symbol or {},
    )
