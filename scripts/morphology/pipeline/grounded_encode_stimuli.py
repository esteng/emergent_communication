"""Grounded encode cells: instantiate placeholder templates with concrete arguments.

The joint paradigm abstracts a code like ``F[f]t#g`` into a template with a face slot
(``[f]`` = "target face(s)") and a number slot (``t#`` = "specified seconds"). The decode
and template-level encode probes stop there. This module fills those slots with the
concrete values the run actually negotiated -- faces (``N``=front, ``S``=back, ``D``,
``all``, ...) mined from real engineer usage and the codebook, and durations/counts mined
as the literal digits agents wrote (``t6``, ``x3``) -- producing GROUNDED cells such as
``F[N]t6g`` = "gentle fan on the front face for 6 seconds".

Each grounded cell is emitted as an ``EncodeStimulus`` (target meaning + expected concrete
code + its concrete morphemes), so the existing ``run_encode_decode_batch`` scores it
unchanged. Cells whose exact concrete surface appeared in the transcript are tagged
``grounded_control`` (the sender should reproduce them); novel face/number combinations are
tagged ``grounded_novel`` (the productivity test).
"""

import json
import random
import re
from itertools import product
from pathlib import Path
from typing import NamedTuple

from morphology.pipeline.morphological_grammar import MorphologicalGrammar
from morphology.pipeline.paradigm_network import SlotAnalysis
from morphology.pipeline.paradigm_encode_decode_batch import EncodeStimulus
from morphology.pipeline.codebook import find_run_jsonl

# Held-out duration/count values (small, plausible, deliberately outside the attested set
# when possible) used to build novel grounded cells that test productive instantiation.
_HELD_OUT_DURATIONS = (6, 8, 3)
_HELD_OUT_COUNTS = (4, 6, 1)
# Number-slot marker -> the unit its digits denote, for rendering the grounded meaning.
_NUMBER_UNIT = {"t": "seconds", "x": "chimes", "p": "pulses"}
_FACE_PLACEHOLDER = "[f]"

class ArgumentInventory(NamedTuple):
    """The concrete argument values this run negotiated, mined from usage + codebook."""

    faces: dict[str, str]  # concrete fill (e.g. "N") -> gloss (e.g. "front face")
    durations: list[int]  # attested t# values
    counts: list[int]  # attested x# values
    attested_codes: frozenset[str]  # every concrete code surface seen in the transcript

def _clean_face_term(fill: str, grammar: MorphologicalGrammar) -> str:
    """Gloss one single face fill, dropping the codebook's ``label`` bookkeeping suffix."""
    gloss = grammar.meaning_by_symbol.get(fill, "")
    gloss = re.sub(r"\s*label\s*$", "", gloss).strip()
    if "face" in gloss.lower():
        return gloss
    return f"{fill} face"

def _face_gloss(fill: str, grammar: MorphologicalGrammar) -> str:
    """Render a face fill as a natural phrase, handling ``all`` and opposite pairs.

    ``all`` becomes "all faces"; a ``X/Y`` opposite pair becomes "<x> and <y>" from each
    side's cleaned gloss; a single face uses its codebook gloss (minus the "label" suffix).
    """
    if fill.lower() == "all":
        return "all faces"
    if "/" in fill:
        sides = [_clean_face_term(side, grammar) for side in fill.split("/")]
        return " and ".join(sides)
    return _clean_face_term(fill, grammar)

def _engineer_messages(run_dir: Path) -> list[str]:
    """Return the encoder (stabilization_engineer) link-channel message texts."""
    jsonl = find_run_jsonl(run_dir)
    if jsonl is None:
        return []
    texts: list[str] = []
    with jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if '"message_sent"' not in line:
                continue
            event = json.loads(line)
            if event.get("event_type") != "message_sent":
                continue
            message = event.get("message", {})
            if message.get("sender_agent_id") == "stabilization_engineer":
                texts.append(str(message.get("text", "")))
    return texts

def mine_argument_inventory(
    run_dir: Path, grammar: MorphologicalGrammar
) -> ArgumentInventory:
    """Mine concrete faces, number values, and attested code surfaces from the run.

    Faces are the bracketed fills the encoder actually wrote (``[N]``, ``[D]``, ``[all]``),
    glossed from single-symbol codebook face entries where available. Durations and counts
    are the literal digits following the ``t``/``x`` markers. Attested code surfaces are the
    whitespace/`+`-delimited code-looking tokens, used to tag control vs novel cells.
    """
    texts = _engineer_messages(run_dir)
    face_counts: dict[str, int] = {}
    durations: set[int] = set()
    counts: set[int] = set()
    attested: set[str] = set()
    for text in texts:
        for fill in re.findall(r"\[([A-Za-z0-9/]+)\]", text):
            if fill != "f":
                face_counts[fill] = face_counts.get(fill, 0) + 1
        for value in re.findall(r"t(\d+)", text):
            durations.add(int(value))
        for value in re.findall(r"x(\d+)", text):
            counts.add(int(value))
        for token in re.split(r"[\s;+]+", text):
            if re.fullmatch(r"[A-Za-z]{1,3}(?:\[[A-Za-z0-9/]+\]|[a-z]?\d+|[a-z])+", token):
                attested.add(token)
    # Codebook symbols are attested too -- a closed construction's forms (e.g. fan12, ws12)
    # live in the negotiated codebook even when the engineer never re-typed them on #link.
    attested |= set(grammar.segmentation)
    face_gloss = {fill: _face_gloss(fill, grammar) for fill in face_counts}
    return ArgumentInventory(
        faces=face_gloss,
        durations=sorted(durations),
        counts=sorted(counts),
        attested_codes=frozenset(attested),
    )

class _SlotOption(NamedTuple):
    """One concrete filler for a template slot: its surface, gloss, and semantic role."""

    surface: str  # concrete morpheme, e.g. "[N]", "t6", "F", "g"
    gloss: str  # e.g. "front face", "6 seconds", "fan / cool air", "gentle"
    role: str  # "face" | "number" | "intensity" | "action"
    pooled: bool = False  # True when this filler was cross-construction role-pooled

def _slot_options(
    slot,
    inventory: ArgumentInventory,
    role_pooled_fillers: dict[str, list[tuple[str, str]]] | None,
) -> list[_SlotOption]:
    """Enumerate the concrete fillers a construction slot can take.

    The face slot expands to every mined face fill; a number slot expands to attested plus
    held-out digit values behind its marker (``t6``, ``x4``); every other slot uses its own
    attested morpheme fillers, tagged intensity or action by the slot label. When a slot's
    own within-run fillers are exhausted (e.g. a single-value locus marker), this is often
    the ONLY source of held-out variation, so ``role_pooled_fillers`` -- other constructions'
    same-role morphemes, meaning-judged as substitutable by ``expand_slot_fillers_by_role``
    -- are appended, marked ``pooled=True``, so combinatorial recombination has something to
    draw on even when the slot itself has no spare fillers.

    Pooling is NOT restricted by slot role (including ``stem``, e.g. the fixed ``HM`` prefix
    on every hum-quality code). Whether substituting into a stem is a category error or a
    legitimate wug-style test of a different root taking the same suffix machinery is an
    empirical question, not one this function should decide by filtering -- ``pooled_slots``
    on the resulting ``EncodeStimulus`` records exactly which slot(s) a cell drew a pooled
    filler from, so downstream reporting can break results out per slot (and per pooled vs.
    native) instead of collapsing the question into a single yes/no inclusion decision here.
    """
    forms = [filler.form for filler in slot.fillers]
    if _FACE_PLACEHOLDER in forms:
        base = [
            _SlotOption(surface=f"[{fill}]", gloss=gloss, role="face")
            for fill, gloss in inventory.faces.items()
        ]
    else:
        number_forms = [form for form in forms if "#" in form]
        if number_forms:
            marker = number_forms[0].replace("#", "")
            is_count = marker.endswith("x")
            unit = _NUMBER_UNIT.get(marker[-1:], "units")
            attested = inventory.counts if is_count else inventory.durations
            held_out = _HELD_OUT_COUNTS if is_count else _HELD_OUT_DURATIONS
            values = list(attested) + [n for n in held_out if n not in attested]
            base = [
                _SlotOption(surface=f"{marker}{n}", gloss=f"{n} {unit}", role="number")
                for n in values
            ]
        else:
            role = "intensity" if "intensity" in slot.label.lower() else "action"
            base = [
                _SlotOption(surface=filler.form, gloss=filler.gloss, role=role)
                for filler in slot.fillers
            ]
    if not role_pooled_fillers:
        return base
    own_surfaces = {option.surface for option in base}
    pooled_role = base[0].role if base else "action"
    extras = [
        _SlotOption(surface=form, gloss=gloss, role=pooled_role, pooled=True)
        for form, gloss in role_pooled_fillers.get(slot.slot_id, [])
        if form not in own_surfaces
    ]
    return base + extras

def _wug_meaning(options: list[_SlotOption]) -> str:
    """Compose a natural grounded meaning from a slot combination's glosses."""
    actions = [o.gloss for o in options if o.role == "action"]
    faces = [o.gloss for o in options if o.role == "face"]
    numbers = [o.gloss for o in options if o.role == "number"]
    intensities = [o.gloss for o in options if o.role == "intensity"]
    meaning = ", ".join(actions) or "code"
    if faces:
        face = faces[0]
        meaning += f" on {face}" if face.startswith("all") else f" on the {face}"
    if numbers:
        meaning += f" for {numbers[0]}"
    if intensities:
        meaning += f", {', '.join(intensities)}"
    return meaning

def build_construction_wugs(
    analysis: SlotAnalysis,
    inventory: ArgumentInventory,
    wugs_per_construction: int,
    controls_per_construction: int,
    seed: int,
    role_pooled_fillers: dict[str, list[tuple[str, str]]] | None = None,
) -> list[EncodeStimulus]:
    """Sample per-construction grounded wugs by filling each construction template.

    For every licensed construction template, the cartesian product of its slots' concrete
    fillers is shuffled (seeded) and drawn from until ``wugs_per_construction`` novel
    surfaces (not seen in the transcript) and up to ``controls_per_construction`` attested
    surfaces are collected. Each cell carries its ``construction_id`` for per-construction
    reporting. Closed constructions (e.g. the fixed-12s routine) contribute whatever few
    novel cells exist from their own fillers -- but when ``role_pooled_fillers`` is given,
    slots also draw on other constructions' same-role morphemes (see ``_slot_options``), so a
    construction whose own slots are exhausted (e.g. a single-value locus marker crossed with
    a closed state enum) can still yield novel cells via cross-construction substitution. Such
    cells are tagged ``grounded_novel_role_pooled`` rather than ``grounded_novel`` so the two
    generalization claims (within-construction recombination vs cross-construction role
    substitution) stay distinguishable downstream.
    """
    slot_by_id = {slot.slot_id: slot for slot in analysis.slots}
    rng = random.Random(seed)
    stimuli: list[EncodeStimulus] = []
    seen_forms: set[str] = set()
    for construction_id, template in analysis.construction_templates.items():
        if any(slot_id not in slot_by_id for slot_id in template):
            continue
        option_lists = [
            _slot_options(slot_by_id[slot_id], inventory, role_pooled_fillers)
            for slot_id in template
        ]
        if not all(option_lists):
            continue
        combos = list(product(*option_lists))
        rng.shuffle(combos)
        label = analysis.construction_labels.get(construction_id, construction_id)
        novel_kept = 0
        control_kept = 0
        for combo in combos:
            if novel_kept >= wugs_per_construction and control_kept >= controls_per_construction:
                break
            surfaces = [option.surface for option in combo]
            form = "".join(surfaces)
            if form in seen_forms:
                continue
            is_control = form in inventory.attested_codes
            pooled_slot_ids: list[str] = []
            if is_control:
                if control_kept >= controls_per_construction:
                    continue
                purpose = "grounded_control"
                control_kept += 1
            else:
                if novel_kept >= wugs_per_construction:
                    continue
                pooled_slot_ids = [
                    slot_id
                    for slot_id, option in zip(template, combo)
                    if option.pooled
                ]
                purpose = "grounded_novel_role_pooled" if pooled_slot_ids else "grounded_novel"
                novel_kept += 1
            seen_forms.add(form)
            stimuli.append(
                EncodeStimulus(
                    target_meaning=_wug_meaning(list(combo)),
                    expected_form=form,
                    expected_morphemes=surfaces,
                    expected_slots=list(template),
                    purpose=purpose,
                    varied_slot=construction_id,
                    varied_slot_label=label,
                    n_morphemes=len(surfaces),
                    construction_id=construction_id,
                    pooled_slots=pooled_slot_ids,
                )
            )
    return stimuli

def build_length_gen_wugs(
    single_cells: list[EncodeStimulus],
    inventory: ArgumentInventory,
    n_wugs: int,
    seed: int,
) -> list[EncodeStimulus]:
    """Coin novel two-routine compound wugs (``A+B``) to test length generalization.

    Compounding with ``+`` is the run's attested length axis, so each compound roughly
    doubles the morpheme depth. Pairs are drawn from the single-construction grounded cells;
    only compounds whose exact surface never appeared in the transcript are kept.
    """
    base = [cell for cell in single_cells if "+" not in cell.expected_form and cell.construction_id]
    if len(base) < 2:
        return []
    rng = random.Random(seed)
    stimuli: list[EncodeStimulus] = []
    seen: set[str] = set()
    for _ in range(n_wugs * 12):
        if len(stimuli) >= n_wugs:
            break
        first, second = rng.sample(base, 2)
        form = f"{first.expected_form}+{second.expected_form}"
        if form in seen or form in inventory.attested_codes:
            continue
        seen.add(form)
        stimuli.append(
            EncodeStimulus(
                target_meaning=f"{first.target_meaning}; then {second.target_meaning}",
                expected_form=form,
                expected_morphemes=first.expected_morphemes + ["+"] + second.expected_morphemes,
                expected_slots=first.expected_slots + ["+"] + second.expected_slots,
                purpose="length_gen_novel",
                varied_slot="COMPOUND",
                varied_slot_label="compound (2 routines)",
                n_morphemes=len(first.expected_morphemes) + 1 + len(second.expected_morphemes),
                construction_id="COMPOUND",
            )
        )
    return stimuli
