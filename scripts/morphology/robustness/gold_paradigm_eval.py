"""Gold-standard accuracy measurement for post-gate paradigm induction.

``pipeline_robustness`` scores induction replicas against each other, which measures
consistency: a judge that reliably over-segments ``P1``/``P2`` on every replica scores
a perfect ARI. This module supplies the missing accuracy axis. Hand-constructed
codebooks whose correct parse is known by construction are run through the real
production path and scored against that gold parse.

The scored unit is the POST-GATE result. What the inducer proposes is not what
downstream consumes: ``paradigm_cache.load_run_paradigm`` applies a
degradation gate (fewer than ``MIN_JOINT_GLOBAL_CODES`` multi-morpheme codes demotes
the whole run to a flat lexicon), and ``_merge_nested_constructions`` /
``_normalize_paradigm_reply`` further rewrite the class paths and templates. Scoring
the raw reply would skip all three.

Gold items are induced with an empty ``negotiation_evidence`` list, because a synthetic
item has no run JSONL to mine. Items whose correct answer depends on the agents'
declared schema therefore test the harder, evidence-free case.
"""

import argparse
import asyncio
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import adjusted_rand_score

from morphology.pipeline.joint_paradigm_induction import (
    JointParadigmEntry,
    JointParadigmReply,
    to_result,
)
from morphology.pipeline.morphological_grammar import MorphologicalGrammar
from morphology.pipeline.paradigm_network import (
    SlotAnalysis,
    ParadigmClass,
    ParadigmConstruction,
    ParadigmMorpheme,
)
from morphology.pipeline.paradigm_cache import (
    MIN_JOINT_GLOBAL_CODES,
    load_run_paradigm,
)
from morphology.robustness.agreement import safe_mean

import morphology.common as C
from morphology.pipeline.semantic_equivalence_judge import equivalence_rate

logger = logging.getLogger(__name__)

_MORPHEME_SEPARATOR = " + "
_GOLD_RUN_SCENARIO = "gold"


# ---------------------------------------------------------------------------
# Gold item format
# ---------------------------------------------------------------------------


class GoldItem(BaseModel):
    """One hand-constructed codebook and its known-correct parse.

    ``codebook`` maps every code to its meaning. ``gold`` maps every code to its parse
    string, or to ``None`` for a code that must stay lexical (an omitted key means the
    same). A parse string is ``form=CLASS:gloss`` segments joined by `` + ``: the forms
    must concatenate to the code exactly, a form may not contain ``=``, and a class name
    may not contain ``:``.
    """

    item_id: str
    phenomenon: str
    notes: str = ""
    codebook: dict[str, str]
    gold: dict[str, str | None] = Field(default_factory=dict)


class GoldMorpheme(BaseModel):
    """One morpheme of a gold parse: its surface form, class label, and gloss."""

    form: str
    class_label: str
    gloss: str


def parse_gold_string(symbol: str, text: str) -> list[GoldMorpheme]:
    """Parse one ``form=CLASS:gloss + ...`` string, checking it reconstructs the code."""
    morphemes: list[GoldMorpheme] = []
    for segment in text.split(_MORPHEME_SEPARATOR):
        if "=" not in segment or ":" not in segment.split("=", 1)[1]:
            raise ValueError(
                f"{symbol!r}: segment {segment!r} is not 'form=CLASS:gloss'"
            )
        form, remainder = segment.split("=", 1)
        class_label, gloss = remainder.split(":", 1)
        if not form.strip() or not class_label.strip() or not gloss.strip():
            raise ValueError(f"{symbol!r}: segment {segment!r} has an empty field")
        morphemes.append(
            GoldMorpheme(
                form=form.strip(), class_label=class_label.strip(), gloss=gloss.strip()
            )
        )
    if len(morphemes) < 2:
        raise ValueError(f"{symbol!r}: a gold parse needs at least two morphemes")
    joined = "".join(morpheme.form for morpheme in morphemes)
    if joined != symbol:
        raise ValueError(f"{symbol!r}: gold morphemes concatenate to {joined!r}")
    return morphemes


def gold_parses(item: GoldItem) -> dict[str, list[GoldMorpheme]]:
    """Return the parsed gold morphemes of every structured code in the item."""
    unknown = set(item.gold) - set(item.codebook)
    if unknown:
        raise ValueError(f"{item.item_id}: gold keys absent from codebook: {sorted(unknown)}")
    return {
        symbol: parse_gold_string(symbol=symbol, text=text)
        for symbol, text in item.gold.items()
        if text is not None
    }


def gold_to_reply(item: GoldItem) -> JointParadigmReply:
    """Expand the compact gold maps into a full ``JointParadigmReply``.

    Constructions are derived, not authored: every distinct class path becomes one
    construction, and the class inventory is the union of the class labels used. The
    head index is always 0 because head placement is deliberately out of scope here.
    """
    parses = gold_parses(item)
    lexical = sorted(set(item.codebook) - set(parses))
    class_labels: list[str] = []
    for morphemes in parses.values():
        for morpheme in morphemes:
            if morpheme.class_label not in class_labels:
                class_labels.append(morpheme.class_label)
    templates: dict[str, list[str]] = {}
    entries: list[JointParadigmEntry] = []
    for symbol, morphemes in parses.items():
        path = [morpheme.class_label for morpheme in morphemes]
        construction_id = "__".join(path)
        templates.setdefault(construction_id, path)
        entries.append(
            JointParadigmEntry(
                symbol=symbol,
                construction_id=construction_id,
                head_morpheme_index=0,
                morphemes=[
                    ParadigmMorpheme(
                        form=morpheme.form,
                        class_id=morpheme.class_label,
                        gloss=morpheme.gloss,
                    )
                    for morpheme in morphemes
                ],
            )
        )
    repeatable = {
        class_label
        for morphemes in parses.values()
        for class_label, count in Counter(
            morpheme.class_label for morpheme in morphemes
        ).items()
        if count > 1
    }
    return JointParadigmReply(
        analysis_type="paradigm_network" if entries else "flat_lexicon",
        rationale=f"gold item {item.item_id}",
        lexical_symbols=lexical,
        classes=[
            ParadigmClass(
                class_id=label,
                label=label,
                description=f"gold class {label}",
                repeatable=label in repeatable,
                declared_fillers=[],
            )
            for label in class_labels
        ],
        constructions=[
            ParadigmConstruction(
                construction_id=construction_id,
                label=construction_id,
                description=f"gold construction {construction_id}",
                class_sequence=path,
            )
            for construction_id, path in templates.items()
        ],
        entries=entries,
    )


def load_gold_item(path: Path) -> GoldItem:
    """Load one gold JSON file and validate its parse against the production validator.

    Running the expanded gold through ``to_result`` catches authoring mistakes -- forms
    that do not reconstruct the code, an incomplete partition, a construction with no
    substitution contrast -- at load time, so they surface as a bad gold file rather than
    as a model error.
    """
    item = GoldItem.model_validate_json(path.read_text(encoding="utf-8"))
    to_result(
        run_key=f"{_GOLD_RUN_SCENARIO}/{item.item_id}",
        codebook=item_codebook(item),
        reply=gold_to_reply(item),
    )
    return item


def item_codebook(item: GoldItem) -> pd.DataFrame:
    """Render a gold item as the codebook DataFrame the production loader expects."""
    return pd.DataFrame(
        [
            {
                "run_key": f"{_GOLD_RUN_SCENARIO}/{item.item_id}",
                "symbol": symbol,
                "meaning": meaning,
                "round_introduced": 1,
            }
            for symbol, meaning in item.codebook.items()
        ],
        columns=["run_key", "symbol", "meaning", "round_introduced"],
    )


# ---------------------------------------------------------------------------
# The comparable view of one analysis, gold or predicted
# ---------------------------------------------------------------------------


class ParseView(BaseModel):
    """Occurrence-level view of one analysis, in the form both sides are scored on.

    Occurrences are keyed by ``symbol@start:end`` so gold and prediction align only on
    identical surface slices, rather than on morpheme index, which would drift as soon as
    the two disagree about a single boundary.
    """

    below_gate: bool
    segmentation: dict[str, list[str]]
    class_by_occurrence: dict[str, str]
    gloss_by_occurrence: dict[str, str]
    construction_by_symbol: dict[str, str]
    construction_templates: dict[str, list[str]]

    def structured(self) -> set[str]:
        """Codes this analysis decomposed into two or more morphemes."""
        return {symbol for symbol, forms in self.segmentation.items() if len(forms) >= 2}

    def boundaries(self) -> set[str]:
        """Every internal cut, as ``symbol@offset`` keys, across all codes."""
        cuts: set[str] = set()
        for symbol, forms in self.segmentation.items():
            offset = 0
            for form in forms[:-1]:
                offset += len(form)
                cuts.add(f"{symbol}@{offset}")
        return cuts


def _occurrence_keys(symbol: str, forms: list[str]) -> list[str]:
    """Return one ``symbol@start:end`` key per morpheme of a segmented code.

    The key carries the full span, not just the start offset. Keying on the start alone
    would align gold's ``AHY`` (chars 0-3) with a prediction's ``A`` (chars 0-1) merely
    because both begin at 0, and then compare glosses for two different morphemes. With
    the end included, a code the two sides segment differently simply contributes no
    aligned occurrences over the divergent region, and the disagreement is charged to the
    segmentation metrics instead of leaking into the class and gloss ones.
    """
    keys: list[str] = []
    offset = 0
    for form in forms:
        keys.append(f"{symbol}@{offset}:{offset + len(form)}")
        offset += len(form)
    return keys


def gate_fired(segmentation: dict[str, list[str]]) -> bool:
    """Apply the production degradation gate rule to a segmentation."""
    n_multi = sum(1 for forms in segmentation.values() if len(forms) >= 2)
    return n_multi < MIN_JOINT_GLOBAL_CODES


def build_parse_view(
    symbols: list[str],
    grammar: MorphologicalGrammar,
    analysis: SlotAnalysis | None,
    below_gate: bool,
) -> ParseView:
    """Project a grammar plus its slot analysis into the scored view.

    When the gate has fired, downstream sees a flat lexicon regardless of what was
    induced, so the view is flattened to match: every code monomorphemic, no classes,
    no constructions. Scoring anything else would credit structure no consumer sees.
    """
    if below_gate or analysis is None:
        return ParseView(
            below_gate=True,
            segmentation={symbol: [symbol] for symbol in symbols},
            class_by_occurrence={},
            gloss_by_occurrence={},
            construction_by_symbol={},
            construction_templates={},
        )
    segmentation = {
        symbol: list(grammar.segmentation.get(symbol, [symbol])) for symbol in symbols
    }
    gloss_by_class_form = {
        f"{slot.slot_id}\t{filler.form}": filler.gloss
        for slot in analysis.slots
        for filler in slot.fillers
    }
    class_by_occurrence: dict[str, str] = {}
    gloss_by_occurrence: dict[str, str] = {}
    for symbol, forms in segmentation.items():
        if len(forms) < 2:
            continue
        slot_ids = analysis.slots_by_symbol.get(symbol, [])
        if len(slot_ids) != len(forms):
            slot_ids = [
                f"pos{position}" for position in grammar.positions_by_symbol.get(symbol, [])
            ]
        if len(slot_ids) != len(forms):
            continue
        for key, form, slot_id in zip(_occurrence_keys(symbol, forms), forms, slot_ids):
            class_by_occurrence[key] = slot_id
            gloss_by_occurrence[key] = gloss_by_class_form.get(f"{slot_id}\t{form}", "")
    return ParseView(
        below_gate=False,
        segmentation=segmentation,
        class_by_occurrence=class_by_occurrence,
        gloss_by_occurrence=gloss_by_occurrence,
        construction_by_symbol=dict(analysis.construction_by_symbol),
        construction_templates={
            construction_id: list(template)
            for construction_id, template in analysis.construction_templates.items()
        },
    )


def gold_parse_view(item: GoldItem) -> ParseView:
    """Build the gold side of the comparison, through the same tail as the prediction.

    Gold is expanded into a reply and pushed through ``to_result``, so it undergoes the
    identical normalization the prediction does -- in particular
    ``_merge_nested_constructions``, which collapses depth variants into one template
    with optional slots. Deriving templates straight from the authored parse strings
    instead would give gold two constructions for an ``A+B`` / ``A+B+C`` family where the
    production path yields one, and would score a correct prediction as a template error.
    """
    result = to_result(
        run_key=f"{_GOLD_RUN_SCENARIO}/{item.item_id}",
        codebook=item_codebook(item),
        reply=gold_to_reply(item),
    )
    return build_parse_view(
        symbols=sorted(item.codebook),
        grammar=result.grammar,
        analysis=result.analysis,
        below_gate=gate_fired(result.grammar.segmentation),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class GoldError(BaseModel):
    """One scored disagreement, for inspecting why an item failed."""

    item_id: str
    replica: int
    kind: str
    symbol: str
    gold: str
    predicted: str


class GoldScore(BaseModel):
    """Accuracy of one prediction against one gold item.

    ``None`` marks a metric that is undefined for this item rather than zero -- for
    example the occurrence-level class ARI when gold and prediction share no aligned
    morpheme slice at all.
    """

    item_id: str
    replica: int
    n_symbols: int

    gate_expected: bool
    gate_predicted: bool
    gate_match: bool

    structured_accuracy: float
    structured_precision: float | None
    structured_recall: float | None
    structured_f1: float | None

    segmentation_exact_match: float
    boundary_precision: float | None
    boundary_recall: float | None
    boundary_f1: float | None

    n_gold_occurrences: int
    n_aligned_occurrences: int
    class_alignment_coverage: float
    class_ari: float | None

    construction_ari: float | None
    n_gold_constructions: int
    template_exact_match: float | None

    n_glosses_compared: int
    gloss_literal_agreement: float | None
    gloss_semantic_agreement: float | None = None

    errors: list[GoldError] = Field(default_factory=list)
    gloss_disagreements: list[list[str]] = Field(default_factory=list)


def _prf(true_positive: int, predicted: int, actual: int) -> tuple[float | None, ...]:
    """Precision, recall, and F1, with ``None`` where the denominator is empty."""
    precision = true_positive / predicted if predicted else None
    recall = true_positive / actual if actual else None
    if precision is None or recall is None or precision + recall == 0.0:
        return precision, recall, None
    return precision, recall, 2 * precision * recall / (precision + recall)


def _class_translation(gold: ParseView, prediction: ParseView) -> dict[str, str]:
    """Map each predicted class to the gold class it most often aligns with."""
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for key, predicted_class in prediction.class_by_occurrence.items():
        if key in gold.class_by_occurrence:
            votes[predicted_class][gold.class_by_occurrence[key]] += 1
    return {
        predicted_class: counts.most_common(1)[0][0]
        for predicted_class, counts in votes.items()
    }


def score_prediction(
    item: GoldItem, prediction: ParseView, replica: int
) -> GoldScore:
    """Score one prediction against one gold item. Pure: no API calls."""
    gold = gold_parse_view(item)
    symbols = sorted(item.codebook)
    errors: list[GoldError] = []

    gold_structured = gold.structured()
    predicted_structured = prediction.structured()
    correct_split = sum(
        (symbol in gold_structured) == (symbol in predicted_structured)
        for symbol in symbols
    )
    split_precision, split_recall, split_f1 = _prf(
        true_positive=len(gold_structured & predicted_structured),
        predicted=len(predicted_structured),
        actual=len(gold_structured),
    )

    exact = 0
    for symbol in symbols:
        gold_forms = gold.segmentation[symbol]
        predicted_forms = prediction.segmentation.get(symbol, [symbol])
        if gold_forms == predicted_forms:
            exact += 1
        else:
            errors.append(
                GoldError(
                    item_id=item.item_id,
                    replica=replica,
                    kind="segmentation",
                    symbol=symbol,
                    gold="|".join(gold_forms),
                    predicted="|".join(predicted_forms),
                )
            )
    gold_cuts = gold.boundaries()
    predicted_cuts = prediction.boundaries()
    boundary_precision, boundary_recall, boundary_f1 = _prf(
        true_positive=len(gold_cuts & predicted_cuts),
        predicted=len(predicted_cuts),
        actual=len(gold_cuts),
    )

    aligned = sorted(set(gold.class_by_occurrence) & set(prediction.class_by_occurrence))
    n_gold_occurrences = len(gold.class_by_occurrence)
    coverage = len(aligned) / n_gold_occurrences if n_gold_occurrences else 1.0
    if not gold.class_by_occurrence and not prediction.class_by_occurrence:
        class_ari = 1.0
    elif len(aligned) >= 2:
        class_ari = float(
            adjusted_rand_score(
                [gold.class_by_occurrence[key] for key in aligned],
                [prediction.class_by_occurrence[key] for key in aligned],
            )
        )
    else:
        class_ari = None

    gold_clusters = {symbol: gold.construction_by_symbol.get(symbol, "LEXICAL") for symbol in symbols}
    predicted_clusters = {
        symbol: prediction.construction_by_symbol.get(symbol, "LEXICAL") for symbol in symbols
    }
    construction_ari = (
        float(
            adjusted_rand_score(
                [gold_clusters[symbol] for symbol in symbols],
                [predicted_clusters[symbol] for symbol in symbols],
            )
        )
        if len(symbols) >= 2
        else None
    )

    translation = _class_translation(gold=gold, prediction=prediction)
    template_matches = 0
    for construction_id, template in gold.construction_templates.items():
        members = {
            symbol
            for symbol, assigned in gold.construction_by_symbol.items()
            if assigned == construction_id
        }
        counterparts = Counter(
            prediction.construction_by_symbol[symbol]
            for symbol in members
            if symbol in prediction.construction_by_symbol
        )
        if not counterparts:
            errors.append(
                GoldError(
                    item_id=item.item_id,
                    replica=replica,
                    kind="template_missing",
                    symbol=construction_id,
                    gold=" ".join(template),
                    predicted="",
                )
            )
            continue
        counterpart = counterparts.most_common(1)[0][0]
        translated = [
            translation.get(class_id, class_id)
            for class_id in prediction.construction_templates.get(counterpart, [])
        ]
        if translated == template:
            template_matches += 1
        else:
            errors.append(
                GoldError(
                    item_id=item.item_id,
                    replica=replica,
                    kind="template",
                    symbol=construction_id,
                    gold=" ".join(template),
                    predicted=" ".join(translated),
                )
            )
    if gold.construction_templates:
        template_exact_match = template_matches / len(gold.construction_templates)
    elif prediction.construction_templates:
        template_exact_match = 0.0
    else:
        template_exact_match = 1.0

    gloss_disagreements: list[list[str]] = []
    gloss_matches = 0
    for key in aligned:
        gold_gloss = gold.gloss_by_occurrence.get(key, "")
        predicted_gloss = prediction.gloss_by_occurrence.get(key, "")
        if gold_gloss.strip().lower() == predicted_gloss.strip().lower():
            gloss_matches += 1
        else:
            gloss_disagreements.append([gold_gloss, predicted_gloss])
            errors.append(
                GoldError(
                    item_id=item.item_id,
                    replica=replica,
                    kind="gloss",
                    symbol=key,
                    gold=gold_gloss,
                    predicted=predicted_gloss,
                )
            )

    if gold.below_gate != prediction.below_gate:
        errors.append(
            GoldError(
                item_id=item.item_id,
                replica=replica,
                kind="gate",
                symbol="",
                gold=str(gold.below_gate),
                predicted=str(prediction.below_gate),
            )
        )

    return GoldScore(
        item_id=item.item_id,
        replica=replica,
        n_symbols=len(symbols),
        gate_expected=gold.below_gate,
        gate_predicted=prediction.below_gate,
        gate_match=gold.below_gate == prediction.below_gate,
        structured_accuracy=correct_split / len(symbols),
        structured_precision=split_precision,
        structured_recall=split_recall,
        structured_f1=split_f1,
        segmentation_exact_match=exact / len(symbols),
        boundary_precision=boundary_precision,
        boundary_recall=boundary_recall,
        boundary_f1=boundary_f1,
        n_gold_occurrences=n_gold_occurrences,
        n_aligned_occurrences=len(aligned),
        class_alignment_coverage=coverage,
        class_ari=class_ari,
        construction_ari=construction_ari,
        n_gold_constructions=len(gold.construction_templates),
        template_exact_match=template_exact_match,
        n_glosses_compared=len(aligned),
        gloss_literal_agreement=gloss_matches / len(aligned) if aligned else None,
        errors=errors,
        gloss_disagreements=gloss_disagreements,
    )


# ---------------------------------------------------------------------------
# Running the production path on a gold item
# ---------------------------------------------------------------------------


def run_gold_replica(
    item: GoldItem,
    replica: int,
    cache_dir: Path,
    joint_model: str,
    codebook_model: str,
) -> ParseView:
    """Induce one replica through the real post-gate loader and project it for scoring.

    Each replica gets its own cache directory, which both isolates replicas from each
    other (the joint-paradigm cache is keyed by run and model, not by replica) and makes
    a re-run resumable. The codebook parquet is pre-seeded at the exact path
    ``load_codebook`` looks for, so no extraction call is ever made. ``role_pool_model``
    is left unset: role pooling is a paid call whose output is not scored here.
    """
    safe_model = "".join(ch if ch.isalnum() else "-" for ch in codebook_model)
    replica_cache = cache_dir / "gold_cache" / f"r{replica}" / item.item_id
    replica_cache.mkdir(parents=True, exist_ok=True)
    runs_root = replica_cache / "runs"
    run_dir = runs_root / _GOLD_RUN_SCENARIO / item.item_id
    run_dir.mkdir(parents=True, exist_ok=True)
    item_codebook(item).to_parquet(replica_cache / f"cb_{item.item_id}_{safe_model}.parquet")

    loaded = load_run_paradigm(
        run_key=f"{_GOLD_RUN_SCENARIO}/{item.item_id}",
        runs_root=runs_root,
        cache_dir=replica_cache,
        model=codebook_model,
        joint_model=joint_model,
        role_pool_model=None,
    )
    return build_parse_view(
        symbols=sorted(item.codebook),
        grammar=loaded.grammar,
        analysis=loaded.analysis,
        below_gate=loaded.below_gate,
    )


class GoldReport(BaseModel):
    """The full accuracy report over a gold set."""

    joint_model: str
    judge_model: str
    n_replicas: int
    item_ids: list[str]
    scores: list[GoldScore]
    macro: dict[str, float | None]


_MACRO_FIELDS = (
    "gate_match",
    "structured_accuracy",
    "structured_precision",
    "structured_recall",
    "structured_f1",
    "segmentation_exact_match",
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "class_alignment_coverage",
    "class_ari",
    "construction_ari",
    "template_exact_match",
    "gloss_literal_agreement",
    "gloss_semantic_agreement",
)


def macro_average(scores: list[GoldScore]) -> dict[str, float | None]:
    """Mean of each metric across scores, ignoring the ones it is undefined for."""
    macro: dict[str, float | None] = {}
    for field in _MACRO_FIELDS:
        values = [
            float(getattr(score, field))
            for score in scores
            if getattr(score, field) is not None
        ]
        macro[field] = safe_mean(values)
    return macro



# The three metrics the paper's Pipeline Accuracy paragraph reports.
_REPORTED_METRICS = (
    ("production_accuracy", "structured_accuracy"),
    ("segmentation_accuracy", "segmentation_exact_match"),
    ("semantic_agreement", "gloss_semantic_agreement"),
)
_CSV_COLUMNS = ["reported_as", "metric", "n_codebooks", "mean", "ci_low", "ci_high"]


def _bootstrap_ci(values: list[float], resamples: int = 20000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap over codebooks, the unit of analysis."""
    import numpy as np
    from scipy.stats import bootstrap

    if len(set(values)) == 1:  # scipy degenerates on a constant sample
        return values[0], values[0]
    result = bootstrap(
        (np.asarray(values, dtype=float),),
        statistic=np.mean,
        n_resamples=resamples,
        confidence_level=0.95,
        method="percentile",
        random_state=np.random.default_rng(seed),
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def write_summary_csv(report: GoldReport, path: Path) -> None:
    """Write the reported accuracy metrics with bootstrapped confidence intervals.

    Writes into `results/tables/`. The values behind the paper are shipped separately as
    `data/morphology/gold_paradigm_accuracy.csv`, because the raw per-replica scores did not
    survive and a re-run cannot land on them again -- compare against that file rather than
    replacing it.

    Replicas are averaged within a codebook before codebooks are resampled, so the
    bootstrap treats the codebook as the independent unit. Codebooks with no morphemes
    contribute no glosses and are absent from the semantic-agreement row.
    """
    by_item: dict[str, list[GoldScore]] = {}
    for score in report.scores:
        by_item.setdefault(score.item_id, []).append(score)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_COLUMNS)
        for reported_as, metric in _REPORTED_METRICS:
            per_codebook = []
            for item_id in report.item_ids:
                values = [
                    getattr(score, metric, None)
                    for score in by_item.get(item_id, [])
                    if getattr(score, metric, None) is not None
                ]
                if values:
                    per_codebook.append(sum(values) / len(values))
            if not per_codebook:
                continue
            low, high = _bootstrap_ci(per_codebook)
            mean = sum(per_codebook) / len(per_codebook)
            writer.writerow([reported_as, metric, len(per_codebook),
                             f"{mean:.3f}", f"{low:.3f}", f"{high:.3f}"])


def main() -> None:
    """Score every gold item against the production induction path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path,
                        default=Path(__file__).resolve().parent / "gold_paradigms")
    parser.add_argument("--joint-model", default="gpt-5.4")
    parser.add_argument("--codebook-model", default="gpt-5.4")
    parser.add_argument("--judge-model", default="claude-sonnet-4-6")
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--cache-dir", type=Path, default=C.ROBUSTNESS / "gold_eval")
    parser.add_argument("--out", type=Path, default=C.TABLES / "gold_paradigm_accuracy.json")
    parser.add_argument("--item", action="append", help="restrict to these item ids")
    parser.add_argument("--skip-gloss-judge", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    cache_dir = Path(args.cache_dir)
    items = [load_gold_item(path) for path in sorted(Path(args.gold_dir).glob("*.json"))]
    if args.item:
        items = [item for item in items if item.item_id in set(args.item)]
    if not items:
        raise SystemExit(f"no gold items found under {args.gold_dir}")

    scores: list[GoldScore] = []
    for item in items:
        for replica in range(args.replicas):
            prediction = run_gold_replica(
                item=item,
                replica=replica,
                cache_dir=cache_dir,
                joint_model=args.joint_model,
                codebook_model=args.codebook_model,
            )
            score = score_prediction(item=item, prediction=prediction, replica=replica)
            if score.gloss_disagreements and not args.skip_gloss_judge:
                pairs = [(pair[0], pair[1]) for pair in score.gloss_disagreements]
                equivalent = asyncio.run(equivalence_rate(pairs, args.judge_model))
                matched = score.n_glosses_compared - len(pairs)
                if score.n_glosses_compared and equivalent is not None:
                    score.gloss_semantic_agreement = (
                        matched + equivalent * len(pairs)
                    ) / score.n_glosses_compared
            else:
                score.gloss_semantic_agreement = score.gloss_literal_agreement
            logger.info(
                "%s r%d: gate_match=%s split_acc=%.2f seg_exact=%.2f class_ari=%s",
                item.item_id,
                replica,
                score.gate_match,
                score.structured_accuracy,
                score.segmentation_exact_match,
                score.class_ari,
            )
            scores.append(score)

    report = GoldReport(
        joint_model=args.joint_model,
        judge_model=args.judge_model,
        n_replicas=args.replicas,
        item_ids=[item.item_id for item in items],
        scores=scores,
        macro=macro_average(scores),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    errors_path = out_path.with_name("gold_paradigm_errors.jsonl")
    with errors_path.open("w", encoding="utf-8") as handle:
        for score in scores:
            for error in score.errors:
                handle.write(error.model_dump_json() + "\n")
    summary_path = C.TABLES / "gold_paradigm_accuracy.csv"
    write_summary_csv(report, summary_path)
    print(json.dumps(report.macro, indent=2))
    print(f"wrote {out_path}, {errors_path} and {summary_path}")


if __name__ == "__main__":
    main()
