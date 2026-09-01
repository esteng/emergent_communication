"""Regression tests for run-global morphology segmentation and slot induction."""

import asyncio
from pathlib import Path

import pandas as pd

import morphology.pipeline.morphological_grammar as morphology
import morphology.robustness.gold_paradigm_eval as gold_eval
import morphology.pipeline.joint_paradigm_induction as joint
import morphology.pipeline.paradigm_network as observational
import morphology.pipeline.semantic_equivalence_judge as semantic_judge
from morphology.pipeline.morphological_grammar import assemble_grammar
from morphology.pipeline.codebook import PostmortemMessage, parse_codebook
from morphology.pipeline.semantic_equivalence_judge import DecodeItem, JudgeOutcome


def test_explicit_codebook_meanings_keep_internal_punctuation() -> None:
    """Commas/semicolons inside a procedure are not mistaken for mapping boundaries."""
    messages = [
        PostmortemMessage(
            round_number=2,
            sender="engineer",
            text=(
                "Bo5p15=alternate bell opposite faces 5 cycles, 15s pause; "
                "WS=warm stone\n"
                "CCL10/F6R=cool cloth over V 10s; remove; fan all 6 faces 10s"
            ),
        )
    ]

    meanings = {entry.symbol: entry.meaning for entry in parse_codebook(messages)}

    assert meanings["Bo5p15"] == "alternate bell opposite faces 5 cycles, 15s pause"
    assert meanings["WS"] == "warm stone"
    assert meanings["CCL10/F6R"] == (
        "cool cloth over V 10s; remove; fan all 6 faces 10s"
    )


def test_joint_induction_can_return_a_complete_flat_lexicon() -> None:
    """The single-step schema needs no fake classes when every code is opaque."""
    codebook = pd.DataFrame(
        {"symbol": ["P1", "P2"], "meaning": ["procedure alpha", "procedure beta"]}
    )
    reply = joint.JointParadigmReply(
        analysis_type="flat_lexicon",
        rationale="The suffixes are arbitrary dictionary indices.",
        lexical_symbols=["P1", "P2"],
        classes=[],
        constructions=[],
        entries=[],
    )

    result = joint.to_result("veyru/flat", codebook, reply)

    assert result.grammar.segmentation == {"P1": ["P1"], "P2": ["P2"]}
    assert result.analysis.n_multimorpheme == 0
    assert result.analysis.construction_templates == {}


def test_joint_induction_builds_segmentation_and_classes_together() -> None:
    """A supported L+d/L+b contrast becomes one construction in one conversion."""
    codebook = pd.DataFrame(
        {"symbol": ["Ld", "Lb", "X"], "meaning": ["dim lamp", "bright lamp", "alarm"]}
    )
    reply = joint.JointParadigmReply(
        analysis_type="paradigm_network",
        rationale="d and b contrast in the shared L frame.",
        lexical_symbols=["X"],
        classes=[
            observational.ParadigmClass(
                class_id="LAMP", label="lamp", description="lamp root", repeatable=False
            ),
            observational.ParadigmClass(
                class_id="MOD", label="modifier", description="lamp state", repeatable=False
            ),
        ],
        constructions=[
            observational.ParadigmConstruction(
                construction_id="LAMP_MOD",
                label="lamp plus modifier",
                description="lamp state construction",
                class_sequence=["LAMP", "MOD"],
            )
        ],
        entries=[
            joint.JointParadigmEntry(
                symbol="Ld", construction_id="LAMP_MOD", head_morpheme_index=0,
                morphemes=[
                    observational.ParadigmMorpheme(
                        form="L", class_id="LAMP", gloss="lamp"
                    ),
                    observational.ParadigmMorpheme(
                        form="d", class_id="MOD", gloss="dim"
                    ),
                ],
            ),
            joint.JointParadigmEntry(
                symbol="Lb", construction_id="LAMP_MOD", head_morpheme_index=0,
                morphemes=[
                    observational.ParadigmMorpheme(
                        form="L", class_id="LAMP", gloss="lamp"
                    ),
                    observational.ParadigmMorpheme(
                        form="b", class_id="MOD", gloss="bright"
                    ),
                ],
            ),
        ],
    )

    result = joint.to_result("veyru/lamps", codebook, reply)

    assert result.grammar.segmentation == {"Ld": ["L", "d"], "Lb": ["L", "b"], "X": ["X"]}
    assert result.analysis.n_multimorpheme == 2
    assert result.analysis.construction_templates == {"LAMP_MOD": ["pos0", "pos1"]}
    assert {filler.form for filler in result.analysis.slots[1].fillers} == {"d", "b"}


def test_old_grammar_cache_remains_readable() -> None:
    """Pre-global JSON caches get empty maps and therefore retain the NW fallback."""
    grammar = morphology.MorphologicalGrammar.model_validate(
        {
            "run_key": "veyru/old",
            "slots": [],
            "segmentation": {"ab": ["a", "b"]},
            "head_index": {"ab": 0},
            "attested_orderings": [["stem", "suffix"]],
            "depth_ceiling": 2,
            "notation_cells": [],
        }
    )

    assert grammar.form_positions == {}
    assert grammar.position_labels == {}


def test_semantic_judge_caps_exhaustive_chunk_concurrency(monkeypatch) -> None:
    """A large exhaustive paradigm cannot fan out every nano request simultaneously."""
    active = 0
    peak = 0

    async def fake_chunk(decode_chunk, equiv_chunk, judge_model=None):  # noqa: ARG001
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return JudgeOutcome(
            decode_scores=[1.0] * len(decode_chunk),
            equivalence_scores=[1.0] * len(equiv_chunk),
            equivalence_same=[True] * len(equiv_chunk),
        )

    monkeypatch.setattr(semantic_judge, "_judge_chunk", fake_chunk)
    outcome = asyncio.run(
        semantic_judge.run_semantic_judge(
            [DecodeItem("intended", "reading")] * 100,
            [],
        )
    )

    assert peak == semantic_judge._JUDGE_CONCURRENCY
    assert outcome.decode_scores == [1.0] * 100


# ---------------------------------------------------------------------------
# Gold-standard accuracy harness
# ---------------------------------------------------------------------------


def _gold_dir() -> Path:
    return Path(gold_eval.__file__).parent / "gold_paradigms"


def test_shipped_gold_items_pass_the_production_validator() -> None:
    """Every gold file expands into a reply the production validator accepts."""
    paths = sorted(_gold_dir().glob("*.json"))
    assert paths, "no gold items shipped"
    for path in paths:
        item = gold_eval.load_gold_item(path)
        assert item.item_id == path.stem
        assert set(item.gold) <= set(item.codebook)


def test_gold_scores_perfectly_against_itself() -> None:
    """The gold view scored as its own prediction is a perfect score on every metric."""
    for path in sorted(_gold_dir().glob("*.json")):
        item = gold_eval.load_gold_item(path)
        view = gold_eval.gold_parse_view(item)
        score = gold_eval.score_prediction(item=item, prediction=view, replica=0)
        assert score.errors == []
        assert score.gate_match
        assert score.structured_accuracy == 1.0
        assert score.segmentation_exact_match == 1.0
        assert score.class_ari == 1.0
        assert score.template_exact_match == 1.0
        assert score.gloss_literal_agreement in (None, 1.0)


def test_opaque_index_over_segmentation_is_penalized() -> None:
    """Splitting P1 into P|1 costs the split/segmentation metrics and nothing else."""
    item = gold_eval.load_gold_item(_gold_dir() / "opaque_index_vs_literal_count.json")
    view = gold_eval.gold_parse_view(item)
    over = view.model_copy(deep=True)
    over.segmentation["P1"] = ["P", "1"]
    over.class_by_occurrence["P1@0:1"] = "OPERATOR"
    over.class_by_occurrence["P1@1:2"] = "COUNT"
    over.gloss_by_occurrence["P1@0:1"] = "repeat the previous step"
    over.gloss_by_occurrence["P1@1:2"] = "one time"
    over.construction_by_symbol["P1"] = "OPERATOR__COUNT"

    score = gold_eval.score_prediction(item=item, prediction=over, replica=0)

    assert score.structured_accuracy < 1.0
    assert score.segmentation_exact_match < 1.0
    assert score.boundary_precision < 1.0
    assert score.boundary_recall == 1.0
    assert score.construction_ari < 1.0
    assert score.template_exact_match == 1.0
    # The invented P|1 morphemes are not in gold at all, so they cannot produce a
    # gloss disagreement; they are penalized by the split/boundary/clustering metrics.
    assert {error.kind for error in score.errors} == {"segmentation"}
    assert score.gloss_literal_agreement == 1.0


def test_class_scoring_is_invariant_to_class_naming() -> None:
    """Renaming every induced class leaves the class ARI at 1.0."""
    item = gold_eval.load_gold_item(_gold_dir() / "action_location_suffixing.json")
    view = gold_eval.gold_parse_view(item)
    aliases = {
        class_id: f"renamed_{index}"
        for index, class_id in enumerate(sorted(set(view.class_by_occurrence.values())))
    }
    assert len(aliases) == 2
    renamed = view.model_copy(deep=True)
    renamed.class_by_occurrence = {
        key: aliases[value] for key, value in view.class_by_occurrence.items()
    }
    renamed.construction_templates = {
        construction_id: [aliases[class_id] for class_id in template]
        for construction_id, template in view.construction_templates.items()
    }

    score = gold_eval.score_prediction(item=item, prediction=renamed, replica=0)

    assert score.class_ari == 1.0
    assert score.template_exact_match == 1.0


def test_merging_two_gold_classes_lowers_the_class_ari() -> None:
    """Collapsing ACTION and LOCATION into one class is scored as a real error."""
    item = gold_eval.load_gold_item(_gold_dir() / "action_location_suffixing.json")
    view = gold_eval.gold_parse_view(item)
    merged = view.model_copy(deep=True)
    merged.class_by_occurrence = {key: "pos0" for key in view.class_by_occurrence}

    score = gold_eval.score_prediction(item=item, prediction=merged, replica=0)

    assert score.class_ari < 1.0
    assert score.segmentation_exact_match == 1.0


def test_gate_flattens_a_single_structured_prediction() -> None:
    """One segmented code is below the gate, so downstream and scoring see a flat lexicon."""
    item = gold_eval.load_gold_item(_gold_dir() / "mnemonic_abbreviation_flat.json")
    grammar = assemble_grammar(
        run_key="gold/mnemonic_abbreviation_flat",
        defined=frozenset(item.codebook),
        segmentation={
            symbol: ["FAN", "2"] if symbol == "FAN2" else [symbol]
            for symbol in item.codebook
        },
        head_index={symbol: 0 for symbol in item.codebook},
        gloss_by_form={},
        form_positions={},
        position_labels={},
        positions_by_symbol={"FAN2": [0, 1]},
        position_glosses={},
        meaning_by_symbol=dict(item.codebook),
    )
    assert gold_eval.gate_fired(grammar.segmentation)

    view = gold_eval.build_parse_view(
        symbols=sorted(item.codebook),
        grammar=grammar,
        analysis=None,
        below_gate=True,
    )
    score = gold_eval.score_prediction(item=item, prediction=view, replica=0)

    assert view.segmentation["FAN2"] == ["FAN2"]
    assert score.gate_match
    assert score.structured_accuracy == 1.0
    assert score.errors == []
