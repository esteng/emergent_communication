"""Replica-robustness measurement for the induction and encode-decode pipeline stages.

Each stage is re-run N times, uncached, on the same frozen input (codebook for
induction; seeded stimuli for encode-decode), and agreement between replicas is
scored pairwise. A third axis -- judge self-consistency -- reruns the SAME judge
call K times on already-generated items to separate generation noise from judge
noise, mirroring ``protocol_probe_replica_self_similarity_metric``'s within-group
replica-agreement design for a different judge.

Outputs are cached under a dedicated ``robustness_cache`` subdirectory so nothing
here touches or invalidates the production caches in ``paradigm_cache.py``.
"""

import argparse
import asyncio
import itertools
import json
import logging
from pathlib import Path
from typing import Any

from sklearn.metrics import adjusted_rand_score

from morphology.pipeline.grounded_encode_stimuli import build_construction_wugs, mine_argument_inventory
from morphology.pipeline.joint_paradigm_induction import induce_joint_paradigm_llm
from morphology.pipeline.paradigm_encode_decode_batch import run_encode_decode_batch
from morphology.pipeline.paradigm_cache import (
    load_codebook,
    load_run_paradigm,
)
from morphology.pipeline.semantic_equivalence_judge import (
    DecodeItem,
    EquivItem,
    JudgeOutcome,
    run_semantic_judge,
)
from morphology.robustness.agreement import kappa, ordinal_labels, safe_mean
from morphology.pipeline.semantic_equivalence_judge import equivalence_rate

logger = logging.getLogger(__name__)

_SAFE_CHARS = "".join


def _safe_model(model: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in model)






# ---------------------------------------------------------------------------
# Induction: generation replicas
# ---------------------------------------------------------------------------


def _load_production_induction_replica(
    run_key: str, cache_dir: Path, joint_model: str
) -> dict[str, Any] | None:
    """Load the existing production joint-paradigm cache as a free replica, if present.

    Same schema as what this script writes (``JointParadigmResult.model_dump_json()``),
    so it drops in directly -- no adapter needed. Returns None if no such cache exists.
    """
    run_dir_name = run_key.split("/", 1)[-1]
    path = (
        cache_dir
        / f"joint_paradigm_{run_dir_name}_{_safe_model(joint_model)}.json"
    )
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_induction_replicas(
    run_key: str,
    runs_root: Path,
    cache_dir: Path,
    codebook_model: str,
    joint_model: str,
    n_replicas: int,
    reuse_production: bool,
) -> list[dict[str, Any]]:
    """Induce the joint paradigm until ``n_replicas`` are collected, uncached, persisted.

    When ``reuse_production`` is set, the existing production joint-paradigm cache
    (from ordinary, non-robustness pipeline runs) counts as the first replica for
    free, and only ``n_replicas - 1`` new inductions are made.
    """
    scenario, run_dir_name = run_key.split("/", 1)
    run_dir = runs_root / scenario / run_dir_name
    safe_key = run_key.replace("/", "_")
    robustness_dir = cache_dir / "robustness_cache"
    robustness_dir.mkdir(parents=True, exist_ok=True)

    replicas: list[dict[str, Any]] = []
    if reuse_production:
        production = _load_production_induction_replica(
            run_key=run_key, cache_dir=cache_dir, joint_model=joint_model
        )
        if production is not None:
            logger.info("%s: reusing production induction cache as replica 0", run_key)
            replicas.append(production)

    n_new = n_replicas - len(replicas)
    if n_new <= 0:
        return replicas

    codebook = load_codebook(
        run_key=run_key,
        run_dir=run_dir,
        cache_dir=cache_dir,
        model=codebook_model,
    )
    start_index = len(replicas)
    for index in range(start_index, start_index + n_new):
        out_path = (
            robustness_dir
            / f"induction_r{index}_{safe_key}_{_safe_model(joint_model)}.json"
        )
        if out_path.exists():
            result = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            outcome = induce_joint_paradigm_llm(
                run_key=run_key, codebook=codebook, model=joint_model
            )
            result = json.loads(outcome.model_dump_json())
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        replicas.append(result)
    return replicas


def _symbol_clusters(reply: dict[str, Any]) -> dict[str, str]:
    """Map every symbol to its construction_id, or the shared label 'LEXICAL'."""
    clusters = {symbol: "LEXICAL" for symbol in reply["lexical_symbols"]}
    for entry in reply["entries"]:
        clusters[entry["symbol"]] = entry["construction_id"]
    return clusters


def _morpheme_glosses(grammar: dict[str, Any]) -> dict[str, str]:
    """Flatten position_glosses (per-position {morpheme: gloss}) into one map.

    A morpheme surface form appearing at more than one position with different
    glosses is not disambiguated here -- the last position wins. Not observed in
    practice; accepted as a simplification for this analysis.
    """
    flattened: dict[str, str] = {}
    for position_glosses in grammar["position_glosses"].values():
        flattened.update(position_glosses)
    return flattened


def score_induction_replicas(
    replicas: list[dict[str, Any]], judge_model: str
) -> dict[str, Any]:
    """Pairwise agreement across induction replicas: segmentation, ARI, gloss equivalence."""
    segmentation_matches: list[float] = []
    aris: list[float] = []
    gloss_disagreement_pairs: list[tuple[str, str]] = []
    n_gloss_common = 0
    for replica_a, replica_b in itertools.combinations(replicas, 2):
        seg_a, seg_b = replica_a["grammar"]["segmentation"], replica_b["grammar"]["segmentation"]
        common_symbols = set(seg_a) & set(seg_b)
        if common_symbols:
            segmentation_matches.append(
                sum(seg_a[symbol] == seg_b[symbol] for symbol in common_symbols)
                / len(common_symbols)
            )
        clusters_a = _symbol_clusters(replica_a["reply"])
        clusters_b = _symbol_clusters(replica_b["reply"])
        common_clustered = sorted(set(clusters_a) & set(clusters_b))
        if len(common_clustered) >= 2:
            aris.append(
                adjusted_rand_score(
                    [clusters_a[symbol] for symbol in common_clustered],
                    [clusters_b[symbol] for symbol in common_clustered],
                )
            )
        glosses_a = _morpheme_glosses(replica_a["grammar"])
        glosses_b = _morpheme_glosses(replica_b["grammar"])
        common_morphemes = set(glosses_a) & set(glosses_b)
        n_gloss_common += len(common_morphemes)
        for morpheme in common_morphemes:
            if glosses_a[morpheme] != glosses_b[morpheme]:
                gloss_disagreement_pairs.append((glosses_a[morpheme], glosses_b[morpheme]))
    gloss_equivalence_rate = asyncio.run(
        equivalence_rate(gloss_disagreement_pairs, judge_model)
    )
    literal_gloss_agreement = (
        1.0 - len(gloss_disagreement_pairs) / n_gloss_common if n_gloss_common else None
    )
    return {
        "n_replicas": len(replicas),
        "n_replica_pairs": len(segmentation_matches),
        "segmentation_exact_match_mean": safe_mean(segmentation_matches),
        "construction_ari_mean": safe_mean(aris),
        "n_common_morphemes_checked": n_gloss_common,
        "literal_gloss_agreement_rate": literal_gloss_agreement,
        "n_gloss_disagreements_sent_to_judge": len(gloss_disagreement_pairs),
        "gloss_semantic_equivalence_rate": gloss_equivalence_rate,
    }


# ---------------------------------------------------------------------------
# Encode-decode: generation replicas
# ---------------------------------------------------------------------------


def _load_production_encode_decode_replica(
    run_key: str, runs_root: Path
) -> dict[str, Any] | None:
    """Adapt the existing production ``encode_decode_items.jsonl`` into a free replica.

    That file uses ``item_scores.py``'s ItemScore field names (``reading``,
    ``exact_match``, ``bag_match``), not this module's ``EncodeDecodeResult`` names
    (``produced_reading``, ``match_expected``, ``match_bag``) -- same content, mapped
    here. Its item set may not exactly match a freshly-generated replica's (different
    original ``wugs_per_construction``/``length_gen`` settings); ``score_encode_decode_
    replicas`` already intersects on shared ``target_meaning`` keys, so a partial
    mismatch just narrows the comparison set rather than breaking it. Returns None if
    no such file exists.
    """
    scenario, run_dir_name = run_key.split("/", 1)
    items_path = runs_root / scenario / run_dir_name / "encode_decode_items.jsonl"
    if not items_path.exists():
        return None
    results = []
    with items_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            results.append(
                {
                    "target_meaning": row["target_meaning"],
                    "expected_form": row.get("expected_form"),
                    "expected_morphemes": None,
                    "expected_slots": row.get("expected_slots"),
                    "produced_form": row.get("produced_form"),
                    "produced_reading": row.get("reading"),
                    "purpose": row.get("purpose"),
                    "varied_slot": row.get("varied_slot"),
                    "varied_slot_label": row.get("varied_slot_label"),
                    "n_morphemes": row.get("n_morphemes"),
                    "construction_id": row.get("construction_id"),
                    "pooled_slots": row.get("pooled_slots"),
                    "is_valid": row.get("is_valid"),
                    "match_expected": row.get("exact_match"),
                    "match_bag": row.get("bag_match"),
                    "match_swapped_order": row.get("swapped_order"),
                    "decodability": row.get("decodability"),
                }
            )
    return {"results": results}


def run_encode_decode_replicas(
    run_key: str,
    runs_root: Path,
    cache_dir: Path,
    codebook_model: str,
    encoder_agent: str,
    decoder_agent: str,
    cutoff_round: int | None,
    wugs_per_construction: int,
    controls_per_construction: int,
    joint_model: str,
    role_pool_model: str,
    judge_model: str,
    n_replicas: int,
    reuse_production: bool,
) -> list[dict[str, Any]]:
    """Run the encode/decode battery until ``n_replicas`` are collected, on identical stimuli.

    When ``reuse_production`` is set, the existing production
    ``encode_decode_items.jsonl`` counts as the first replica for free, and only
    ``n_replicas - 1`` new battery runs are made.
    """
    scenario, run_dir_name = run_key.split("/", 1)
    run_dir = runs_root / scenario / run_dir_name

    replicas: list[dict[str, Any]] = []
    if reuse_production:
        production = _load_production_encode_decode_replica(run_key=run_key, runs_root=runs_root)
        if production is not None:
            logger.info("%s: reusing production encode-decode items as replica 0", run_key)
            replicas.append(production)

    n_new = n_replicas - len(replicas)
    if n_new <= 0:
        return replicas

    loaded = load_run_paradigm(
        run_key=run_key,
        runs_root=runs_root,
        cache_dir=cache_dir,
        model=codebook_model,
        joint_model=joint_model,
        role_pool_model=role_pool_model,
    )
    if loaded.below_gate:
        raise ValueError(
            f"{run_key} is below the flat-lexicon gate; not eligible for encode-decode robustness"
        )
    assert loaded.analysis is not None and loaded.role_expansion is not None
    inventory = mine_argument_inventory(run_dir=run_dir, grammar=loaded.grammar)
    stimuli = build_construction_wugs(
        analysis=loaded.analysis,
        inventory=inventory,
        wugs_per_construction=wugs_per_construction,
        controls_per_construction=controls_per_construction,
        seed=42,
        role_pooled_fillers=loaded.role_expansion.per_slot,
    )
    safe_key = run_key.replace("/", "_")
    robustness_dir = cache_dir / "robustness_cache"
    robustness_dir.mkdir(parents=True, exist_ok=True)
    start_index = len(replicas)
    for index in range(start_index, start_index + n_new):
        out_path = robustness_dir / f"encode_decode_r{index}_{safe_key}.json"
        if out_path.exists():
            result = json.loads(out_path.read_text(encoding="utf-8"))
        else:
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
            result = json.loads(outcome.model_dump_json())
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        replicas.append(result)
    return replicas


def score_encode_decode_replicas(
    replicas: list[dict[str, Any]], judge_model: str
) -> dict[str, Any]:
    """Pairwise agreement: Cohen's kappa on exact/bag/decodability, form + reading agreement."""
    exact_kappas: list[float] = []
    bag_kappas: list[float] = []
    decode_kappas: list[float] = []
    sender_exact_agree: list[bool] = []
    reading_disagreement_pairs: list[tuple[str, str]] = []
    n_reading_common = 0
    for replica_a, replica_b in itertools.combinations(replicas, 2):
        by_meaning_a = {row["target_meaning"]: row for row in replica_a["results"]}
        by_meaning_b = {row["target_meaning"]: row for row in replica_b["results"]}
        common = sorted(set(by_meaning_a) & set(by_meaning_b))
        if len(common) < 2:
            continue
        exact_kappas.append(
            kappa(
                [by_meaning_a[m]["match_expected"] for m in common],
                [by_meaning_b[m]["match_expected"] for m in common],
            )
        )
        bag_kappas.append(
            kappa(
                [by_meaning_a[m]["match_bag"] for m in common],
                [by_meaning_b[m]["match_bag"] for m in common],
            )
        )
        decode_kappas.append(
            kappa(
                ordinal_labels([by_meaning_a[m]["decodability"] for m in common]),
                ordinal_labels([by_meaning_b[m]["decodability"] for m in common]),
                weights="linear",
            )
        )
        for meaning in common:
            form_a = by_meaning_a[meaning]["produced_form"]
            form_b = by_meaning_b[meaning]["produced_form"]
            sender_exact_agree.append(form_a == form_b)
            reading_a = by_meaning_a[meaning]["produced_reading"]
            reading_b = by_meaning_b[meaning]["produced_reading"]
            if reading_a and reading_b:
                n_reading_common += 1
                if reading_a != reading_b:
                    reading_disagreement_pairs.append((reading_a, reading_b))
    reading_equivalence_rate = asyncio.run(
        equivalence_rate(reading_disagreement_pairs, judge_model)
    )
    return {
        "n_replicas": len(replicas),
        "n_replica_pairs": len(exact_kappas),
        "exact_match_kappa_mean": safe_mean(exact_kappas),
        "bag_match_kappa_mean": safe_mean(bag_kappas),
        "decodability_kappa_mean": safe_mean(decode_kappas),
        "sender_form_exact_agreement": safe_mean([float(x) for x in sender_exact_agree]),
        "n_common_readings_checked": n_reading_common,
        "n_reading_disagreements_sent_to_judge": len(reading_disagreement_pairs),
        "reading_semantic_equivalence_rate": reading_equivalence_rate,
    }


# ---------------------------------------------------------------------------
# Judge self-consistency: rerun the SAME judge call K times on frozen items
# ---------------------------------------------------------------------------


def _outcome_agreement(outcomes: list[JudgeOutcome], field: str) -> float | None:
    """Mean pairwise item-level RAW agreement across K reruns of the identical judge call.

    Uncorrected for chance -- if the judge's verdict distribution is skewed (e.g. mostly
    "same"), two independent reruns agree a lot just from both defaulting to the common
    answer. See _outcome_kappa for the chance-corrected companion.
    """
    sequences = [getattr(outcome, field) for outcome in outcomes]
    if not sequences or not sequences[0]:
        return None
    n_items = len(sequences[0])
    agreements = []
    for pair_a, pair_b in itertools.combinations(sequences, 2):
        agreements.append(
            sum(1 for i in range(n_items) if pair_a[i] == pair_b[i]) / n_items
        )
    return safe_mean(agreements)


def _outcome_kappa(outcomes: list[JudgeOutcome], field: str) -> float | None:
    """Mean pairwise chance-corrected agreement (Cohen's kappa) across K judge reruns.

    Both decode_scores and equivalence_scores are the judge's 0.0/0.5/1.0 ordinal scale
    (see _DECODE_SCORE / _EQUIV_SCORE in semantic_equivalence_judge.py), so this needs
    the same integer-label rescale as decodability_kappa in score_encode_decode_replicas
    -- sklearn's cohen_kappa_score rejects raw floats containing 0.5 as "continuous".
    """
    sequences = [getattr(outcome, field) for outcome in outcomes]
    if not sequences or not sequences[0]:
        return None
    kappas = [
        kappa(ordinal_labels(list(a)), ordinal_labels(list(b)), weights="linear")
        for a, b in itertools.combinations(sequences, 2)
    ]
    return safe_mean(kappas)


async def _judge_self_consistency(
    decode_items: list[DecodeItem],
    equiv_items: list[EquivItem],
    judge_model: str,
    k_replicas: int,
    raw_out_path: Path | None,
) -> dict[str, Any]:
    """Rerun one fixed judge call k_replicas times; measure raw + chance-corrected agreement.

    When raw_out_path is given, persists every rerun's raw decode_scores/equivalence_scores
    so a later true pooled kappa (across runs, not just within-run pairwise-averaged) can be
    computed without re-spending on the judge calls -- the earlier version of this function
    only kept the aggregate mean, which made retroactive pooling impossible.
    """
    if raw_out_path is not None and raw_out_path.exists():
        cached = json.loads(raw_out_path.read_text(encoding="utf-8"))
        outcomes = [
            JudgeOutcome(
                decode_scores=decode_scores,
                equivalence_scores=equivalence_scores,
                equivalence_same=[score == 1.0 for score in equivalence_scores],
            )
            for decode_scores, equivalence_scores in zip(
                cached["decode_scores"], cached["equivalence_scores"]
            )
        ]
    else:
        outcomes = [
            await run_semantic_judge(
                decode_items=decode_items, equiv_items=equiv_items, judge_model=judge_model
            )
            for _ in range(k_replicas)
        ]
        if raw_out_path is not None:
            raw_out_path.write_text(
                json.dumps(
                    {
                        "decode_scores": [list(o.decode_scores) for o in outcomes],
                        "equivalence_scores": [list(o.equivalence_scores) for o in outcomes],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    return {
        "k_replicas": k_replicas,
        "n_decode_items": len(decode_items),
        "n_equiv_items": len(equiv_items),
        "decode_verdict_agreement_mean": _outcome_agreement(outcomes, "decode_scores"),
        "decode_verdict_kappa_mean": _outcome_kappa(outcomes, "decode_scores"),
        "equivalence_verdict_agreement_mean": _outcome_agreement(outcomes, "equivalence_scores"),
        "equivalence_verdict_kappa_mean": _outcome_kappa(outcomes, "equivalence_scores"),
    }


def judge_self_consistency_for_induction(
    run_key: str,
    replicas: list[dict[str, Any]],
    judge_model: str,
    k_replicas: int,
    cache_dir: Path,
) -> dict[str, Any]:
    """Rerun the gloss-equivalence judge call from replica 0 vs replica 1, k times."""
    if len(replicas) < 2:
        return {"skipped": "fewer than 2 induction replicas"}
    glosses_a = _morpheme_glosses(replicas[0]["grammar"])
    glosses_b = _morpheme_glosses(replicas[1]["grammar"])
    common = set(glosses_a) & set(glosses_b)
    pairs = [
        (glosses_a[m], glosses_b[m]) for m in common if glosses_a[m] != glosses_b[m]
    ]
    equiv_items = [EquivItem(reading_a=a, reading_b=b) for a, b in pairs]
    safe_key = run_key.replace("/", "_")
    raw_out_path = (
        cache_dir / "robustness_cache" / f"judge_consistency_induction_{safe_key}.json"
    )
    return asyncio.run(
        _judge_self_consistency(
            decode_items=[],
            equiv_items=equiv_items,
            judge_model=judge_model,
            k_replicas=k_replicas,
            raw_out_path=raw_out_path,
        )
    )


def judge_self_consistency_for_encode_decode(
    run_key: str,
    replicas: list[dict[str, Any]],
    judge_model: str,
    k_replicas: int,
    cache_dir: Path,
) -> dict[str, Any]:
    """Rerun replica 0's decode + reading-equivalence judge call k times, unchanged."""
    if not replicas:
        return {"skipped": "no encode-decode replicas"}
    rows = replicas[0]["results"]
    decode_items = [
        DecodeItem(intended=row["target_meaning"], reading=row["produced_reading"])
        for row in rows
        if row["produced_reading"]
    ]
    equiv_items = []
    if len(replicas) >= 2:
        by_meaning_b = {row["target_meaning"]: row for row in replicas[1]["results"]}
        for row in rows:
            partner = by_meaning_b.get(row["target_meaning"])
            if partner and row["produced_reading"] and partner["produced_reading"]:
                equiv_items.append(
                    EquivItem(
                        reading_a=row["produced_reading"], reading_b=partner["produced_reading"]
                    )
                )
    safe_key = run_key.replace("/", "_")
    raw_out_path = (
        cache_dir / "robustness_cache" / f"judge_consistency_encode_decode_{safe_key}.json"
    )
    return asyncio.run(
        _judge_self_consistency(
            decode_items=decode_items,
            equiv_items=equiv_items,
            judge_model=judge_model,
            k_replicas=k_replicas,
            raw_out_path=raw_out_path,
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-keys", required=True, help="Comma-separated scenario/run_dir_name")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--codebook-model", default="gpt-5.4")
    parser.add_argument("--joint-model", default="gpt-5.4")
    parser.add_argument("--role-pool-model", default="gpt-5.4")
    parser.add_argument("--judge-model", default="claude-sonnet-4-6")
    parser.add_argument("--encoder-agent", default="stabilization_engineer")
    parser.add_argument("--decoder-agent", default="field_observer")
    parser.add_argument("--cutoff-round", type=int, default=None)
    parser.add_argument("--wugs-per-construction", type=int, default=30)
    parser.add_argument("--controls-per-construction", type=int, default=5)
    parser.add_argument("--n-replicas", type=int, default=3)
    parser.add_argument("--k-judge-replicas", type=int, default=3)
    parser.add_argument(
        "--no-reuse-production",
        action="store_true",
        help="Force all n_replicas to be freshly generated; skip loading an existing "
        "production induction cache / encode_decode_items.jsonl as a free replica 0.",
    )
    parser.add_argument("--out", default="pipeline_robustness_report.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    runs_root = Path(args.runs_root)
    cache_dir = Path(args.cache_dir)
    run_keys = [key.strip() for key in args.run_keys.split(",") if key.strip()]

    report: dict[str, Any] = {}
    for run_key in run_keys:
        logger.info("=== %s ===", run_key)
        induction_replicas = run_induction_replicas(
            run_key=run_key,
            runs_root=runs_root,
            cache_dir=cache_dir,
            codebook_model=args.codebook_model,
            joint_model=args.joint_model,
            n_replicas=args.n_replicas,
            reuse_production=not args.no_reuse_production,
        )
        induction_scores = score_induction_replicas(induction_replicas, args.judge_model)
        induction_judge_consistency = judge_self_consistency_for_induction(
            run_key, induction_replicas, args.judge_model, args.k_judge_replicas, cache_dir
        )

        encode_decode_replicas = run_encode_decode_replicas(
            run_key=run_key,
            runs_root=runs_root,
            cache_dir=cache_dir,
            codebook_model=args.codebook_model,
            encoder_agent=args.encoder_agent,
            decoder_agent=args.decoder_agent,
            cutoff_round=args.cutoff_round,
            wugs_per_construction=args.wugs_per_construction,
            controls_per_construction=args.controls_per_construction,
            joint_model=args.joint_model,
            role_pool_model=args.role_pool_model,
            judge_model=args.judge_model,
            n_replicas=args.n_replicas,
            reuse_production=not args.no_reuse_production,
        )
        encode_decode_scores = score_encode_decode_replicas(
            encode_decode_replicas, args.judge_model
        )
        encode_decode_judge_consistency = judge_self_consistency_for_encode_decode(
            run_key, encode_decode_replicas, args.judge_model, args.k_judge_replicas, cache_dir
        )

        report[run_key] = {
            "induction": induction_scores,
            "induction_judge_self_consistency": induction_judge_consistency,
            "encode_decode": encode_decode_scores,
            "encode_decode_judge_self_consistency": encode_decode_judge_consistency,
        }
        logger.info("%s: %s", run_key, json.dumps(report[run_key], indent=2))

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
