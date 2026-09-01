"""Shared paradigm loading for the decode and encode->decode morphology pipelines.

Both drivers need the same prefix: extract the negotiated codebook, induce the
run-global joint paradigm (segmentation + classes + construction templates), apply the
degradation gate (a flat lexicon is not worth eliciting), and pool same-role fillers for
wug stimuli. This module owns that prefix behind one public ``load_run_paradigm`` plus
``read_run_metadata`` so neither driver imports the other's private helpers.
"""

import json
import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from morphology.pipeline.joint_paradigm_induction import (
    JointParadigmResult,
    induce_joint_paradigm_llm,
)
from morphology.pipeline.codebook import extract_codebook, find_run_jsonl, gather_postmortem_messages
from morphology.pipeline.morphological_grammar import MorphologicalGrammar
from morphology.pipeline.paradigm_network import SlotAnalysis
from morphology.pipeline.role_pooled_stimuli import RoleExpansion, expand_slot_fillers_by_role

logger = logging.getLogger(__name__)

# The joint validator requires every construction to hold a slot with two or more
# contrasting fillers, so a single minimal pair is sufficient evidence there.
MIN_JOINT_GLOBAL_CODES = 2


class RunMetadata(NamedTuple):
    """Communication budget and the agent's own model, read from the run summary."""

    context_budget: int | None
    model: str


class LoadedParadigm(NamedTuple):
    """The run-global grammar/paradigm plus role expansion and the gate decision.

    ``analysis`` and ``role_expansion`` are ``None`` when ``below_gate`` is True: a flat
    lexicon short-circuits before paradigm-network induction or role pooling, so those
    paid calls are never made. Callers must check ``below_gate`` before using them.
    """

    grammar: MorphologicalGrammar
    analysis: SlotAnalysis | None
    role_expansion: RoleExpansion | None
    n_multimorpheme: int
    below_gate: bool


def read_run_metadata(run_dir: Path, agent_id: str) -> RunMetadata:
    """Read communication budget and the agent's model from the run summary cache."""
    path = run_dir / "run_summary_cache.json"
    if not path.exists():
        return RunMetadata(context_budget=None, model="unknown")
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data.get("scenario_config", {})
    budget = config.get("round_time_budget_seconds")
    model = next(
        (
            item.get("model", "unknown")
            for item in data.get("agent_models", [])
            if item.get("agent_id") == agent_id
        ),
        "unknown",
    )
    return RunMetadata(
        context_budget=int(budget) if budget is not None else None, model=str(model)
    )


def load_codebook(
    *, run_key: str, run_dir: Path, cache_dir: Path, model: str
) -> pd.DataFrame:
    """Load or extract the meaning-bearing codebook shared by both induction paths.

    Cached per (run, model): the cache filename used to omit the model entirely, so
    calling this with a different ``model`` silently reused whatever extraction was
    cached first instead of re-extracting -- a real run/model mismatch bug, not just a
    missed cache hit. Every other cache in this module (joint paradigm, role expansion)
    already keys on model; this one now does too.
    """
    run_dir_name = run_key.split("/", 1)[-1]
    safe_model = "".join(ch if ch.isalnum() else "-" for ch in model)
    cache = cache_dir / f"cb_{run_dir_name}_{safe_model}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    jsonl_path = find_run_jsonl(run_dir)
    if jsonl_path is None:
        raise FileNotFoundError(f"no run JSONL found under {run_dir}")
    messages = gather_postmortem_messages(jsonl_path)
    entries = extract_codebook(messages=messages, model=model)
    codebook = pd.DataFrame(
        [
            {
                "run_key": run_key,
                "symbol": entry.symbol,
                "meaning": entry.meaning,
                "round_introduced": entry.round_introduced,
            }
            for entry in entries
        ],
        columns=["run_key", "symbol", "meaning", "round_introduced"],
    )
    codebook.to_parquet(cache)
    return codebook


def load_joint_paradigm(
    *,
    run_key: str,
    run_dir: Path,
    cache_dir: Path,
    codebook_model: str,
    joint_model: str,
) -> JointParadigmResult:
    """Load or jointly induce segmentation, classes, and constructions in one call."""
    run_dir_name = run_key.split("/", 1)[-1]
    safe_model = "".join(ch if ch.isalnum() else "-" for ch in joint_model)
    cache = cache_dir / f"joint_paradigm_{run_dir_name}_{safe_model}.json"
    if cache.exists():
        result = JointParadigmResult.model_validate_json(cache.read_text(encoding="utf-8"))
    else:
        codebook = load_codebook(
            run_key=run_key,
            run_dir=run_dir,
            cache_dir=cache_dir,
            model=codebook_model,
        )
        jsonl_path = find_run_jsonl(run_dir)
        source_messages = (
            gather_postmortem_messages(jsonl_path) if jsonl_path is not None else []
        )
        evidence = [
            {
                "round": message.round_number,
                "sender": message.sender,
                "text": message.text,
            }
            for message in source_messages
        ]
        result = induce_joint_paradigm_llm(
            run_key=run_key,
            codebook=codebook,
            model=joint_model,
            negotiation_evidence=evidence,
        )
        cache.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def _load_role_expansion(
    *,
    run_dir_name: str,
    cache_dir: Path,
    grammar: MorphologicalGrammar,
    analysis: SlotAnalysis,
    role_pool_model: str,
) -> RoleExpansion:
    """Load or compute role-pooled filler expansion, caching the paid call.

    ``RoleExpansion`` is a plain NamedTuple, not a Pydantic model, so the cache round-trips
    it through a small JSON dict of lists (JSON has no tuple type -- pairs come back as
    2-element lists and are re-tupled on load).
    """
    safe_model = "".join(ch if ch.isalnum() else "-" for ch in role_pool_model)
    cache = (
        cache_dir
        / f"role_expansion_{run_dir_name}_{safe_model}.json"
    )
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        return RoleExpansion(
            per_slot={
                slot_id: [tuple(pair) for pair in pairs]
                for slot_id, pairs in payload["per_slot"].items()
            },
            stackable=[tuple(pair) for pair in payload["stackable"]],
        )
    result = expand_slot_fillers_by_role(grammar, analysis, model=role_pool_model)
    cache.write_text(
        json.dumps({"per_slot": result.per_slot, "stackable": result.stackable}, indent=2),
        encoding="utf-8",
    )
    return result


def load_run_paradigm(
    *,
    run_key: str,
    runs_root: Path,
    cache_dir: Path,
    model: str,
    joint_model: str,
    role_pool_model: str | None,
) -> LoadedParadigm:
    """Load the run-global grammar/paradigm, apply the degradation gate, pool fillers.

    The single-step codebook-to-paradigm inducer supplies both the grammar and the
    paradigm network. A run with fewer than the minimum multi-morpheme codes is flagged
    ``below_gate`` and no paradigm-network or role-pooling call is issued.
    """
    scenario, run_dir_name = run_key.split("/", 1)
    run_dir = runs_root / scenario / run_dir_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    joint_result = load_joint_paradigm(
        run_key=run_key,
        run_dir=run_dir,
        cache_dir=cache_dir,
        codebook_model=model,
        joint_model=joint_model,
    )
    grammar = joint_result.grammar
    n_multi = sum(1 for parts in grammar.segmentation.values() if len(parts) >= 2)
    if n_multi < MIN_JOINT_GLOBAL_CODES:
        return LoadedParadigm(
            grammar=grammar,
            analysis=None,
            role_expansion=None,
            n_multimorpheme=n_multi,
            below_gate=True,
        )
    analysis = joint_result.analysis
    role_expansion = (
        _load_role_expansion(
            run_dir_name=run_dir_name,
            cache_dir=cache_dir,
            grammar=grammar,
            analysis=analysis,
            role_pool_model=role_pool_model,
        )
        if role_pool_model is not None
        else RoleExpansion(per_slot={}, stackable=[])
    )
    return LoadedParadigm(
        grammar=grammar,
        analysis=analysis,
        role_expansion=role_expansion,
        n_multimorpheme=n_multi,
        below_gate=False,
    )
