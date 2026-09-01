"""Assemble the three tidy dataframes for downstream statistical analysis of the
encode->decode (and decode-only) morphology batteries: item-level, construction-level,
and run-level.

Item-level is the base table -- one row per tested cell, everything needed to average
across slots, constructions, or pipelines without a join. Construction-level rolls that up
to one row per (run, construction), carrying native/pooled capacity and grid/effective
saturation (reusing ``paradigm_saturation``'s existing machinery rather than recomputing
it). Run-level rolls construction-level up to one row per run, plus the run's scenario
config, denormalized into every table so none of the three needs a join for basic
filtering or grouping (e.g. "constructions per run per model").
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from morphology.pipeline.item_scores import (
    ENCODE_DECODE_PIPELINE,
    ItemScore,
    read_item_scores,
)
from morphology.pipeline.paradigm_cache import load_joint_paradigm
from morphology.analysis.paradigm_saturation import (
    CONTROL_PURPOSES,
    EffectiveSaturation,
    construction_saturations,
    effective_saturations,
)
from morphology.pipeline.role_pooled_stimuli import RoleExpansion

import morphology.common as C

logger = logging.getLogger(__name__)

def _run_config(run_dir_name: str) -> dict:
    """Read the scenario config + model/provider fields to denormalize into every table."""
    path = C.RUNS / "veyru" / run_dir_name / "run_summary_cache.json"
    if not path.exists():
        return {"model": "unknown", "provider": "unknown",
                "round_time_budget_seconds": None, "seed": None, "round_count": None}
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data.get("scenario_config", {})
    models = {m for m in data.get("models", []) if m}
    model = next(iter(models)) if len(models) == 1 else "|".join(sorted(models))
    return {
        "model": model,
        "provider": data.get("provider", "unknown"),
        "round_time_budget_seconds": config.get("round_time_budget_seconds"),
        "seed": config.get("seed"),
        "round_count": config.get("round_count"),
    }


def _load_run_items(run_key: str) -> list[ItemScore]:
    """Load the per-item scores shipped for a run (decode battery and/or encode->decode)."""
    _, run_dir_name = run_key.split("/", 1)
    items: list[ItemScore] = []
    for filename in ("decode_items.jsonl", "items.jsonl"):
        path = C.ENCODE_DECODE / run_dir_name / filename
        if path.exists():
            items += read_item_scores(path)
    return items


def build_item_table(run_keys: list[str]) -> pd.DataFrame:
    """One row per tested item, with run config denormalized onto every row."""
    rows: list[dict] = []
    for run_key in run_keys:
        scenario, run_dir_name = run_key.split("/", 1)
        config = _run_config(run_dir_name)
        for item in _load_run_items(run_key):
            row = item.model_dump()
            row.update(config)
            row["is_attested"] = item.purpose in CONTROL_PURPOSES
            row["n_slots_total"] = len(item.expected_slots)
            row["n_slots_pooled"] = len(item.pooled_slots)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_construction_table(
    run_keys: list[str], runs_root: Path, cache_dir: Path, joint_model: str,
) -> pd.DataFrame:
    """One row per (run, construction): capacity, grid saturation, effective saturation.

    Reuses ``construction_saturations`` (grid: attested / licensed product) and
    ``effective_saturations`` (credits the unattested remainder at the novel-cell success
    rate) so this table's saturation numbers are computed the same way as everywhere else
    in the codebase, not a parallel reimplementation. Pooled capacity additionally folds in
    ``expand_slot_fillers_by_role``'s cached output, so ``pooled_capacity`` reflects what
    the grounded battery could actually reach with cross-construction substitution.
    """
    rows: list[dict] = []
    for run_key in run_keys:
        scenario, run_dir_name = run_key.split("/", 1)
        run_dir = runs_root / scenario / run_dir_name
        config = _run_config(run_dir_name)
        joint_result = load_joint_paradigm(
            run_key=run_key, run_dir=run_dir, cache_dir=cache_dir,
            codebook_model="gpt-5.4", joint_model=joint_model,
        )
        grids = {
            row.construction_id: row
            for row in construction_saturations(joint_result, run_key)
        }
        items = _load_run_items(run_key)
        effective = effective_saturations(list(grids.values()), items)
        effective_by_cx: dict[str, dict[str, EffectiveSaturation]] = {}
        for row in effective:
            if row.pipeline != ENCODE_DECODE_PIPELINE:
                continue
            effective_by_cx.setdefault(row.construction_id, {})[row.metric] = row

        role_expansion = _cached_role_expansion(run_dir_name, cache_dir, joint_model)
        slot_by_id = {s["slot_id"]: s for s in joint_result.analysis.model_dump()["slots"]}

        for construction_id, template in joint_result.analysis.construction_templates.items():
            grid = grids.get(construction_id)
            pooled_capacity = 1
            for slot_id in template:
                slot = slot_by_id.get(slot_id)
                if slot is None:
                    pooled_capacity = None
                    break
                n_native = max(len(slot["fillers"]), 1)
                n_pooled = len(role_expansion.per_slot.get(slot_id, []))
                pooled_capacity *= n_native + n_pooled

            construction_items = [i for i in items if i.construction_id == construction_id
                                   and i.pipeline == ENCODE_DECODE_PIPELINE]
            n_control = sum(1 for i in construction_items if i.purpose in CONTROL_PURPOSES)
            n_novel = sum(1 for i in construction_items if i.purpose not in CONTROL_PURPOSES)
            n_novel_pooled = sum(
                1 for i in construction_items
                if i.purpose not in CONTROL_PURPOSES and i.pooled_slots
            )

            row: dict = {
                **config,
                "run_key": run_key,
                "construction_id": construction_id,
                "construction_label": joint_result.analysis.construction_labels.get(
                    construction_id, construction_id
                ),
                "n_slots": len(template),
                "native_capacity": grid.possible if grid else None,
                "pooled_capacity": pooled_capacity,
                "n_attested_codebook": grid.attested if grid else None,
                "grid_saturation": grid.saturation if grid else None,
                "n_control_tested": n_control,
                "n_novel_tested": n_novel,
                "n_novel_pooled": n_novel_pooled,
            }
            for metric in ("exact", "bag", "decodability"):
                eff = effective_by_cx.get(construction_id, {}).get(metric)
                row[f"novel_success_{metric}"] = eff.novel_success if eff else None
                row[f"effective_saturation_{metric}"] = eff.effective if eff else None
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _cached_role_expansion(
    run_dir_name: str, cache_dir: Path, role_pool_model: str
) -> RoleExpansion:
    """Read a role-expansion cache file without paying for a miss (construction table is
    read-only over data the battery run already produced)."""
    safe_model = "".join(ch if ch.isalnum() else "-" for ch in role_pool_model)
    cache = cache_dir / f"role_expansion_{run_dir_name}_{safe_model}.json"
    if not cache.exists():
        return RoleExpansion(per_slot={}, stackable=[])
    payload = json.loads(cache.read_text(encoding="utf-8"))
    return RoleExpansion(
        per_slot={
            sid: [tuple(pair) for pair in pairs] for sid, pairs in payload["per_slot"].items()
        },
        stackable=[tuple(pair) for pair in payload["stackable"]],
    )


def build_run_table(item_df: pd.DataFrame, construction_df: pd.DataFrame) -> pd.DataFrame:
    """One row per run: config, gate status, and item-weighted + construction-weighted rates."""
    if construction_df.empty:
        return pd.DataFrame()
    config_cols = [
        "run_key", "model", "provider", "round_time_budget_seconds", "seed", "round_count",
    ]
    base = construction_df[config_cols].drop_duplicates(subset="run_key").set_index("run_key")

    cx_agg = construction_df.groupby("run_key").agg(
        n_constructions=("construction_id", "count"),
        n_controls_tested=("n_control_tested", "sum"),
        n_novel_tested=("n_novel_tested", "sum"),
        n_novel_pooled=("n_novel_pooled", "sum"),
        grid_saturation_mean=("grid_saturation", "mean"),
        grid_saturation_median=("grid_saturation", "median"),
        effective_saturation_decode_mean=("effective_saturation_decodability", "mean"),
        cx_weighted_exact_rate=("novel_success_exact", "mean"),
        cx_weighted_decode_rate=("novel_success_decodability", "mean"),
    )

    ed = item_df[item_df["pipeline"] == ENCODE_DECODE_PIPELINE]
    item_agg = ed.groupby("run_key").agg(
        item_weighted_exact_rate=("exact_match", "mean"),
        item_weighted_bag_rate=("bag_match", "mean"),
        item_weighted_decode_rate=("decodability", "mean"),
        item_weighted_valid_rate=("is_valid", "mean"),
    )

    out = base.join(cx_agg, how="left").join(item_agg, how="left").reset_index()
    return out


def main() -> None:
    """Write the item, construction and run tables the figures and statistics read."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=C.RUNS,
                        help="the shipped event logs; also used when an induction cache misses")
    parser.add_argument("--cache-dir", type=Path, default=C.CACHE)
    parser.add_argument("--joint-model", default="gpt-5.4")
    args = parser.parse_args()

    manifest = C.load_manifest()
    scenario = manifest["scenario"]
    run_keys = [f"{scenario}/{run_id}" for run_id in manifest["tested_run_ids"]]

    item_df = build_item_table(run_keys)
    construction_df = build_construction_table(
        run_keys=run_keys, runs_root=args.runs_root,
        cache_dir=args.cache_dir, joint_model=args.joint_model,
    )
    run_df = build_run_table(item_df=item_df, construction_df=construction_df)

    for frame, name in ((run_df, "run_table.csv"),
                        (construction_df, "construction_table.csv"),
                        (item_df, "item_table.csv")):
        out = C.CSV / name
        frame.to_csv(out, index=False)
        print(f"wrote {out.relative_to(C.ROOT)}  ({len(frame)} rows)")


if __name__ == "__main__":
    main()
