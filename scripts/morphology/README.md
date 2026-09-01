# Morphosyntactic Productivity

Reproduces the figures, tables and reported statistics of the morphology experiments
appendices.

```
pip install -r requirements.txt
export PYTHONPATH=scripts
python -m morphology.analysis.build_tidy_tables
```

Outputs are under `results/` (`figures/`, `tables/`, `csv/`).
Run `build_tidy_tables.py`, `make_emergence_runs.py` and `make_decode_production_items.py`
first; everything else reads their output or `data/` directly. Nothing in this section
needs an API key or the simulation corpus.

### Productive rules (`scripts/morphology/analysis/`)


- `build_tidy_tables.py`: prepares data for subsequent scripts — one table per item, per
  construction and per run.
- `make_emergence_runs.py`, `make_decode_production_items.py`: the two inputs the R scripts
  read.
- `plot_panel_emergence.py`: percentage of runs at each budget that develop a morphological
  inventory, with bootstrap CIs.
- `plot_panel_decode_production.py`: decode and production accuracy by construction, control
  against novel cells, exact match solid and any-order hatched.
- `plot_saturation.py`: per-construction grid saturation with and without semantic-role
  pooling — the appendix figure motivating why pooling is needed.
- `stat_decode_production.py`: Wilson intervals on the novel-cell decode and production
  rates; all six lower bounds are above zero.
- `emergence_paper_numbers.R`: the emergence model, and the source of the reported contrasts
  (pooled budget slope −0.452, Opus vs GPT 1.79, Sonnet vs GPT 1.18, Opus vs Sonnet 0.610).
  Fits both the mixed-effects model the paper describes and the plain GLM, and prints them
  side by side: the seed random effect comes out singular with a variance of 0, so the two
  are numerically identical and the GLM is what the contrasts are read off.
- `decode_production.R`: the decode and production models, including that decode accuracy
  does not differ across models (all pairwise p>0.8).

`paradigm_saturation.py` and `item_table_view.py` are the supporting library: grid and
effective saturation, and the row filter plus control/novel split that the figure and the
intervals beside it must share. Both R scripts need `lme4` and `emmeans`.

### Robustness and accuracy (`scripts/morphology/robustness/`)


- `pool_robustness.py`: the pipeline-robustness statistics, pooled across the five re-run
  roots rather than averaged per run — every replica pair contributes its item-level
  observations to one confusion matrix before the statistic is computed once.
- `judge_agreement.py`: the semantic judge against a blind human annotator, all 150
  judgments pooled — 124/150 exact, linear-weighted κ=0.642, and 16 of the 26 mismatches
  graded more harshly by the judge.
- `gold_paradigm_eval.py`: scores the induction against ten hand-built codebooks whose
  paradigms are known by construction, giving the reported production accuracy (98.9%),
  segmentation accuracy (98.2%) and semantic agreement (92.9%).
- `pipeline_robustness.py`: re-runs induction and encode→decode N times on the same frozen
  input to measure replica agreement, with judge self-consistency as a separate axis so
  generation noise and judge noise stay distinguishable.

`agreement.py` is the supporting library: the kappa conventions and the ordinal rescaling
they need, shared so the same treatment applies whether the raters are two induction
replicas, two reruns of the judge, or the judge and a human. The last two scripts reach an
LLM through the pipeline they re-run (see below); the first two read cached output and do
not.

### The elicitation pipeline (`scripts/morphology/pipeline/`)

The apparatus that produced the data the sections above read. It reaches back into agents
frozen at a chosen round, asks them to coin and to read novel forms, and judges the result.
Six of its modules issue an LLM call; the rest is schema, scoring and cache handling that
runs without credentials.

- `codebook.py`: reads a run's event log and recovers the negotiated codebook — an LLM
  extraction, corrected by a deterministic parser wherever the agents wrote an explicit
  `SYMBOL=meaning` definition.
- `joint_paradigm_induction.py`: segments the codes and assigns them to substitution classes
  in one pass, yielding a `MorphologicalGrammar` and a `SlotAnalysis`.
- `paradigm_cache.py`: loads all of the above, cache first, and applies the productivity
  gate that demotes a flat lexicon before anything is elicited.
- `role_pooled_stimuli.py`, `grounded_encode_stimuli.py`, `novel_cell_stimuli.py`: derive the
  cells to test, including the cross-construction role pooling that makes a saturated slot
  testable. Pooling is the only one of the three that needs a model to build a stimulus; the
  other two mine the transcript. `novel_cell_stimuli.py` also carries the decode call, which
  reads the cells back rather than building them.
- `paradigm_encode_decode_batch.py`: the encode→decode round trip against the two frozen
  agents.
- `semantic_equivalence_judge.py`: scores the receiver's reading against the meaning the
  sender was given, never seeing the codes themselves.
- `run_encode_decode_pipeline.py`: the entry point, reducing one run into a report.

`morphological_grammar.py` and `paradigm_network.py` are the two shared descriptions a run's
morphology is expressed in — schema, validation and projection only, no calls.
`analysis_mode.py` is how a frozen agent is reached.

```
python -m morphology.pipeline.run_encode_decode_pipeline \
  --run-key veyru/1778877868 --runs-root <corpus> --cache-dir <cache> --max-cells 2
```

Reconstructing a frozen agent's thread is delegated to the GlossoGen simulation platform, a
separate codebase; `analysis_mode.py` imports it lazily, so every other module here runs
without it.

## LLM scripts

Backend selection is environment-driven — there is no provider flag. Claude models route to
Portkey whenever `PORTKEY_API_KEY` is set, since Azure cannot serve them. Otherwise: Azure
when `AZURE_ENDPOINT` and `AZURE_API_KEY` are both set, else Portkey, else the OpenAI SDK
reading `OPENAI_API_KEY`.

Below shows the files which call an LLM, in the order the pipeline passes through them.
Every call goes through `call_structured` in `analysis_mode.py` against a Pydantic output schema.

| Stage | Site | Prompt |
|---|---|---|
| codebook extraction | `codebook.py` | `codebook_extraction.txt` |
| paradigm induction | `joint_paradigm_induction.py` | `joint_paradigm_induction.txt` |
| role pooling | `role_pooled_stimuli.py` | `role_pooling.txt` |
| encode | `paradigm_encode_decode_batch.py` | `encode_batch.txt` |
| decode | `novel_cell_stimuli.py` | `decode_batch.txt` |
| semantic judge | `semantic_equivalence_judge.py` | `semantic_judge.txt` |

`analysis_mode.py` is the shared client these files issue their calls through.

Four further modules need credentials without issuing calls of their own.
`run_encode_decode_pipeline.py` drives the stages above. `paradigm_cache.py` reaches three
of them — codebook, induction and role pooling — when there is no cached data.
In `robustness/`, `pipeline_robustness.py` and `gold_paradigm_eval.py` likewise 
all through the pipeline rather than directly.

`morphological_grammar.py`, `paradigm_network.py`, `grounded_encode_stimuli.py` and
`item_scores.py` never make calls, and neither does the stimulus-building half of
`novel_cell_stimuli.py`.

Their outputs are cached in `data/morphology/`, and everything downstream reads the cache,
so the reported statistics are reproducible without re-running any of them. Re-running
produces different numbers; these are LLM calls.

The prompts are the files in `data/morphology/prompts/`, loaded at call time rather than
embedded in the source.

## Data

Exported from the GlossoGen simulation platform, which is a separate codebase; these exports
are this release's entry point. `scripts/make_slim_morphology_data.py` records how each file
was cut down, and which steps are reductions rather than straight copies.

```
data/morphology/
  transcripts/           the 135 induced roots' event logs, verbatim — the transcripts and
                         postmortems everything else is derived from
  cache/                 the induction caches, one generation, under unversioned names
  encode_decode/         per tested root: report, scored items, raw model responses
  robustness/            cached replica outputs, plus the per-run report the pooled
                         statistics are weighted by
  judge_validation/      one flat 150-row human-vs-judge table, and the rubric both raters
                         scored against
  prompts/               the six prompts the pipeline issues
  induction_results.jsonl   gate outcome per root, for all 135 rather than the 43 tested
  run_manifest.json      the 135 induced and 43 tested run ids
  gold_paradigm_accuracy.csv   the reported pipeline-accuracy values; `gold_paradigm_eval.py`
                         re-derives them into `results/tables/`, so compare rather than replace
```

## Tests

```
python -m pytest tests/
```

This runs the eleven offline regression tests over joint induction, the cache loader, the gold-eval scorer
and the judge's concurrency cap. No network and no credentials are needed. 
