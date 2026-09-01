# [GlossoGen: Emergent Language in Complex Multi-Agent LLM Interactions]()

Elias Stengel-Eskin, Newton Sander, Carlos Bonetti, Sasha Boguraev, James Bowler, Hale Sirin, Simon Kirby

Reproduces the figures, tables and reported statistics of the paper. 

```
pip install -r requirements.txt
python scripts/emergence/prep_per_round.py      # run from anywhere; paths are relative
```

Outputs are under `results/` (`figures/`, `tables/`, `csv/`). 
Run `prep_per_round.py` and `compute_components.py` first; everything else reads
their output or `data/` directly. The morphology section has its own entry point.

### Conditions for emergence (`scripts/emergence/`)


- `prep_per_round.py`: prepares data for subsequent scripts
- `perplexity_success_closed.py`: computes perplexity for a proprietary models.
- `perplexity_success_open.py`: the same plot for the open-weight models, one column each
  for Qwen and Llama.
- `stat_success_by_budget.py`: mean success rate by time budget over postmortem-off runs,
  proprietary against open-weight (0.921 vs 0.308 at the 2000s budget).


### Transmission to newcomers (`scripts/transmission/`)


- `transmission_by_history.py`: post-swap success against how many rounds of the
  transcript the newcomer saw, one panel per source model, 15 roots each.
- `llama_transmission.py`: the same experiment with a Llama-3.3-70B newcomer
- `stats_metalinguistic.py`: tests whether newcomers query compositional terms less than
  atomic ones, and whether that falls off with more history
- `judge_agreement.py`: agreement between the LLM judge and a blind human annotator.

### Description length predicts success (`scripts/mdl/`)

- `compute_components.py`: reuses each team's grammar to score the held-out rounds after
  the postmortem stage is removed, giving 135 seed-runs over 45 roots.
- `stats_and_partials.py`: correlates those description lengths with success, averaging
  seeds to roots first.
- `mdl_2panel.py`: plots both against success, one point per root, sized by coverage.
- `robustness.py`: sweeps the smoothing constant
- 
Everything here uses one grammar induction — the `corpus_swap14` set, induced over the
14 language-development rounds with grammars in `data/mdl/grammars/`. 

`scfg.py`, `mdl.py` and `ontology.py` are the supporting library: the grammar parser, the
description-length computation, and the fixed inventory of referents the codes ground out
into. `robustness.py` re-runs the component computation five times and takes a couple of
minutes.

### Morphosyntactic productivity (`scripts/morphology/`)

This section has its own README at `scripts/morphology/README.md` — read that for the
full module breakdown. It is packaged as a Python package rather than loose scripts, so
it is invoked differently from the rest:

```
export PYTHONPATH=scripts
python -m morphology.analysis.build_tidy_tables
```

- `analysis/`: builds the tidy per-item, per-construction and per-run tables, then the
  emergence, decode/production and saturation figures and the Wilson intervals beside
  them. Two of the models are R (`lme4`, `emmeans`) rather than Python.
- `robustness/`: pipeline replica agreement, the semantic judge against a blind human
  annotator (124/150 exact, weighted κ=0.642), and scoring the induction against ten
  hand-built codebooks of known answer (98.9% production, 98.2% segmentation).
- `pipeline/`: the elicitation apparatus that produced the data — it reaches back into
  agents frozen at a chosen round, asks them to coin and read novel forms, and judges the
  result. Six modules issue LLM calls; their outputs are cached under `data/morphology/`,
  so nothing downstream needs credentials.

## LLM annotation scripts

Require a `FOUNDRY_API_KEY` (or `OPENAI_API_KEY`) from the environment. 

- `scripts/transmission/annotate_metalinguistic.py` — the GPT-5.5 judge that labels which
  newcomer messages are metalinguistic queries and what they target. Cached as
  `data/annotations/metaling_*.csv`.
- `scripts/mdl/induce_grammars.py` — the GPT-5.5 induction that writes one grammar per
  protocol. Cached as `data/mdl/grammars/` and `data/mdl/mdl_table_joint.csv`. 

The morphology pipeline issues calls of its own, from six modules and against a different
set of backends; `scripts/morphology/README.md` lists them stage by stage.

## Data

Exported from the GlossoGen simulation platform, which is a separate codebase; these
exports are this release's entry point.

```
data/
  baseline_message_level.csv        one row per message, the 15-round budget/postmortem sweep
  baseline_run_level.csv            one row per run of the same sweep
  learnability_run_level.csv        one row per run of the swap experiments
  learnability_message_level_slim.csv   messages of the baseline and replace_learned runs
  annotations/                      cached GPT-5.5 metalinguistic labels, judge validation set,
                                    and the grammars used in the annotation prompt
  prompts/                          the SCFG induction and annotation prompts
  mdl/
    protocol_roots.txt              the 45 baseline roots
    resume_nopm_runs.json           root -> its 3 resume seeds
    grammars/                       the induced SCFG per root
    mdl_table_joint.csv             per-root DL(G), coverage, agent model
    grounded/                       per-round agent messages and the meanings they denote
    round_success/                  the per-round success series of each resume seed
    postmortems/                    the language-design discussion the grammars are induced from
  morphology/                       inputs for the morphosyntax section: run transcripts,
                                    induction and encode/decode caches, robustness replicas,
                                    the judge validation set and the six pipeline prompts
                                    (see `scripts/morphology/README.md` for the breakdown)
```

## Tests

```
export PYTHONPATH=scripts
python -m pytest tests/
```

This runs the eleven offline regression tests over joint induction, the cache loader, the
gold-eval scorer and the judge's concurrency cap. No network and no credentials are needed.

## Citation

```bibtex
@article{stengeleskin2026glossogen,
  title={GlossoGen: Emergent Language in Complex Multi-Agent LLM Interactions},
  author={Stengel-Eskin, Elias and Sander, Newton and Bonetti, Carlos and Boguraev, Sasha and Bowler, James and Sirin, Hale and Kirby, Simon},
  journal={arXiv preprint arXiv:},
  year={2026}
}
```


