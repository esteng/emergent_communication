"""LLM semantic judge for decodability and meaning-equivalence.

Two batched judgments, both over short natural-language MEANING pairs only -- the
judge never sees the frozen agent thread or the shorthand codes, so it cannot
re-derive a reading from a form:

* decodability -- does an agent's free-text READING capture an INTENDED meaning
  (``full`` / ``partial`` / ``none``), robust to paraphrase and attentive to
  argument roles rather than shared words;
* equivalence -- do two agent readings denote the SAME situation, a PARTIALLY
  overlapping one, or a DIFFERENT one (``same`` / ``partial`` / ``different``);
  this is the order-invariance test grounded in the agent's own two decodings
  instead of a leading "are these the same?" prompt.

Both run in one call on the ``claude-sonnet-4-6`` Portkey deployment via ``call_structured``
(cost logged under the ``semantic_judge`` caller). Verdicts are addressed by index
so a dropped or reordered item degrades to a conservative default rather than
misaligning the batch.
"""

import asyncio
import random
from typing import NamedTuple

from pydantic import BaseModel

from morphology.pipeline.analysis_mode import call_structured

import morphology.common as C

# The deployment name of the judge model (also the token_pricing key). ``claude-*``
# models route via Portkey; ``gpt-*`` models via Azure.
JUDGE_MODEL = "claude-sonnet-4-6"
# The judge emits one short verdict per item; give it ample output budget and minimal
# reasoning so a reasoning model does not spend the whole budget thinking and return
# empty content.
_JUDGE_MAX_OUTPUT_TOKENS = 16384
# ``medium`` reasoning is required: at ``minimal``/``low`` nano stops discriminating
# and blanket-returns "partial"/"different" for the whole list. ``medium`` makes it
# evaluate each item (it correctly separates paraphrase from genuine misreads).
_JUDGE_REASONING_EFFORT = "medium"
# Items per judge call. Even at ``medium`` a very long list drifts toward laziness;
# small chunks keep every item attended to. Chunks are judged in parallel, so this
# costs latency-of-one-call, not N.
_JUDGE_CHUNK = 10
# Exhaustive order paradigms can produce dozens of chunks. Unbounded fan-out caused
# a single Azure request to sit behind the SDK's long timeout while also leaking a
# client per chunk; a modest cap avoids connection-pool/rate pressure at sweep scale.
_JUDGE_CONCURRENCY = 8

# Ordinal decodability verdict -> score. ``full`` = meaning captured (paraphrase
# ok); ``partial`` = right pieces, wrong relation or something missing/added;
# ``none`` = unrelated.
_DECODE_SCORE = {"full": 1.0, "partial": 0.5, "none": 0.0}
# Ordinal equivalence verdict -> score, same ordinal shape as decodability.
_EQUIV_SCORE = {"same": 1.0, "partial": 0.5, "different": 0.0}

_SYSTEM = C.load_prompt("semantic_judge")

class DecodeItem(NamedTuple):
    """One decodability comparison: an intended meaning vs an agent's reading."""

    intended: str
    reading: str

class EquivItem(NamedTuple):
    """One equivalence comparison: two agent readings to judge same/different."""

    reading_a: str
    reading_b: str

class _DecodeVerdict(BaseModel):
    """The judge's decodability verdict for one indexed decode item."""

    index: int
    verdict: str  # "full" | "partial" | "none"

class _EquivVerdict(BaseModel):
    """The judge's equivalence verdict for one indexed equivalence item."""

    index: int
    verdict: str  # "same" | "partial" | "different"

class _JudgeReply(BaseModel):
    """The whole judge reply: one verdict list per task."""

    decode_verdicts: list[_DecodeVerdict]
    equivalence_verdicts: list[_EquivVerdict]

class JudgeOutcome(NamedTuple):
    """Scored judge output aligned position-for-position to the input item lists.

    ``decode_scores[i]`` is the 0/0.5/1.0 score for ``decode_items[i]``.
    ``equivalence_scores[j]`` is the 0/0.5/1.0 score for ``equiv_items[j]``
    ('same'/'partial'/'different'). ``equivalence_same[j]`` is the strict boolean
    view (True only for a full 'same', i.e. score == 1.0) that existing callers
    (the order-invariance test) use -- a 'partial' equivalence means the swap DID
    change something, so it is correctly treated as not-same there, same as
    'different'. Items the judge failed to return degrade to 0.0 / False
    (conservative).
    """

    decode_scores: list[float]
    equivalence_scores: list[float]
    equivalence_same: list[bool]

def _render(decode_items: list[DecodeItem], equiv_items: list[EquivItem]) -> str:
    """Render both item lists as an indexed prompt body for the judge."""
    lines: list[str] = ["DECODE ITEMS:"]
    for index, item in enumerate(decode_items):
        lines.append(f"[{index}] INTENDED: {item.intended}")
        lines.append(f"    READING:  {item.reading}")
    lines.append("")
    lines.append("EQUIVALENCE ITEMS:")
    for index, item in enumerate(equiv_items):
        lines.append(f"[{index}] A: {item.reading_a}")
        lines.append(f"    B: {item.reading_b}")
    return "\n".join(lines)

async def _judge_chunk(
    decode_chunk: list[DecodeItem], equiv_chunk: list[EquivItem], judge_model: str
) -> JudgeOutcome:
    """Judge one small chunk (decode-only or equivalence-only) in a single call."""
    call = await call_structured(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _render(decode_chunk, equiv_chunk)},
        ],
        output_schema=_JudgeReply,
        model=judge_model,
        caller="semantic_judge",
        max_output_tokens=_JUDGE_MAX_OUTPUT_TOKENS,
        reasoning_effort=_JUDGE_REASONING_EFFORT,
    )
    reply: _JudgeReply = call.result
    decode_by_index = {v.index: v.verdict.strip().lower() for v in reply.decode_verdicts}
    equiv_by_index = {v.index: v.verdict.strip().lower() for v in reply.equivalence_verdicts}
    decode_scores = [
        _DECODE_SCORE.get(decode_by_index.get(index, "none"), 0.0)
        for index in range(len(decode_chunk))
    ]
    equivalence_scores = [
        _EQUIV_SCORE.get(equiv_by_index.get(index, "different"), 0.0)
        for index in range(len(equiv_chunk))
    ]
    equivalence_same = [score == 1.0 for score in equivalence_scores]
    return JudgeOutcome(
        decode_scores=decode_scores,
        equivalence_scores=equivalence_scores,
        equivalence_same=equivalence_same,
    )

def content_words(text: str) -> set[str]:
    """Lowercase alphabetic tokens of length > 2 -- the meaning-overlap vocabulary."""
    return {
        "".join(character for character in word if character.isalpha())
        for word in text.lower().split()
        if len("".join(character for character in word if character.isalpha())) > 2
    }

def build_negative_control_items(
    meaning_readings: list[tuple[str, str]], max_items: int, seed: int
) -> list[DecodeItem]:
    """Pair each reading with an UNRELATED intended meaning; a trustworthy judge says 'none'.

    The partner's meaning must share NO content word with what the READING ITSELF says --
    not with the reading's own true meaning. Those two are not interchangeable: a short,
    on-topic decode paraphrase makes them nearly equivalent, but a reading that instead
    explains why a form does NOT decode (e.g. "bell + silent hum -- reads like mixing a
    tool code with a hum symptom code, not negotiated") routinely namedrops several
    domains' vocabulary regardless of its own cell's true meaning. Checking meaning-vs-
    meaning disjointness in that case does not guarantee the reading's actual words are
    disjoint from the partner, and the control silently stops being a true negative --
    exactly the same failure mode this function was built to rule out, just one level
    removed. Checking disjointness directly against the reading's own words is correct
    for both short paraphrases and long rejection explanations alike.

    Also requiring the two MEANINGS to differ is not enough on its own: in a paradigm
    every cell is built from the same small vocabulary, so two different cells routinely
    share their action, duration, or intensity ("cool air, 12 seconds" vs "cool cloth on
    all faces for 12 seconds"). Such a pair is a textbook ``partial`` under the judge's own
    rubric -- right action, wrong object -- so scoring it as a failed ``none`` measures the
    stimulus, not the judge. Disjointness makes the control a true negative.

    The partner is drawn at random from every disjoint candidate (seeded, so a run is
    reproducible), because taking the first match makes one meaning the partner of every
    control and confounds judge trust with that meaning's peculiarities.
    """
    rng = random.Random(seed)
    reading_words = [content_words(reading) for _, reading in meaning_readings]
    meaning_words = [content_words(meaning) for meaning, _ in meaning_readings]
    items: list[DecodeItem] = []
    for index, (_, reading) in enumerate(meaning_readings):
        if len(items) >= max_items:
            break
        candidates = [
            other
            for other in range(len(meaning_readings))
            if other != index
            and reading_words[index]
            and reading_words[index].isdisjoint(meaning_words[other])
        ]
        if not candidates:
            continue
        partner = rng.choice(candidates)
        items.append(
            DecodeItem(intended=meaning_readings[partner][0], reading=reading)
        )
    return items

def build_positive_control_items(
    meanings: list[str], max_items: int
) -> list[DecodeItem]:
    """Pair each meaning with itself; a judge with any discrimination scores 'full'.

    The negative controls alone cannot distinguish a discriminating judge from one that
    blanket-answers 'none' -- both score 1.00 on them. This is the mirror guard: the
    reading IS the intended meaning, so anything below 1.00 means the judge has stopped
    reading items (the known nano failure mode at low reasoning effort, where it returns
    one verdict for the whole list). Trivially easy by design -- it is a regression floor,
    not a discrimination test.
    """
    return [DecodeItem(intended=meaning, reading=meaning) for meaning in meanings[:max_items]]

async def run_semantic_judge(
    decode_items: list[DecodeItem], equiv_items: list[EquivItem], judge_model: str = JUDGE_MODEL
) -> JudgeOutcome:
    """Score all decode and equivalence items, chunked into small parallel judge calls.

    Decode and equivalence are chunked SEPARATELY (never mixed in one call) so no
    single call exceeds ``_JUDGE_CHUNK`` items -- the size at which nano stops
    attending item-by-item. Chunks run with bounded concurrency to avoid saturating
    the Azure client during exhaustive order sweeps. Returns empty results when there
    is nothing to judge. ``judge_model`` defaults to the Claude judge routed through
    Portkey; pass a ``gpt-*`` model to route via Azure instead (e.g. the cheap nano
    judge, to compare judge quality).
    """
    if not decode_items and not equiv_items:
        return JudgeOutcome(decode_scores=[], equivalence_scores=[], equivalence_same=[])

    semaphore = asyncio.Semaphore(_JUDGE_CONCURRENCY)

    async def limited_judge(
        decode_chunk: list[DecodeItem], equiv_chunk: list[EquivItem]
    ) -> JudgeOutcome:
        async with semaphore:
            return await _judge_chunk(decode_chunk, equiv_chunk, judge_model)

    decode_tasks = [
        limited_judge(decode_items[start:start + _JUDGE_CHUNK], [])
        for start in range(0, len(decode_items), _JUDGE_CHUNK)
    ]
    equiv_tasks = [
        limited_judge([], equiv_items[start:start + _JUDGE_CHUNK])
        for start in range(0, len(equiv_items), _JUDGE_CHUNK)
    ]
    results = await asyncio.gather(*decode_tasks, *equiv_tasks)
    decode_outs = results[: len(decode_tasks)]
    equiv_outs = results[len(decode_tasks):]
    decode_scores = [score for out in decode_outs for score in out.decode_scores]
    equivalence_scores = [score for out in equiv_outs for score in out.equivalence_scores]
    equivalence_same = [same for out in equiv_outs for same in out.equivalence_same]
    return JudgeOutcome(
        decode_scores=decode_scores,
        equivalence_scores=equivalence_scores,
        equivalence_same=equivalence_same,
    )


async def equivalence_rate(pairs: list[tuple[str, str]], judge_model: str) -> float | None:
    """Fraction of literal-text-differing pairs the judge still scores as equivalent."""
    if not pairs:
        return None
    items = [EquivItem(reading_a=a, reading_b=b) for a, b in pairs]
    outcome = await run_semantic_judge(decode_items=[], equiv_items=items, judge_model=judge_model)
    return sum(outcome.equivalence_same) / len(outcome.equivalence_same)
