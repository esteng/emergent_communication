"""Reading a run and recovering the codebook its agents negotiated.

Three steps, in order. ``find_run_jsonl`` / ``run_agent_ids`` locate the event log and the
agents that played. ``gather_postmortem_messages`` pulls the postmortem-channel transcript
out of it. Then the codebook itself is recovered two ways that are combined, not
alternatives: ``extract_codebook`` asks a model for every negotiated mapping, including the
prose-only conventions a parser would miss, and ``parse_codebook`` deterministically
recovers explicit ``SYMBOL=meaning`` definitions, which override an LLM paraphrase
wherever they are fuller. That override is what protects multi-step meanings containing
commas and semicolons.
"""

import json
import re
from pathlib import Path

from pydantic import BaseModel

import morphology.common as C

def find_run_jsonl(run_dir):
    """Return the primary event-log JSONL inside a run directory, or None.

    Prefers ``<scenario>.jsonl`` (stem matches the parent scenario dir name),
    otherwise falls back to the first ``.jsonl`` file found.
    """
    candidates = [c for c in run_dir.glob("*.jsonl") if not c.stem.endswith("_debug")]
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.stem == run_dir.parent.name:
            return candidate
    return candidates[0]


def run_agent_ids(run_dir):
    """Return the agent ids registered in a run, in log order.

    Reads ``agent_registered`` events straight from the run's JSONL so the caller stays
    free of any simulation-platform import.
    """
    jsonl = find_run_jsonl(run_dir)
    if jsonl is None:
        return []
    seen: list[str] = []
    with jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if '"agent_registered"' not in line:
                continue
            event = json.loads(line)
            if event.get("event_type") != "agent_registered":
                continue
            agent_id = event.get("agent_id")
            if agent_id and agent_id not in seen:
                seen.append(agent_id)
    return seen


# Category-prefixed lines like "Tools: b=bell, ..." or "Faces: F/B/L/R/T/Bo (...)".
_CATEGORY_RE = re.compile(
    r"^(tools?|faces?|edges?|corners?|format|intensity|codes?|symptom|duration|"
    r"location|loc|cues?|tool|face)\s*:\s*(.*)$",
    re.IGNORECASE,
)

_QUOTE_CHARS = "\"'‘’“”"

# A short token in a slash-enumerated set, e.g. the F/B/L/R/T/Bo in a Faces line.
_SET_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,3}$")

# Gate: the line introduces codes via "use X for ...".
_USE_FOR_RE = re.compile(r"\buse\s+[\"']?[A-Za-z]", re.IGNORECASE)
# Within such a line, each "SYM for meaning" pair (uppercase-initial symbol so prose
# like "wait for hum" is not matched); handles chains "OH for overheating, DI for dimness".
_FOR_PAIR_RE = re.compile(r"\b[\"']?([A-Z][A-Za-z0-9/+-]{0,5})[\"']?\s+for\s+([a-z][^,;.]*)")

# Space-separated mapping inside a category line, e.g. "T top" in "Faces: T top, B back".
_SPACE_PAIR_RE = re.compile(r"^[\"']?([A-Za-z][A-Za-z0-9]{0,5})[\"']?\s+([a-z].*)$")

# Markdown table rows: "| shorthand | meaning |"; the separator row "|---|---|" is skipped.
_TABLE_ROW_RE = re.compile(r"^\|([^|]+)\|([^|]+)\|")
_TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+$")
_TABLE_HEADER_WORDS = frozenset({
    "shorthand", "code", "codes", "symbol", "symbols", "meaning", "canonical",
    "expansion", "note", "notes", "action", "procedure",
})

# Determiners / pronouns / connectives that can land on the symbol side of a pattern
# but are never codes. (Status words like "done"/"NO"/"OK" are NOT here — they are real
# codes — so the case-insensitive check below must not include them.)
_NON_SYMBOL_WORDS = frozenset({
    "this", "that", "it", "them", "these", "those", "the", "one", "both", "each",
    "so", "then", "next", "and", "but", "if", "let", "or", "also", "use", "see",
    "add", "will", "yes", "got", "your", "our", "keep", "they", "we", "my", "for",
})

class PostmortemMessage(BaseModel):
    """One postmortem-channel message with its round and sender."""

    round_number: int
    sender: str
    text: str

class CodebookEntry(BaseModel):
    """A single negotiated shorthand mapping parsed from a postmortem transcript."""

    symbol: str
    raw_symbol: str
    meaning: str
    category: str
    round_introduced: int

def gather_postmortem_messages(jsonl_path: Path) -> list[PostmortemMessage]:
    """Return every postmortem-channel ``message_sent`` in a run, in log order."""
    messages: list[PostmortemMessage] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if '"message_sent"' not in line:
                continue
            event = json.loads(line)
            if event.get("event_type") != "message_sent":
                continue
            message = event.get("message") or {}
            channel_id = message.get("channel_id") or ""
            if "postmortem" not in channel_id:
                continue
            text = message.get("text") or ""
            if not text:
                continue
            round_number = event.get("round_number")
            if not isinstance(round_number, int):
                continue
            sender = message.get("sender_id") or message.get("sender") or "unknown"
            messages.append(
                PostmortemMessage(round_number=round_number, sender=sender, text=text)
            )
    return messages

def _normalize_symbol(raw_symbol: str) -> str:
    """Strip quotes, a ``<placeholder>`` argument, and a trailing slash from a symbol.

    ``"FA6/<startF>"`` -> ``FA6``; ``"+opp"`` -> ``+opp``; ``b`` -> ``b``.
    """
    symbol = raw_symbol.strip().strip(_QUOTE_CHARS)
    symbol = re.sub(r"<[^>]*>", "", symbol)
    symbol = symbol.rstrip("/.,;:!?").strip()
    return symbol

def _looks_like_symbol(symbol: str) -> bool:
    """Accept short / non-lexical tokens as code symbols; reject plain long words.

    A symbol qualifies if it is short (<=4 chars), contains a digit, or contains a
    non-alphanumeric code character. This keeps ``b``, ``FA6``, ``+opp``, ``EX``
    while rejecting a full English meaning accidentally captured on the left side.
    """
    if not symbol:
        return False
    if len(symbol) <= 4:
        return True
    if any(ch.isdigit() for ch in symbol):
        return True
    if any(not ch.isalnum() for ch in symbol):
        return True
    return False

def _parse_slash_set(category: str, body: str, round_number: int) -> list[CodebookEntry]:
    """Parse a slash-enumerated set such as ``F/B/L/R/T/Bo (front/back/...)``."""
    paren_match = re.search(r"\(([^)]*)\)", body)
    glosses: list[str] = []
    if paren_match:
        glosses = [g.strip() for g in paren_match.group(1).split("/")]
        body = body[: paren_match.start()]
    tokens = [tok.strip().strip(_QUOTE_CHARS) for tok in body.split("/")]
    tokens = [tok for tok in tokens if _SET_TOKEN_RE.match(tok)]
    if len(tokens) < 2:
        return []
    entries: list[CodebookEntry] = []
    for index, token in enumerate(tokens):
        if index < len(glosses) and glosses[index]:
            meaning = glosses[index]
        else:
            meaning = category
        entries.append(
            CodebookEntry(
                symbol=token,
                raw_symbol=token,
                meaning=meaning,
                category=category,
                round_introduced=round_number,
            )
        )
    return entries

def _clean_meaning(text: str) -> str:
    """Trim a meaning to its first gloss, dropping trailing prose / parentheticals."""
    meaning = text.strip().strip(_QUOTE_CHARS).strip()
    meaning = re.split(r"\.\s|\s—\s|\s-\s|\s\(|\s*\|\s*", meaning, maxsplit=1)[0].strip()
    return meaning.strip(_QUOTE_CHARS).strip()

def _make_entry(
    raw_symbol: str, meaning: str, category: str, round_number: int
) -> CodebookEntry | None:
    """Build a CodebookEntry if the symbol is code-shaped and not a determiner."""
    symbol = _normalize_symbol(raw_symbol)
    if not symbol or symbol.lower() in _NON_SYMBOL_WORDS or not _looks_like_symbol(symbol):
        return None
    return CodebookEntry(
        symbol=symbol,
        raw_symbol=raw_symbol.strip(_QUOTE_CHARS),
        meaning=_clean_meaning(meaning),
        category=category,
        round_introduced=round_number,
    )

def _parse_eq_clauses(body: str, category: str, round_number: int) -> list[CodebookEntry]:
    """Parse every explicit ``SYMBOL=meaning`` clause without truncating its meaning.

    Commas and semicolons often belong to a multi-step definition (``cool cloth;
    remove; fan all faces``).  A delimiter ends a definition only when the text after
    it starts another code-shaped ``SYMBOL=`` mapping.
    """
    starts = list(
        re.finditer(
            rf"(?:^|[;,|])\s*[{_QUOTE_CHARS}]?"
            rf"([A-Za-z][\w/<>@+:.\-]{{0,14}}?)\s*[{_QUOTE_CHARS}]?\s*=\s*",
            body,
        )
    )
    entries: list[CodebookEntry] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(body)
        meaning = body[match.end():end].strip().rstrip(";,|").strip()
        entry = _make_entry(
            raw_symbol=match.group(1),
            meaning=meaning,
            category=category,
            round_number=round_number,
        )
        if entry is not None:
            entries.append(entry)
    return entries

def _parse_space_pairs(category: str, body: str, round_number: int) -> list[CodebookEntry]:
    """Parse space-separated ``symbol meaning`` clauses, e.g. ``T top, B back, F front``."""
    entries: list[CodebookEntry] = []
    for clause in re.split(r"[;,|]", body):
        clause = clause.strip().strip("-•* ").strip()
        pair_match = _SPACE_PAIR_RE.match(clause)
        if pair_match is None:
            continue
        entry = _make_entry(
            raw_symbol=pair_match.group(1),
            meaning=pair_match.group(2),
            category=category,
            round_number=round_number,
        )
        if entry is not None:
            entries.append(entry)
    return entries

def _parse_use_for(line: str, round_number: int) -> list[CodebookEntry]:
    """Parse ``use X for Y`` prose definitions, including chained ``X for Y, Z for W``."""
    if _USE_FOR_RE.search(line) is None:
        return []
    entries: list[CodebookEntry] = []
    for match in _FOR_PAIR_RE.finditer(line):
        entry = _make_entry(
            raw_symbol=match.group(1),
            meaning=match.group(2),
            category="use_for",
            round_number=round_number,
        )
        if entry is not None:
            entries.append(entry)
    return entries

def _parse_table_row(line: str, round_number: int) -> list[CodebookEntry]:
    """Parse a markdown table row ``| shorthand | meaning |`` into a codebook entry."""
    if _TABLE_SEP_RE.match(line):
        return []
    row_match = _TABLE_ROW_RE.match(line)
    if row_match is None:
        return []
    symbol_cell = row_match.group(1).strip().strip(_QUOTE_CHARS).strip()
    meaning_cell = row_match.group(2).strip()
    if not symbol_cell or symbol_cell.lower() in _TABLE_HEADER_WORDS:
        return []
    # The symbol cell may carry a parametric template ("fanAll [int] Ns"); take the
    # leading fixed token and drop bracketed placeholders.
    raw_symbol = re.sub(r"\[[^\]]*\]", "", symbol_cell.split()[0]).strip()
    entry = _make_entry(
        raw_symbol=raw_symbol, meaning=meaning_cell, category="table", round_number=round_number
    )
    return [entry] if entry is not None else []

# Keywords that, on a line ending in ":", open a multi-line "SYMBOL meaning" code block.
_BLOCK_HEADER_KEYWORDS = (
    ("procedure", "procedure"), ("code", "code"), ("symptom", "symptom"),
    ("tag", "symptom"), ("shorthand", "shorthand"), ("abbrev", "shorthand"),
    ("face", "faces"), ("tool", "tools"), ("legend", "general"), ("format", "format"),
)

def _block_header_category(line: str) -> str | None:
    """Return a category if ``line`` is a code-block header (ends ':', has a code keyword)."""
    if not line.rstrip().endswith(":"):
        return None
    low = line.lower()
    for keyword, category in _BLOCK_HEADER_KEYWORDS:
        if keyword in low:
            return category
    return None

def _is_strict_code_symbol(symbol: str) -> bool:
    """Stricter than ``_looks_like_symbol``: for block bodies, reject titlecase words.

    Inside a multi-line block, lines look like ``P1 all-6 press`` / ``A chaotic hum``.
    Accept only symbols that are unambiguously code-shaped so a prose line like
    ``Got the code map`` does not register ``Got`` as a symbol.
    """
    if not symbol:
        return False
    if any(ch.isdigit() for ch in symbol):
        return True
    if len(symbol) == 1 and symbol.isupper():
        return True
    if symbol.isupper() and 2 <= len(symbol) <= 4:
        return True
    return any(ch in symbol for ch in "/+-:@")

def _parse_block_pair(line: str, category: str, round_number: int) -> CodebookEntry | None:
    """Parse a standalone ``SYMBOL meaning`` line inside a code block."""
    pair_match = re.match(r"^(\S{1,8})\s+(\S.+)$", line)
    if pair_match is None:
        return None
    if not _is_strict_code_symbol(_normalize_symbol(pair_match.group(1))):
        return None
    return _make_entry(
        raw_symbol=pair_match.group(1),
        meaning=pair_match.group(2),
        category=category,
        round_number=round_number,
    )

def parse_codebook(messages: list[PostmortemMessage]) -> list[CodebookEntry]:
    """Parse the consolidated codebook from one run's postmortem messages.

    Deduplicates on the normalized symbol, keeping the earliest round in which it
    appeared. Returns entries sorted by ``round_introduced`` then symbol.
    """
    found: dict[str, CodebookEntry] = {}

    def _register(entry: CodebookEntry) -> None:
        existing = found.get(entry.symbol)
        if existing is None or entry.round_introduced < existing.round_introduced:
            found[entry.symbol] = entry

    for message in messages:
        round_number = message.round_number
        block_category = None  # set while inside a multi-line "SYMBOL meaning" block
        for raw_line in message.text.splitlines():
            line = raw_line.strip().lstrip("-•*").strip()
            if not line:
                block_category = None
                continue

            # Markdown table rows: "| shorthand | meaning |".
            table_entries = _parse_table_row(line, round_number)
            if table_entries:
                for entry in table_entries:
                    _register(entry)
                block_category = None
                continue

            # A header line ending in ":" opens (or switches) a multi-line code block.
            header_category = _block_header_category(line)
            if header_category is not None:
                block_category = header_category

            category = "general"
            body = line
            category_match = _CATEGORY_RE.match(line)
            if category_match:
                category = category_match.group(1).lower()
                body = category_match.group(2)

            eq_entries = _parse_eq_clauses(body, category, round_number)
            matched_any = bool(eq_entries)
            for entry in eq_entries:
                _register(entry)

            # Category lines without "=": try slash sets and space-separated pairs.
            if category_match and not matched_any:
                for entry in _parse_slash_set(
                    category=category, body=body, round_number=round_number
                ):
                    _register(entry)
                for entry in _parse_space_pairs(
                    category=category, body=body, round_number=round_number
                ):
                    _register(entry)

            # A standalone "SYMBOL meaning" line inside an open block (e.g. "P1 all-6 press").
            # A line that is not such a pair closes the block.
            if block_category is not None and header_category is None and not matched_any:
                block_entry = _parse_block_pair(line, block_category, round_number)
                if block_entry is not None:
                    _register(block_entry)
                else:
                    block_category = None

            # "Use X for Y" prose definitions anywhere in the line.
            for entry in _parse_use_for(line, round_number):
                _register(entry)

    return sorted(found.values(), key=lambda entry: (entry.round_introduced, entry.symbol))

# Punctuation stripped from a link token before matching it against the codebook.
# Separators inside a parametric code (e.g. "FA6/T", "A:dim", "x@2") whose head is
# the fixed symbol the agents locked in.


_EXTRACTION_MAX_OUTPUT_TOKENS = 16384
_EXTRACTION_REASONING_EFFORT = "low"
_EXTRACTION_ROUNDS_PER_CHUNK = 20

def extract_codebook(
    messages: list[PostmortemMessage], model: str
) -> list[CodebookEntry]:
    """Extract a codebook from postmortem messages with a structured LLM call.

    Routes through the analysis-mode ``call_structured`` dispatcher, which selects a
    backend from the environment. Messages are sent in bounded groups of rounds (one
    call for the current 15-round runs); the model returns one entry per negotiated
    shorthand mapping, projected onto ``CodebookEntry``.
    """
    import asyncio

    from pydantic import BaseModel

    from morphology.pipeline.analysis_mode import call_structured

    class _LLMEntry(BaseModel):
        symbol: str
        meaning: str
        category: str
        round_introduced: int

    class _CodebookExtraction(BaseModel):
        entries: list[_LLMEntry]

    system_prompt = C.load_prompt("codebook_extraction")

    round_groups: dict[int, list[PostmortemMessage]] = {}
    for message in messages:
        round_groups.setdefault(message.round_number, []).append(message)
    ordered_rounds = sorted(round_groups)
    chunks = [
        [message for round_number in round_chunk for message in round_groups[round_number]]
        for start in range(0, len(ordered_rounds), _EXTRACTION_ROUNDS_PER_CHUNK)
        for round_chunk in [ordered_rounds[start:start + _EXTRACTION_ROUNDS_PER_CHUNK]]
    ]

    async def _extract_chunks() -> list[_CodebookExtraction]:
        calls = [
            call_structured(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": "\n".join(
                            f"[round {m.round_number}] {m.sender}: {m.text}"
                            for m in chunk
                        ),
                    },
                ],
                output_schema=_CodebookExtraction,
                model=model,
                caller="codebook_extractor",
                max_output_tokens=_EXTRACTION_MAX_OUTPUT_TOKENS,
                reasoning_effort=_EXTRACTION_REASONING_EFFORT,
            )
            for chunk in chunks
        ]
        return [call.result for call in await asyncio.gather(*calls)]

    extractions = asyncio.run(_extract_chunks())
    llm_entries = [
        entry
        for extraction in extractions
        for entry in extraction.entries
    ]
    # Hard guard on the prompt's "no internal spaces" rule: a symbol containing a
    # space is a rule or phrase the model let through, not a compact code token.
    entries = [
        CodebookEntry(
            symbol=entry.symbol,
            raw_symbol=entry.symbol,
            meaning=entry.meaning,
            category=entry.category,
            round_introduced=entry.round_introduced,
        )
        for entry in llm_entries
        if " " not in entry.symbol.strip()
    ]
    entries = _dedup_exact_symbols(entries)
    return _merge_explicit_meanings(messages, _dedup_case_variants(entries))

def _content_words(text: str) -> set[str]:
    """Lowercase alphabetic tokens of length > 2, for meaning-overlap comparison."""
    words = set()
    for word in text.lower().split():
        cleaned = "".join(ch for ch in word if ch.isalpha())
        if len(cleaned) > 2:
            words.add(cleaned)
    return words

def _dedup_exact_symbols(entries: list[CodebookEntry]) -> list[CodebookEntry]:
    """Merge chunk overlap, keeping the fullest meaning and earliest attestation."""
    by_symbol: dict[str, CodebookEntry] = {}
    for entry in entries:
        existing = by_symbol.get(entry.symbol)
        if existing is None:
            by_symbol[entry.symbol] = entry
            continue
        fuller = max((existing, entry), key=lambda item: len(_content_words(item.meaning)))
        by_symbol[entry.symbol] = fuller.model_copy(
            update={"round_introduced": min(existing.round_introduced, entry.round_introduced)}
        )
    return list(by_symbol.values())

def _merge_explicit_meanings(
    messages: list[PostmortemMessage], entries: list[CodebookEntry]
) -> list[CodebookEntry]:
    """Prefer a fuller literal ``SYMBOL=meaning`` definition over an LLM truncation.

    The LLM remains responsible for prose-only recall, but explicit definitions are
    stronger evidence and can be recovered deterministically.  This particularly
    protects multi-step meanings containing commas and semicolons.
    """
    explicit = {entry.symbol: entry for entry in parse_codebook(messages)}
    merged: list[CodebookEntry] = []
    for entry in entries:
        literal = explicit.get(entry.symbol)
        if literal is not None and len(_content_words(literal.meaning)) > len(
            _content_words(entry.meaning)
        ):
            entry = entry.model_copy(update={"meaning": literal.meaning})
        merged.append(entry)
    return merged

def _meaning_overlap(first: str, second: str) -> float:
    """Jaccard overlap of two meanings' content words (0.0 when either is empty)."""
    left, right = _content_words(first), _content_words(second)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)

def _dedup_case_variants(entries: list[CodebookEntry]) -> list[CodebookEntry]:
    """Merge case-insensitive duplicate symbols whose meanings overlap.

    Backstops the prompt's case-unification instruction: two entries whose symbols
    are equal ignoring case and whose meanings share enough content words are the
    same code captured twice (``Bc1``/``BC1``); the first occurrence is kept.
    Case-insensitive matches with unrelated meanings are preserved as distinct, so a
    genuine case contrast is not collapsed.
    """
    kept: list[CodebookEntry] = []
    for entry in entries:
        duplicate = any(
            other.symbol.lower() == entry.symbol.lower()
            and _meaning_overlap(other.meaning, entry.meaning) >= 0.5
            for other in kept
        )
        if not duplicate:
            kept.append(entry)
    return kept
