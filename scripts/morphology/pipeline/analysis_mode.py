"""Reaching a frozen agent: thread reconstruction plus a structured model call.

Defines the request/response contract and implements it. Used two ways: as a library,
where the pipeline calls ``call_structured`` directly for every LLM stage, and as a CLI,
which reads an ``AnalysisModeRequest`` on stdin and writes an ``AnalysisModeResponse`` on
stdout, one request per process.

Thread reconstruction is the one step it delegates to the GlossoGen simulation platform
(a separate codebase), through ``export_agent_thread``, which imports it lazily so every
other module in this release runs without it. The frozen agent's own model
(``export.meta.model``) drives the request model.

Three backends. Claude models (``claude-*``) route to Portkey whenever it is
configured, since Azure cannot serve them regardless of what else is set;
every other model follows environment priority (checked in this order):

* **Azure** (when ``AZURE_ENDPOINT`` + ``AZURE_API_KEY`` are set) -- an
  ``AsyncAzureOpenAI`` chat-completions call with JSON-schema structured output.
  The deployment name defaults to the frozen agent's model (override with
  ``AZURE_DEPLOYMENT``); the API version defaults to a recent preview (override
  with ``AZURE_API_VERSION``).
* **Portkey** (when ``PORTKEY_API_KEY`` is set) -- routes Claude models through
  the Portkey AI gateway's OpenAI-compatible endpoint
  (``https://api.portkey.ai/v1``). Anthropic has no native strict-JSON-schema
  response format, so structured output is enforced via forced tool-use: the
  Pydantic schema becomes a single tool and ``tool_choice`` forces the model to
  call it. Optional ``PORTKEY_VIRTUAL_KEY`` / ``PORTKEY_PROVIDER`` env vars are
  forwarded as Portkey headers when the account needs them; a bare
  ``PORTKEY_API_KEY`` is enough when the key is already scoped to a provider on
  the Portkey dashboard.
* **OpenAI** (fallback) -- the OpenAI SDK directly, which reads ``OPENAI_API_KEY``
  and honors ``OPENAI_BASE_URL``.

Backend selection is entirely environment-driven -- there is no provider flag. Every
structured call goes through ``call_structured``, which reads the environment once and
dispatches accordingly.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit

from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel

from morphology.pipeline.codebook import CodebookEntry

class TokenUsage(BaseModel):
    """Token counts for one structured call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


def _enforce_strict_schema(schema: dict) -> None:
    """Recursively add ``additionalProperties: false`` to every object schema.

    OpenAI's strict mode requires this at every level, including ``$defs`` and nested
    ``items``.
    """
    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
        for prop_schema in schema["properties"].values():
            _enforce_strict_schema(schema=prop_schema)
    if "items" in schema and isinstance(schema["items"], dict):
        _enforce_strict_schema(schema=schema["items"])
    if "$defs" in schema:
        for def_schema in schema["$defs"].values():
            _enforce_strict_schema(schema=def_schema)


def export_agent_thread(**kwargs):
    """Reconstruct a frozen agent's thread via the simulation platform.

    This is the one place the pipeline reaches back into the GlossoGen platform, which
    is a separate codebase. It is imported lazily so that every other module here --
    induction, stimulus construction, scoring, plotting -- imports and runs without it.
    Only the live probe path needs it; everything the paper reports replays from the
    cached responses shipped in ``data/morphology/``.
    """
    try:
        from schmidt.thread_export.export_agent_thread import export_agent_thread_from_run_dir
    except ImportError as exc:  # pragma: no cover - depends on the platform being installed
        raise RuntimeError(
            "Reaching a frozen agent requires the GlossoGen simulation platform "
            "(a separate codebase) on the import path. The cached encode/decode "
            "responses under data/morphology/encode_decode/ reproduce every reported "
            "number without it."
        ) from exc
    return export_agent_thread_from_run_dir(**kwargs)


logger = logging.getLogger(__name__)

_AZURE_ENDPOINT_ENV = "AZURE_ENDPOINT"
_AZURE_KEY_ENV = "AZURE_API_KEY"
_AZURE_VERSION_ENV = "AZURE_API_VERSION"
_AZURE_DEPLOYMENT_ENV = "AZURE_DEPLOYMENT"
_DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"
_AZURE_TIMEOUT_SECONDS = 180.0
_PORTKEY_API_KEY_ENV = "PORTKEY_API_KEY"
_PORTKEY_VIRTUAL_KEY_ENV = "PORTKEY_VIRTUAL_KEY"
_PORTKEY_PROVIDER_ENV = "PORTKEY_PROVIDER"
_PORTKEY_CONFIG_ENV = "PORTKEY_CONFIG"
_PORTKEY_BASE_URL = "https://api.portkey.ai/v1"
_PORTKEY_TIMEOUT_SECONDS = 180.0
# Output-token cap per probe reply. Generous so a reasoning model's internal
# tokens do not truncate the structured JSON; also the conservative figure the
# pre-call budget projection assumes.
_OUTPUT_TOKEN_ALLOWANCE = 4096
# Fallback chars-per-token ratio when a real tokenizer is unavailable.


class AnalysisModeRequest(BaseModel):
    """One frozen-agent prompt request handed to analysis mode.

    ``cutoff_round`` is **exclusive**: ``R`` freezes state through round ``R-1``;
    pass ``R+1`` to capture the end of round ``R`` (matching the platform's
    ``build_message_history`` semantics). ``None`` uses the full end-of-run state.
    ``output_schema_name`` names the structured shape the agent must emit
    (``encode`` / ``decode`` / ``tolerance``). ``injected_codebook`` carries
    artificially constructed codes to splice into the frozen context for the
    artificial-codebook stimuli; ``None`` leaves the run's own state untouched.
    """

    run_dir: str
    agent_id: str
    cutoff_round: int | None
    prompt: str
    output_schema_name: str
    injected_codebook: list[CodebookEntry] | None


class AnalysisModeResponse(BaseModel):
    """The structured reply analysis mode returns for one request.

    ``output_fields`` carries the schema-specific payload (e.g.
    ``{"produced_form": ...}`` for ``encode``; ``{"interpreted_meaning": ...,
    "is_valid": "true"}`` for ``decode``/``tolerance``) as strings so the contract
    stays schema-agnostic. ``usage`` carries token counts keyed by name.
    """

    reasoning: str
    output_fields: dict[str, str]
    model: str
    provider: str
    usage: dict[str, int]


class EncodeOutput(BaseModel):
    """Structured reply for an ``encode`` probe: coin a form for a target meaning."""

    reasoning: str
    produced_form: str


class DecodeOutput(BaseModel):
    """Structured reply for a ``decode``/``tolerance`` probe: read back a form."""

    reasoning: str
    interpreted_meaning: str
    is_valid: bool


class StructuredCall(NamedTuple):
    """A validated structured reply (any Pydantic schema) plus reported token usage."""

    result: BaseModel
    usage: TokenUsage


def _codebook_note(entries: list[CodebookEntry]) -> str:
    """Render an injected codebook as a short instruction block for the probe."""
    lines = [f"{entry.symbol} = {entry.meaning}" for entry in entries]
    joined = "\n".join(lines)
    return f"Assume these codes are in force:\n{joined}"


def _chat_messages(request, probe: AnalysisModeRequest) -> list[dict[str, str]]:
    """Build the chat-completions messages: exported thread + optional codebook + probe.

    ``flatten_tools=True`` on export renders every historical tool call as plain
    text, so each exported message is a clean ``role``/``content`` string that maps
    straight to a chat-completions turn.
    """
    messages: list[dict[str, str]] = [
        {"role": message.role, "content": (message.content or "")}
        for message in request.messages
    ]
    if probe.injected_codebook:
        messages.append({"role": "user", "content": _codebook_note(entries=probe.injected_codebook)})
    messages.append({"role": "user", "content": probe.prompt})
    return messages


def _output_schema(output_schema_name: str) -> type[BaseModel]:
    """Return the Pydantic output schema for the named probe shape."""
    if output_schema_name == "encode":
        return EncodeOutput
    if output_schema_name in ("decode", "tolerance"):
        return DecodeOutput
    raise ValueError(f"Unknown output_schema_name: {output_schema_name!r}")


def _output_fields(result: BaseModel) -> dict[str, str]:
    """Flatten a structured reply into the contract's string-valued output_fields."""
    if isinstance(result, EncodeOutput):
        return {"produced_form": result.produced_form}
    if isinstance(result, DecodeOutput):
        return {
            "interpreted_meaning": result.interpreted_meaning,
            "is_valid": "true" if result.is_valid else "false",
        }
    raise ValueError(f"Unexpected result type: {type(result).__name__}")


def _use_azure() -> bool:
    """Return True when Azure endpoint + key are configured."""
    return bool(os.environ.get(_AZURE_ENDPOINT_ENV)) and bool(os.environ.get(_AZURE_KEY_ENV))


def _use_portkey() -> bool:
    """Return True when a Portkey API key is configured."""
    return bool(os.environ.get(_PORTKEY_API_KEY_ENV))


def _is_claude_model(model: str) -> bool:
    """Return True for Anthropic model names, which Azure cannot serve."""
    return model.startswith("claude-")


def _portkey_headers() -> dict[str, str]:
    """Build Portkey gateway headers from environment.

    Portkey requires the request to resolve to a provider by one of: a Config ID
    (``x-portkey-config``, set by a project admin and bundling provider
    credentials), a Virtual Key (``x-portkey-virtual-key``, same idea, scoped to
    one provider), or a raw ``x-portkey-provider`` name paired with the real
    provider key forwarded as the ``Authorization`` bearer token -- the last of
    those does NOT work with only a Portkey API key, since Portkey then expects
    an actual Anthropic key in the Authorization header. A member of a
    Portkey project (as opposed to its admin) almost always has a Config ID or
    Virtual Key to use instead; ``PORTKEY_PROVIDER`` is kept only for the case
    where a real provider key is also supplied via ``OPENAI_API_KEY``-style
    passthrough on the client.
    """
    headers = {"x-portkey-api-key": os.environ[_PORTKEY_API_KEY_ENV]}
    config = os.environ.get(_PORTKEY_CONFIG_ENV)
    if config:
        headers["x-portkey-config"] = config
    virtual_key = os.environ.get(_PORTKEY_VIRTUAL_KEY_ENV)
    if virtual_key:
        headers["x-portkey-virtual-key"] = virtual_key
    provider = os.environ.get(_PORTKEY_PROVIDER_ENV)
    if provider:
        headers["x-portkey-provider"] = provider
    return headers


def _response_format(output_schema: type[BaseModel]) -> dict[str, Any]:
    """Build a strict JSON-schema ``response_format`` for a Pydantic output model."""
    schema = output_schema.model_json_schema()
    _enforce_strict_schema(schema=schema)
    return {
        "type": "json_schema",
        "json_schema": {"name": output_schema.__name__, "schema": schema, "strict": True},
    }


async def _azure_structured(
    messages: list[dict[str, str]],
    output_schema: type[BaseModel],
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> StructuredCall:
    """Call Azure OpenAI chat-completions with JSON-schema structured output.

    ``reasoning_effort`` (e.g. ``"minimal"``) is forwarded only when set; a reasoning
    model given a large classification batch otherwise spends the whole
    ``max_output_tokens`` budget on hidden reasoning and returns empty content, so a
    tighter effort plus a larger token budget keeps structured output from truncating.
    """
    endpoint = os.environ[_AZURE_ENDPOINT_ENV]
    parts = urlsplit(endpoint)
    resource_root = f"{parts.scheme}://{parts.netloc}"
    deployment = os.environ.get(_AZURE_DEPLOYMENT_ENV) or model
    api_version = os.environ.get(_AZURE_VERSION_ENV) or _DEFAULT_AZURE_API_VERSION
    client = AsyncAzureOpenAI(
        azure_endpoint=resource_root,
        api_key=os.environ[_AZURE_KEY_ENV],
        api_version=api_version,
        timeout=_AZURE_TIMEOUT_SECONDS,
    )
    extra: dict[str, Any] = {}
    if reasoning_effort is not None:
        extra["reasoning_effort"] = reasoning_effort
    try:
        completion = await client.chat.completions.create(
            model=deployment,
            messages=messages,
            response_format=_response_format(output_schema=output_schema),
            max_completion_tokens=max_output_tokens,
            **extra,
        )
    finally:
        await client.close()
    content = completion.choices[0].message.content
    if not content:
        choice = completion.choices[0]
        reported = completion.usage
        raise RuntimeError(
            f"Azure deployment {deployment!r} returned empty content "
            f"(finish_reason={choice.finish_reason}, "
            f"completion_tokens={reported.completion_tokens if reported else '?'}, "
            f"max_completion_tokens={max_output_tokens}); "
            "likely reasoning-token truncation -- raise max_output_tokens or lower reasoning_effort."
        )
    result = output_schema.model_validate_json(content)
    reported = completion.usage
    usage = TokenUsage(
        input_tokens=reported.prompt_tokens if reported else 0,
        output_tokens=reported.completion_tokens if reported else 0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return StructuredCall(result=result, usage=usage)


def _portkey_tool(output_schema: type[BaseModel]) -> dict[str, Any]:
    """Wrap a Pydantic output schema as a single OpenAI-style function tool.

    Anthropic (unlike OpenAI) has no strict JSON-schema response format; forced
    tool-use is its own recommended mechanism for reliable structured output, and
    Portkey's OpenAI-compatible endpoint passes ``tools``/``tool_choice`` straight
    through to the underlying Anthropic call.
    """
    schema = output_schema.model_json_schema()
    _enforce_strict_schema(schema=schema)
    return {
        "type": "function",
        "function": {
            "name": output_schema.__name__,
            "description": f"Return a {output_schema.__name__} reply.",
            "parameters": schema,
        },
    }


async def _portkey_structured(
    messages: list[dict[str, str]],
    output_schema: type[BaseModel],
    model: str,
    max_output_tokens: int,
) -> StructuredCall:
    """Call a Claude model through the Portkey gateway via forced tool-use.

    Uses the plain ``openai`` SDK pointed at Portkey's OpenAI-compatible base URL
    with Portkey auth carried in headers (``x-portkey-api-key`` and, if
    configured, ``x-portkey-virtual-key`` / ``x-portkey-provider``); the SDK's own
    ``api_key`` argument is unused by Portkey but required by the client
    constructor, so it is set to the Portkey key as a placeholder.
    """
    tool = _portkey_tool(output_schema=output_schema)
    tool_name = tool["function"]["name"]
    client = AsyncOpenAI(
        base_url=_PORTKEY_BASE_URL,
        api_key=os.environ[_PORTKEY_API_KEY_ENV],
        default_headers=_portkey_headers(),
        timeout=_PORTKEY_TIMEOUT_SECONDS,
    )
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            max_tokens=max_output_tokens,
        )
    finally:
        await client.close()
    choice = completion.choices[0]
    tool_calls = choice.message.tool_calls
    if not tool_calls:
        raise RuntimeError(
            f"Portkey/Claude model {model!r} returned no tool call "
            f"(finish_reason={choice.finish_reason}); expected a forced "
            f"{tool_name!r} call."
        )
    result = output_schema.model_validate_json(tool_calls[0].function.arguments)
    reported = completion.usage
    usage = TokenUsage(
        input_tokens=reported.prompt_tokens if reported else 0,
        output_tokens=reported.completion_tokens if reported else 0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return StructuredCall(result=result, usage=usage)


async def _openai_structured(
    messages: list[dict[str, str]],
    output_schema: type[BaseModel],
    model: str,
    reasoning_effort: str | None,
) -> StructuredCall:
    """Call the OpenAI API directly (chat completions, JSON-schema structured output)."""
    client = AsyncOpenAI()
    extra: dict[str, Any] = {}
    if reasoning_effort is not None:
        extra["reasoning_effort"] = reasoning_effort
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=_response_format(output_schema=output_schema),
            **extra,
        )
    finally:
        await client.close()
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError(
            f"model {model!r} returned empty content "
            f"(finish_reason={completion.choices[0].finish_reason})"
        )
    reported = completion.usage
    return StructuredCall(
        result=output_schema.model_validate_json(content),
        usage=TokenUsage(
            input_tokens=reported.prompt_tokens if reported else 0,
            output_tokens=reported.completion_tokens if reported else 0,
        ),
    )


def _select_backend(model: str) -> str:
    """Pick the backend from the environment.

    Claude models can only be served through Portkey, so they route there whenever it is
    configured regardless of what else is set; every other model follows environment
    priority.
    """
    if _is_claude_model(model) and _use_portkey():
        return "portkey"
    if _use_azure():
        return "azure"
    if _use_portkey():
        return "portkey"
    return "openai"


async def call_structured(
    messages: list[dict[str, str]],
    output_schema: type[BaseModel],
    model: str,
    caller: str,
    max_output_tokens: int = _OUTPUT_TOKEN_ALLOWANCE,
    reasoning_effort: str | None = None,
) -> StructuredCall:
    """Route a structured-output call to Azure, Portkey, or the OpenAI provider.

    Single entry point shared by the analysis-mode probe handler and other
    ling-analysis LLM callers (e.g. the meaning-grounded morpheme segmenter), so
    backend selection lives in exactly one place: Azure when configured, else
    Portkey when configured, else the built-in OpenAI provider. ``max_output_tokens``
    and ``reasoning_effort`` default to the probe-tuned values; the semantic
    judge overrides them for its larger, reasoning-light batch. ``reasoning_effort``
    is ignored on the Portkey/Claude branch (Anthropic has no equivalent knob).
    Every call's token usage and dollar cost is appended to the API cost ledger,
    attributed to ``caller``.
    """
    backend = _select_backend(model)
    if backend == "azure":
        call = await _azure_structured(
            messages=messages, output_schema=output_schema, model=model,
            max_output_tokens=max_output_tokens, reasoning_effort=reasoning_effort,
        )
    elif backend == "portkey":
        call = await _portkey_structured(
            messages=messages, output_schema=output_schema, model=model,
            max_output_tokens=max_output_tokens,
        )
    else:
        call = await _openai_structured(
            messages=messages, output_schema=output_schema, model=model,
            reasoning_effort=reasoning_effort,
        )
    logger.info(
        "%s: %s/%s in=%d out=%d", caller, backend, model,
        call.usage.input_tokens, call.usage.output_tokens,
    )
    return call


async def _handle(request: AnalysisModeRequest) -> AnalysisModeResponse:
    """Export the frozen thread, run one probe under the agent's own model, reply."""
    run_dir = Path(request.run_dir)
    export = await export_agent_thread(
        run_dir=run_dir,
        scenario_name=run_dir.parent.name,
        agent_id=request.agent_id,
        cutoff_round=request.cutoff_round,
        output_format="openai_chat",
        include_thinking=False,
        flatten_tools=True,
    )
    if not hasattr(export.request, "messages"):
        raise TypeError("openai_chat export did not yield a message list")

    model = export.meta.model
    output_schema = _output_schema(output_schema_name=request.output_schema_name)
    messages = _chat_messages(request=export.request, probe=request)

    call = await call_structured(
        messages=messages,
        output_schema=output_schema,
        model=model,
        caller="analysis_mode_probe",
    )

    backend_name = _select_backend(model)
    sys.stderr.write(
        f"[analysis_mode] {backend_name}/{model} "
        f"in={call.usage.input_tokens} out={call.usage.output_tokens}\n"
    )

    return AnalysisModeResponse(
        reasoning=call.result.reasoning,
        output_fields=_output_fields(result=call.result),
        model=model,
        provider=backend_name,
        usage={
            "input_tokens": call.usage.input_tokens,
            "output_tokens": call.usage.output_tokens,
        },
    )


def main() -> None:
    """Read one request from stdin, run the probe, write one response to stdout."""
    logging.basicConfig(level=logging.INFO)
    request = AnalysisModeRequest.model_validate_json(sys.stdin.read())
    response = asyncio.run(_handle(request=request))
    sys.stdout.write(response.model_dump_json())


if __name__ == "__main__":
    main()
