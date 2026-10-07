"""Hidden claim-level verifier for the v2 turn pipeline.

Semantic support is decided by hidden per-unit verifier model calls through
the OpenCode/LangChain adapter using OpenCode native JSON-schema structured
output backed by Pydantic, with a bounded plain-text JSON fallback when the
native structured channel itself is unavailable on the provider path
(kodmial/aa#192). No keyword overlap, token stems, hand-written
phrase lists, or source-ID presence heuristics decide semantic support.

Transport contract (kodmial/aa#190): per-unit only, concurrent. Each
response unit gets exactly one verifier request; units run concurrently in
one round. The model emits only ``requires_book_evidence`` (boolean),
``supported`` (boolean) and ``evidence_passage_ids`` (string list). AA code
binds ``unit_id`` from the input unit, derives the internal scope
deterministically (``book`` when evidence is required, otherwise a
non-book glue-compatible scope), and computes ``all_required_supported``
as the conjunction. The model never copies ids and never computes the
aggregate.

Deterministic code remains only where semantics are deterministic: exact
quotation/provenance validation, source/checksum/version validation, and
the ID-completeness gate (exactly one verdict per response unit).
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Sequence
from typing import Any
from xml.sax.saxutils import escape as _xml_escape
from xml.sax.saxutils import quoteattr as _xml_quoteattr

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from aa.conversation.response_units import ResponseUnitDraft
from aa.conversation.v2_prompts import load_verifier_system_v2
from aa.conversation.verifier_schema import (
    NON_BOOK_SCOPE,
    VERIFIER_MAX_ATTEMPTS,
    GroundingResult,
    UnitVerdict,
    VerifierValidationError,
    validate_grounding_result,
    validate_unit_decision,
    verifier_single_json_schema,
)
from aa.opencode.errors import (
    OpenCodeDeterministicError,
    OpenCodeError,
    OpenCodeProviderAccessError,
    OpenCodeRateLimitError,
    OpenCodeTimeoutError,
    OpenCodeTransientError,
)

logger = logging.getLogger("aa.conversation.verifier")

VERIFIER_AGENT_V2 = "aa-verifier-v2"

# Bounded verifier evidence window: the full 16k-token Evidence Pack (up to
# 12 passages) makes the verifier prompt the largest per-turn structured
# model input. The display window keeps the top-ranked passages only,
# cutting verifier input tokens and latency while the stored-pack
# deterministic checks below still use the full pack, so grounding
# strictness is unchanged: the model may only cite listed ids, and every
# cited id is still validated against the full pack plus
# checksum/quote gates. Turn-independent, never an exact-question special
# case.
VERIFIER_MAX_EVIDENCE_PASSAGES = 6

# Bounded per-passage display length for the verifier prompt only.
# Truncating display text bounds input tokens and latency while
# deterministic cite/quote/checksum gates still use the full stored pack.
# Display truncation is explicitly marked with ``... [truncated ...]`` so
# the model can see the passage is incomplete and withhold support instead
# of judging on a silently cut prefix. Turn-independent, never an
# exact-question special case.
VERIFIER_MAX_PASSAGE_CHARS = 800

VERIFIER_TRUNCATION_SUFFIX_FORMAT = "... [truncated {omitted} chars omitted]"


def _display_passage_text(text: str) -> str:
    """Bound one passage display text for the verifier prompt (no semantics)."""
    if len(text) <= VERIFIER_MAX_PASSAGE_CHARS:
        return text
    omitted = len(text) - VERIFIER_MAX_PASSAGE_CHARS
    return text[:VERIFIER_MAX_PASSAGE_CHARS] + VERIFIER_TRUNCATION_SUFFIX_FORMAT.format(
        omitted=omitted
    )


def _escape(value: str) -> str:
    return _xml_escape(value, {"'": "&apos;", '"': "&quot;"})


def display_id_map_for_window(window: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Map short ordinal display ids (``p1``..``pN``) to full passage ids.

    Short ordinal ids are trivially copyable; the deterministic
    cite/quote/checksum gates below still validate the resolved full ids
    against the full stored pack, so grounding strictness is unchanged.
    Turn-independent, never an exact-question special case. Positions with
    a missing/empty full id or empty text are skipped here exactly as in
    the prompt builders below, keeping the display order and the map
    consistent.
    """
    mapping: dict[str, str] = {}
    position = 0
    for passage in window:
        if not isinstance(passage, dict):
            continue
        full_id = passage.get("passage_id", "")
        text = passage.get("text", "")
        if not isinstance(full_id, str) or not full_id:
            continue
        if not isinstance(text, str) or not text:
            continue
        position += 1
        mapping[f"p{position}"] = full_id
    return mapping


def resolve_cited_passage_ids(
    cited: Sequence[object],
    *,
    short_to_full: dict[str, str],
    full_ids: set[str],
) -> list[str]:
    """Resolve model-cited ids to full stored passage ids (fail-closed).

    Short display ids (``p1``..``pN``, surrounding whitespace tolerated)
    resolve through ``short_to_full``; full stored ids pass through
    unchanged for back-compat; anything else passes through untouched so
    the strict cite gate below still rejects it. Never invents ids.
    """
    resolved: list[str] = []
    for item in cited:
        text = item.strip() if isinstance(item, str) else item
        if isinstance(text, str) and text in short_to_full:
            resolved.append(short_to_full[text])
        elif isinstance(text, str) and text in full_ids:
            resolved.append(text)
        elif isinstance(item, str):
            resolved.append(item.strip())
        else:
            resolved.append(str(item))
    return resolved


def _pack_index(passages: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for passage in passages:
        passage_id = passage.get("passage_id")
        text = passage.get("text")
        if isinstance(passage_id, str) and passage_id and isinstance(text, str) and text:
            index[passage_id] = passage
    return index


def check_cited_passage_ids(result: GroundingResult, *, pack_ids: set[str]) -> GroundingResult:
    """Fail closed when a supported verdict cites unknown or missing evidence.

    An unsupported verdict (supported False) is already blocked and
    narrowed away from the user; its citation list is irrelevant to
    grounding, so only supported verdicts must cite valid evidence. A
    supported book unit with no/unknown citation still fails closed.
    Turn-independent, never an exact-question special case.
    """
    for verdict in result.units:
        if not verdict.supported:
            continue
        if verdict.scope == "book":
            if not verdict.evidence_passage_ids:
                raise VerifierValidationError(f"book unit {verdict.unit_id!r} cites no evidence")
            for cited in verdict.evidence_passage_ids:
                if cited not in pack_ids:
                    raise VerifierValidationError(
                        f"unit {verdict.unit_id!r} cites unknown passage {cited!r}"
                    )
        else:
            for cited in verdict.evidence_passage_ids:
                if cited not in pack_ids:
                    raise VerifierValidationError(
                        f"unit {verdict.unit_id!r} cites unknown passage {cited!r}"
                    )
    return result


def check_passage_checksums(passages: Sequence[dict[str, Any]]) -> None:
    """Fail closed when stored passage text does not match its checksum."""
    for passage in passages:
        text = passage.get("text")
        expected = passage.get("text_sha256")
        if not isinstance(text, str) or not text:
            continue
        if not isinstance(expected, str) or not expected:
            continue
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != expected:
            raise VerifierValidationError("evidence passage checksum mismatch")


def check_exact_quotes(
    *,
    units: Sequence[ResponseUnitDraft],
    result: GroundingResult,
    passages: Sequence[dict[str, Any]],
) -> GroundingResult:
    """Fail closed when a supported verdict quotes text not in cited passages.

    Quoted spans use the deterministic quote-span detector. A non-empty
    quoted span in a supported unit must appear verbatim in at least one
    cited exact passage for its unit; otherwise the draft cannot cross the
    product boundary. Unsupported units are already blocked and narrowed
    away, so their quotes are irrelevant. Turn-independent, never an
    exact-question special case.
    """
    from aa.conversation.output_limits import extract_quoted_spans

    by_id = _pack_index(passages)
    verdict_by_id = {verdict.unit_id: verdict for verdict in result.units}
    for unit in units:
        verdict = verdict_by_id.get(unit.unit_id)
        if verdict is None:
            raise VerifierValidationError(f"missing verdict for {unit.unit_id!r}")
        if not verdict.supported:
            continue
        spans = [span for span in extract_quoted_spans(unit.text) if span.strip()]
        if not spans:
            continue
        cited_texts: list[str] = []
        for cited in verdict.evidence_passage_ids:
            entry = by_id.get(cited)
            if entry is not None:
                cited_texts.append(str(entry.get("text", "")))
        # Quotes in non-book units without cited passages cannot be
        # proven exact: they fail closed as well.
        if not cited_texts:
            raise VerifierValidationError(
                f"unit {unit.unit_id!r} quotes text without cited exact passages"
            )
        for span in spans:
            if not any(span in candidate for candidate in cited_texts):
                raise VerifierValidationError(
                    f"unit {unit.unit_id!r} quotes text absent from cited passages"
                )
    return result


def _reply_structured_content(reply: object) -> object:
    if isinstance(reply, BaseMessage):
        return reply.content
    return getattr(reply, "content", reply)


# Capability-compatible text fallback (kodmial/aa#192, Gate C live repair):
# provider-native ``json_schema`` structured output is an optional
# capability, not an assumption. Live evidence on exact main showed the
# simplified structured verifier request never reaching a served verdict on
# the Space Bunny fallback path (planner structured serves on the same
# fallback while verifier stays ``unavailable`` with generic clarification
# collapse and max latency over the 30s budget) while ordinary text calls on
# the same fallback serve. When the native structured channel is
# unavailable, the verifier retries the same single-unit decision once as
# bounded plain-text JSON through the ordinary text path (already proven to
# serve) and strictly Pydantic-validates it. Grounding semantics are
# unchanged: only a fully validated decision is accepted; anything else
# fails closed. Provider 429 always propagates immediately and never
# triggers the text fallback. Turn-independent, never an exact-question
# special case.
VERIFIER_TEXT_JSON_SUFFIX = (
    "\n\nReturn ONLY a JSON object with exactly these keys: "
    '{"requires_book_evidence": boolean, "supported": boolean, '
    '"evidence_passage_ids": array of strings}. '
    'Example: {"requires_book_evidence": true, "supported": true, '
    '"evidence_passage_ids": ["p1"]}. '
    "Cite only short passage ids (p1..pN) from <book_evidence>. "
    "No other text, no markdown, no explanation."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# Bounded native-structured capability cache (Gate C live repair):
# a deterministic structured-channel failure proves the provider path does
# not serve native verifier structured output. Later units/turns on the
# same model path then go directly to the bounded text JSON path instead
# of burning a second slow provider round-trip per unit (the max-latency
# pathology: structured attempt + text fallback per unit makes every
# verifier call pay twice). The cache is per model path with a TTL so a
# recovered provider is retried and one transient timeout or one bad
# decision never permanently disables structured calls process-wide.
# Turn-independent, never question-specific. Tests reset it via
# clear_verifier_capability_cache().
VERIFIER_CAPABILITY_TTL_S = 300.0

_TEXT_JSON_PREFERRED_AT: dict[str, float] = {}
_CAPABILITY_LOCK = threading.Lock()


def _verifier_capability_key(model: Any | None) -> str:
    """Return the capability-cache key for one verifier model path."""
    try:
        primary = getattr(model, "primary_model", "")
        fallback = getattr(model, "fallback_model", "")
        agent = getattr(model, "agent", "")
        if (
            (isinstance(primary, str) and primary.strip())
            or (isinstance(fallback, str) and fallback.strip())
            or (isinstance(agent, str) and agent.strip())
        ):
            return (
                f"{primary.strip() if isinstance(primary, str) else ''}"
                f"|{fallback.strip() if isinstance(fallback, str) else ''}"
                f"|{agent.strip() if isinstance(agent, str) else ''}"
            )
    except Exception:
        pass
    if model is None:
        return ""
    return "default"


def _capability_entry_fresh(recorded_at: float) -> bool:
    try:
        return (time.monotonic() - float(recorded_at)) < VERIFIER_CAPABILITY_TTL_S
    except (TypeError, ValueError):
        return False


def structured_text_fallback_preferred(model: Any | None = None) -> bool:
    """Whether the process should skip native structured verifier calls.

    With no model, reports whether any cached model path is currently
    preferred (backward-compatible probe used by tests). With a model,
    reports only that model path so a different path still tries native
    structured output first.
    """
    with _CAPABILITY_LOCK:
        if model is None:
            fresh: list[str] = []
            for key, recorded_at in _TEXT_JSON_PREFERRED_AT.items():
                if _capability_entry_fresh(recorded_at):
                    fresh.append(key)
            # Drop expired entries so a recovered provider is retried.
            for key in list(_TEXT_JSON_PREFERRED_AT):
                if key not in fresh:
                    _TEXT_JSON_PREFERRED_AT.pop(key, None)
            return bool(fresh)
        key = _verifier_capability_key(model)
        if key == "":
            return any(_capability_entry_fresh(stamp) for stamp in _TEXT_JSON_PREFERRED_AT.values())
        cached_at: float | None = _TEXT_JSON_PREFERRED_AT.get(key)
        if cached_at is None:
            return False
        if not _capability_entry_fresh(cached_at):
            _TEXT_JSON_PREFERRED_AT.pop(key, None)
            return False
        return True


def mark_structured_unavailable(model: Any | None = None) -> None:
    """Remember that native structured verifier output is unavailable.

    Only deterministic capability failures call this (schema rejected,
    missing structured payload). Transient/timeout and content validation
    failures fall back once without poisoning the cache. Entries expire
    via ``VERIFIER_CAPABILITY_TTL_S`` and are keyed per model path.
    """
    key = _verifier_capability_key(model)
    if not key:
        key = "default"
    with _CAPABILITY_LOCK:
        _TEXT_JSON_PREFERRED_AT[key] = time.monotonic()


def clear_verifier_capability_cache() -> None:
    """Reset the structured-capability cache (tests only)."""
    with _CAPABILITY_LOCK:
        _TEXT_JSON_PREFERRED_AT.clear()


def _model_on_fallback_path(model: Any | None) -> bool:
    """Whether ``model`` is currently serving via its configured fallback.

    Gate C live repair (run 37598043365 on exact main 8f9ed2a): every live
    turn served the fallback while the verifier still attempted native
    structured output first, paying a slow structured round-trip per unit
    before the text fallback (the max-latency pathology, max 54s over the
    30s budget) and collapsing to verifier-unavailable when the weak
    fallback path rejected structured output. When the shared primary
    circuit is already open the structured attempt on the fallback path is
    doomed, so later units/turns go directly to the bounded text path.
    Turn-independent, never an exact-question special case; provider 429
    still propagates and never triggers this path.
    """
    try:
        fast = getattr(model, "_fast_fallback_available", None)
        if callable(fast) and bool(fast()):
            return True
        circuit = getattr(model, "_primary_circuit_open", None)
        if callable(circuit) and bool(circuit()):
            return True
    except Exception:
        return False
    return False


def _verifier_prefers_text(model: Any | None) -> bool:
    """Whether one verifier model path should use the text path directly."""
    if structured_text_fallback_preferred(model):
        return True
    return _model_on_fallback_path(model)


def _strip_text_json_fences(text: str) -> str:
    """Remove one markdown code fence wrapper from a text JSON reply."""
    match = _FENCE_RE.search(text)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return text.strip()


def _scan_double_quoted(text: str) -> list[bool]:
    """Mark characters inside double-quoted JSON strings (escape-aware)."""
    inside = [False] * len(text)
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            inside[i] = True
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
                inside[i] = True
    return inside


def _strip_trailing_commas_safe(text: str) -> str:
    """Strip trailing commas only outside double-quoted strings."""
    inside = _scan_double_quoted(text)
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch == "," and not inside[i]:
            j = i + 1
            while j < len(text) and text[j] in " \t\r\n":
                j += 1
            if j < len(text) and text[j] in ("}", "]"):
                continue
        out.append(ch)
    return "".join(out)


def _replace_python_literals_safe(text: str) -> str:
    """Replace ``True``/``False``/``None`` only outside string values.

    Blind ``str.replace`` rewrites substrings inside quoted
    ``evidence_passage_ids`` values (for example ``"True"`` becomes
    ``"true"``), corrupting cited ids before validation. This scanner
    replaces whole-word Python literals only when they appear outside
    double-quoted strings, leaving quoted values byte-identical.
    """
    inside = _scan_double_quoted(text)
    out: list[str] = []
    i = 0
    while i < len(text):
        if inside[i]:
            out.append(text[i])
            i += 1
            continue
        rest = text[i:]
        replaced = False
        for word, json_word in (("True", "true"), ("False", "false"), ("None", "null")):
            if rest.startswith(word):
                before = text[i - 1] if i > 0 else ""
                after = text[i + len(word)] if i + len(word) < len(text) else ""
                if (not before.isalnum() and before != "_") and (
                    not after.isalnum() and after != "_"
                ):
                    out.append(json_word)
                    i += len(word)
                    replaced = True
                    break
        if not replaced:
            out.append(text[i])
            i += 1
    return "".join(out)


def _tolerant_json_candidates(candidate: str) -> list[str]:
    """Build bounded normalized candidates for one text JSON payload.

    Order: strict, trailing-comma stripped, Python literals, Python
    literals plus trailing commas. All rewrites apply only outside
    double-quoted string values so quoted ``evidence_passage_ids`` are
    never corrupted. Single-quoted payloads are handled via
    ``ast.literal_eval`` in :func:`_tolerant_json_loads`, never via blind
    quote replacement that would corrupt apostrophes. Deduplicated, at
    most four entries.
    """
    out: list[str] = [candidate]
    stripped = _strip_trailing_commas_safe(candidate)
    if stripped != candidate:
        out.append(stripped)
    python_fixed = _replace_python_literals_safe(candidate)
    if python_fixed not in out:
        out.append(python_fixed)
    normalized_python = _strip_trailing_commas_safe(python_fixed)
    if normalized_python not in out:
        out.append(normalized_python)
    return out[:4]


def _tolerant_json_loads(candidate: str) -> object:
    """Parse one JSON object tolerating common small-model deviations.

    Strict JSON is tried first, then the bounded string-safe normalized
    candidates above (trailing commas, Python literals outside strings).
    Single-quoted Python-dict payloads are parsed with
    ``ast.literal_eval`` (which preserves quoted values exactly) instead
    of blind quote/literal replacement. Key/type strictness is enforced
    later by Pydantic (extra keys still rejected), so grounding is
    unchanged.
    """
    last_error: Exception | None = None
    for text in _tolerant_json_candidates(candidate):
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
    # Single-quoted or Python-literal payloads: literal_eval preserves
    # quoted string values exactly (no blind True/None/apostrophe rewrite).
    for text in _tolerant_json_candidates(candidate):
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as exc:
            last_error = exc
            continue
        if isinstance(value, (dict, list)):
            return value
        return value
    assert last_error is not None
    raise last_error


def parse_text_json_decision(text: str) -> dict[str, Any]:
    """Strictly parse one text-path verifier decision (fail-closed).

    Accepts only a single JSON object with exactly the transport decision
    keys; extra keys, missing keys, wrong types, or non-object JSON all
    raise :class:`VerifierValidationError`. Small-model text deviations
    (fences, leading explanations, trailing commas, single quotes, Python
    literals) are normalized before parsing, but Pydantic key/type
    strictness is unchanged. Turn-independent, never an exact-question
    special case.
    """
    cleaned = _strip_text_json_fences(text or "")
    if not cleaned:
        raise VerifierValidationError("verifier text output is empty")
    candidate = cleaned
    if not candidate.lstrip().startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise VerifierValidationError("verifier text output is not a JSON object")
        candidate = candidate[start : end + 1]
    try:
        data = _tolerant_json_loads(candidate)
    except (json.JSONDecodeError, ValueError, SyntaxError, TypeError) as exc:
        raise VerifierValidationError(f"verifier text output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise VerifierValidationError("verifier text output is not a JSON object")
    # Strict validation (extra keys rejected, types checked) happens in
    # validate_unit_decision via coerce_single_verdict; surface its error
    # shape here for a uniform fail-closed boundary.
    validate_unit_decision(data)
    return dict(data)


# Errors that mean the native structured channel itself is unavailable on
# this provider path (deterministic schema rejection, provider access
# rejection). Only these poison the bounded per-model capability cache.
# Transient/timeout failures still fall back once to text but never mark
# the cache, so one slow timeout cannot permanently disable structured
# calls after the provider recovers. Provider 429 is deliberately absent:
# it must propagate for runner retire/restart, never fall back.
_STRUCTURED_CAPABILITY_ERRORS: tuple[type[BaseException], ...] = (
    OpenCodeDeterministicError,
    OpenCodeProviderAccessError,
)

# Transient/timeout after model fallback: retry once via the text path
# without marking the capability cache (recovery-friendly).
_STRUCTURED_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    OpenCodeTransientError,
    OpenCodeTimeoutError,
)


async def _ainvoke_verifier_text(
    model: Any,
    user_text: str,
    system_text: str,
) -> str:
    """Invoke the ordinary text path and return its raw text (429 propagates)."""
    text_invoke = getattr(model, "_ainvoke_text", None)
    if callable(text_invoke):
        reply = await text_invoke(user_text, system=system_text)
        if not isinstance(reply, str):
            raise VerifierValidationError("verifier text output is not text")
        return reply
    plain_invoke = getattr(model, "ainvoke", None)
    if not callable(plain_invoke):
        raise VerifierValidationError("verifier model has no text invocation path")
    reply = await plain_invoke(
        [SystemMessage(content=system_text), HumanMessage(content=user_text)]
    )
    content = _reply_structured_content(reply)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(str(item.get("text")))
        return "\n".join(parts)
    raise VerifierValidationError("verifier text output is not text")


def build_single_unit_text(
    *,
    unit: ResponseUnitDraft,
    passages: Sequence[dict[str, Any]],
) -> str:
    """Render the minimal single-unit verifier payload (no id copying).

    The model judges exactly one unit and returns only booleans plus the
    citation list. It never copies unit ids, never emits a scope string,
    and never computes an aggregate flag: AA code binds the id, derives
    the internal scope, and computes the conjunction. Turn-independent,
    never an exact-question special case.
    """
    lines: list[str] = [
        "Judge exactly one response unit below.",
        "Return requires_book_evidence true when the unit contains any substantive "
        "external claim about recovery, the program, the world, the user, or a "
        "recommended action; return false only for pure conversation glue or a "
        "truthful assistant identity or capability statement grounded in stable "
        "system instructions. A general offer to help discuss common topics in "
        "general terms needs no book evidence; any specific program fact, "
        "mechanism, or recommended action always needs book evidence.",
        "Set supported true only when every substantive proposition in the unit is "
        "semantically established by the cited passages; otherwise set supported false. "
        "A unit that needs book evidence but cites no passage is unsupported.",
        "Cite only passage ids listed in <book_evidence> in evidence_passage_ids; "
        "passages are numbered p1..pN, cite those short ids.",
        "<response_unit>",
        _escape(unit.text),
        "</response_unit>",
        "<book_evidence>",
    ]
    window = list(passages[:VERIFIER_MAX_EVIDENCE_PASSAGES])
    if window:
        position = 0
        for passage in window:
            if not isinstance(passage, dict):
                continue
            raw_id = passage.get("passage_id", "")
            raw_text = passage.get("text", "")
            if not isinstance(raw_id, str) or not raw_id:
                continue
            if not isinstance(raw_text, str) or not raw_text:
                continue
            raw_source = passage.get("source_id", passage.get("source", ""))
            raw_section = passage.get("section_id", passage.get("section", ""))
            source_id = raw_source if isinstance(raw_source, str) else ""
            section_id = raw_section if isinstance(raw_section, str) else ""
            text = raw_text
            position += 1
            lines.append(
                f"<passage id={_xml_quoteattr(f'p{position}')} "
                f"source={_xml_quoteattr(source_id)} "
                f"section={_xml_quoteattr(section_id)}>"
                f"{_escape(_display_passage_text(text))}</passage>"
            )
        if position == 0:
            lines.append("(no book evidence supplied for this turn)")
    else:
        lines.append("(no book evidence supplied for this turn)")
    lines.append("</book_evidence>")
    return "\n".join(lines)


def coerce_single_verdict(
    data: object,
    *,
    unit_id: str,
    short_to_full: dict[str, str] | None = None,
    full_ids: set[str] | None = None,
) -> UnitVerdict:
    """Validate one single-unit decision and bind it to ``unit_id``.

    The transport decision carries only booleans plus citations. AA code
    binds ``unit_id`` from the input unit, derives the internal scope
    deterministically (``book`` when book evidence is required, otherwise
    the non-book glue-compatible scope), and never lets the model copy ids
    or compute the aggregate. Short display ids resolve to full stored
    passage ids before validation so the strict cite gate sees full ids;
    unknown ids still fail closed.
    """
    decision = validate_unit_decision(data)
    raw_ids: list[str] = list(decision.evidence_passage_ids)
    stripped: list[str] = [item.strip() if isinstance(item, str) else str(item) for item in raw_ids]
    if short_to_full:
        resolved = resolve_cited_passage_ids(
            stripped,
            short_to_full=short_to_full,
            full_ids=full_ids or set(),
        )
    else:
        resolved = stripped
    scope = "book" if decision.requires_book_evidence else NON_BOOK_SCOPE
    return UnitVerdict(
        unit_id=unit_id,
        scope=scope,
        supported=bool(decision.supported),
        evidence_passage_ids=resolved,
    )


async def _verify_single_unit(
    unit: ResponseUnitDraft,
    passages: Sequence[dict[str, Any]],
    *,
    model: Any,
    _prefer_text_snapshot: bool | None = None,
) -> UnitVerdict:
    """Verify exactly one unit with the minimal boolean decision schema.

    Provider-native ``json_schema`` structured output is an optional
    capability (kodmial/aa#192, Gate C live repair): when the structured
    channel itself is unavailable on this provider path (deterministic
    schema rejection, missing structured payload), the same single-unit
    decision is retried once as bounded plain-text JSON through the
    ordinary text path and strictly validated. Transient/timeout and
    structured content-validation failures also retry once via text but
    never poison the capability cache, so one slow call or one bad
    decision cannot permanently disable structured calls. After a
    deterministic capability failure the model path prefers the text path
    directly (bounded by TTL, per model path) so later units/turns do not
    burn two slow provider round-trips per unit (the live max-latency
    pathology). When the shared primary circuit is already open (live
    fallback path serving after a primary rejection) the structured
    attempt is skipped proactively for the same reason. The text path
    itself retries at most once on strict-validation failure so one
    weak-model non-compliant reply does not collapse the turn to
    verifier-unavailable without a second bounded attempt; grounding
    stays strict because every attempt is fully Pydantic-validated.
    This stays inside the one concurrent per-unit round (no
    extra verifier round, no planner/retrieval/answer rerun). Provider 429
    always propagates immediately for runner retire/restart.
    """

    async def _text_decision() -> UnitVerdict:
        try:
            text_reply = await _ainvoke_verifier_text(
                model, user_text + VERIFIER_TEXT_JSON_SUFFIX, system_text
            )
            data = parse_text_json_decision(text_reply)
            return coerce_single_verdict(
                data, unit_id=unit.unit_id, short_to_full=short_to_full, full_ids=full_ids
            )
        except VerifierValidationError:
            # One weak-model non-compliant text reply (prose instead of
            # JSON, wrong keys/types, unknown passage ids): retry once
            # with the same bounded prompt before failing closed. Only
            # validation failures retry here; provider errors (including
            # 429) propagate immediately and never retry.
            logger.info("verifier text validation retry used")
            text_retry = await _ainvoke_verifier_text(
                model, user_text + VERIFIER_TEXT_JSON_SUFFIX, system_text
            )
            retry_data = parse_text_json_decision(text_retry)
            return coerce_single_verdict(
                retry_data,
                unit_id=unit.unit_id,
                short_to_full=short_to_full,
                full_ids=full_ids,
            )

    system_text = load_verifier_system_v2()
    user_text = build_single_unit_text(unit=unit, passages=passages)
    window = list(passages[:VERIFIER_MAX_EVIDENCE_PASSAGES])
    short_to_full = display_id_map_for_window(window)
    full_ids = set(_pack_index(passages).keys())
    structured_invoke = getattr(model, "ainvoke_structured", None)
    # Snapshot the capability at round start so concurrent units in one
    # turn behave consistently (all try structured or all use text).
    # The cache is still marked on deterministic failure for future turns.
    # An already-open primary circuit (fallback path serving) also prefers
    # text proactively so the fallback path never burns a doomed
    # structured round-trip per unit.
    prefer_text = (
        _prefer_text_snapshot
        if _prefer_text_snapshot is not None
        else _verifier_prefers_text(model)
    )
    if callable(structured_invoke) and not prefer_text:
        try:
            raw = await structured_invoke(
                user_text,
                system=system_text,
                schema=verifier_single_json_schema(),
                retry_count=VERIFIER_MAX_ATTEMPTS,
            )
        except OpenCodeRateLimitError:
            raise
        except _STRUCTURED_CAPABILITY_ERRORS as exc:
            logger.info(
                "verifier structured channel unavailable; text fallback used",
                extra={"category": type(exc).__name__},
            )
            mark_structured_unavailable(model)
            return await _text_decision()
        except _STRUCTURED_TRANSIENT_ERRORS as exc:
            # One transient/timeout: retry once via text and prefer the text
            # path for subsequent turns (TTL-bounded) so every later turn
            # does not burn two slow provider round-trips per unit (the live
            # max-latency pathology). The TTL expiry re-probes structured
            # output after recovery.
            logger.info(
                "verifier structured transient; text fallback used",
                extra={"category": type(exc).__name__},
            )
            mark_structured_unavailable(model)
            return await _text_decision()
        except OpenCodeError as exc:
            # Any other provider-side structured failure (for example a
            # startup/not-ready/session error from the OpenCode boundary)
            # retries once via the bounded text path instead of collapsing
            # the turn to verifier-unavailable without a text attempt.
            # Provider 429 is already re-raised above and never falls back.
            logger.info(
                "verifier structured provider error; text fallback used",
                extra={"category": type(exc).__name__},
            )
            mark_structured_unavailable(model)
            return await _text_decision()
        if not isinstance(raw, dict):
            logger.info(
                "verifier structured payload missing; text fallback used",
            )
            mark_structured_unavailable(model)
            return await _text_decision()
        try:
            return coerce_single_verdict(
                raw, unit_id=unit.unit_id, short_to_full=short_to_full, full_ids=full_ids
            )
        except VerifierValidationError as exc:
            # The provider returned a structured object that fails the
            # transport decision schema (for example loose enforcement on a
            # fallback route emitting extra keys, or a content grounding
            # failure like bad passage ids). The text path with its
            # explicit single-object instruction may still serve; try it
            # once before failing closed and prefer text subsequently
            # (TTL-bounded) so later turns do not pay two provider
            # round-trips per unit. Grounding stays strict: the text
            # decision is still fully Pydantic-validated.
            logger.info(
                "verifier structured decision invalid; text fallback used",
                extra={"category": type(exc).__name__},
            )
            mark_structured_unavailable(model)
            return await _text_decision()
    elif callable(structured_invoke) and prefer_text:
        return await _text_decision()
    reply = await model.ainvoke(
        [SystemMessage(content=system_text), HumanMessage(content=user_text)]
    )
    content = _reply_structured_content(reply)
    if isinstance(content, str):
        data = parse_text_json_decision(content)
        return coerce_single_verdict(
            data, unit_id=unit.unit_id, short_to_full=short_to_full, full_ids=full_ids
        )
    return coerce_single_verdict(
        content, unit_id=unit.unit_id, short_to_full=short_to_full, full_ids=full_ids
    )


async def _verify_per_unit_concurrent(
    units: Sequence[ResponseUnitDraft],
    passages: Sequence[dict[str, Any]],
    *,
    model: Any,
) -> GroundingResult:
    """Verify each unit concurrently with the boolean decision schema.

    Exactly one concurrent round. Deterministic cite/quote/checksum gates
    run on the assembled result, so grounding strictness is unchanged.
    A non-429 failure in one unit fails only that unit closed and preserves
    independently verified units for deterministic narrowing. Provider 429
    always propagates for runner retire/restart.
    """
    import asyncio as _asyncio

    prefer_text_snapshot = _verifier_prefers_text(model)
    raw_results = await _asyncio.gather(
        *(
            _verify_single_unit(
                unit, passages, model=model, _prefer_text_snapshot=prefer_text_snapshot
            )
            for unit in units
        ),
        return_exceptions=True,
    )

    # A single provider/format failure must not discard independently
    # verified units from the same draft. Failed units are represented as
    # deterministic unsupported book claims, so they can never cross the
    # product boundary; successfully verified units remain eligible for
    # deterministic narrowing. Provider 429 is different: the whole runner
    # must retire/restart and therefore always propagates.
    expected_unavailable_errors = (OpenCodeError, VerifierValidationError)
    for item in raw_results:
        if isinstance(item, OpenCodeRateLimitError):
            raise item
        if isinstance(item, BaseException) and not isinstance(item, expected_unavailable_errors):
            # Cancellation/system exceptions and programming defects are not
            # transport unavailability. Do not silently turn them into a
            # partial user answer.
            raise item

    verdicts: list[UnitVerdict] = []
    unavailable_unit_ids: list[str] = []
    first_unavailable: BaseException | None = None
    for unit, item in zip(units, raw_results, strict=True):
        if isinstance(item, expected_unavailable_errors):
            if first_unavailable is None:
                first_unavailable = item
            unavailable_unit_ids.append(unit.unit_id)
            verdicts.append(
                UnitVerdict(
                    unit_id=unit.unit_id,
                    scope="book",
                    supported=False,
                    evidence_passage_ids=[],
                )
            )
            logger.info(
                "verifier unit unavailable; failing only that unit closed",
                extra={"category": type(item).__name__},
            )
        else:
            assert isinstance(item, UnitVerdict)
            verdicts.append(item)

    # If every unit failed at the provider/format boundary there is no
    # verified material to salvage. Preserve the historical unavailable
    # signal so Gate C can attribute a total verifier outage correctly.
    if unavailable_unit_ids and len(unavailable_unit_ids) == len(units):
        assert first_unavailable is not None
        raise first_unavailable

    ordered = sorted(verdicts, key=lambda verdict: verdict.unit_id)
    all_supported = not unavailable_unit_ids and all(verdict.supported for verdict in ordered)
    assembled = GroundingResult(
        units=list(ordered),
        all_required_supported=all_supported,
        unavailable_unit_ids=unavailable_unit_ids,
    )
    result = validate_grounding_result(
        assembled, expected_unit_ids=[unit.unit_id for unit in units]
    )
    pack_ids = set(_pack_index(passages).keys())
    check_passage_checksums(passages)
    check_cited_passage_ids(result, pack_ids=pack_ids)
    check_exact_quotes(units=units, result=result, passages=passages)
    return result


async def run_verifier(
    units: Sequence[ResponseUnitDraft],
    passages: Sequence[dict[str, Any]],
    *,
    model: Any,
) -> GroundingResult:
    """Invoke the hidden verifier once per unit, concurrently.

    Per-unit only production path (kodmial/aa#190): no batch-first round,
    no fallback round, no transport-format retry. One concurrent round of
    minimal boolean decisions keeps the live SLO to a single round.
    Validation-shaped failures fail closed immediately without re-running
    planner/retrieval/answer. Provider 429 always propagates immediately
    for runner retire/restart and never triggers extra calls.
    Turn-independent, never an exact-question special case.
    """
    if not units:
        raise VerifierValidationError("verifier needs at least one response unit")
    result = await _verify_per_unit_concurrent(units, passages, model=model)
    logger.info("verifier output accepted", extra={"units": len(units)})
    return result


__all__ = [
    "VERIFIER_AGENT_V2",
    "VERIFIER_CAPABILITY_TTL_S",
    "VERIFIER_MAX_EVIDENCE_PASSAGES",
    "VERIFIER_MAX_PASSAGE_CHARS",
    "VERIFIER_TEXT_JSON_SUFFIX",
    "build_single_unit_text",
    "check_cited_passage_ids",
    "check_exact_quotes",
    "check_passage_checksums",
    "clear_verifier_capability_cache",
    "coerce_single_verdict",
    "display_id_map_for_window",
    "mark_structured_unavailable",
    "parse_text_json_decision",
    "resolve_cited_passage_ids",
    "run_verifier",
    "structured_text_fallback_preferred",
]
