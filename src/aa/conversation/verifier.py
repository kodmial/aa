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

import hashlib
import json
import logging
import re
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


# Capability-compatible text fallback (kodmial/aa#192): provider-native
# ``json_schema`` structured output is an optional capability, not an
# assumption. Live evidence on exact main showed the simplified structured
# verifier request never reaching a served verdict on the Space Bunny
# fallback path (14/14 ``unavailable``, no ``aa-verifier-v2`` served
# identity) while ordinary text calls on the same fallback serve. When the
# native structured channel is unavailable, the verifier retries the same
# single-unit decision once as bounded plain-text JSON through the ordinary
# text path (already proven to serve) and strictly Pydantic-validates it.
# Grounding semantics are unchanged: only a fully validated decision is
# accepted; anything else fails closed. Provider 429 always propagates
# immediately and never triggers the text fallback.
VERIFIER_TEXT_JSON_SUFFIX = (
    "\n\nReturn ONLY a JSON object with exactly these keys: "
    '{"requires_book_evidence": boolean, "supported": boolean, '
    '"evidence_passage_ids": array of strings}. '
    "No other text, no markdown, no explanation."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_text_json_fences(text: str) -> str:
    """Remove one markdown code fence wrapper from a text JSON reply."""
    match = _FENCE_RE.search(text)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return text.strip()


def parse_text_json_decision(text: str) -> dict[str, Any]:
    """Strictly parse one text-path verifier decision (fail-closed).

    Accepts only a single JSON object with exactly the transport decision
    keys; extra keys, missing keys, wrong types, or non-object JSON all
    raise :class:`VerifierValidationError`. Turn-independent, never an
    exact-question special case.
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
        data = json.loads(candidate)
    except (json.JSONDecodeError, ValueError) as exc:
        raise VerifierValidationError(f"verifier text output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise VerifierValidationError("verifier text output is not a JSON object")
    # Strict validation (extra keys rejected, types checked) happens in
    # validate_unit_decision via coerce_single_verdict; surface its error
    # shape here for a uniform fail-closed boundary.
    validate_unit_decision(data)
    return dict(data)


# Errors that mean the native structured channel itself is unavailable on
# this provider path (schema rejected, structured payload missing,
# transient/timeout after model fallback). Provider 429 is deliberately
# absent: it must propagate for runner retire/restart, never fall back.
_STRUCTURED_CAPABILITY_ERRORS: tuple[type[BaseException], ...] = (
    OpenCodeDeterministicError,
    OpenCodeProviderAccessError,
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
) -> UnitVerdict:
    """Verify exactly one unit with the minimal boolean decision schema.

    Provider-native ``json_schema`` structured output is an optional
    capability (kodmial/aa#192): when the structured channel itself is
    unavailable on this provider path (deterministic schema rejection,
    missing structured payload, transient/timeout after model fallback),
    the same single-unit decision is retried once as bounded plain-text
    JSON through the ordinary text path and strictly validated. This stays
    inside the one concurrent per-unit round (no extra verifier round, no
    planner/retrieval/answer rerun). Provider 429 always propagates
    immediately for runner retire/restart.
    """
    system_text = load_verifier_system_v2()
    user_text = build_single_unit_text(unit=unit, passages=passages)
    window = list(passages[:VERIFIER_MAX_EVIDENCE_PASSAGES])
    short_to_full = display_id_map_for_window(window)
    full_ids = set(_pack_index(passages).keys())
    structured_invoke = getattr(model, "ainvoke_structured", None)
    if callable(structured_invoke):
        try:
            raw = await structured_invoke(
                user_text,
                system=system_text,
                schema=verifier_single_json_schema(),
                retry_count=VERIFIER_MAX_ATTEMPTS,
            )
        except OpenCodeRateLimitError:
            raise
        except _STRUCTURED_CAPABILITY_ERRORS:
            text_reply = await _ainvoke_verifier_text(
                model, user_text + VERIFIER_TEXT_JSON_SUFFIX, system_text
            )
            data = parse_text_json_decision(text_reply)
            return coerce_single_verdict(
                data, unit_id=unit.unit_id, short_to_full=short_to_full, full_ids=full_ids
            )
        if not isinstance(raw, dict):
            raise VerifierValidationError("verifier output is not a structured object")
        return coerce_single_verdict(
            raw, unit_id=unit.unit_id, short_to_full=short_to_full, full_ids=full_ids
        )
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
    Errors propagate fail-closed; provider 429 always propagates for
    runner retire/restart.
    """
    import asyncio as _asyncio

    verdicts = await _asyncio.gather(
        *(_verify_single_unit(unit, passages, model=model) for unit in units)
    )
    ordered = sorted(verdicts, key=lambda verdict: verdict.unit_id)
    all_supported = all(verdict.supported for verdict in ordered)
    assembled = GroundingResult(units=list(ordered), all_required_supported=all_supported)
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
    "VERIFIER_MAX_EVIDENCE_PASSAGES",
    "VERIFIER_MAX_PASSAGE_CHARS",
    "VERIFIER_TEXT_JSON_SUFFIX",
    "build_single_unit_text",
    "check_cited_passage_ids",
    "check_exact_quotes",
    "check_passage_checksums",
    "coerce_single_verdict",
    "display_id_map_for_window",
    "parse_text_json_decision",
    "resolve_cited_passage_ids",
    "run_verifier",
]
