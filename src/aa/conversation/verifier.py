"""Hidden claim-level verifier for the v2 turn pipeline.

Semantic support is decided by one hidden verifier model call through the
OpenCode/LangChain adapter using OpenCode native JSON-schema structured
output backed by Pydantic. No keyword overlap, token stems, hand-written
phrase lists, or source-ID presence heuristics decide semantic support.

Deterministic code remains only where semantics are deterministic: exact
quotation/provenance validation, source/checksum/version validation, and
the ID-completeness gate (exactly one verdict per response unit).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from typing import Any
from xml.sax.saxutils import escape as _xml_escape
from xml.sax.saxutils import quoteattr as _xml_quoteattr

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError as PydanticValidationError

from aa.conversation.response_units import ResponseUnitDraft
from aa.conversation.v2_prompts import load_verifier_system_v2
from aa.conversation.verifier_schema import (
    VERIFIER_MAX_ATTEMPTS,
    GroundingResult,
    VerifierValidationError,
    validate_grounding_result,
    verifier_json_schema,
)

logger = logging.getLogger("aa.conversation.verifier")

VERIFIER_AGENT_V2 = "aa-verifier-v2"


def _escape(value: str) -> str:
    return _xml_escape(value, {"'": "&apos;", '"': "&quot;"})


def build_verifier_user_text(
    *,
    units: Sequence[ResponseUnitDraft],
    passages: Sequence[dict[str, Any]],
) -> str:
    """Render verifier input: ordered units plus exact book evidence.

    The payload repeats the closed output contract uniformly for every
    turn (Gate C live repair, run 37538518277: the scope vocabulary lived
    only in the system prompt, and the weak fallback model emitted
    schema-invalid scopes/ids so the verifier never served, with 13
    clarifications, missing verifier identity, and p95 34.4s/max 42.7s
    over the 30s budget). Repeating the exact ``unit_id`` copy rule, the
    closed scope vocabulary, the cite-only-supplied-ids rule, and the
    all_required_supported derivation rule here is turn-independent
    hardening, never an exact-question special case: AA-side Pydantic +
    completeness + checksum + quote checks stay strict.
    """
    lines: list[str] = [
        "Return one verdict per <unit> in order, copying each id attribute exactly;",
        'scope must be exactly one of "book", "product_meta", "conversation_glue".',
        "Cite only passage ids listed in <book_evidence> in evidence_passage_ids.",
        "Set all_required_supported true only when every unit is supported.",
        "<response_units>",
    ]
    for unit in units:
        lines.append(f"<unit id={_xml_quoteattr(unit.unit_id)}>{_escape(unit.text)}</unit>")
    lines.append("</response_units>")
    lines.append("<book_evidence>")
    if passages:
        for passage in passages:
            passage_id = str(passage.get("passage_id", ""))
            source_id = str(passage.get("source_id", passage.get("source", "")))
            section_id = str(passage.get("section_id", passage.get("section", "")))
            text = str(passage.get("text", ""))
            if not passage_id or not text:
                continue
            lines.append(
                f"<passage id={_xml_quoteattr(passage_id)} "
                f"source={_xml_quoteattr(source_id)} "
                f"section={_xml_quoteattr(section_id)}>"
                f"{_escape(text)}</passage>"
            )
    else:
        lines.append("(no book evidence supplied for this turn)")
    lines.append("</book_evidence>")
    return "\n".join(lines)


def _pack_index(passages: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for passage in passages:
        passage_id = passage.get("passage_id")
        text = passage.get("text")
        if isinstance(passage_id, str) and passage_id and isinstance(text, str) and text:
            index[passage_id] = passage
    return index


def check_cited_passage_ids(result: GroundingResult, *, pack_ids: set[str]) -> GroundingResult:
    """Fail closed when a verdict cites unknown or missing book evidence."""
    for verdict in result.units:
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
    """Fail closed when a verbatim quote is not in its cited passage(s).

    Quoted spans use the deterministic quote-span detector. A non-empty
    quoted span must appear verbatim in at least one cited exact passage
    for its unit; otherwise the draft cannot cross the product boundary.
    """
    from aa.conversation.output_limits import extract_quoted_spans

    by_id = _pack_index(passages)
    verdict_by_id = {verdict.unit_id: verdict for verdict in result.units}
    for unit in units:
        verdict = verdict_by_id.get(unit.unit_id)
        if verdict is None:
            raise VerifierValidationError(f"missing verdict for {unit.unit_id!r}")
        spans = [span for span in extract_quoted_spans(unit.text) if span.strip()]
        if not spans:
            continue
        cited_texts: list[str] = []
        for cited in verdict.evidence_passage_ids:
            entry = by_id.get(cited)
            if entry is not None:
                cited_texts.append(str(entry.get("text", "")))
        # Quotes in glue/meta units without cited passages cannot be
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


def coerce_grounding_result(
    data: object,
    *,
    units: Sequence[ResponseUnitDraft],
    passages: Sequence[dict[str, Any]],
) -> GroundingResult:
    """Validate one verifier object end to end (schema + deterministic)."""
    expected = [unit.unit_id for unit in units]
    if isinstance(data, GroundingResult):
        result = validate_grounding_result(data, expected_unit_ids=expected)
    elif isinstance(data, dict):
        try:
            parsed = GroundingResult.model_validate(data)
        except PydanticValidationError as exc:
            raise VerifierValidationError(f"verifier output invalid: {exc}") from exc
        result = validate_grounding_result(parsed, expected_unit_ids=expected)
    else:
        raise VerifierValidationError("verifier output is not a structured object")
    pack_ids = set(_pack_index(passages).keys())
    check_passage_checksums(passages)
    check_cited_passage_ids(result, pack_ids=pack_ids)
    check_exact_quotes(units=units, result=result, passages=passages)
    return result


def _reply_structured_content(reply: object) -> object:
    if isinstance(reply, BaseMessage):
        return reply.content
    return getattr(reply, "content", reply)


async def run_verifier(
    units: Sequence[ResponseUnitDraft],
    passages: Sequence[dict[str, Any]],
    *,
    model: Any,
) -> GroundingResult:
    """Invoke the hidden verifier and validate its structured verdict.

    A single OpenCode ``json_schema`` request is issued; OpenCode owns the
    bounded validation retry (``retryCount``). AA code performs exactly one
    Pydantic validation plus the deterministic completeness and
    exact-quote/provenance checks, and never reparses text or retries.
    """
    if not units:
        raise VerifierValidationError("verifier needs at least one response unit")
    system_text = load_verifier_system_v2()
    user_text = build_verifier_user_text(units=units, passages=passages)
    structured_invoke = getattr(model, "ainvoke_structured", None)
    if callable(structured_invoke):
        raw = await structured_invoke(
            user_text,
            system=system_text,
            schema=verifier_json_schema(),
            retry_count=VERIFIER_MAX_ATTEMPTS,
        )
        result = coerce_grounding_result(raw, units=units, passages=passages)
        logger.info("verifier output accepted", extra={"units": len(units)})
        return result
    reply = await model.ainvoke(
        [SystemMessage(content=system_text), HumanMessage(content=user_text)]
    )
    content = _reply_structured_content(reply)
    if isinstance(content, str):
        raise VerifierValidationError("verifier output is not a structured object")
    result = coerce_grounding_result(content, units=units, passages=passages)
    logger.info("verifier output accepted", extra={"units": len(units)})
    return result


__all__ = [
    "VERIFIER_AGENT_V2",
    "build_verifier_user_text",
    "check_cited_passage_ids",
    "check_exact_quotes",
    "check_passage_checksums",
    "coerce_grounding_result",
    "run_verifier",
]
