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
    UnitVerdict,
    VerifierValidationError,
    validate_grounding_result,
    verifier_json_schema,
    verifier_single_json_schema,
)

logger = logging.getLogger("aa.conversation.verifier")

VERIFIER_AGENT_V2 = "aa-verifier-v2"

# Bounded verifier evidence window (Gate C live repair, run 37544234331:
# 14/14 ordinary turns collapsed to generic clarification with the verifier
# never served (missing aa-verifier-v2 identity) and p50 18.8s/p95 38.7s/
# max 41.1s over the 30s hard budget on the weak fallback model. The full
# 16k-token Evidence Pack (up to 12 passages) makes the verifier prompt the
# largest per-turn model input; weak providers are slow and flaky on large
# structured prompts while the planner (small prompt) serves. The display
# window keeps the top-ranked passages only, cutting verifier input tokens
# and latency while the stored-pack deterministic checks below still use
# the full pack, so grounding strictness is unchanged: the model may only
# cite listed ids, and every cited id is still validated against the full
# pack plus checksum/quote gates. Turn-independent, never an
# exact-question special case.
VERIFIER_MAX_EVIDENCE_PASSAGES = 8

# Bounded per-passage display length for the verifier prompt only (Gate C
# live repair, run 37551226807: 14/14 clarifications with the verifier
# never served in the served-model audit, answer collapse, diversity fail,
# and max 57.9s over the 30s budget on the weak fallback). The verifier
# prompt is the largest per-turn model input; weak fallback providers
# reject or time out large structured requests (no served audit) while the
# small-prompt planner serves. Truncating display text bounds input tokens
# and latency while deterministic cite/quote/checksum gates still use the
# full stored pack. Display truncation is explicitly marked with
# ``... [truncated ...]`` so the model can see the passage is incomplete
# and withhold support instead of judging on a silently cut prefix
# (a qualifier or contradiction after the cut is invisible to the model,
# so silent truncation could cause false-supported, not only safe
# false-unsupported). Turn-independent, never an exact-question
# special case.
VERIFIER_MAX_PASSAGE_CHARS = 1200

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

    Gate C live repair, run 37556798996: 13/14 ordinary turns collapsed
    to generic clarification with ``verifier_outcome=unavailable``,
    diversity fail, and max 50.7s over the 30s budget while every model
    call served the weak fallback. Stored passage ids are long
    provenance strings (``section#exp0001``/``section#atom-...``) that a
    weak model must copy exactly; flaked citations fail
    ``check_cited_passage_ids`` deterministically (no per-unit retry),
    so the turn clarifies and the batch + per-unit double slow round
    burns the latency budget. Short ordinal ids are trivially copyable;
    the deterministic cite/quote/checksum gates below still validate the
    resolved full ids against the full stored pack, so grounding
    strictness is unchanged. Turn-independent, never an exact-question
    special case. Positions with a missing/empty full id or empty text
    are skipped here exactly as in the prompt builders below, keeping the
    display order and the map consistent.
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
        "Cite only passage ids listed in <book_evidence> in evidence_passage_ids; "
        "passages are numbered p1..pN, cite those short ids.",
        "Set all_required_supported true only when every unit is supported.",
        "<response_units>",
    ]
    for unit in units:
        lines.append(f"<unit id={_xml_quoteattr(unit.unit_id)}>{_escape(unit.text)}</unit>")
    lines.append("</response_units>")
    lines.append("<book_evidence>")
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

    Gate C live repair, run 37551226807: 14/14 clarifications with the
    verifier never served. An unsupported verdict (supported False) is
    already blocked and narrowed away from the user; its citation list is
    irrelevant to grounding, so only supported verdicts must cite valid
    evidence. A supported book unit with no/unknown citation still fails
    closed. Turn-independent, never an exact-question special case.
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
    away, so their quotes are irrelevant (Gate C live repair, run
    37551226807: turning correctly-unsupported verdicts into unavailable
    generic clarifications). Turn-independent, never an exact-question
    special case.
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


def _normalize_verifier_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize weak-provider formatting before strict Pydantic validation.

    Gate C live repair, run 37544234331: the weak fallback model emits
    schema-valid-intent verdicts with systematic formatting variance
    (capitalized scopes like ``Book``, surrounding whitespace in ids,
    hyphen/underscore confusion in ``conversation-glue``) that fail the
    strict native/Pydantic enum and the ID-completeness gate, so the
    verifier never serves and every ordinary turn clarifies. Normalization
    is turn-independent hardening, never an exact-question special case:
    the closed scope vocabulary, exact id matching (after trim), and the
    cite/quote/checksum gates below stay strict; only case, surrounding
    whitespace, and the hyphen/underscore separator are tolerated.
    """
    normalized: dict[str, Any] = dict(data)
    raw_units = normalized.get("units")
    if not isinstance(raw_units, list):
        return normalized
    fixed_units: list[Any] = []
    for entry in raw_units:
        if not isinstance(entry, dict):
            fixed_units.append(entry)
            continue
        fixed = dict(entry)
        unit_id = fixed.get("unit_id")
        if isinstance(unit_id, str):
            stripped = unit_id.strip()
            fixed["unit_id"] = stripped
        scope = fixed.get("scope")
        if isinstance(scope, str):
            cleaned = scope.strip().lower().replace("-", "_")
            fixed["scope"] = cleaned
        evidence_ids = fixed.get("evidence_passage_ids")
        if isinstance(evidence_ids, list):
            fixed["evidence_passage_ids"] = [
                item.strip() if isinstance(item, str) else item for item in evidence_ids
            ]
        fixed_units.append(fixed)
    normalized["units"] = fixed_units
    return normalized


def _remap_units_by_order(
    raw_units: list[Any], expected_ids: list[str]
) -> list[dict[str, Any]] | None:
    """Reassign verdict ids by position when counts match (0 extra calls).

    Gate C live repair, run 37551226807: the weak fallback systematically
    flakes exact ``u1``..``uN`` id copying (wrong ids, off-by-one,
    ``1``/``unit1`` variants) while preserving order and verdict content,
    so the batch verifier never serves and every ordinary turn clarifies
    with per-unit fallback burning a second slow-model round over the 30s
    budget. When the model returns the correct number of verdicts in order,
    reassigning ``expected_ids`` by position preserves grounding without
    extra latency: scope/supported/evidence content is untouched, and the
    full cite/quote/checksum gates below still validate against the full
    stored pack. Turn-independent, never an exact-question special case.
    Returns ``None`` when counts mismatch, entries are not dicts, or the
    returned id set already matches expected (count equality does not
    prove order preservation: remapping a shuffled but id-correct list
    would silently misattribute supported status and evidence to the
    wrong unit). Remap applies only when ids are systematically wrong;
    any overlap with expected ids fails closed instead of clobbering a
    correct binding by position.
    """
    if len(raw_units) != len(expected_ids) or not expected_ids:
        return None
    raw_ids = [entry.get("unit_id") if isinstance(entry, dict) else None for entry in raw_units]
    if set(raw_ids) == set(expected_ids):
        return None
    expected_set = set(expected_ids)
    if any(raw_id in expected_set for raw_id in raw_ids):
        return None
    remapped: list[dict[str, Any]] = []
    for entry, expected_id in zip(raw_units, expected_ids, strict=True):
        if not isinstance(entry, dict):
            return None
        fixed = dict(entry)
        fixed["unit_id"] = expected_id
        remapped.append(fixed)
    return remapped


def coerce_grounding_result(
    data: object,
    *,
    units: Sequence[ResponseUnitDraft],
    passages: Sequence[dict[str, Any]],
) -> GroundingResult:
    """Validate one verifier object end to end (schema + deterministic)."""
    expected = [unit.unit_id for unit in units]
    if isinstance(data, GroundingResult):
        try:
            result = validate_grounding_result(data, expected_unit_ids=expected)
        except VerifierValidationError:
            remapped_units = _remap_units_by_order(
                [
                    {
                        "unit_id": verdict.unit_id,
                        "scope": verdict.scope,
                        "supported": verdict.supported,
                        "evidence_passage_ids": list(verdict.evidence_passage_ids),
                    }
                    for verdict in data.units
                ],
                expected,
            )
            if remapped_units is None:
                raise
            try:
                remapped = GroundingResult.model_validate(
                    {
                        "units": remapped_units,
                        "all_required_supported": bool(data.all_required_supported),
                    }
                )
            except PydanticValidationError as exc:
                raise VerifierValidationError(f"verifier output invalid: {exc}") from exc
            result = validate_grounding_result(remapped, expected_unit_ids=expected)
            logger.info("verifier ids remapped by order", extra={"units": len(expected)})
    elif isinstance(data, dict):
        try:
            parsed = GroundingResult.model_validate(_normalize_verifier_payload(data))
        except PydanticValidationError as exc:
            raise VerifierValidationError(f"verifier output invalid: {exc}") from exc
        try:
            result = validate_grounding_result(parsed, expected_unit_ids=expected)
        except VerifierValidationError:
            raw_units = _normalize_verifier_payload(data).get("units")
            remapped_units = (
                _remap_units_by_order(raw_units, expected) if isinstance(raw_units, list) else None
            )
            if remapped_units is None:
                raise
            try:
                remapped = GroundingResult.model_validate(
                    {
                        "units": remapped_units,
                        "all_required_supported": bool(data.get("all_required_supported", False)),
                    }
                )
            except PydanticValidationError as exc:
                raise VerifierValidationError(f"verifier output invalid: {exc}") from exc
            result = validate_grounding_result(remapped, expected_unit_ids=expected)
            logger.info("verifier ids remapped by order", extra={"units": len(expected)})
    else:
        raise VerifierValidationError("verifier output is not a structured object")
    window = list(passages[:VERIFIER_MAX_EVIDENCE_PASSAGES])
    short_to_full = display_id_map_for_window(window)
    pack_ids = set(_pack_index(passages).keys())
    if short_to_full:
        translated_units: list[UnitVerdict] = []
        for verdict in result.units:
            translated_units.append(
                UnitVerdict(
                    unit_id=verdict.unit_id,
                    scope=verdict.scope,
                    supported=verdict.supported,
                    evidence_passage_ids=resolve_cited_passage_ids(
                        list(verdict.evidence_passage_ids),
                        short_to_full=short_to_full,
                        full_ids=pack_ids,
                    ),
                )
            )
        result = GroundingResult(
            units=translated_units,
            all_required_supported=result.all_required_supported,
        )
    check_passage_checksums(passages)
    check_cited_passage_ids(result, pack_ids=pack_ids)
    check_exact_quotes(units=units, result=result, passages=passages)
    return result


def _reply_structured_content(reply: object) -> object:
    if isinstance(reply, BaseMessage):
        return reply.content
    return getattr(reply, "content", reply)


def build_single_unit_text(
    *,
    unit: ResponseUnitDraft,
    passages: Sequence[dict[str, Any]],
) -> str:
    """Render the minimal single-unit verifier payload (no id copying).

    Turn-independent hardening for weak fallback models: the model judges
    exactly one unit and returns one verdict object without copying
    ``u1``..``uN`` ids, which removes the systematic id-completeness flake
    seen live (14/14 clarifications with the verifier never served). The
    closed scope vocabulary and cite-only-supplied-ids rules are repeated
    verbatim; AA-side strict checks stay unchanged.
    """
    lines: list[str] = [
        "Judge exactly one response unit below.",
        'scope must be exactly one of "book", "product_meta", "conversation_glue".',
        "Cite only passage ids listed in <book_evidence> in evidence_passage_ids; "
        "passages are numbered p1..pN, cite those short ids.",
        "A book unit that cites no evidence passage is unsupported.",
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
    """Validate one single-unit verdict and bind it to ``unit_id``.

    Short display ids (``p1``..``pN``) resolve to full stored passage ids
    before validation so the strict cite gate below sees full ids; full
    stored ids pass through unchanged and unknown ids still fail closed.
    """
    from pydantic import ValidationError as PydanticValidationError

    if not isinstance(data, dict):
        raise VerifierValidationError("verifier output is not a structured object")
    payload = dict(data)
    scope = payload.get("scope")
    if isinstance(scope, str):
        payload["scope"] = scope.strip().lower().replace("-", "_")
    evidence_ids = payload.get("evidence_passage_ids")
    if isinstance(evidence_ids, list):
        stripped = [item.strip() if isinstance(item, str) else item for item in evidence_ids]
        if short_to_full:
            payload["evidence_passage_ids"] = resolve_cited_passage_ids(
                stripped,
                short_to_full=short_to_full,
                full_ids=full_ids or set(),
            )
        else:
            payload["evidence_passage_ids"] = stripped
    payload["unit_id"] = unit_id
    try:
        return UnitVerdict.model_validate(payload)
    except PydanticValidationError as exc:
        raise VerifierValidationError(f"verifier output invalid: {exc}") from exc


async def _verify_single_unit(
    unit: ResponseUnitDraft,
    passages: Sequence[dict[str, Any]],
    *,
    model: Any,
) -> UnitVerdict:
    """Verify exactly one unit with the minimal single-verdict schema."""
    system_text = load_verifier_system_v2()
    user_text = build_single_unit_text(unit=unit, passages=passages)
    window = list(passages[:VERIFIER_MAX_EVIDENCE_PASSAGES])
    short_to_full = display_id_map_for_window(window)
    full_ids = set(_pack_index(passages).keys())
    structured_invoke = getattr(model, "ainvoke_structured", None)
    if callable(structured_invoke):
        raw = await structured_invoke(
            user_text,
            system=system_text,
            schema=verifier_single_json_schema(),
            retry_count=VERIFIER_MAX_ATTEMPTS,
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
        raise VerifierValidationError("verifier output is not a structured object")
    return coerce_single_verdict(
        content, unit_id=unit.unit_id, short_to_full=short_to_full, full_ids=full_ids
    )


async def _verify_per_unit_concurrent(
    units: Sequence[ResponseUnitDraft],
    passages: Sequence[dict[str, Any]],
    *,
    model: Any,
) -> GroundingResult:
    """Verify each unit concurrently with the single-verdict schema.

    Concurrency keeps the fallback within one slow-model round instead of
    N sequential rounds (live SLO). Deterministic cite/quote/checksum
    gates run on the assembled result exactly as in the batch path, so
    grounding strictness is unchanged. Provider/transient errors
    propagate (fail-closed); only validation-shaped failures are raised
    as ``VerifierValidationError`` by the callers.
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
    """Invoke the hidden verifier and validate its structured verdict.

    Fast path first: a single OpenCode ``json_schema`` batch request with
    OpenCode-owned bounded validation retry (``retryCount``). On a
    validation-shaped failure (weak fallback models systematically flake
    on exact ``u1``..``uN`` id copying even for trivial glue/meta drafts,
    collapsing every ordinary turn to generic clarification with the
    verifier never served), fall back once to concurrent per-unit
    single-verdict verification with the minimal schema (no id copying).
    AA code performs strict Pydantic validation plus the deterministic
    completeness and exact-quote/provenance checks on both paths, and
    never reparses text. Provider/transient/timeout errors propagate
    immediately without fallback (the model adapter already exhausted
    primary/fallback).
    """
    if not units:
        raise VerifierValidationError("verifier needs at least one response unit")
    system_text = load_verifier_system_v2()
    user_text = build_verifier_user_text(units=units, passages=passages)
    structured_invoke = getattr(model, "ainvoke_structured", None)
    try:
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
    except (VerifierValidationError, ValueError) as exc:
        # Bounded single fallback to the simpler per-unit task for
        # id/format flake only; deterministic grounding rejections for the
        # same draft/pack (unknown passage, missing evidence, bad quote,
        # checksum) would repeat per unit and only burn live latency, so
        # they fail closed immediately. Provider/transient errors are not
        # VerifierValidationError and propagate above without fallback.
        message = str(exc).lower()
        deterministic = (
            "cites no evidence" in message
            or "cites unknown passage" in message
            or "quotes text" in message
            or "without cited exact passages" in message
            or "absent from cited passages" in message
            or "checksum mismatch" in message
            or "needs at least one response unit" in message
        )
        if deterministic:
            raise
        logger.info(
            "verifier batch invalid, per-unit fallback scheduled",
            extra={"category": "verifier-invalid"},
        )
        result = await _verify_per_unit_concurrent(units, passages, model=model)
        logger.info("verifier per-unit output accepted", extra={"units": len(units)})
        return result


__all__ = [
    "VERIFIER_AGENT_V2",
    "VERIFIER_MAX_EVIDENCE_PASSAGES",
    "VERIFIER_MAX_PASSAGE_CHARS",
    "build_single_unit_text",
    "build_verifier_user_text",
    "check_cited_passage_ids",
    "check_exact_quotes",
    "check_passage_checksums",
    "coerce_grounding_result",
    "coerce_single_verdict",
    "display_id_map_for_window",
    "resolve_cited_passage_ids",
    "run_verifier",
]
