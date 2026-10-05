"""Natural grounded answer pipeline with claim-level verification and repair.

Flow per normal turn::

    Evidence Pack -> AA Agent draft -> split into razdel units ->
    hidden model verifier -> [targeted re-plan/retrieve, at most 2 rounds] ->
    natural grounded Russian response inside the #83 envelope.

Retrieval and grounding stay invisible. Internal failures remain
internal: the user sees a natural Russian continuation or clarification,
never corpus/retrieval/provider mechanics and never a technical
fail-closed reply.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage

from aa.conversation.answer_node import generate_draft, recent_history
from aa.conversation.output_limits import (
    HARD_CHARS,
    HARD_WORDS,
    QUOTE_BUDGET_CHARS,
    aggregate_quote_chars,
    compact_retry_instruction,
    compact_text_to_envelope,
    envelope_passes,
    is_bulk_reproduction_request,
    is_continuation_request,
)
from aa.conversation.quote_state import (
    is_adjacent_to_recent,
    merge_recent_ranges,
    pack_pages_recent,
    ranges_from_pack,
)
from aa.conversation.response_units import (
    ResponseUnitDraft,
    ResponseUnitError,
    split_response_units,
)
from aa.conversation.retrieval_node import state_passages_to_prompt
from aa.conversation.verifier import run_verifier
from aa.conversation.verifier_schema import GroundingResult, VerifierValidationError

logger = logging.getLogger("aa.conversation.turn_pipeline")

MAX_TARGETED_REPAIR_ROUNDS = 2
MAX_PACK_PASSAGES = 12

NATURAL_CLARIFICATION_REPLY = (
    "Расскажите чуть подробнее, что сейчас важнее всего? "
    "Помогу разобрать конкретную ситуацию и ближайшие шаги."
)

NATURAL_RETRY_REPLY = "Давайте продолжим спокойно. Расскажите, что сейчас беспокоит сильнее всего?"

_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")

_INTERNAL_TERMS = (
    "corpus",
    "retrieval",
    "grounding",
    "evidence",
    "index",
    "embedding",
    "planner",
    "provider",
    "qualification",
    "fail_closed",
    "fail closed",
    "search failure",
    "structured output",
    "json_schema",
    "json schema",
    "rerank",
    "bm25",
    "faiss",
)


def contains_cyrillic(text: str) -> bool:
    """Whether user-visible text obeys the Russian-output contract."""
    return _CYRILLIC_RE.search(text) is not None


def leaks_internal_terms(text: str) -> bool:
    """Whether user-visible text exposes hidden mechanics."""
    lowered = text.casefold()
    return any(term in lowered for term in _INTERNAL_TERMS)


def verdict_by_id(result: GroundingResult) -> dict[str, Any]:
    """Index verifier verdicts by unit id."""
    return {verdict.unit_id: verdict for verdict in result.units}


def unsupported_unit_texts(
    units: Sequence[ResponseUnitDraft], result: GroundingResult | None
) -> list[str]:
    """Collect texts of unsupported book units for targeted re-planning."""
    if result is None:
        return [unit.text for unit in units]
    by_id = verdict_by_id(result)
    missing: list[str] = []
    for unit in units:
        verdict = by_id.get(unit.unit_id)
        if verdict is None:
            missing.append(unit.text)
            continue
        scope = str(getattr(verdict, "scope", ""))
        supported = bool(getattr(verdict, "supported", False))
        if scope == "book" and not supported:
            missing.append(unit.text)
        elif not supported:
            # Unsupported glue/meta also blocks the draft; include it so
            # the repair focuses on the same wording.
            missing.append(unit.text)
    return missing


def keep_supported_text(units: Sequence[ResponseUnitDraft], result: GroundingResult | None) -> str:
    """Narrow a failed draft to its supported units only."""
    if result is None:
        return ""
    by_id = verdict_by_id(result)
    kept: list[str] = []
    for unit in units:
        verdict = by_id.get(unit.unit_id)
        if verdict is not None and bool(getattr(verdict, "supported", False)):
            kept.append(unit.text)
    return " ".join(kept).strip()


def grounding_result_to_state(result: GroundingResult | None) -> dict[str, Any]:
    """Serialize a verifier outcome into graph state (no prompt text)."""
    if result is None:
        return {"verified": False, "units": [], "all_required_supported": False}
    return {
        "verified": True,
        "all_required_supported": bool(result.all_required_supported),
        "units": [
            {
                "unit_id": verdict.unit_id,
                "scope": str(verdict.scope),
                "supported": bool(verdict.supported),
                "evidence_passage_ids": list(verdict.evidence_passage_ids),
            }
            for verdict in result.units
        ],
    }


def merge_pack_dicts(
    current: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge new exact passages into a bounded Evidence Pack."""
    merged: list[dict[str, Any]] = [dict(item) for item in current]
    seen = {str(item.get("passage_id", "")) for item in merged}
    for item in incoming:
        if len(merged) >= MAX_PACK_PASSAGES:
            break
        key = str(item.get("passage_id", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(dict(item))
    return merged


def compact_supported_to_envelope(
    units: Sequence[ResponseUnitDraft], result: GroundingResult | None, *, text: str
) -> str:
    """Deterministically keep leading supported units fitting the envelope."""
    if envelope_passes(text):
        return text
    by_id = verdict_by_id(result) if result is not None else {}
    kept: list[str] = []
    for unit in units:
        verdict = by_id.get(unit.unit_id)
        if verdict is not None and not bool(getattr(verdict, "supported", False)):
            continue
        candidate = " ".join([*kept, unit.text]) if kept else unit.text
        if not envelope_passes(candidate):
            break
        kept.append(unit.text)
    if kept:
        return " ".join(kept)
    return compact_text_to_envelope(text)


def strip_adjacent_quotes(
    text: str, *, pack_dicts: list[dict[str, Any]], recent: list[dict[str, Any]]
) -> str:
    """Prevent serial paging: drop verbatim quotes from adjacent ranges."""
    from aa.conversation.output_limits import extract_quoted_spans

    if not recent:
        return text
    adjacent_texts: list[str] = []
    for item in pack_dicts:
        candidate = {
            "source_id": str(item.get("source_id", item.get("source", ""))),
            "section_id": str(item.get("section_id", item.get("section", ""))),
            "char_start": item.get("char_start", 0),
            "char_end": item.get("char_end", 0),
        }

        if is_adjacent_to_recent(candidate, recent):
            passage_text = str(item.get("text", ""))
            if passage_text:
                adjacent_texts.append(passage_text)
    if not adjacent_texts:
        return text
    filtered = text
    for span in extract_quoted_spans(text):
        if not span.strip():
            continue
        if any(span in candidate for candidate in adjacent_texts):
            filtered = filtered.replace(span, "этот отрывок", 1)
    return " ".join(filtered.split()).strip()


async def _verify_draft(
    draft: str,
    pack_dicts: list[dict[str, Any]],
    *,
    verifier_model: Any,
) -> tuple[list[ResponseUnitDraft], GroundingResult | None, bool]:
    """Split and verify one draft; never raises verification errors."""
    try:
        units = split_response_units(draft)
    except (ResponseUnitError, ValueError) as exc:
        logger.info("v2 draft split failed", extra={"category": "split-invalid"})
        _ = exc
        return [], None, False
    if not units:
        return [], None, False
    if not contains_cyrillic(draft) or leaks_internal_terms(draft):
        logger.info("v2 draft failed language/leak guard")
        return units, None, False
    try:
        result = await run_verifier(units, pack_dicts, model=verifier_model)
    except (VerifierValidationError, ValueError) as exc:
        logger.info("v2 verification failed closed", extra={"category": "verifier-invalid"})
        _ = exc
        return units, None, False
    except Exception as exc:  # provider/transient/timeout after fallback
        logger.info("v2 verifier unavailable", extra={"category": type(exc).__name__})
        return units, None, False
    if not result.all_required_supported:
        return units, result, False
    if any(not verdict.supported for verdict in result.units):
        return units, result, False
    return units, result, True


async def run_v2_answer_turn(
    *,
    user_message: str,
    summary: str,
    recent: list[BaseMessage],
    evidence_pack: list[dict[str, Any]],
    answer_model: Any,
    verifier_model: Any,
    planner_model: Any | None = None,
    retrieval_index: Any | None = None,
    retrieval_config: Any | None = None,
    recent_quote_ranges: list[dict[str, Any]] | None = None,
    max_repair_rounds: int = MAX_TARGETED_REPAIR_ROUNDS,
) -> dict[str, Any]:
    """Run draft -> verify -> bounded repair -> envelope for one turn.

    Returns ``{"text": ..., "units": ..., "verification": ...,
    "rounds": ..., "recent_quote_ranges": ...}``. The returned text is
    always natural Russian inside the #83 envelope, with no unsupported
    substantive claim and no internal mechanics leaked.
    """
    recent_ranges = [dict(item) for item in (recent_quote_ranges or []) if isinstance(item, dict)]
    pack = [dict(item) for item in evidence_pack if isinstance(item, dict)]

    async def _draft_with_pack(active_pack: list[dict[str, Any]], prompt_text: str) -> str | None:
        passages = state_passages_to_prompt(active_pack)
        try:
            return await generate_draft(
                model=answer_model,
                recent=recent,
                summary=summary,
                passages=passages,
                user_message=prompt_text,
            )
        except Exception as exc:
            logger.info("v2 answer generation failed", extra={"category": type(exc).__name__})
            return None

    draft = await _draft_with_pack(pack, user_message)
    if draft is None:
        return {
            "text": NATURAL_RETRY_REPLY,
            "units": [],
            "verification": grounding_result_to_state(None),
            "rounds": 0,
            "recent_quote_ranges": recent_ranges,
        }

    units, result, passed = await _verify_draft(draft, pack, verifier_model=verifier_model)
    rounds = 0
    current_draft: str = draft

    # Bounded targeted retrieval/regeneration for unsupported claims.
    while not passed and rounds < max_repair_rounds:
        missing = unsupported_unit_texts(units, result) if units else [draft]
        missing = [text for text in missing if text.strip()]
        if planner_model is None or retrieval_index is None:
            break
        focus = user_message
        if missing:
            joined = " | ".join(missing)[:800]
            focus = f"{user_message}\nНедостающая поддержка: {joined}"
        try:
            from aa.conversation.planner_node import run_planner as _run_planner

            plan = await _run_planner(focus, model=planner_model, summary=summary, recent=recent)
            queries = list(plan.queries)
        except Exception as exc:
            logger.info("v2 targeted re-plan failed", extra={"category": type(exc).__name__})
            break
        if not queries:
            break
        try:
            from aa.retrieval.evidence import RetrievalConfig, retrieve_evidence

            active_config = retrieval_config if retrieval_config is not None else RetrievalConfig()
            new_pack = retrieve_evidence(retrieval_index, queries, config=active_config)
            from aa.conversation.retrieval_node import pack_to_state as _pack_to_state

            _, new_dicts = _pack_to_state(new_pack)
        except Exception as exc:
            logger.info("v2 targeted retrieval failed", extra={"category": type(exc).__name__})
            break
        if not new_dicts:
            break
        merged = merge_pack_dicts(pack, new_dicts)
        if len(merged) == len(pack):
            break
        pack = merged
        rounds += 1
        next_draft = await _draft_with_pack(pack, user_message)
        if next_draft is None:
            break
        current_draft = next_draft
        units, result, passed = await _verify_draft(
            current_draft, pack, verifier_model=verifier_model
        )

    if passed and units and result is not None:
        final = current_draft
        # Serial-paging guard: a continuation after a prior quote must not
        # page adjacent canonical ranges verbatim.
        if (
            is_continuation_request(user_message) or is_bulk_reproduction_request(user_message)
        ) and (pack_pages_recent(pack, recent_ranges)):
            narrowed_paging = strip_adjacent_quotes(final, pack_dicts=pack, recent=recent_ranges)
            if narrowed_paging and contains_cyrillic(narrowed_paging):
                final = narrowed_paging
            else:
                final = NATURAL_CLARIFICATION_REPLY
                return {
                    "text": final,
                    "units": units,
                    "verification": grounding_result_to_state(result),
                    "rounds": rounds,
                    "recent_quote_ranges": merge_recent_ranges(
                        recent_ranges, ranges_from_pack(pack)
                    ),
                }
        if not envelope_passes(final):
            # At most one compact regeneration from the same pack.
            compact_hint = compact_retry_instruction(
                remaining_chars=HARD_CHARS,
                remaining_words=HARD_WORDS,
                quote_remaining=QUOTE_BUDGET_CHARS,
            )
            second = await _draft_with_pack(pack, f"{user_message}\n{compact_hint}")
            if second is not None:
                second_units, second_result, second_passed = await _verify_draft(
                    second, pack, verifier_model=verifier_model
                )
                if second_passed and second_units and second_result is not None:
                    if envelope_passes(second):
                        final = second
                        units, result = second_units, second_result
                    else:
                        final = compact_supported_to_envelope(
                            second_units, second_result, text=second
                        )
                        units, result = second_units, second_result
                        if not envelope_passes(final) or not contains_cyrillic(final):
                            final = NATURAL_CLARIFICATION_REPLY
                else:
                    final = compact_supported_to_envelope(units, result, text=final)
                    if not envelope_passes(final):
                        final = NATURAL_CLARIFICATION_REPLY
            else:
                final = compact_supported_to_envelope(units, result, text=final)
                if not envelope_passes(final):
                    final = NATURAL_CLARIFICATION_REPLY
        # Quote-budget deterministic guard.
        if aggregate_quote_chars(final) > QUOTE_BUDGET_CHARS:
            compacted = compact_text_to_envelope(final)
            if envelope_passes(compacted) and contains_cyrillic(compacted):
                final = compacted
            else:
                final = NATURAL_CLARIFICATION_REPLY
        if not contains_cyrillic(final) or leaks_internal_terms(final):
            final = NATURAL_CLARIFICATION_REPLY
        return {
            "text": final,
            "units": units,
            "verification": grounding_result_to_state(result),
            "rounds": rounds,
            "recent_quote_ranges": merge_recent_ranges(recent_ranges, ranges_from_pack(pack)),
        }

    # Repair budget exhausted: narrow to supported material or clarify.
    narrowed = keep_supported_text(units, result) if units else ""
    if (
        narrowed
        and contains_cyrillic(narrowed)
        and not leaks_internal_terms(narrowed)
        and envelope_passes(narrowed)
        and aggregate_quote_chars(narrowed) <= QUOTE_BUDGET_CHARS
    ):
        return {
            "text": narrowed,
            "units": units,
            "verification": grounding_result_to_state(result),
            "rounds": rounds,
            "recent_quote_ranges": merge_recent_ranges(recent_ranges, ranges_from_pack(pack)),
        }
    if narrowed and (not envelope_passes(narrowed)):
        compacted = compact_text_to_envelope(narrowed)
        if (
            envelope_passes(compacted)
            and contains_cyrillic(compacted)
            and not leaks_internal_terms(compacted)
        ):
            return {
                "text": compacted,
                "units": units,
                "verification": grounding_result_to_state(result),
                "rounds": rounds,
                "recent_quote_ranges": merge_recent_ranges(recent_ranges, ranges_from_pack(pack)),
            }
    return {
        "text": NATURAL_CLARIFICATION_REPLY,
        "units": units,
        "verification": grounding_result_to_state(result),
        "rounds": rounds,
        "recent_quote_ranges": recent_ranges,
    }


async def answer_pipeline_node(
    state: Any,
    *,
    answer_model: Any,
    verifier_model: Any,
    planner_model: Any | None = None,
    retrieval_index: Any | None = None,
    retrieval_config: Any | None = None,
) -> dict[str, Any]:
    """LangGraph answer node: state in, grounded Russian reply out."""
    from aa.conversation.graph_state import NORMAL_ROUTE

    if str(state.get("route", NORMAL_ROUTE)) != NORMAL_ROUTE:
        return {}
    user_message = str(state.get("current_user_message", ""))
    if not user_message.strip():
        return {
            "draft_response": NATURAL_CLARIFICATION_REPLY,
            "final_response": NATURAL_CLARIFICATION_REPLY,
            "grounding_result": grounding_result_to_state(None),
            "retry_state": {"answer_rounds": 0},
        }
    messages = [item for item in state.get("messages", []) if isinstance(item, BaseMessage)]
    outcome = await run_v2_answer_turn(
        user_message=user_message,
        summary=str(state.get("conversation_summary", "")),
        recent=recent_history(messages, current_user_message=user_message),
        evidence_pack=[item for item in state.get("evidence_pack", []) if isinstance(item, dict)],
        answer_model=answer_model,
        verifier_model=verifier_model,
        planner_model=planner_model if planner_model is not None else answer_model,
        retrieval_index=retrieval_index,
        retrieval_config=retrieval_config,
        recent_quote_ranges=[
            item for item in state.get("recent_quote_ranges", []) if isinstance(item, dict)
        ],
    )
    return {
        "draft_response": str(outcome["text"]),
        "final_response": str(outcome["text"]),
        "grounding_result": dict(outcome["verification"]),
        "retry_state": {"answer_rounds": int(outcome["rounds"])},
        "recent_quote_ranges": list(outcome["recent_quote_ranges"]),
    }


__all__ = [
    "MAX_PACK_PASSAGES",
    "MAX_TARGETED_REPAIR_ROUNDS",
    "NATURAL_CLARIFICATION_REPLY",
    "NATURAL_RETRY_REPLY",
    "answer_pipeline_node",
    "compact_supported_to_envelope",
    "contains_cyrillic",
    "grounding_result_to_state",
    "keep_supported_text",
    "leaks_internal_terms",
    "merge_pack_dicts",
    "run_v2_answer_turn",
    "strip_adjacent_quotes",
    "unsupported_unit_texts",
]
