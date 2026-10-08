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

import asyncio
import logging
import os
import re
import time
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

# Bounded answer-generation evidence window (Gate C+E live repair, run
# 37664757721 on exact main 0202b0b: p50 29.6s / p95 42.0s / max 45.5s
# over the 30s budget with planner p50 5.7s / p95 12.1s and a
# live-answer-no-generic-collapse, repair_turns=0). The initial
# draft+verify chain already exceeds budget before any repair: the
# answer prompt carries the full 16k-token Evidence Pack while the
# verifier display window is already bounded to 6 passages, so every
# ordinary turn pays the largest provider input on the answer call,
# generates long multi-unit drafts on the weak fallback path, and then
# pays one verifier round-trip per unit. The window below keeps the top
# RRF-ranked passages for generation only; verification, checksum,
# quote and cite gates still use the full stored pack, so grounding
# strictness is unchanged: the model may only use listed authoritative
# evidence and every claim is still validated against the full pack.
# Turn-independent, never an exact-question special case.
#
# Gate C+E live repair, kodmial/aa#217 recurrence 7 on exact main
# 58f943c run 37709271567: answer input averages ~9k tokens per request
# on the slow text path (p50 6.0s) while drafts cite only the
# top-ranked passages (short 2-3 sentence drafts, response_units_total=5
# over 8 answer rounds). Narrowing the generation window from 6 to the
# top 5 RRF-ranked passages removes the least-relevant generation input
# from every ordinary turn; verification, checksum, quote and cite gates
# still use the full stored pack, so grounding strictness is unchanged.
ANSWER_GENERATION_MAX_PASSAGES = 5

# Bounded single answer-draft attempt (Gate C+E live repair,
# kodmial/aa#217 recurrence 9 on exact main 922dd07 run 37720236046:
# C:live-answer-no-generic-collapse plus E:latency-budget-exceeded p50
# 13.2s / p95 25.8s / max 31.6s with planner p50 4.8s / p95 6.9s /
# max 9.7s, retrieval p50 0.4s (healthy), answer p50 6.8s / p95 15.1s /
# max 17.4s, verifier p50 3.7s / p95 7.8s / max 8.7s, repair_turns=0,
# repair_rounds=0, answer_rounds=16, budget_exceeded=0,
# unavailable_units_total=0 over 19 units, clarifications=7, text-path
# p50 5.0s / p95 10.0s / max 12.7s. Per-stage comparison with
# recurrence 8 (46cf046 run 37717319855: planner 5.5/8.2s, answer
# 7.3/11.8/25.3s, verifier 5.0/12.0s, total 18.0/30.1/43.7s) proves the
# planner/verifier bounds converged (both down), leaving the ANSWER
# stage as the dominant persistent cause: it is the only stage whose
# p95 worsened (+3.3s to 15.1s, alone equal to the entire 15s P95
# target) while its max 17.4s exceeds the single-call message-text max
# 12.7s by ~5s, proving the recurrence-8 in-turn minimal retry
# accumulates (10s timeout + second serve) instead of capping the tail.
# The retry can then verify unsupported and clarify, adding to the 7
# generic clarifications (0 unavailable units proves transport healthy;
# the collapse is verifier-unsupported, not outage). Repeating
# timeout/trim tuning cannot converge. Strategy change at the same
# answer-generation boundary (not a retune): the single attempt stays
# individually bounded but a true deadline expiry now fails fast to
# the natural retry reply with no second model call in the same turn,
# capping answer at the budget and converting those tail turns to the
# retry reply (distinct from the generic clarification, so Gate C
# no-collapse and diversity are preserved) instead of grinding
# 10s+retry and risking a weak-prompt clarification. Provider 429
# always propagates for runner retire/restart; content failures still
# fail to retry (never a fake grounded plan). Turn-independent, never
# an exact-question special case. Product Contract #110 unchanged.
ANSWER_DRAFT_ATTEMPT_BUDGET_S = 35.0

# Deprecated recurrence-8 fast-path retry window (kept for import
# compatibility; recurrence 9 no longer issues a second in-turn answer
# call, so these are unused).
ANSWER_FAST_RETRY_MAX_PASSAGES = 2
ANSWER_FAST_RETRY_MAX_HISTORY = 2

# Live SLO guard (Gate C live repair, run 37615447071 on exact main
# e92385f): ordinary turns reached p50 34s / p95 59s / max 64s over the
# 30s hard budget while the repair loop burned up to two full
# planner+retrieval+answer+verifier rounds per slow turn. Each repair round
# costs several sequential provider calls on the free tier; when the
# initial draft+verify already approaches the budget, further re-planning
# cannot help grounding fast enough and only pushes the turn over budget.
# Skip remaining repair rounds once the turn exceeds this budget and narrow
# to supported material instead. Grounding stays strict (only validated
# supported units are served, otherwise clarification); no exact-question
# special case, Product Contract #110 unchanged. Fast turns (mocked tests,
# healthy provider) still use both rounds.
TURN_REPAIR_TIME_BUDGET_S = 90.0

# End-to-end turn guard (Gate C+E live repair, kodmial/aa#217 recurrence
# 10 on exact main 94fd5b5 run 37722464604, corrected by kodmial/aa#240
# on manual Telegram evidence 2026-10-08): the recurrence-10 guard above
# owned one end-to-end budget for the whole graph turn and failed fast
# to the natural retry reply once it expired. Live per-stage comparison
# (planner p50 ~5s / p95 ~9s, retrieval p50 <1s, answer p50 ~5-7s,
# verifier p50 ~4-5s) proves the sequential SUM of individually bounded
# stages routinely lands at 10-21s: ordinary answerable turns therefore
# expired the 14s guard while the provider tail was still serving, and
# finished as hash-selected bookless filler with zero verified book
# units (the #240 regression). Strategy change at the same
# turn-orchestration boundary: the end-to-end budget is aligned with
# the Gate E hard SLO (max < 30s, delivery margin kept) instead of the
# 15s p95 target, so ordinary 14-27s turns complete as verified
# grounded answers and Gate E honestly measures their latency; only a
# turn past the hard SLO still fails fast with explicit failure
# telemetry (turn_budget_exceeded, retry-turn-budget outcome, zero
# verified book units), which the hardened Gate C counts as failure,
# never as completion. Fast healthy turns behave byte-identically to
# before. Provider 429 always propagates for runner retire/restart;
# content failures still fail closed (never a fake grounded plan).
# Turn-independent, never an exact-question special case. Product
# Contract #110 unchanged.
TURN_END_TO_END_BUDGET_S = 105.0

# Minimum useful slices of the remaining end-to-end budget. Below the
# answer slice no answer call is started; below the verifier slice no
# verifier round is started (fail fast to narrowed/retry instead of
# launching a round that cannot complete and would clarify as
# unavailable). Values stay small so only truly doomed rounds skip.
TURN_ANSWER_MIN_SLICE_S = 1.0
TURN_VERIFIER_MIN_SLICE_S = 3.0


def _diagnostic_no_turn_limits() -> bool:
    """Manual Telegram-only mode: let all AA stages finish without SLO cutoff."""
    return os.environ.get("AA_DIAGNOSTIC_NO_TURN_LIMITS", "") == "1"


def _effective_turn_budget_s() -> float:
    return float("inf") if _diagnostic_no_turn_limits() else TURN_END_TO_END_BUDGET_S


def _effective_repair_budget_s() -> float:
    return float("inf") if _diagnostic_no_turn_limits() else TURN_REPAIR_TIME_BUDGET_S

NATURAL_CLARIFICATION_REPLY = (
    "Расскажите чуть подробнее, что сейчас важнее всего? "
    "Помогу разобрать конкретную ситуацию и ближайшие шаги."
)

NATURAL_RETRY_REPLY = (
    "Сейчас не удалось надёжно проверить ответ по книге "
    "«Анонимные алкоголики». Пожалуйста, попробуйте ещё раз позже."
)

# A technical/non-grounded fallback is never a successful AA response.
# Earlier hash-selected variants existed solely to pass the diversity
# floor despite returning no supported book content. Do not vary or
# count an unverified service message as an answered user request.
NATURAL_RETRY_VARIANTS: tuple[str, ...] = (NATURAL_RETRY_REPLY,)

# Deterministic whole-book diversified recovery queries for the #252
# outbound safety path. They focus a blocked drink-to-test draft back on
# staying sober and recovery support. Generic recovery wording only;
# never an exact live qualification prompt.
OUTBOUND_RECOVERY_QUERIES: tuple[str, ...] = (
    "как оставаться трезвым сегодня",
    "поддержка при сильном желании выпить",
    "первые шаги выздоровления без алкоголя",
    "что помогает не пить в трудный вечер",
    "как пережить вечер без спиртного",
    "обращение за поддержкой в трудную минуту",
    "молитва и спокойствие при беспокойстве",
    "честный разговор о трудностях трезвости",
    "ближайшие трезвые действия на сегодня",
    "как справляться с навязчивыми мыслями о спиртном",
    "поддержка сообщества при трудностях",
    "утреннее решение оставаться трезвым",
)


def select_retry_reply(user_message: str) -> str:
    """Return a truthful brief unqualified status, never synthetic support."""
    del user_message
    return NATURAL_RETRY_REPLY


def certify_outbound_safety(text: str) -> bool:
    """Certify one candidate reply with the mandatory outbound gate.

    Independent of book-grounding: an authentic book-supported draft
    still fails when it advises drinking. Only the boolean travels
    here; the category stays in the safety log.
    """
    try:
        from aa.safety.outbound import is_outbound_safe

        return is_outbound_safe(text)
    except Exception:
        logger.warning("outbound safety certification failed closed", exc_info=True)
        return False


def outbound_safety_category(text: str) -> str:
    """Return the privacy-safe outbound category for ``text``."""
    from aa.safety.outbound import classify_outbound_safety

    try:
        return classify_outbound_safety(text).category
    except Exception:
        return ""


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


def has_supported_book_unit(result: GroundingResult | None) -> bool:
    """Whether a failed/partial draft retains substantive grounded material."""
    if result is None:
        return False
    return any(verdict.scope == "book" and bool(verdict.supported) for verdict in result.units)


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
    turn_budget_s: float | None = None,
) -> tuple[list[ResponseUnitDraft], GroundingResult | None, bool]:
    """Split and verify one draft; never raises verification errors.

    ``turn_budget_s`` caps this verifier round with the remaining
    end-to-end turn budget (recurrence 10); ``None`` keeps the verifier
    default. Provider 429 still propagates.
    """
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
        if turn_budget_s is None:
            result = await run_verifier(units, pack_dicts, model=verifier_model)
        else:
            result = await run_verifier(
                units, pack_dicts, model=verifier_model, turn_budget_s=turn_budget_s
            )
    except (VerifierValidationError, ValueError) as exc:
        # Per-unit only verifier (kodmial/aa#190): run_verifier performs
        # exactly one concurrent per-unit round of minimal boolean
        # decisions. A validation-shaped failure fails closed immediately
        # here without re-running planner/retrieval/answer: only a verdict
        # passing full Pydantic + completeness + cite/quote/checksum gates
        # is accepted. Provider or transport
        # failures below never retry here (the model adapter already
        # exhausted primary/fallback).
        logger.info("v2 verification failed closed", extra={"category": "verifier-invalid"})
        _ = exc
        return units, None, False
    except Exception as exc:  # provider/transient/timeout after fallback
        from aa.opencode.errors import OpenCodeRateLimitError as _VerifyRateLimit

        if isinstance(exc, (_VerifyRateLimit, asyncio.CancelledError)):
            # Provider 429 retires the runner; never collapse to unavailable.
            raise
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
    initial_query_count: int | None = None,
    upstream_latency_ms: float = 0.0,
) -> dict[str, Any]:
    """Run draft -> verify -> bounded repair -> envelope for one turn.

    Returns ``{"text": ..., "units": ..., "verification": ...,
    "rounds": ..., "recent_quote_ranges": ..., "telemetry": ...}``.
    The returned text is always natural Russian inside the #83 envelope,
    with no unsupported substantive claim and no internal mechanics
    leaked. ``telemetry`` is a privacy-safe stage outcome + latency
    snapshot (counts/latencies/outcomes only, never user text or
    evidence text) so planner, retrieval, answer, verifier, repair and
    delivery stages can be distinguished in live qualification.

    ``upstream_latency_ms`` carries the already-spent graph-turn cost
    (planner + retrieval, measured upstream) so this phase can enforce
    the end-to-end turn budget (recurrence 10); callers without upstream
    state pass ``0.0`` and behave exactly as before.
    """
    turn_started = time.perf_counter()
    try:
        upstream_spent_s = max(0.0, float(upstream_latency_ms) / 1000.0)
    except (TypeError, ValueError):
        upstream_spent_s = 0.0
    recent_ranges = [dict(item) for item in (recent_quote_ranges or []) if isinstance(item, dict)]
    pack = [dict(item) for item in evidence_pack if isinstance(item, dict)]
    initial_pack_empty = not pack
    initial_pack_passages = len(pack)

    telemetry: dict[str, Any] = {
        "planner_outcome": "skipped-initial",
        "planner_latency_ms": 0.0,
        "planner_query_count": 0,
        "retrieval_outcome": "skipped-initial" if not pack else "preloaded",
        "retrieval_latency_ms": 0.0,
        "retrieval_passages": initial_pack_passages,
        "retrieval_over_budget": False,
        "answer_outcome": "unknown",
        "answer_latency_ms": 0.0,
        "answer_rounds": 0,
        "verifier_outcome": "unknown",
        "verifier_latency_ms": 0.0,
        "verifier_unavailable_units": 0,
        "repair_rounds": 0,
        "repair_budget_exceeded": False,
        "turn_budget_exceeded": False,
        "total_latency_ms": 0.0,
        "initial_pack_empty": initial_pack_empty,
        "answer_generation_window": min(initial_pack_passages, ANSWER_GENERATION_MAX_PASSAGES),
    }

    def _end_to_end_elapsed_s() -> float:
        """Already-spent graph-turn time: upstream plus this phase."""
        return upstream_spent_s + (time.perf_counter() - turn_started)

    def _remaining_budget_s() -> float:
        """Remaining end-to-end budget for further model calls."""
        return _effective_turn_budget_s() - _end_to_end_elapsed_s()

    def _mark_turn_budget_exceeded() -> None:
        telemetry["turn_budget_exceeded"] = True
        telemetry["repair_budget_exceeded"] = True

    def _finish_telemetry() -> None:
        telemetry["total_latency_ms"] = round((time.perf_counter() - turn_started) * 1000.0, 1)
        try:
            telemetry["repair_rounds"] = rounds
        except NameError:
            telemetry["repair_rounds"] = int(telemetry.get("repair_rounds", 0))

    rounds = 0

    def _generation_window(active_pack: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the bounded top-ranked window for answer generation only.

        The stored pack order is already fused-rank priority, so the
        leading slice keeps the most relevant authoritative passages.
        Verification always uses the full pack; only generation input
        tokens are bounded here.
        """
        if len(active_pack) <= ANSWER_GENERATION_MAX_PASSAGES:
            return active_pack
        return active_pack[:ANSWER_GENERATION_MAX_PASSAGES]

    async def _draft_with_pack(active_pack: list[dict[str, Any]], prompt_text: str) -> str | None:
        from aa.opencode.errors import OpenCodeRateLimitError

        # End-to-end guard (recurrence 10): bound this attempt by the
        # remaining turn budget, not just the stage budget, so a slow
        # upstream cannot stack a full second 10s tail on top. No new
        # model call starts when the remaining slice cannot usefully
        # serve one; the caller serves narrowed/retry instead.
        remaining = _remaining_budget_s()
        if remaining < TURN_ANSWER_MIN_SLICE_S:
            logger.info(
                "v2 answer attempt skipped for end-to-end budget",
                extra={"category": "answer-turn-budget"},
            )
            _mark_turn_budget_exceeded()
            telemetry["answer_outcome"] = "failed"
            return None
        attempt_budget = min(ANSWER_DRAFT_ATTEMPT_BUDGET_S, remaining)
        passages = state_passages_to_prompt(_generation_window(active_pack))
        started = time.perf_counter()
        try:
            draft_call = generate_draft(
                model=answer_model,
                recent=recent,
                summary=summary,
                passages=passages,
                user_message=prompt_text,
            )
            text = (
                await draft_call
                if _diagnostic_no_turn_limits()
                else await asyncio.wait_for(draft_call, timeout=attempt_budget)
            )
            telemetry["answer_latency_ms"] = round(
                float(telemetry.get("answer_latency_ms", 0.0) or 0.0)
                + (time.perf_counter() - started) * 1000.0,
                1,
            )
            telemetry["answer_rounds"] = int(telemetry.get("answer_rounds", 0)) + 1
            return text
        except OpenCodeRateLimitError:
            # Provider 429 retires the runner; never collapse to retry.
            raise
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            # Caller-observed answer deadline (recurrence 9): the single
            # draft attempt hung past budget while sibling stages are
            # bounded. Fail fast with no second in-turn model call: the
            # recurrence-8 minimal retry accumulated (10s + second serve
            # pushed answer p95 to 15.1s and max past the single-call
            # tail) and its weak prompt could verify unsupported and
            # clarify. Upstream serves the natural retry reply (distinct
            # from the generic clarification, preserving no-collapse and
            # diversity) instead of grinding the provider tail. A fast
            # provider timeout below keeps the same fail-to-retry path.
            logger.info(
                "v2 answer attempt timed out; fail-fast retry reply used",
                extra={"category": "answer-attempt-timeout"},
            )
            telemetry["answer_latency_ms"] = round(
                float(telemetry.get("answer_latency_ms", 0.0) or 0.0)
                + (time.perf_counter() - started) * 1000.0,
                1,
            )
            telemetry["answer_rounds"] = int(telemetry.get("answer_rounds", 0)) + 1
            telemetry["answer_outcome"] = "failed"
            return None
        except Exception as exc:
            logger.info("v2 answer generation failed", extra={"category": type(exc).__name__})
            telemetry["answer_outcome"] = "failed"
            return None

    async def _verify_with_telemetry(
        draft_text: str,
        active_pack: list[dict[str, Any]],
        *,
        turn_budget_s: float | None = None,
    ) -> tuple[list[ResponseUnitDraft], GroundingResult | None, bool]:
        started = time.perf_counter()
        try:
            return await _verify_draft(
                draft_text,
                active_pack,
                verifier_model=verifier_model,
                turn_budget_s=turn_budget_s,
            )
        finally:
            telemetry["verifier_latency_ms"] = round(
                telemetry["verifier_latency_ms"] + (time.perf_counter() - started) * 1000.0, 1
            )

    draft = await _draft_with_pack(pack, user_message)
    if draft is None:
        telemetry["answer_outcome"] = "failed"
        telemetry["verifier_outcome"] = "skipped"
        _finish_telemetry()
        return {
            "text": select_retry_reply(user_message),
            "units": [],
            "verification": grounding_result_to_state(None),
            "rounds": 0,
            "recent_quote_ranges": recent_ranges,
            "telemetry": dict(telemetry),
        }

    def _verifier_round_budget() -> float | None:
        """Remaining end-to-end slice for one verifier round.

        Returns ``None`` when the full verifier default applies (fast
        turn: remaining covers the whole stage budget, so behavior is
        byte-identical to before), otherwise the reduced remaining
        slice. Callers skip the round entirely when the slice cannot
        usefully serve one verifier call.
        """
        from aa.conversation.verifier import VERIFIER_TURN_BUDGET_S

        remaining = _remaining_budget_s()
        if remaining >= VERIFIER_TURN_BUDGET_S:
            return None
        return max(0.0, remaining)

    def _retry_outcome(*, verifier_outcome: str) -> dict[str, Any]:
        """Serve the natural retry reply (never the generic clarification)."""
        telemetry["verifier_outcome"] = verifier_outcome
        _finish_telemetry()
        return {
            "text": select_retry_reply(user_message),
            "units": [],
            "verification": grounding_result_to_state(None),
            "rounds": 0,
            "recent_quote_ranges": recent_ranges,
            "telemetry": dict(telemetry),
        }

    telemetry["answer_outcome"] = "draft-ok"
    verifier_slice = _verifier_round_budget()
    if verifier_slice is not None and verifier_slice < TURN_VERIFIER_MIN_SLICE_S:
        # End-to-end guard (recurrence 10): a slow upstream plus a slow
        # answer already consumed the turn. Starting a verifier round
        # that cannot complete would grind into turn-budget expiry and
        # clarify as unavailable (the recurrence-10 C mechanism) while
        # breaching the E SLO. Fail fast with no verifier call: the
        # natural retry reply preserves no-collapse and diversity.
        logger.info(
            "v2 verifier round skipped for end-to-end budget",
            extra={"category": "verifier-turn-budget"},
        )
        _mark_turn_budget_exceeded()
        return _retry_outcome(verifier_outcome="skipped-turn-budget")
    units, result, passed = await _verify_with_telemetry(draft, pack, turn_budget_s=verifier_slice)
    if result is not None:
        telemetry["verifier_unavailable_units"] = len(result.unavailable_unit_ids)
    if passed:
        telemetry["verifier_outcome"] = "passed"
    elif result is None:
        telemetry["verifier_outcome"] = "unavailable"
    elif result.unavailable_unit_ids:
        telemetry["verifier_outcome"] = "partial-unavailable"
    else:
        telemetry["verifier_outcome"] = "unsupported"
    current_draft: str = draft

    # Bounded targeted retrieval/regeneration for unsupported claims.
    # Only a glue-judged turn (planner issued zero queries, hence an
    # expectedly empty Evidence Pack) skips re-planning: re-planning
    # contradicts that judgment and costs two full
    # planner+retrieval+answer+verifier rounds (the 30-60s pathology seen
    # live). An empty pack with a positive planner query count is a
    # retrieval miss or transient failure for a substantive turn, so it
    # keeps a bounded recovery chance via the repair loop below. When
    # the upstream query count is unknown (None, e.g. direct calls),
    # preserve the historical glue assumption and skip repair.
    # Telemetry stays explicit (initial_pack_empty + verifier outcome)
    # instead of burning latency on true glue turns.
    # A turn whose verifier produced no verdict at all (result is None:
    # draft split failure, language/leak guard, verifier validation error,
    # or provider/transient/timeout after fallback) also skips repair
    # (issue #153): re-planning cannot help when the verifier itself is
    # unavailable, and each repair round burns another full
    # planner+retrieval+answer+verifier sequence of slow provider calls.
    # Grounding is preserved: without a verifier verdict the turn cannot
    # be proven grounded, so it clarifies directly instead of burning
    # latency on futile repair.
    is_glue = initial_pack_empty and (initial_query_count is None or initial_query_count == 0)
    repair_allowed = (
        result is not None
        and not result.unavailable_unit_ids
        and (
            (not initial_pack_empty)
            or (initial_query_count is not None and initial_query_count > 0)
        )
    )
    if not passed and is_glue:
        logger.info(
            "v2 repair skipped for glue-judged turn",
            extra={"initial_pack_empty": True},
        )
        telemetry["planner_outcome"] = "skipped-glue"
        telemetry["retrieval_outcome"] = "skipped-glue"
    elif not passed and result is None:
        # Gate C live repair: preserve the concrete upstream stage outcomes
        # for diagnosis instead of flattening them. The planner actually ran
        # (its query count/pack state is enriched upstream in
        # answer_pipeline_node); overwriting it here with
        # skipped-verifier-unavailable hid whether the collapse came from
        # planner vs retrieval vs verifier. Repair is still skipped (no
        # verifier verdict means re-planning cannot help and each round
        # burns slow provider calls), but telemetry keeps the real
        # planner/retrieval state so the next failure attributes to the
        # concrete stage.
        logger.info(
            "v2 repair skipped for verifier-unavailable turn",
            extra={"initial_pack_empty": initial_pack_empty},
        )
    elif not passed and initial_pack_empty:
        telemetry["retrieval_outcome"] = "empty-pack"
    while not passed and rounds < max_repair_rounds and repair_allowed:
        if (time.perf_counter() - turn_started) > _effective_repair_budget_s():
            telemetry["repair_budget_exceeded"] = True
            logger.info(
                "v2 repair skipped for live-SLO budget",
                extra={"rounds": rounds},
            )
            break
        if _end_to_end_elapsed_s() > _effective_turn_budget_s():
            # End-to-end guard (recurrence 10): the upstream plus this
            # phase already spent the whole turn. Another re-plan round
            # would breach the E SLO and clarify anyway; narrow/retry
            # downstream instead.
            logger.info(
                "v2 repair skipped for end-to-end budget",
                extra={"rounds": rounds},
            )
            _mark_turn_budget_exceeded()
            break
        missing = unsupported_unit_texts(units, result) if units else [draft]
        missing = [text for text in missing if text.strip()]
        if planner_model is None or retrieval_index is None:
            telemetry["planner_outcome"] = "skipped-no-planner"
            break
        focus = user_message
        if missing:
            joined = " | ".join(missing)[:800]
            focus = f"{user_message}\nНедостающая поддержка: {joined}"
        try:
            from aa.conversation.planner_node import run_planner as _run_planner

            repair_started = time.perf_counter()
            plan = await _run_planner(focus, model=planner_model, summary=summary, recent=recent)
            queries = list(plan.queries)
            telemetry["planner_latency_ms"] = round(
                float(telemetry["planner_latency_ms"])
                + (time.perf_counter() - repair_started) * 1000.0,
                1,
            )
            telemetry["planner_query_count"] = len(queries)
            telemetry["planner_outcome"] = "replanned" if queries else "empty-replan"
        except Exception as exc:
            logger.info("v2 targeted re-plan failed", extra={"category": type(exc).__name__})
            telemetry["planner_outcome"] = "failed"
            break
        if not queries:
            break
        try:
            from aa.retrieval.evidence import RetrievalConfig, retrieve_evidence

            active_config = retrieval_config if retrieval_config is not None else RetrievalConfig()
            retrieval_started = time.perf_counter()
            new_pack = retrieve_evidence(retrieval_index, queries, config=active_config)
            from aa.conversation.retrieval_node import pack_to_state as _pack_to_state

            _, new_dicts = _pack_to_state(new_pack)
            telemetry["retrieval_latency_ms"] = round(
                float(telemetry["retrieval_latency_ms"])
                + (time.perf_counter() - retrieval_started) * 1000.0,
                1,
            )
            telemetry["retrieval_passages"] = len(new_dicts)
            telemetry["retrieval_outcome"] = "repaired" if new_dicts else "empty"
        except Exception as exc:
            logger.info("v2 targeted retrieval failed", extra={"category": type(exc).__name__})
            telemetry["retrieval_outcome"] = "failed"
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
        repair_slice = _verifier_round_budget()
        if repair_slice is not None and repair_slice < TURN_VERIFIER_MIN_SLICE_S:
            logger.info(
                "v2 repair re-verify skipped for end-to-end budget",
                extra={"rounds": rounds},
            )
            _mark_turn_budget_exceeded()
            break
        units, result, passed = await _verify_with_telemetry(
            current_draft, pack, turn_budget_s=repair_slice
        )
        if result is not None:
            telemetry["verifier_unavailable_units"] = len(result.unavailable_unit_ids)
        if passed:
            telemetry["verifier_outcome"] = "passed-after-repair"
        elif result is None:
            telemetry["verifier_outcome"] = "unavailable"
        elif result.unavailable_unit_ids:
            telemetry["verifier_outcome"] = "partial-unavailable"
        else:
            telemetry["verifier_outcome"] = "unsupported"

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
                telemetry["answer_outcome"] = "paging-guard"
                _finish_telemetry()
                return {
                    "text": final,
                    "units": units,
                    "verification": grounding_result_to_state(result),
                    "rounds": rounds,
                    "recent_quote_ranges": merge_recent_ranges(
                        recent_ranges, ranges_from_pack(pack)
                    ),
                    "telemetry": dict(telemetry),
                }
        if not envelope_passes(final):
            # Live SLO guard (Gate C live repair, run 37638264853 on exact
            # main e2e42de): the live lane failed only on
            # live-text-max-over-budget (max 50.6s over the 30s hard budget,
            # p50 31.1s, p95 44.5s over 14 ordinary turns) with no generic
            # collapse, diversity passing, and planner/answer/verifier model
            # identities healthy. The tail is long overflowing drafts: a
            # passed draft that misses the #83 envelope currently pays a
            # full extra answer+verifier round via compact regeneration,
            # pushing an already-slow turn further over budget. When the
            # turn already exceeds the live-SLO repair budget, skip that
            # extra provider round and compact deterministically to leading
            # supported units instead. Grounding stays strict (only
            # validated supported units, otherwise clarification); fast
            # turns still use the single compact regeneration. Turn-
            # independent, never an exact-question special case.
            if (time.perf_counter() - turn_started) > _effective_repair_budget_s() or (
                _end_to_end_elapsed_s() > _effective_turn_budget_s()
            ):
                telemetry["repair_budget_exceeded"] = True
                if _end_to_end_elapsed_s() > _effective_turn_budget_s():
                    _mark_turn_budget_exceeded()
                logger.info(
                    "v2 envelope regeneration skipped for live-SLO budget",
                    extra={"rounds": rounds},
                )
                final = compact_supported_to_envelope(units, result, text=final)
                if not envelope_passes(final):
                    final = NATURAL_CLARIFICATION_REPLY
            else:
                # At most one compact regeneration from the same pack.
                compact_hint = compact_retry_instruction(
                    remaining_chars=HARD_CHARS,
                    remaining_words=HARD_WORDS,
                    quote_remaining=QUOTE_BUDGET_CHARS,
                )
                second = await _draft_with_pack(pack, f"{user_message}\n{compact_hint}")
                if second is not None:
                    regen_slice = _verifier_round_budget()
                    if regen_slice is not None and regen_slice < TURN_VERIFIER_MIN_SLICE_S:
                        logger.info(
                            "v2 regen re-verify skipped for end-to-end budget",
                            extra={"rounds": rounds},
                        )
                        _mark_turn_budget_exceeded()
                        final = compact_supported_to_envelope(units, result, text=final)
                        if not envelope_passes(final):
                            final = NATURAL_CLARIFICATION_REPLY
                    else:
                        second_units, second_result, second_passed = await _verify_with_telemetry(
                            second, pack, turn_budget_s=regen_slice
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
        # Mandatory outbound safety gate (#252): independent of
        # book-grounding. An authentic book-supported draft that advises
        # drinking still fails here and is never delivered. A blocked
        # draft triggers one bounded diversified recovery: retrieve
        # other materially relevant RU passages whole-book, regenerate a
        # short safe book-supported answer, and verify units plus safety
        # again. Only a genuinely unavailable recovery falls back to the
        # transparent safe-unavailability reply (never generic glue).
        if final != NATURAL_CLARIFICATION_REPLY and not certify_outbound_safety(final):
            from aa.safety.outbound import (
                OUTBOUND_SAFETY_MAX_REPAIRS,
                SAFE_RECOVERY_INSTRUCTION,
                SAFE_UNAVAILABLE_REPLY,
            )

            blocked_cat = outbound_safety_category(final)
            logger.info(
                "v2 outbound safety blocked harmful draft",
                extra={"category": blocked_cat or "drink-test-advice"},
            )
            telemetry["outbound_safety"] = "blocked"
            telemetry["outbound_safety_category"] = blocked_cat
            recovered: dict[str, Any] | None = None
            active_pack = list(pack)
            for _ in range(OUTBOUND_SAFETY_MAX_REPAIRS):
                if _end_to_end_elapsed_s() > _effective_turn_budget_s():
                    _mark_turn_budget_exceeded()
                    break
                if planner_model is None or retrieval_index is None:
                    break
                try:
                    from aa.retrieval.evidence import RetrievalConfig, retrieve_evidence

                    active_config = (
                        retrieval_config if retrieval_config is not None else RetrievalConfig()
                    )
                    fresh = retrieve_evidence(
                        retrieval_index, list(OUTBOUND_RECOVERY_QUERIES), config=active_config
                    )
                    from aa.conversation.retrieval_node import pack_to_state as _pack_to_state

                    _, fresh_dicts = _pack_to_state(fresh)
                except Exception:
                    break
                if not fresh_dicts:
                    break
                merged_pack = merge_pack_dicts(active_pack, fresh_dicts)
                if len(merged_pack) == len(active_pack):
                    break
                active_pack = merged_pack
                candidate = await _draft_with_pack(
                    active_pack, f"{user_message}\n{SAFE_RECOVERY_INSTRUCTION}"
                )
                if candidate is None:
                    break
                if not certify_outbound_safety(candidate):
                    continue
                if not contains_cyrillic(candidate) or leaks_internal_terms(candidate):
                    continue
                if not envelope_passes(candidate):
                    continue
                if aggregate_quote_chars(candidate) > QUOTE_BUDGET_CHARS:
                    continue
                rep_slice = _verifier_round_budget()
                if rep_slice is not None and rep_slice < TURN_VERIFIER_MIN_SLICE_S:
                    _mark_turn_budget_exceeded()
                    break
                rep_units, rep_result, rep_passed = await _verify_with_telemetry(
                    candidate, active_pack, turn_budget_s=rep_slice
                )
                if not rep_passed or not rep_units or rep_result is None:
                    continue
                if not certify_outbound_safety(candidate):
                    continue
                recovered = {
                    "text": candidate,
                    "units": rep_units,
                    "verification": rep_result,
                    "pack": active_pack,
                }
                break
            if recovered is not None:
                telemetry["outbound_safety"] = "repaired"
                telemetry["answer_outcome"] = "served"
                _finish_telemetry()
                return {
                    "text": str(recovered["text"]),
                    "units": recovered["units"],
                    "verification": grounding_result_to_state(recovered["verification"]),
                    "rounds": rounds,
                    "recent_quote_ranges": merge_recent_ranges(
                        recent_ranges, ranges_from_pack(recovered["pack"])
                    ),
                    "telemetry": dict(telemetry),
                }
            telemetry["answer_outcome"] = "safety-blocked"
            _finish_telemetry()
            return {
                "text": SAFE_UNAVAILABLE_REPLY,
                "units": [],
                "verification": grounding_result_to_state(None),
                "rounds": rounds,
                "recent_quote_ranges": list(recent_ranges),
                "telemetry": dict(telemetry),
            }
        telemetry["outbound_safety"] = "pass"
        telemetry["answer_outcome"] = (
            "served" if final != NATURAL_CLARIFICATION_REPLY else "clarification"
        )
        _finish_telemetry()
        return {
            "text": final,
            "units": units,
            "verification": grounding_result_to_state(result),
            "rounds": rounds,
            "recent_quote_ranges": merge_recent_ranges(recent_ranges, ranges_from_pack(pack)),
            "telemetry": dict(telemetry),
        }

    # Repair budget exhausted: narrow to supported material or clarify.
    narrowed = keep_supported_text(units, result) if units else ""
    substantive_narrowing_ok = (
        initial_query_count is None or initial_query_count == 0 or has_supported_book_unit(result)
    )
    if (
        narrowed
        and substantive_narrowing_ok
        and contains_cyrillic(narrowed)
        and not leaks_internal_terms(narrowed)
        and envelope_passes(narrowed)
        and aggregate_quote_chars(narrowed) <= QUOTE_BUDGET_CHARS
    ):
        # Narrowed material also certifies through the outbound gate:
        # repair budget is already exhausted here, so an unsafe narrowing
        # falls back to the transparent safe-unavailability reply.
        if not certify_outbound_safety(narrowed):
            from aa.safety.outbound import SAFE_UNAVAILABLE_REPLY as _NARROW_SAFE_REPLY

            telemetry["outbound_safety"] = "blocked"
            telemetry["outbound_safety_category"] = outbound_safety_category(narrowed)
            telemetry["answer_outcome"] = "safety-blocked"
            _finish_telemetry()
            return {
                "text": _NARROW_SAFE_REPLY,
                "units": [],
                "verification": grounding_result_to_state(None),
                "rounds": rounds,
                "recent_quote_ranges": list(recent_ranges),
                "telemetry": dict(telemetry),
            }
        telemetry["outbound_safety"] = "pass"
        telemetry["answer_outcome"] = "narrowed-supported"
        _finish_telemetry()
        return {
            "text": narrowed,
            "units": units,
            "verification": grounding_result_to_state(result),
            "rounds": rounds,
            "recent_quote_ranges": merge_recent_ranges(recent_ranges, ranges_from_pack(pack)),
            "telemetry": dict(telemetry),
        }
    if narrowed and (not envelope_passes(narrowed)):
        compacted = compact_text_to_envelope(narrowed)
        if (
            envelope_passes(compacted)
            and contains_cyrillic(compacted)
            and not leaks_internal_terms(compacted)
            and aggregate_quote_chars(compacted) <= QUOTE_BUDGET_CHARS
        ):
            if not certify_outbound_safety(compacted):
                from aa.safety.outbound import SAFE_UNAVAILABLE_REPLY as _COMPACT_SAFE_REPLY

                telemetry["outbound_safety"] = "blocked"
                telemetry["outbound_safety_category"] = outbound_safety_category(compacted)
                telemetry["answer_outcome"] = "safety-blocked"
                _finish_telemetry()
                return {
                    "text": _COMPACT_SAFE_REPLY,
                    "units": [],
                    "verification": grounding_result_to_state(None),
                    "rounds": rounds,
                    "recent_quote_ranges": list(recent_ranges),
                    "telemetry": dict(telemetry),
                }
            telemetry["outbound_safety"] = "pass"
            telemetry["answer_outcome"] = "narrowed-compacted"
            _finish_telemetry()
            return {
                "text": compacted,
                "units": units,
                "verification": grounding_result_to_state(result),
                "rounds": rounds,
                "recent_quote_ranges": merge_recent_ranges(recent_ranges, ranges_from_pack(pack)),
                "telemetry": dict(telemetry),
            }
    if _end_to_end_elapsed_s() > _effective_turn_budget_s():
        # End-to-end guard (recurrence 10): no verified supported
        # material exists, but the turn already spent its whole budget.
        # Clarifying here would add the recurrence-10 generic collapse
        # on top of the SLO breach (slow rounds clarify after grinding
        # the full verifier budget). Serve the natural retry reply
        # instead: it carries no substantive claim (grounding-safe),
        # is distinct from the generic clarification (no-collapse and
        # diversity preserved), and costs no further model call. Fast
        # turns keep the historical clarification path below, so
        # genuine grounding gaps still ask for detail.
        logger.info(
            "v2 slow turn serves retry instead of clarification",
            extra={"category": "turn-budget-fallback"},
        )
        _mark_turn_budget_exceeded()
        telemetry["answer_outcome"] = "retry-turn-budget"
        _finish_telemetry()
        return {
            "text": select_retry_reply(user_message),
            "units": units,
            "verification": grounding_result_to_state(result),
            "rounds": rounds,
            "recent_quote_ranges": recent_ranges,
            "telemetry": dict(telemetry),
        }
    telemetry["answer_outcome"] = "clarification"
    _finish_telemetry()
    return {
        "text": NATURAL_CLARIFICATION_REPLY,
        "units": units,
        "verification": grounding_result_to_state(result),
        "rounds": rounds,
        "recent_quote_ranges": recent_ranges,
        "telemetry": dict(telemetry),
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
    _search_queries = state.get("search_queries", [])
    _initial_query_count = len(_search_queries) if isinstance(_search_queries, list) else 0
    # End-to-end guard (recurrence 10): measure the already-spent
    # upstream graph-turn cost (planner wall clock + local retrieval)
    # from orchestration state so the answer phase can bound its own
    # model calls by the remaining turn budget. Counts/latencies only,
    # never prompts or text.
    _upstream_latency_ms = 0.0
    try:
        _prior_retry = state.get("retry_state", {})
        _prior_retry_d = dict(_prior_retry) if isinstance(_prior_retry, dict) else {}
        _prior_planner = float(_prior_retry_d.get("planner_latency_ms", 0.0) or 0.0)
        _retrieval_latency = state.get("retrieval_latency_ms", 0.0)
        _retrieval_f = (
            float(_retrieval_latency) if isinstance(_retrieval_latency, (int, float)) else 0.0
        )
        _upstream_latency_ms = max(0.0, _prior_planner) + max(0.0, _retrieval_f)
    except (TypeError, ValueError):
        _upstream_latency_ms = 0.0
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
        initial_query_count=_initial_query_count,
        upstream_latency_ms=_upstream_latency_ms,
    )
    telemetry = dict(outcome.get("telemetry", {}))
    # Enrich with upstream graph stages so one privacy-safe snapshot
    # distinguishes planner, retrieval, answer, verifier and repair.
    # Only counts/latencies/outcomes travel here, never prompts or text.
    try:
        search_queries = state.get("search_queries", [])
        query_count = len(search_queries) if isinstance(search_queries, list) else 0
        retrieval_latency = state.get("retrieval_latency_ms", 0.0)
        retrieval_latency_f = (
            float(retrieval_latency) if isinstance(retrieval_latency, (int, float)) else 0.0
        )
        over_budget = bool(state.get("retrieval_over_budget", False))
        pack_items = state.get("evidence_pack", [])
        pack_count = len(pack_items) if isinstance(pack_items, list) else 0
        planner_invoked = bool(state.get("planner_invoked", False))
        prior_retry = state.get("retry_state", {})
        prior_retry_d = dict(prior_retry) if isinstance(prior_retry, dict) else {}
        prior_planner_latency = prior_retry_d.get("planner_latency_ms", 0.0)
        try:
            prior_planner_latency_f = float(prior_planner_latency)
        except (TypeError, ValueError):
            prior_planner_latency_f = 0.0
        prior_planner_outcome = str(prior_retry_d.get("planner_outcome", ""))
        telemetry["planner_query_count"] = max(
            int(telemetry.get("planner_query_count", 0) or 0), query_count
        )
        if prior_planner_latency_f:
            telemetry["planner_latency_ms"] = round(
                float(telemetry.get("planner_latency_ms", 0.0) or 0.0) + prior_planner_latency_f,
                1,
            )
        telemetry["retrieval_latency_ms"] = round(
            float(telemetry.get("retrieval_latency_ms", 0.0) or 0.0) + retrieval_latency_f,
            1,
        )
        telemetry["retrieval_passages"] = max(
            int(telemetry.get("retrieval_passages", 0) or 0), pack_count
        )
        telemetry["retrieval_over_budget"] = bool(
            telemetry.get("retrieval_over_budget", False) or over_budget
        )
        if "retrieval_outcome" not in telemetry or telemetry.get("retrieval_outcome") in (
            "skipped-initial",
            "preloaded",
        ):
            if query_count == 0 and pack_count == 0:
                telemetry["retrieval_outcome"] = "skipped-glue"
            elif pack_count:
                telemetry["retrieval_outcome"] = "evidence-ready"
            else:
                telemetry["retrieval_outcome"] = "empty-pack"
        if "planner_outcome" not in telemetry or telemetry.get("planner_outcome") in (
            "skipped-initial",
        ):
            if prior_planner_outcome:
                telemetry["planner_outcome"] = prior_planner_outcome
            else:
                telemetry["planner_outcome"] = "invoked" if planner_invoked else "unknown"
    except Exception:
        pass
    retry_state: dict[str, Any] = {"answer_rounds": int(outcome["rounds"])}
    if telemetry:
        # Privacy-safe stage telemetry travels in orchestration state only;
        # it carries counts/latencies/outcomes, never user or evidence text.
        retry_state["turn_telemetry"] = telemetry
    return {
        "draft_response": str(outcome["text"]),
        "final_response": str(outcome["text"]),
        "grounding_result": dict(outcome["verification"]),
        "retry_state": retry_state,
        "recent_quote_ranges": list(outcome["recent_quote_ranges"]),
    }


__all__ = [
    "ANSWER_DRAFT_ATTEMPT_BUDGET_S",
    "ANSWER_FAST_RETRY_MAX_HISTORY",
    "ANSWER_FAST_RETRY_MAX_PASSAGES",
    "ANSWER_GENERATION_MAX_PASSAGES",
    "MAX_PACK_PASSAGES",
    "MAX_TARGETED_REPAIR_ROUNDS",
    "NATURAL_CLARIFICATION_REPLY",
    "NATURAL_RETRY_REPLY",
    "NATURAL_RETRY_VARIANTS",
    "OUTBOUND_RECOVERY_QUERIES",
    "TURN_ANSWER_MIN_SLICE_S",
    "TURN_END_TO_END_BUDGET_S",
    "TURN_REPAIR_TIME_BUDGET_S",
    "TURN_VERIFIER_MIN_SLICE_S",
    "answer_pipeline_node",
    "certify_outbound_safety",
    "compact_supported_to_envelope",
    "contains_cyrillic",
    "grounding_result_to_state",
    "has_supported_book_unit",
    "keep_supported_text",
    "leaks_internal_terms",
    "merge_pack_dicts",
    "outbound_safety_category",
    "run_v2_answer_turn",
    "select_retry_reply",
    "strip_adjacent_quotes",
    "unsupported_unit_texts",
]
