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
from aa.conversation.failures import TurnFailed
from aa.conversation.output_limits import (
    HARD_CHARS,
    HARD_WORDS,
    MAX_TRANSPORT_SEGMENTS,
    QUOTE_BUDGET_CHARS,
    aggregate_quote_chars,
    compact_retry_instruction,
    compact_text_to_envelope,
    envelope_passes,
    is_bulk_reproduction_request,
    is_continuation_request,
    split_text_to_envelope_segments,
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
# Bounded Evidence Pack size (issue #295): wide enough that decisive
# deep-ranked candidates (fused rank >16) survive to generation and
# verification inside the 16k source-token budget; the budget selector
# stays authoritative so real RAM/latency/token limits still bind.
MAX_PACK_PASSAGES = 20

# The generator receives the entire retrieved Evidence Pack (no top-5/top-8
# evidence window). A passage's completeness is a source-fidelity invariant,
# independent of answer latency or qualification thresholds.
# Retained only for backwards import compatibility; zero means no
# generation-stage passage-count cap. Consume only via an explicit
# ``<= 0`` check (as in ``assess_evidence_window_coverage``) or
# :func:`apply_answer_generation_window`: never ``pack[:CONST]``
# (``pack[:0]`` is empty) or ``if CONST`` (``bool(0)`` is False).
ANSWER_GENERATION_MAX_PASSAGES = 0


def apply_answer_generation_window(
    pack: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the generation window for an Evidence Pack.

    A non-positive cap means the whole pack reaches generation with full
    text; a positive cap preserves legacy bounded slicing for
    observability. Centralizes the ``<= 0 means uncapped`` rule so callers
    never rely on ``pack[:CAP]`` or truthiness, both of which invert for
    ``0``.
    """
    if int(ANSWER_GENERATION_MAX_PASSAGES) <= 0:
        return pack
    return pack[: int(ANSWER_GENERATION_MAX_PASSAGES)]


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
#
# Gate C+E live repair, kodmial/aa#269 on exact main 90ff8b8 run
# 37813016105 (C:live-answer-relevance-substantive-drinking-2 plus
# E:latency-budget-exceeded p50 25027ms / p95 71451ms / max 81435ms with
# planner p50 5926ms, answer p50 9162ms, verifier p50 9632ms, repair_turns=2,
# answer_rounds=24, message-text p50 5310ms / p95 14271ms over 150 text
# calls): the 90s budget never binds while the p95 tail breaches the 60s
# Gate E target, so slow tail turns still burn a second full
# planner+retrieval+answer+verifier sequence after an already-slow initial
# chain. Tightening to 50s keeps one full repair reachable on ordinary
# turns (initial draft+verify plus one repair round still fit) while slow
# tail turns skip further re-planning and narrow to verified supported
# material instead of grinding another ~20s sequence. Turn-independent,
# never an exact-question special case.
TURN_REPAIR_TIME_BUDGET_S = 50.0

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
# the temporary Gate E hard SLO (max < 120s, delivery margin kept;
# 105s < 120s) instead of the 60s p95 target, so ordinary slow turns
# complete as verified
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


# Issue #301: no hardcoded AA conversational replies. Model/provider /
# retrieval / timeout / verifier failures are typed unsuccessful
# outcomes (:class:`TurnFailed`), never synthetic successful AA
# conversation. Greeting, identity/capabilities, clarification,
# follow-up, disambiguation and safety-recovered answers are composed
# by the AA model from the user message and conversational state; only
# the transport boundary may emit the clearly marked service-error
# signal (never a substantive answer).

# Outbound-safety recovery uses the same generic semantic fallback as any
# other retrieval need: the resolved intent plus bounded conversation
# context. No canned query-expansion tables are maintained here.
OUTBOUND_RECOVERY_QUERIES: tuple[str, ...] = ()


def safety_recovery_request(*, resolved_intent: str, user_message: str) -> str:
    """Return the intent-preserving request for safety recovery.

    The planner's resolved intent (with relevant history already folded
    in upstream) stays primary; the raw turn is only a fallback. No
    keyword, stem, domain-vocabulary or exact-utterance routing is
    applied here: the caller passes model-resolved text through verbatim.
    The safety control instruction is never merged into this string.
    """
    intent = " ".join(str(resolved_intent or "").split()).strip()
    if intent:
        return intent
    return " ".join(str(user_message or "").split()).strip()


def build_safety_recovery_queries(
    resolved_request: str,
    *,
    summary: str = "",
    recent_texts: Sequence[str] | None = None,
    max_queries: int = 12,
) -> list[str]:
    """Build safety-recovery retrieval queries from the genuine user need.

    Only the resolved request plus bounded conversation context travel
    here. The safety control instruction is never part of a semantic
    query, so retrieval cannot be polluted by policy wording.
    """
    from aa.conversation.answer_adequacy import build_generic_fallback_queries

    cleaned = " ".join(str(resolved_request or "").split()).strip()
    if not cleaned:
        return []
    recent = [str(item) for item in (recent_texts or []) if str(item).strip()]
    try:
        return build_generic_fallback_queries(
            cleaned, summary=summary, recent=recent, max_queries=max_queries
        )
    except Exception:
        return [cleaned]


def safety_candidate_fingerprint(text: str) -> str:
    """Return a privacy-safe digest identifying one recovery candidate.

    Delegates to the adequacy-layer digest so this hot-path module keeps
    no hash-selection machinery of its own (kodmial/aa#240 invariant).
    """
    from aa.conversation.answer_adequacy import safety_candidate_fingerprint as _fingerprint

    return _fingerprint(text)


async def generate_dialogue_reply(
    *,
    user_message: str,
    summary: str,
    recent: list[Any],
    answer_model: Any,
) -> str | None:
    """Generate one model-composed dialogue reply with no book evidence.

    Issue #301: greeting, identity/capabilities, clarification,
    follow-up and disambiguation are composed by the AA model from the
    user message and conversational state, never from a fixed string.
    The same generation contract is used (empty Evidence Pack, live
    request verbatim last); the caller still enforces envelope,
    language, leak, quote-budget and outbound-safety gates. Returns
    ``None`` when the model cannot produce a usable candidate (typed
    failure upstream, never a canned substitute).
    """
    from aa.conversation.answer_node import generate_draft

    cleaned = " ".join(str(user_message or "").split()).strip()
    if not cleaned or answer_model is None:
        return None
    try:
        candidate = await generate_draft(
            model=answer_model,
            recent=list(recent or []),
            summary=str(summary or ""),
            passages=[],
            user_message=cleaned,
        )
    except Exception:
        return None
    text = str(candidate or "").strip()
    if not text:
        return None
    if not contains_cyrillic(text) or leaks_internal_terms(text):
        return None
    if not envelope_passes(text):
        return None
    if aggregate_quote_chars(text) > QUOTE_BUDGET_CHARS:
        return None
    if not certify_outbound_safety(text):
        return None
    return text


# Supplementary repair context label shared by all focused-regeneration
# prompts below. Generic wording only, never an exact live prompt.
MISSING_SUPPORT_PREFIX = "Недостающая поддержка"


def anchored_repair_focus(user_message: str, missing_texts: Sequence[str]) -> str:
    """Build a repair focus that keeps the live request topically primary.

    Gate C+E live repair, kodmial/aa#269 on exact main 90ff8b8 run
    37813016105 (C:live-answer-relevance-substantive-drinking-2): repair
    and fallback regeneration prompts previously carried the live user
    message first and appended model-generated unsupported unit texts
    after it, so the model attended to trailing supplementary material
    (which may itself be off-topic) instead of the actual request and
    served verified-but-irrelevant replies. Supplementary context now
    travels first and the live request travels last, matching the answer
    prompt contract where ``<user_message>`` is always last. Generic
    ordering only, never an exact-question special case.
    """
    cleaned = [text.strip() for text in missing_texts if text.strip()]
    if not cleaned:
        return user_message
    joined = " | ".join(cleaned)[:800]
    return f"{MISSING_SUPPORT_PREFIX}: {joined}\n{user_message}"


def anchored_adequacy_regen_prompt(resolved_request: str) -> str:
    """Build the adequacy-regeneration prompt with the request last.

    Same kodmial/aa#269 relevance mechanism as
    :func:`anchored_repair_focus`: the practical-answer instruction
    travels first and the resolved live request travels last, so the
    regeneration stays anchored to the user's actual topic instead of
    drifting toward trailing instruction prose. Generic ordering only,
    never an exact-question special case.
    """
    return (
        "Дайте один практичный ответ по книге: "
        "конкретное объяснение и ближайший шаг только из приведённых отрывков, "
        "сохраняя тот же предмет и шаг, о котором спрашивает пользователь.\n"
        f"{resolved_request}"
    )


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


def certified_user_span_texts(
    *,
    draft: str,
    units: Sequence[ResponseUnitDraft],
    result: GroundingResult | None,
    pack: list[dict[str, Any]],
    recent: Sequence[Any] | None = None,
    user_message: str = "",
) -> frozenset[str]:
    """Return trusted user-attributed span texts for one verified draft.

    Carries the #308 candidate-level origin trust across textual
    modifications owned by later stages (narrowing, compaction,
    transport splitting): a span counts as a faithful user quotation in
    repaired/final text only when it was certified ``user_report`` on
    the checked candidate AND still anchors verbatim to a trusted
    HumanMessage slice. Anything else counts toward the book quota fail
    closed. Never raises; empty set means no trusted exemption.
    """
    if result is None or not units or not draft.strip():
        return frozenset()
    try:
        from aa.conversation.quote_provenance import certify_answer_candidate

        try:
            from aa.conversation.quote_provenance import (
                build_user_message_index,
                extract_answer_quotes,
            )
        except ImportError:
            return frozenset()
        certificate = certify_answer_candidate(
            answer=draft,
            units=units,
            verdicts=result.units,
            passages=[dict(item) for item in pack if isinstance(item, dict)],
            user_index=build_user_message_index(list(recent or []), user_message),
        )
        user_ranges = {
            (span.answer_char_start, span.answer_char_end)
            for span in certificate.spans
            if span.origin == "user_report"
        }
        return frozenset(
            span.span_text
            for span in extract_answer_quotes(draft).spans
            if (span.answer_char_start, span.answer_char_end) in user_ranges
            and span.span_text.strip()
        )
    except Exception:
        return frozenset()


def final_book_quote_chars(
    text: str,
    *,
    user_span_texts: frozenset[str] | set[str] | None = None,
    pack: list[dict[str, Any]] | None = None,
    recent: Sequence[Any] | None = None,
    user_message: str = "",
) -> int:
    """Return origin-specific verbatim-book quote chars for final text.

    The shared extractor detects every quoted span, but the book
    copyright quota applies to verified ``book_claim`` spans, not to
    faithful ``user_report`` quotations of a person's own messages.
    A span is exempt only when it carries trusted origin (certified
    ``user_report`` on the checked candidate) and still anchors
    verbatim to a HumanMessage slice; ambiguity or missing source
    mapping counts fail closed so an apparent book quote can never
    evade the quota by masquerading as a user report.
    """
    from aa.conversation.quote_provenance import anchor_user_span, extract_answer_quotes

    trusted = set(user_span_texts or set())
    try:
        from aa.conversation.quote_provenance import build_user_message_index

        user_index = build_user_message_index(list(recent or []), user_message)
    except ImportError:
        user_index = []
    _ = pack
    total = 0
    for span in extract_answer_quotes(text).spans:
        span_text = span.span_text
        if not span_text.strip():
            continue
        if span_text in trusted and anchor_user_span(span_text, user_index) is not None:
            continue
        total += len(span_text)
    return total


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


def keep_supported_units(
    units: Sequence[ResponseUnitDraft], result: GroundingResult | None
) -> list[ResponseUnitDraft]:
    """Narrow units to the verifier-supported subset matching ``keep_supported_text``."""
    if result is None:
        return []
    by_id = verdict_by_id(result)
    kept: list[ResponseUnitDraft] = []
    for unit in units:
        verdict = by_id.get(unit.unit_id)
        if verdict is not None and bool(getattr(verdict, "supported", False)):
            kept.append(unit)
    return kept


def keep_relevant_supported_units(
    units: Sequence[ResponseUnitDraft], result: GroundingResult | None
) -> list[ResponseUnitDraft]:
    """Narrow units to the supported subset that also addresses the intent.

    Architecture-level granularity repair (kodmial/aa#284 systemic
    recurrence 2): the verifier derives turn relevance from per-unit
    model verdicts and requires every served supported book unit to
    carry ``addresses_intent == True`` (padding or an off-topic
    digression alongside one relevant sentence fails the whole turn).
    The historical support-only narrowing therefore cannot rescue a
    mixed draft: it keeps the irrelevant padding, the narrowed subset
    still fails adequacy as ``irrelevant-citation``, and the turn
    collapses to a generic retry while a clean sibling passes, moving
    the fingerprint across SHAs and paraphrases. This helper keeps
    only model-certified relevant material -- supported book units
    with an explicit ``addresses_intent`` true verdict plus supported
    non-book glue, which needs no book passage -- so the served subset
    is relevant by construction. Model verdicts only: no keyword,
    stem, step-number, token-overlap, or exact-question heuristics.
    """
    if result is None:
        return []
    by_id = verdict_by_id(result)
    kept: list[ResponseUnitDraft] = []
    for unit in units:
        verdict = by_id.get(unit.unit_id)
        if verdict is None or not bool(getattr(verdict, "supported", False)):
            continue
        if str(getattr(verdict, "scope", "")) == "book" and not bool(
            getattr(verdict, "addresses_intent", False)
        ):
            continue
        kept.append(unit)
    return kept


def keep_relevant_supported_text(
    units: Sequence[ResponseUnitDraft], result: GroundingResult | None
) -> str:
    """Join the relevant supported subset matching ``keep_relevant_supported_units``."""
    return " ".join(unit.text for unit in keep_relevant_supported_units(units, result)).strip()


def has_relevant_supported_book_unit(result: GroundingResult | None) -> bool:
    """Whether any supported book unit explicitly addresses the intent."""
    if result is None:
        return False
    return any(
        verdict.scope == "book"
        and bool(verdict.supported)
        and bool(getattr(verdict, "addresses_intent", False))
        for verdict in result.units
    )


def narrowed_grounding_state(
    kept_units: Sequence[ResponseUnitDraft], result: GroundingResult | None
) -> dict[str, Any]:
    """Build verification state for exactly the served narrowed subset."""
    if result is None or not kept_units:
        return {"verified": False, "units": [], "all_required_supported": False}
    kept_ids = {unit.unit_id for unit in kept_units}
    filtered = [
        {
            "unit_id": verdict.unit_id,
            "scope": str(verdict.scope),
            "supported": bool(verdict.supported),
            "evidence_passage_ids": list(verdict.evidence_passage_ids),
            "addresses_intent": bool(getattr(verdict, "addresses_intent", False)),
            "origin": str(getattr(verdict, "origin", "conversation_glue") or "conversation_glue"),
            "origin_ref": dict(getattr(verdict, "origin_ref", None) or {})
            if isinstance(getattr(verdict, "origin_ref", None), dict)
            else {},
        }
        for verdict in result.units
        if verdict.unit_id in kept_ids and bool(verdict.supported)
    ]
    if not filtered:
        return {"verified": False, "units": [], "all_required_supported": False}
    # kodmial/aa#286: every supported book unit in the served subset must
    # address the intent; one relevant sentence no longer passes padding.
    supported_book = [item for item in filtered if item.get("scope") == "book"]
    relevant = bool(supported_book) and all(
        bool(item.get("addresses_intent", False)) for item in supported_book
    )
    needs_book = any(item.get("scope") == "book" for item in filtered)
    return {
        "verified": True,
        "all_required_supported": True,
        "units": filtered,
        "answer_relevant": bool(relevant or not needs_book),
        "relevance_category": "" if (relevant or not needs_book) else "irrelevant-citation",
    }


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
        "answer_relevant": bool(getattr(result, "answer_relevant", False)),
        "relevance_category": str(getattr(result, "relevance_category", "") or ""),
        "units": [
            {
                "unit_id": verdict.unit_id,
                "scope": str(verdict.scope),
                "supported": bool(verdict.supported),
                "evidence_passage_ids": list(verdict.evidence_passage_ids),
                "addresses_intent": bool(getattr(verdict, "addresses_intent", False)),
                "origin": str(
                    getattr(verdict, "origin", "conversation_glue") or "conversation_glue"
                ),
                "origin_ref": dict(getattr(verdict, "origin_ref", None) or {})
                if isinstance(getattr(verdict, "origin_ref", None), dict)
                else {},
            }
            for verdict in result.units
        ],
    }


def _union_id_lists(first: Any, second: Any) -> list[str]:
    """Union two optional id lists preserving order (no fabrication)."""
    out: list[str] = []
    seen: set[str] = set()
    for source in (first, second):
        if not isinstance(source, list):
            continue
        for raw in source:
            clean = str(raw or "").strip()
            if clean and clean not in seen:
                seen.add(clean)
                out.append(clean)
    return out


def merge_pack_dicts(
    current: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    resolved_intent: str = "",
    conversation_context: str = "",
    budget_tokens: int | None = None,
    max_passages: int | None = None,
    information_needs: Any | None = None,
    need_ids_by_passage: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Merge new exact passages with coverage-aware budgeting (#311).

    All eligible old and new candidates are unioned first (stable-id
    dedup with integrity failure on same-id/different-content, never a
    silent keep-older; duplicate passages union their ``need_ids`` /
    ``query_ids`` so shared evidence keeps all associations), then the
    combined pool is reconsidered under an explicit total token budget
    plus the pack count bound. A newly relevant source can therefore
    displace an older irrelevant one in an already full pack instead of
    being rejected for insertion order.

    Ordering inside this function is a lexical fallback (generic token
    overlap via :func:`order_pack_semantically`), never an LLM semantic
    verdict. When typed ``information_needs`` (plus per-passage need
    associations carried on the pack dicts or via
    ``need_ids_by_passage``) are available, budgeting preserves
    per-need representation round-robin so a second need's sole relevant
    passage survives the budget boundary even with lower lexical
    overlap; without need provenance the legacy lexical order applies.
    Provenance and all stored passage fields are preserved verbatim.
    """
    from aa.corpus.budget import RETRIEVED_PASSAGES_BUDGET_TOKENS, estimate_text_tokens
    from aa.retrieval.evidence import check_passage_id_consistency, enrich_pack_provenance

    cap = int(MAX_PACK_PASSAGES if max_passages is None else max_passages)
    budget = int(RETRIEVED_PASSAGES_BUDGET_TOKENS if budget_tokens is None else budget_tokens)
    combined: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in [*list(current or []), *list(incoming or [])]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("passage_id", ""))
        text = item.get("text")
        if not key or not isinstance(text, str) or not text:
            continue
        if key in combined:
            check_passage_id_consistency(key, combined[key], item)
            enrich_pack_provenance(combined[key], item)
            try:
                merged_needs = _union_id_lists(combined[key].get("need_ids"), item.get("need_ids"))
                if merged_needs:
                    combined[key]["need_ids"] = merged_needs
                merged_queries = _union_id_lists(
                    combined[key].get("query_ids"), item.get("query_ids")
                )
                if merged_queries:
                    combined[key]["query_ids"] = merged_queries
            except Exception:
                pass
            continue
        combined[key] = dict(item)
        order.append(key)
    # Explicit per-passage need map (same typed seam as retrieval)
    # supplements pack-embedded provenance without overwriting it.
    if isinstance(need_ids_by_passage, dict):
        for pid, nids in need_ids_by_passage.items():
            key = str(pid or "").strip()
            if not key or key not in combined:
                continue
            try:
                merged = _union_id_lists(combined[key].get("need_ids"), nids)
                if merged:
                    combined[key]["need_ids"] = merged
            except Exception:
                continue
    pool = [combined[key] for key in order]
    context = f"{resolved_intent or ''} {conversation_context or ''}".strip()
    if context:
        try:
            from aa.conversation.semantic_selection import order_pack_semantically

            pool = order_pack_semantically(
                pool,
                resolved_intent=str(resolved_intent or ""),
                conversation_context=str(conversation_context or ""),
            )
        except Exception:
            pool = [combined[key] for key in order]
    # Coverage-aware budgeting: when need provenance exists, interleave
    # per-need best-first so each represented need keeps exposure at the
    # budget boundary; remaining slots fill in pool order.
    ordered_pool = pool
    try:
        need_order: list[str] = []
        if information_needs:
            for entry in list(information_needs):
                nid = ""
                try:
                    nid = str(
                        entry.get("need_id", "")
                        if isinstance(entry, dict)
                        else getattr(entry, "need_id", "")
                    )
                except Exception:
                    nid = ""
                nid = nid.strip()
                if nid and nid not in need_order:
                    need_order.append(nid)
        has_provenance = any(
            isinstance(item.get("need_ids"), list)
            and [n for n in item["need_ids"] if str(n).strip()]
            for item in pool
            if isinstance(item, dict)
        )
        if need_order and has_provenance:
            buckets: dict[str, list[dict[str, Any]]] = {nid: [] for nid in need_order}
            for item in pool:
                raw = item.get("need_ids", [])
                ids = (
                    [str(n).strip() for n in list(raw) if str(n or "").strip()]
                    if isinstance(raw, list)
                    else []
                )
                for nid in ids:
                    if nid in buckets and item not in buckets[nid]:
                        buckets[nid].append(item)
            interleaved: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            progressed = True
            while progressed:
                progressed = False
                for nid in need_order:
                    for candidate in list(buckets.get(nid, [])):
                        pid = str(candidate.get("passage_id", ""))
                        if pid not in seen_ids:
                            interleaved.append(candidate)
                            seen_ids.add(pid)
                            progressed = True
                            break
            for item in pool:
                pid = str(item.get("passage_id", ""))
                if pid not in seen_ids:
                    interleaved.append(item)
                    seen_ids.add(pid)
            if interleaved:
                ordered_pool = interleaved
    except Exception:
        ordered_pool = pool
    selected: list[dict[str, Any]] = []
    total = 0
    for item in ordered_pool:
        if cap > 0 and len(selected) >= cap:
            break
        try:
            need = int(estimate_text_tokens(str(item.get("text", ""))))
        except Exception:
            continue
        if need <= 0:
            continue
        if need > budget:
            continue
        if total + need > budget:
            continue
        selected.append(item)
        total += need
    return selected


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
    resolved_intent: str = "",
    user_message: str = "",
    conversation_context: str = "",
    recent: Sequence[Any] | None = None,
    context_digest: str = "",
) -> tuple[list[ResponseUnitDraft], GroundingResult | None, bool]:
    """Split and verify one draft; never raises verification errors.

    The same invocation returns groundedness and intent-relevance
    verdicts. ``turn_budget_s`` caps this verifier round with the
    remaining end-to-end turn budget; ``None`` keeps the verifier
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

    async def _invoke_verifier() -> Any:
        # Backward-compatible: scripted test doubles may predate the
        # canonical digest parameter; retry without it on TypeError.
        try:
            if turn_budget_s is None:
                return await run_verifier(
                    units,
                    pack_dicts,
                    model=verifier_model,
                    resolved_intent=resolved_intent,
                    user_message=user_message,
                    conversation_context=conversation_context,
                    recent=recent,
                    answer_text=draft,
                    context_digest=context_digest,
                )
            return await run_verifier(
                units,
                pack_dicts,
                model=verifier_model,
                turn_budget_s=turn_budget_s,
                resolved_intent=resolved_intent,
                user_message=user_message,
                conversation_context=conversation_context,
                recent=recent,
                answer_text=draft,
                context_digest=context_digest,
            )
        except TypeError as exc:
            if "context_digest" not in str(exc) and "unexpected keyword" not in str(exc):
                raise
            if turn_budget_s is None:
                return await run_verifier(
                    units,
                    pack_dicts,
                    model=verifier_model,
                    resolved_intent=resolved_intent,
                    user_message=user_message,
                    conversation_context=conversation_context,
                    recent=recent,
                    answer_text=draft,
                )
            return await run_verifier(
                units,
                pack_dicts,
                model=verifier_model,
                turn_budget_s=turn_budget_s,
                resolved_intent=resolved_intent,
                user_message=user_message,
                conversation_context=conversation_context,
                recent=recent,
                answer_text=draft,
            )

    try:
        result = await _invoke_verifier()
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
    if not bool(getattr(result, "answer_relevant", False)):
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
    planner_reason: str | None = None,
    planner_outcome_raw: str | None = None,
    turn_trace_id: str | None = None,
    planner_mode: str | None = None,
    resolved_intent: str | None = None,
    whole_turn_judge_model: Any | None = None,
    conversation_context: dict[str, Any] | None = None,
    resolved_turn: dict[str, Any] | None = None,
    context_digest: str | None = None,
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
    from aa.conversation.answer_adequacy import (
        ADEQUACY_FAIL as _ADEQ_FAIL,
    )
    from aa.conversation.answer_adequacy import (
        ADEQUACY_PASS as _ADEQ_PASS,
    )
    from aa.conversation.answer_adequacy import (
        FAILURE_BUDGET_EXCEEDED as _FAIL_BUDGET,
    )
    from aa.conversation.answer_adequacy import (
        FAILURE_REPAIR_FAILED as _FAIL_REPAIR,
    )
    from aa.conversation.answer_adequacy import (
        assess_turn_adequacy as _assess_adequacy,
    )
    from aa.conversation.answer_adequacy import (
        build_generic_fallback_queries as _build_recovery,
    )
    from aa.conversation.answer_adequacy import (
        effective_request as _effective_request_fn,
    )
    from aa.conversation.answer_adequacy import (
        is_conversational_plan as _is_conversational,
    )
    from aa.conversation.answer_adequacy import (
        new_turn_trace_id as _new_trace_id,
    )
    from aa.conversation.answer_adequacy import (
        planner_reason_for as _planner_reason_for,
    )
    from aa.conversation.answer_adequacy import (
        runtime_sha as _runtime_sha,
    )

    recent_ranges = [dict(item) for item in (recent_quote_ranges or []) if isinstance(item, dict)]
    pack = [dict(item) for item in evidence_pack if isinstance(item, dict)]
    # Authoritative pack integrity before any answer/verifier model call
    # (kodmial/aa#310): corrupt packs fail here, before generation and
    # before verification, including every repair path below. Empty pack
    # (glue/user-report) passes.
    try:
        from aa.conversation.evidence_integrity import validate_book_pack_for_model_use

        validate_book_pack_for_model_use(pack)
    except Exception as exc:
        from aa.conversation.failures import TurnFailed as _PackTurnFailed

        raise _PackTurnFailed("evidence-integrity", f"evidence pack invalid: {exc}") from exc
    # Lexical-fallback ordering over the broad Evidence Pack (issue
    # #295 heuristic, #311 provenance): reorder by generic intent
    # relevance without dropping any passage, so decisive deep-ranked
    # material surfaces for generation and verification while the full
    # pack stays available. Deterministic and model-free here (the
    # retrieval layer already applied the same promotion); a dedicated
    # selection-model pass may reorder again upstream without ever
    # pruning before budgeting. This is a lexical fallback, never a
    # model semantic verdict (#311/#312).
    _pack_order = "fused"
    try:
        from aa.conversation.semantic_selection import order_pack_semantically as _order_pack

        _order_intent = str(resolved_intent or "").strip() or " ".join(user_message.split()).strip()
        if pack and _order_intent:
            pack = _order_pack(
                pack, resolved_intent=_order_intent, conversation_context=str(summary or "")
            )
            _pack_order = "lexical_fallback"
            logger.debug(
                "evidence pack lexically reordered",
                extra={"passages": len(pack)},
            )
        else:
            logger.debug(
                "evidence pack lexical reorder skipped; fused order kept",
                extra={"pack_empty": not pack, "intent_empty": not _order_intent},
            )
    except Exception as exc:
        _pack_order = "fused-fallback"
        logger.debug(
            "evidence pack lexical reorder failed; fused order kept",
            extra={"category": type(exc).__name__},
        )
    initial_pack_empty = not pack
    initial_pack_passages = len(pack)
    _query_hint = int(initial_query_count) if isinstance(initial_query_count, int) else 0
    _raw_reason_in = str(planner_reason or planner_outcome_raw or "")
    if _raw_reason_in:
        _effective_reason = (
            _raw_reason_in
            if _raw_reason_in
            in (
                "legitimate-glue",
                "substantive-with-queries",
                "provider-error",
                "timeout",
                "invalid",
                "unknown",
            )
            else _planner_reason_for(_query_hint, _raw_reason_in)
        )
    else:
        _effective_reason = _planner_reason_for(_query_hint, "ok" if _query_hint else "empty")
    # Model-driven turn understanding: the planner already resolved the
    # intent. When upstream did not supply a mode, derive it from the
    # reason without inspecting text; provider errors never count as glue.
    _mode = str(planner_mode or "").strip()
    if not _mode:
        _mode = (
            "conversational"
            if _effective_reason == "legitimate-glue" and _query_hint == 0
            else "retrieval"
        )
    _resolved_intent = str(resolved_intent or "").strip()
    if not _resolved_intent:
        _resolved_intent = " ".join(user_message.split()).strip()
    try:
        _resolved_request = _effective_request_fn(
            resolved_intent=_resolved_intent, user_message=user_message
        )
    except Exception:
        _resolved_request = user_message
    # Canonical shared view (#305): planner, selector, generator, repair
    # and verifier consume the same resolved-turn bytes and digest. The
    # canonical context (when supplied by the graph) is authoritative;
    # legacy ad-hoc assembly remains only for direct calls. Tail-preserving
    # truncation keeps end-of-message conditions; no stage secretly clips
    # first-N while claiming the same digest.
    _canonical_source: dict[str, Any] | None = None
    if isinstance(resolved_turn, dict) and resolved_turn.get("conversation_context"):
        try:
            _canonical_source = dict(resolved_turn["conversation_context"])
        except Exception:
            _canonical_source = None
    elif isinstance(conversation_context, dict) and conversation_context.get("snapshot_version"):
        _canonical_source = dict(conversation_context)
    _canonical_digest = str(context_digest or "")
    if not _canonical_digest and isinstance(resolved_turn, dict):
        _canonical_digest = str(resolved_turn.get("context_digest", "") or "")
    if _canonical_source is not None:
        try:
            from aa.conversation.conversation_context import canonical_model_view as _cc_view
            from aa.conversation.conversation_context import (
                truncate_preserving_tail as _cc_tail,
            )

            _view = _cc_view(_canonical_source)
            _conversation_context = str(_view.get("combined", "") or "")
            _recent_texts = [
                line for line in str(_view.get("conversation", "")).splitlines() if line.strip()
            ]
            # Resolve the canonical intent for downstream prompts.
            if (
                isinstance(resolved_turn, dict)
                and str(resolved_turn.get("resolved_intent", "")).strip()
            ):
                _resolved_intent = " ".join(str(resolved_turn["resolved_intent"]).split()).strip()
                try:
                    _resolved_request = _effective_request_fn(
                        resolved_intent=_resolved_intent, user_message=user_message
                    )
                except Exception:
                    _resolved_request = user_message
        except Exception:
            _canonical_source = None
    if _canonical_source is None:
        from aa.conversation.conversation_context import truncate_preserving_tail as _cc_tail

        _recent_texts = []
        for _msg in list(recent or []):
            _content = getattr(_msg, "content", "")
            if isinstance(_content, str) and _content.strip():
                _recent_texts.append(_cc_tail(_content.strip(), 1200))
        _conversation_context = _cc_tail(
            " ".join([summary.strip(), *_recent_texts[-8:]]).strip(), 4000
        )
    # Schema-only glue decision: conversational mode with zero queries.
    try:
        _proven_glue = bool(_is_conversational(mode=_mode, query_count=_query_hint))
    except Exception:
        _proven_glue = bool(_mode == "conversational" and _query_hint == 0)
    _trace_id = str(turn_trace_id or _new_trace_id())

    telemetry: dict[str, Any] = {
        "planner_outcome": "skipped-initial",
        "planner_latency_ms": 0.0,
        "planner_query_count": 0,
        "planner_reason": _effective_reason,
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
        "answer_generation_window": initial_pack_passages,
        "evidence_window_omitted_generation": 0,
        "evidence_window_omitted_verifier": 0,
        "semantic_selection_applied": bool(initial_pack_passages > 0),
        "pack_order": _pack_order,
        "semantic_deep_rank_promoted": False,
        "adequacy_verdict": "unknown",
        "failure_category": "",
        "answers_request": False,
        "technically_grounded": False,
        "qualified": False,
        "turn_trace_id": _trace_id,
        "runtime_sha": _runtime_sha(),
        "context_digest": _canonical_digest,
    }

    def _record_adequacy(
        *,
        reply_text: str,
        verification_state: dict[str, Any] | None,
    ) -> None:
        """Assess the whole turn and record split statuses in telemetry."""
        try:
            assessment = _assess_adequacy(
                user_message=user_message,
                reply=reply_text,
                evidence_pack=pack,
                grounding_result=verification_state,
                planner_reason=str(telemetry.get("planner_reason", _effective_reason)),
                verifier_outcome=str(telemetry.get("verifier_outcome", "unknown")),
                unavailable_units=int(telemetry.get("verifier_unavailable_units", 0) or 0),
                turn_budget_exceeded=bool(telemetry.get("turn_budget_exceeded", False)),
                planner_mode=_mode,
                resolved_intent=_resolved_intent,
                planner_query_count=_query_hint,
            )
        except Exception as exc:
            # Fail closed: an assessment error on a substantive turn must
            # never fall through as unknown and serve as success.
            logger.info(
                "v2 adequacy assessment failed closed",
                extra={"category": type(exc).__name__},
            )
            telemetry["adequacy_verdict"] = _ADEQ_FAIL
            if not str(telemetry.get("failure_category", "") or "").strip():
                telemetry["failure_category"] = _FAIL_REPAIR
            telemetry["answers_request"] = False
            telemetry["technically_grounded"] = False
            telemetry["qualified"] = False
            return
        telemetry["adequacy_verdict"] = assessment.verdict
        # Preserve a concrete earlier failure category (for example a
        # recovery or repair failure) instead of overwriting it with the
        # generic adequacy label for the same turn.
        if not str(telemetry.get("failure_category", "") or "").strip():
            telemetry["failure_category"] = assessment.failure_category
        telemetry["answers_request"] = bool(assessment.answers_request)
        telemetry["technically_grounded"] = bool(assessment.technically_grounded)
        telemetry["qualified"] = bool(
            assessment.verdict == _ADEQ_PASS and assessment.answers_request
        )

    def _safe_assess_adequacy(
        *,
        reply_text: str,
        verification_state: dict[str, Any] | None,
        verifier_outcome: str | None = None,
    ) -> Any:
        """Assess adequacy without ever falling through on unknown.

        Direct adequacy reads (repair gate) must fail closed like
        ``_record_adequacy``: an assessment error on a substantive turn
        returns an explicit failure instead of raising or serving the
        candidate as success. Pure glue keeps its pass.
        """
        try:
            return _assess_adequacy(
                user_message=user_message,
                reply=reply_text,
                evidence_pack=pack,
                grounding_result=verification_state,
                planner_reason=str(telemetry.get("planner_reason", _effective_reason)),
                verifier_outcome=str(
                    verifier_outcome
                    if verifier_outcome is not None
                    else telemetry.get("verifier_outcome", "unknown")
                ),
                unavailable_units=int(telemetry.get("verifier_unavailable_units", 0) or 0),
                turn_budget_exceeded=bool(telemetry.get("turn_budget_exceeded", False)),
                planner_mode=_mode,
                resolved_intent=_resolved_intent,
                planner_query_count=_query_hint,
            )
        except Exception as exc:
            logger.info(
                "v2 adequacy assessment failed closed",
                extra={"category": type(exc).__name__},
            )
            from aa.conversation.answer_adequacy import AdequacyAssessment as _AdequacyCls

            if _proven_glue:
                return _AdequacyCls(
                    substantive_request=False,
                    technically_grounded=False,
                    answers_request=True,
                    verdict=_ADEQ_PASS,
                    failure_category="",
                    verified_book_units=0,
                    evidence_passages=len(pack),
                )

            try:
                verified = len(
                    [
                        item
                        for item in (
                            (verification_state or {}).get("units", [])
                            if isinstance(verification_state, dict)
                            else []
                        )
                        if isinstance(item, dict)
                        and item.get("scope") == "book"
                        and item.get("supported") is True
                        and item.get("evidence_passage_ids")
                    ]
                )
            except Exception:
                verified = 0
            return _AdequacyCls(
                substantive_request=True,
                technically_grounded=False,
                answers_request=False,
                verdict=_ADEQ_FAIL,
                failure_category=_FAIL_REPAIR,
                verified_book_units=verified,
                evidence_passages=len(pack),
            )

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
        try:
            telemetry["evidence_window_omitted_generation"] = 0
            telemetry["evidence_window_omitted_verifier"] = 0
            telemetry["retrieval_passages"] = len(pack)
        except Exception:
            pass

    rounds = 0

    # No silent empty-evidence success (kodmial/aa#251): a planner error
    # is never legitimate glue. When the pack is empty and the turn is
    # not positively proven glue, perform one bounded recovery against
    # the canonical RU book using the actual turn context (never canned
    # generic queries), then generate and recheck from the rebuilt pack.
    if not pack and retrieval_index is not None and not _proven_glue:
        _needs_recovery = _effective_reason in (
            "provider-error",
            "timeout",
            "invalid",
            "unknown",
            "legitimate-glue",
        ) or (initial_query_count is None or _query_hint == 0)
        if _needs_recovery:
            try:
                from aa.conversation.conversation_context import truncate_preserving_tail as _r_tail

                _recovery_recent = []
                for _msg in list(recent or [])[-2:]:
                    _content = getattr(_msg, "content", "")
                    if isinstance(_content, str) and _content.strip():
                        _recovery_recent.append(_r_tail(_content.strip(), 200))
                _recovery_queries = _build_recovery(
                    user_message, summary=summary, recent=_recovery_recent
                )
            except Exception:
                _recovery_queries = []
            if _recovery_queries:
                try:
                    from aa.retrieval.evidence import (
                        RetrievalConfig,
                        retrieve_evidence_for_recovery,
                    )

                    _active_cfg = (
                        retrieval_config if retrieval_config is not None else RetrievalConfig()
                    )
                    _rec_started = time.perf_counter()
                    _rec_pack = retrieve_evidence_for_recovery(
                        retrieval_index, _recovery_queries, config=_active_cfg
                    )
                    from aa.conversation.retrieval_node import pack_to_state as _pack_state

                    _, _rec_dicts = _pack_state(_rec_pack)
                    telemetry["retrieval_latency_ms"] = round(
                        float(telemetry.get("retrieval_latency_ms", 0.0) or 0.0)
                        + (time.perf_counter() - _rec_started) * 1000.0,
                        1,
                    )
                    if _rec_dicts:
                        pack = merge_pack_dicts(
                            pack,
                            _rec_dicts,
                            resolved_intent=_resolved_intent,
                            conversation_context=_conversation_context,
                        )
                        telemetry["retrieval_passages"] = len(pack)
                        telemetry["retrieval_outcome"] = "recovered"
                        telemetry["answer_generation_window"] = len(pack)
                        logger.info(
                            "v2 empty-pack recovery rebuilt evidence",
                            extra={"passages": len(pack)},
                        )
                    else:
                        telemetry["retrieval_outcome"] = "empty-after-recovery"
                except Exception as exc:
                    logger.info(
                        "v2 empty-pack recovery failed",
                        extra={"category": type(exc).__name__},
                    )
                    telemetry["retrieval_outcome"] = "recovery-failed"

    def _generation_window(
        active_pack: list[dict[str, Any]], *, wider: bool = False
    ) -> list[dict[str, Any]]:
        """Send all retrieved book passages to OpenCode, without rank slicing.

        The same source-complete Evidence Pack is used for initial drafts
        and revisions. Relevance selection belongs to retrieval/reranking,
        not an arbitrary top-N cutoff in answer generation.
        """
        del wider  # Backwards-compatible repair caller; no narrower window.
        return apply_answer_generation_window(active_pack)

    async def _draft_with_pack(
        active_pack: list[dict[str, Any]],
        prompt_text: str,
        *,
        wider: bool = False,
        safety_policy: str = "",
    ) -> str | None:
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
        # Re-validate on every draft, including repair paths (kodmial/aa#310):
        # merged/recovered packs fail before any answer LLM call.
        try:
            from aa.conversation.evidence_integrity import validate_book_pack_for_model_use

            validate_book_pack_for_model_use(active_pack)
        except Exception as exc:
            logger.info(
                "v2 answer pack integrity failed closed",
                extra={"category": type(exc).__name__},
            )
            telemetry["answer_outcome"] = "failed"
            return None
        passages = state_passages_to_prompt(_generation_window(active_pack, wider=wider))
        started = time.perf_counter()
        try:
            draft_call = generate_draft(
                model=answer_model,
                recent=recent,
                summary=summary,
                passages=passages,
                user_message=prompt_text,
                safety_policy=safety_policy,
                resolved_intent=_resolved_intent,
                conversation_context=_canonical_source,
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
                resolved_intent=_resolved_intent,
                user_message=user_message,
                conversation_context=_conversation_context,
                recent=list(recent or []),
                context_digest=_canonical_digest,
            )
        finally:
            telemetry["verifier_latency_ms"] = round(
                telemetry["verifier_latency_ms"] + (time.perf_counter() - started) * 1000.0, 1
            )

    draft = await _draft_with_pack(pack, user_message)
    if draft is None:
        telemetry["answer_outcome"] = "failed"
        telemetry["verifier_outcome"] = "skipped"
        if _proven_glue:
            _dialogue = await generate_dialogue_reply(
                user_message=user_message,
                summary=summary,
                recent=list(recent or []),
                answer_model=answer_model,
            )
            if _dialogue is not None:
                _finish_telemetry()
                return {
                    "text": _dialogue,
                    "units": [],
                    "verification": grounding_result_to_state(None),
                    "rounds": 0,
                    "recent_quote_ranges": recent_ranges,
                    "telemetry": dict(telemetry),
                    "evidence_pack": list(pack),
                }
        _finish_telemetry()
        raise TurnFailed(
            "answer-failed", "answer model produced no draft", telemetry=dict(telemetry)
        )

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
        """Fail the turn as a typed unsuccessful outcome (never canned text)."""
        telemetry["verifier_outcome"] = verifier_outcome
        if not str(telemetry.get("failure_category", "") or "").strip():
            telemetry["failure_category"] = _FAIL_BUDGET
        _finish_telemetry()
        raise TurnFailed(
            verifier_outcome or "verifier-unavailable",
            "verifier round skipped",
            telemetry=dict(telemetry),
        )

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
    # No silent empty-evidence success (kodmial/aa#251): a planner
    # error/timeout/invalid outcome is never legitimate glue, even with
    # zero queries. Only a positively proven contentless turn with an
    # explicit legitimate-glue reason skips repair.
    # kodmial/aa#286 item 4 signal (model-driven, never a keyword
    # table): whether the verifier reported any book-required unit. Used
    # below to guard the qualified fallback boundary when a judge is
    # present; the historical glue skip-repair contract (kodmial/aa#145)
    # is preserved when no separately instantiated judge is supplied.
    _verifier_claims_book = False
    try:
        if result is not None:
            _unavailable_ids = set(getattr(result, "unavailable_unit_ids", ()) or ())
            _verifier_claims_book = any(
                str(getattr(item, "scope", "")) == "book"
                and str(getattr(item, "unit_id", "")) not in _unavailable_ids
                for item in result.units
            )
    except Exception:
        _verifier_claims_book = False
    is_glue = bool(
        _proven_glue
        and _effective_reason == "legitimate-glue"
        and (initial_query_count is None or _query_hint == 0)
    )
    repair_allowed = (
        result is not None
        and not result.unavailable_unit_ids
        and (
            (not initial_pack_empty)
            or (initial_query_count is not None and initial_query_count > 0)
            or (not _proven_glue)
            or (_effective_reason in ("provider-error", "timeout", "invalid", "unknown"))
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
        focus = anchored_repair_focus(user_message, missing)
        try:
            from aa.conversation.planner_node import run_planner as _run_planner

            repair_started = time.perf_counter()
            # Repair re-planning consumes the same canonical context, never
            # a divergent reassembly, so corrections stay resolvable.
            # Scripted test doubles may predate the canonical parameter.
            try:
                plan = await _run_planner(
                    focus,
                    model=planner_model,
                    summary=summary,
                    recent=recent,
                    conversation_context=_canonical_source,
                )
            except TypeError as _planner_type_error:
                if "conversation_context" not in str(
                    _planner_type_error
                ) and "unexpected keyword" not in str(_planner_type_error):
                    raise
                plan = await _run_planner(
                    focus, model=planner_model, summary=summary, recent=recent
                )
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
            # Focused additional search (issue #295): repair retrieval
            # carries the resolved intent for bounded semantic promotion
            # so a decisive deep-ranked passage surfaces instead of
            # regenerating on the same misleading top prefixes.
            new_pack = retrieve_evidence(
                retrieval_index,
                queries,
                config=active_config,
                resolved_intent=_resolved_intent,
                conversation_context=_conversation_context,
            )
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
            # Gate C live repair, kodmial/aa#234 recurrence 2 on exact
            # main c34dd9e run 37805421560: the prior recurrence fixed
            # the omitted-wire circuit (verifier now healthy: 0
            # unavailable units over 49 response units), yet the gate
            # still fails live-answer-no-generic-collapse with
            # repair_turns=0, repair_rounds=0, answer_rounds=17 and no
            # budget breach. Per-stage comparison proves the collapse no
            # longer comes from the circuit: planner p50 6.5s / p95
            # 15.6s, answer p50 7.5s / p95 14.1s and verifier p50 7.0s
            # / p95 13.0s are all slow but within budget, while the
            # targeted repair never fires because retrieval returns no
            # new passages and the loop breaks with zero generations.
            # Repeating the circuit patch cannot converge. Strategy
            # change at this retrieval/repair boundary (not a retune):
            # when repair retrieval adds nothing, attempt one bounded
            # focused regeneration from the current pack before
            # collapsing, so a duplicate retrieval cannot force a
            # zero-round generic fallback. Turn-independent, never an
            # exact-question special case.
            if pack and rounds == 0:
                _fallback_focus = user_message
                try:
                    _missing_now = (
                        unsupported_unit_texts(units, result) if units else [current_draft]
                    )
                    _fallback_focus = anchored_repair_focus(
                        user_message, [text for text in _missing_now if text.strip()]
                    )
                except Exception:
                    pass
                _fallback_draft = await _draft_with_pack(pack, _fallback_focus, wider=True)
                if _fallback_draft is not None:
                    _fallback_slice = _verifier_round_budget()
                    _fallback_ok = not (
                        _fallback_slice is not None and _fallback_slice < TURN_VERIFIER_MIN_SLICE_S
                    )
                    if _fallback_ok:
                        _fb_units, _fb_result, _fb_passed = await _verify_with_telemetry(
                            _fallback_draft, pack, turn_budget_s=_fallback_slice
                        )
                        if _fb_result is not None:
                            telemetry["verifier_unavailable_units"] = len(
                                _fb_result.unavailable_unit_ids
                            )
                        if _fb_passed and _fb_units and _fb_result is not None:
                            current_draft = _fallback_draft
                            units, result, passed = _fb_units, _fb_result, True
                            telemetry["verifier_outcome"] = "passed-after-repair-existing-pack"
                            rounds += 1
                            break
                    else:
                        _mark_turn_budget_exceeded()
            break
        merged = merge_pack_dicts(
            pack,
            new_dicts,
            resolved_intent=_resolved_intent,
            conversation_context=_conversation_context,
        )
        if len(merged) == len(pack):
            # Same duplicate-retrieval collapse as above: the pack already
            # holds the only passages retrieval can find, so merging adds
            # nothing and the historical break left zero repair rounds.
            # Regenerate once from the existing pack with the same
            # missing-support focus instead of collapsing immediately.
            if rounds == 0:
                _dup_focus = user_message
                try:
                    _missing_dup = (
                        unsupported_unit_texts(units, result) if units else [current_draft]
                    )
                    _dup_focus = anchored_repair_focus(
                        user_message, [text for text in _missing_dup if text.strip()]
                    )
                except Exception:
                    pass
                _dup_draft = await _draft_with_pack(pack, _dup_focus, wider=True)
                if _dup_draft is not None:
                    _dup_slice = _verifier_round_budget()
                    _dup_ok = not (
                        _dup_slice is not None and _dup_slice < TURN_VERIFIER_MIN_SLICE_S
                    )
                    if _dup_ok:
                        _dup_units, _dup_result, _dup_passed = await _verify_with_telemetry(
                            _dup_draft, pack, turn_budget_s=_dup_slice
                        )
                        if _dup_result is not None:
                            telemetry["verifier_unavailable_units"] = len(
                                _dup_result.unavailable_unit_ids
                            )
                        if _dup_passed and _dup_units and _dup_result is not None:
                            current_draft = _dup_draft
                            units, result, passed = _dup_units, _dup_result, True
                            telemetry["verifier_outcome"] = "passed-after-repair-existing-pack"
                            rounds += 1
                            break
                    else:
                        _mark_turn_budget_exceeded()
            break
        pack = merged
        rounds += 1
        next_draft = await _draft_with_pack(pack, user_message, wider=True)
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

    # Relevance-padding rescue (kodmial/aa#284 systemic recurrence 2,
    # closed by breaker kodmial/aa#290 for verifier-unavailable poisoning).
    #
    # Architecture-level granularity repair at the delivery boundary:
    # the verifier requires every served supported book unit to carry
    # an explicit model ``addresses_intent`` verdict, so a draft that
    # mixes one relevant supported book unit with irrelevant padding
    # fails the whole turn (``answer_relevant`` false, adequacy
    # ``irrelevant-citation``) while a clean sibling passes, moving
    # the Gate C fingerprint across SHAs and paraphrases. Repair
    # replanning cannot converge this: the pack is adequate and only
    # the draft composition varies. Before collapsing to a generic
    # retry, deterministically narrow to the relevant supported
    # subset (model verdicts only, never keyword/stem/step-number/
    # token-overlap heuristics or exact-question branches) and serve
    # it when the subset itself passes the envelope, quote-budget,
    # outbound-safety and whole-turn adequacy gates. No extra model
    # call, so the Gate E SLO is preserved; per-claim grounding holds
    # for exactly what is delivered. Turns with no relevant
    # supported book unit fall through to the historical collapse.
    #
    # Breaker kodmial/aa#290 invariant: one verifier-unavailable unit is
    # synthesized as book-scoped unsupported and poisons the whole-turn
    # telemetry (repair skipped, rescue blocked, conversational fallback
    # withheld via the claims-book guard, adequacy unavailable-verifier
    # fail) even when a verified relevant supported subset exists to
    # serve. With verifier p95 51s / max 65s, whichever family hits the
    # transient collapses while clean siblings pass, moving the
    # fingerprint across runs (local 3 same, systemic 3 cross) with no
    # convergence. The narrowed subset excludes unavailable units, so its
    # telemetry must describe the served subset (zero unavailable, passed
    # outcome) rather than the discarded draft; otherwise qualification
    # still counts a grounded delivery as failure.
    if not passed and units and result is not None:
        _pad_kept = keep_relevant_supported_units(units, result)
        if _pad_kept and len(_pad_kept) < len(units) and has_relevant_supported_book_unit(result):
            _pad_candidate = keep_relevant_supported_text(_pad_kept, result)
            # Origin-specific book quota (#308): faithful user_report
            # quotations are exempt; ambiguity counts fail closed.
            _pad_user_spans = certified_user_span_texts(
                draft=current_draft,
                units=units,
                result=result,
                pack=pack,
                recent=list(recent or []),
                user_message=user_message,
            )
            if (
                _pad_candidate
                and _pad_candidate != current_draft
                and contains_cyrillic(_pad_candidate)
                and not leaks_internal_terms(_pad_candidate)
                and envelope_passes(_pad_candidate)
                and final_book_quote_chars(
                    _pad_candidate,
                    user_span_texts=_pad_user_spans,
                    pack=pack,
                    recent=list(recent or []),
                    user_message=user_message,
                )
                <= QUOTE_BUDGET_CHARS
                and certify_outbound_safety(_pad_candidate)
            ):
                _pad_state = narrowed_grounding_state(_pad_kept, result)
                _pad_saved_unavailable = int(telemetry.get("verifier_unavailable_units", 0) or 0)
                _pad_saved_outcome = str(telemetry.get("verifier_outcome", "unknown"))
                # The served subset contains no unavailable units by
                # construction (only supported units kept), so judge it
                # with subset telemetry; restore on failure.
                telemetry["verifier_unavailable_units"] = 0
                if _pad_saved_outcome in (
                    "unavailable",
                    "partial-unavailable",
                    "skipped-turn-budget",
                ):
                    telemetry["verifier_outcome"] = "passed"
                _pad_adequacy = _safe_assess_adequacy(
                    reply_text=_pad_candidate,
                    verification_state=_pad_state,
                    verifier_outcome="passed",
                )
                if _pad_adequacy.verdict == _ADEQ_PASS:
                    telemetry["answer_outcome"] = "narrowed-adequacy"
                    telemetry["outbound_safety"] = "pass"
                    telemetry["verifier_unavailable_units"] = 0
                    if str(telemetry.get("verifier_outcome", "")) in (
                        "unavailable",
                        "partial-unavailable",
                        "skipped-turn-budget",
                    ):
                        telemetry["verifier_outcome"] = "passed"
                    _record_adequacy(
                        reply_text=_pad_candidate,
                        verification_state=_pad_state,
                    )
                    _finish_telemetry()
                    return {
                        "text": _pad_candidate,
                        "units": _pad_kept,
                        "verification": _pad_state,
                        "rounds": rounds,
                        "recent_quote_ranges": merge_recent_ranges(
                            recent_ranges, ranges_from_pack(pack)
                        ),
                        "telemetry": dict(telemetry),
                        "evidence_pack": list(pack),
                    }
                telemetry["verifier_unavailable_units"] = _pad_saved_unavailable
                telemetry["verifier_outcome"] = _pad_saved_outcome

    if passed and units and result is not None:
        # kodmial/aa#286 item 4 double-misclassification guard: a
        # planner-certified glue turn with an empty pack whose verifier
        # also reports pure glue may still carry substantive advice when
        # both model judgements err. When a separately instantiated
        # whole-turn judge is supplied, it adjudicates the served draft:
        # a substantive-claim verdict overturns the glue pass so the
        # turn can never serve unverified substantive advice as a
        # qualified glue success. Without a judge the historical path is
        # preserved (live Gate C always supplies the judge). Bounded by
        # the remaining end-to-end budget; no keyword tables consulted.
        if _proven_glue and initial_pack_empty and not pack and whole_turn_judge_model is not None:
            try:
                from aa.conversation.whole_turn_judge import (
                    JUDGE_INDEPENDENCE_LIMITATION as _JUDGE_LIMIT,
                )
                from aa.conversation.whole_turn_judge import judge_whole_turn as _judge_turn

                telemetry["judge_independence_limitation"] = _JUDGE_LIMIT
                _judge_slice = _remaining_budget_s()
                if _judge_slice >= TURN_VERIFIER_MIN_SLICE_S:
                    _judgement = await _judge_turn(
                        resolved_intent=_resolved_request,
                        reply=current_draft,
                        context=_conversation_context,
                        model=whole_turn_judge_model,
                    )
                    telemetry["independent_judge_helpful"] = bool(_judgement.helpful)
                    telemetry["independent_judge_addresses_intent"] = bool(
                        _judgement.addresses_intent
                    )
                    telemetry["independent_judge_substantive"] = bool(
                        _judgement.contains_substantive_claim
                    )
                    if bool(_judgement.contains_substantive_claim):
                        logger.info(
                            "v2 glue turn overturned by whole-turn judge",
                            extra={"category": "glue-substantive-overturn"},
                        )
                        # A misclassified substantive request must never
                        # count as a qualified glue success: record an
                        # explicit failure (the shared glue adequacy path
                        # would otherwise PASS a claim-free retry as glue).
                        telemetry["adequacy_verdict"] = _ADEQ_FAIL
                        telemetry["failure_category"] = _FAIL_REPAIR
                        telemetry["answers_request"] = False
                        telemetry["technically_grounded"] = False
                        telemetry["qualified"] = False
                        _finish_telemetry()
                        raise TurnFailed(
                            "glue-substantive-overturn",
                            "judge overturned glue",
                            telemetry=dict(telemetry),
                        )
                else:
                    _mark_turn_budget_exceeded()
            except TurnFailed:
                raise
            except Exception as exc:
                logger.info(
                    "v2 whole-turn glue judge failed closed",
                    extra={"category": type(exc).__name__},
                )
        # Whole-turn answer adequacy gate (kodmial/aa#251): per-unit
        # success alone never serves an all-glue or irrelevant answer for
        # a substantive request. With evidence available regenerate once
        # from the existing precise passages, then re-verify and reassess;
        # otherwise fail explicitly with a repair failure, never a helpful
        # success claim. Bounded by the remaining end-to-end budget.
        _adequacy_now = _safe_assess_adequacy(
            reply_text=current_draft,
            verification_state=grounding_result_to_state(result),
        )
        if _adequacy_now.substantive_request and _adequacy_now.verdict == _ADEQ_FAIL and pack:
            _regen_prompt = anchored_adequacy_regen_prompt(_resolved_request)
            _regen_units: list[ResponseUnitDraft] = []
            _regen_result: GroundingResult | None = None
            _regen_passed = False
            _regen_draft = await _draft_with_pack(pack, _regen_prompt, wider=True)
            if _regen_draft is not None:
                _regen_slice = _verifier_round_budget()
                _regen_ok = not (
                    _regen_slice is not None and _regen_slice < TURN_VERIFIER_MIN_SLICE_S
                )
                if _regen_ok:
                    _regen_units, _regen_result, _regen_passed = await _verify_with_telemetry(
                        _regen_draft, pack, turn_budget_s=_regen_slice
                    )
                    if _regen_result is not None:
                        telemetry["verifier_unavailable_units"] = len(
                            _regen_result.unavailable_unit_ids
                        )
                    if _regen_passed and _regen_units and _regen_result is not None:
                        _regen_state = grounding_result_to_state(_regen_result)
                        _regen_adequacy = _safe_assess_adequacy(
                            reply_text=_regen_draft,
                            verification_state=_regen_state,
                            verifier_outcome="passed",
                        )
                        if _regen_adequacy.verdict == _ADEQ_PASS:
                            current_draft = _regen_draft
                            units, result, passed = _regen_units, _regen_result, True
                            telemetry["verifier_outcome"] = "passed-after-adequacy-repair"
                            rounds += 1
                        else:
                            telemetry["failure_category"] = _FAIL_REPAIR
                            logger.info(
                                "v2 adequacy repair failed; explicit failure served",
                                extra={"category": "adequacy-repair-failed"},
                            )
                            # Recurrence-2 partial-progress rescue: the
                            # focused regen may hold one relevant supported
                            # unit alongside unsupported filler. The full
                            # regen still fails, but its verified subset
                            # can be served narrowed (per-claim grounding
                            # holds for what is delivered) instead of
                            # discarding the progress to a generic retry.
                            # Systemic recurrence 2 (kodmial/aa#284): keep
                            # the relevant supported subset only, using the
                            # per-unit model verdicts (supported plus an
                            # explicit addresses_intent true for book
                            # units). Support-only narrowing keeps
                            # irrelevant padding, still fails adequacy as
                            # irrelevant-citation, and collapses while a
                            # clean sibling passes.
                            _regen_kept = (
                                keep_relevant_supported_units(_regen_units, _regen_result)
                                if _regen_units
                                else []
                            )
                            _regen_narrowed = (
                                keep_relevant_supported_text(_regen_kept, _regen_result)
                                if _regen_kept
                                else ""
                            )
                            if _regen_narrowed:
                                from aa.conversation.output_limits import (
                                    QUOTE_BUDGET_CHARS as _RQB,
                                )

                                _regen_user_spans = certified_user_span_texts(
                                    draft=_regen_draft,
                                    units=_regen_units,
                                    result=_regen_result,
                                    pack=pack,
                                    recent=list(recent or []),
                                    user_message=user_message,
                                )
                                if (
                                    contains_cyrillic(_regen_narrowed)
                                    and not leaks_internal_terms(_regen_narrowed)
                                    and envelope_passes(_regen_narrowed)
                                    and final_book_quote_chars(
                                        _regen_narrowed,
                                        user_span_texts=_regen_user_spans,
                                        pack=pack,
                                        recent=list(recent or []),
                                        user_message=user_message,
                                    )
                                    <= _RQB
                                    and certify_outbound_safety(_regen_narrowed)
                                ):
                                    _rn_state = narrowed_grounding_state(_regen_kept, _regen_result)
                                    _rn_adequacy = _safe_assess_adequacy(
                                        reply_text=_regen_narrowed,
                                        verification_state=_rn_state,
                                        verifier_outcome="passed",
                                    )
                                    if _rn_adequacy.verdict == _ADEQ_PASS:
                                        telemetry["answer_outcome"] = "narrowed-adequacy-regen"
                                        telemetry["outbound_safety"] = "pass"
                                        telemetry["verifier_unavailable_units"] = 0
                                        _record_adequacy(
                                            reply_text=_regen_narrowed,
                                            verification_state=_rn_state,
                                        )
                                        _finish_telemetry()
                                        return {
                                            "text": _regen_narrowed,
                                            "units": _regen_kept,
                                            "verification": _rn_state,
                                            "rounds": rounds,
                                            "recent_quote_ranges": merge_recent_ranges(
                                                recent_ranges, ranges_from_pack(pack)
                                            ),
                                            "telemetry": dict(telemetry),
                                            "evidence_pack": list(pack),
                                        }
                    else:
                        telemetry["failure_category"] = _FAIL_REPAIR
                        # Same partial-progress rescue when the regen never
                        # fully verified: serve its supported relevant
                        # subset when it exists and itself passes adequacy.
                        # ``_regen_units``/``_regen_result`` are pre-initialized
                        # above, so this branch always sees current-round
                        # evidence and never stale prior-round state.
                        _part_units = list(_regen_units)
                        _part_result = _regen_result
                        _part_kept = (
                            keep_relevant_supported_units(_part_units, _part_result)
                            if _part_units and _part_result is not None
                            else []
                        )
                        _part_narrowed = ""
                        try:
                            if _part_kept and _part_result is not None:
                                _part_narrowed = keep_relevant_supported_text(
                                    _part_kept, _part_result
                                )
                        except Exception:
                            _part_narrowed = ""
                        if _part_narrowed:
                            from aa.conversation.output_limits import (
                                QUOTE_BUDGET_CHARS as _PQB,
                            )

                            _part_draft_text = (
                                _regen_draft if isinstance(_regen_draft, str) else current_draft
                            )
                            _part_user_spans = certified_user_span_texts(
                                draft=_part_draft_text,
                                units=_part_units,
                                result=_part_result,
                                pack=pack,
                                recent=list(recent or []),
                                user_message=user_message,
                            )
                            if (
                                contains_cyrillic(_part_narrowed)
                                and not leaks_internal_terms(_part_narrowed)
                                and envelope_passes(_part_narrowed)
                                and final_book_quote_chars(
                                    _part_narrowed,
                                    user_span_texts=_part_user_spans,
                                    pack=pack,
                                    recent=list(recent or []),
                                    user_message=user_message,
                                )
                                <= _PQB
                                and certify_outbound_safety(_part_narrowed)
                            ):
                                _pn_state = narrowed_grounding_state(_part_kept, _part_result)
                                _pn_adequacy = _safe_assess_adequacy(
                                    reply_text=_part_narrowed,
                                    verification_state=_pn_state,
                                    verifier_outcome="passed",
                                )
                                if _pn_adequacy.verdict == _ADEQ_PASS:
                                    telemetry["answer_outcome"] = "narrowed-adequacy-regen"
                                    telemetry["outbound_safety"] = "pass"
                                    telemetry["verifier_unavailable_units"] = 0
                                    _record_adequacy(
                                        reply_text=_part_narrowed,
                                        verification_state=_pn_state,
                                    )
                                    _finish_telemetry()
                                    return {
                                        "text": _part_narrowed,
                                        "units": _part_kept,
                                        "verification": _pn_state,
                                        "rounds": rounds,
                                        "recent_quote_ranges": merge_recent_ranges(
                                            recent_ranges, ranges_from_pack(pack)
                                        ),
                                        "telemetry": dict(telemetry),
                                        "evidence_pack": list(pack),
                                    }
                else:
                    _mark_turn_budget_exceeded()
                    telemetry["failure_category"] = _FAIL_BUDGET
            else:
                telemetry["failure_category"] = _FAIL_REPAIR
            _adequacy_now = _safe_assess_adequacy(
                reply_text=current_draft,
                verification_state=grounding_result_to_state(result),
            )
            if _adequacy_now.substantive_request and _adequacy_now.verdict == _ADEQ_FAIL:
                # Same recurrence-2 collapse: the draft is verifier-passing
                # (supported book units exist) but whole-turn inadequate
                # (all-glue or irrelevant), and the historical path served
                # the exact generic retry without trying the verified
                # subset it already holds. Before collapsing, attempt one
                # deterministic narrowing to the relevant supported units
                # (model verdicts only: supported plus explicit
                # addresses_intent for book units, glue kept): when the
                # narrowed text itself passes envelope, safety and
                # adequacy, serving it preserves a grounded answer instead
                # of adding another generic collapse. Turn-independent,
                # never an exact-question special case.
                _narrow_kept = keep_relevant_supported_units(units, result) if units else []
                _narrowed_candidate = (
                    keep_relevant_supported_text(_narrow_kept, result) if _narrow_kept else ""
                )
                if _narrowed_candidate and _narrowed_candidate != current_draft:
                    from aa.conversation.output_limits import QUOTE_BUDGET_CHARS as _QB

                    _narrow_user_spans = certified_user_span_texts(
                        draft=current_draft,
                        units=units,
                        result=result,
                        pack=pack,
                        recent=list(recent or []),
                        user_message=user_message,
                    )
                    _narrow_ok = (
                        contains_cyrillic(_narrowed_candidate)
                        and not leaks_internal_terms(_narrowed_candidate)
                        and envelope_passes(_narrowed_candidate)
                        and final_book_quote_chars(
                            _narrowed_candidate,
                            user_span_texts=_narrow_user_spans,
                            pack=pack,
                            recent=list(recent or []),
                            user_message=user_message,
                        )
                        <= _QB
                        and certify_outbound_safety(_narrowed_candidate)
                    )
                    if _narrow_ok:
                        _narrow_state = narrowed_grounding_state(_narrow_kept, result)
                        _narrow_adequacy = _safe_assess_adequacy(
                            reply_text=_narrowed_candidate,
                            verification_state=_narrow_state,
                            verifier_outcome="passed",
                        )
                        if _narrow_adequacy.verdict == _ADEQ_PASS:
                            telemetry["answer_outcome"] = "narrowed-adequacy"
                            telemetry["outbound_safety"] = "pass"
                            _record_adequacy(
                                reply_text=_narrowed_candidate,
                                verification_state=_narrow_state,
                            )
                            _finish_telemetry()
                            return {
                                "text": _narrowed_candidate,
                                "units": _narrow_kept,
                                "verification": _narrow_state,
                                "rounds": rounds,
                                "recent_quote_ranges": merge_recent_ranges(
                                    recent_ranges, ranges_from_pack(pack)
                                ),
                                "telemetry": dict(telemetry),
                                "evidence_pack": list(pack),
                            }
                telemetry["answer_outcome"] = "adequacy-repair-failed"
                telemetry["failure_category"] = str(
                    telemetry.get("failure_category", "") or _FAIL_REPAIR
                )
                _finish_telemetry()
                raise TurnFailed(
                    "adequacy-repair-failed", "adequacy repair failed", telemetry=dict(telemetry)
                )
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
                telemetry["answer_outcome"] = "paging-guard"
                _finish_telemetry()
                raise TurnFailed(
                    "paging-guard", "adjacent paging blocked", telemetry=dict(telemetry)
                )
        transport_segments: list[str] | None = None
        # Origin-specific book quota (#308): user_report spans exempt,
        # ambiguity counts fail closed. Computed once for the served
        # draft; quote-budget calculation before and after Telegram
        # splitting uses the same canonical spans.
        _served_user_spans = certified_user_span_texts(
            draft=current_draft,
            units=units,
            result=result,
            pack=pack,
            recent=list(recent or []),
            user_message=user_message,
        )

        def _served_book_quotes(candidate: str) -> int:
            return final_book_quote_chars(
                candidate,
                user_span_texts=_served_user_spans,
                pack=pack,
                recent=list(recent or []),
                user_message=user_message,
            )

        if not envelope_passes(final):
            # Issue #295: a fully verified grounded complete answer keeps
            # its essential final points across sequential envelope-passing
            # transport segments instead of leading-sentence truncation.
            # Every segment carries only verifier-supported content (the
            # full draft passed before splitting), each passes the envelope,
            # quote-aggregate and outbound-safety gates; bulk/attack,
            # unverified or unsafe drafts keep the single-message compact
            # path below. Bounded to MAX_TRANSPORT_SEGMENTS; overflow falls
            # back to clarification, never unbounded paging.
            if (
                passed
                and units
                and result is not None
                and _served_book_quotes(final) <= QUOTE_BUDGET_CHARS
                and contains_cyrillic(final)
                and not leaks_internal_terms(final)
                and certify_outbound_safety(final)
                and not is_bulk_reproduction_request(final)
            ):
                try:
                    _segments = split_text_to_envelope_segments(final)
                except ValueError:
                    _segments = []
                if 1 < len(_segments) <= MAX_TRANSPORT_SEGMENTS and all(
                    envelope_passes(seg)
                    and _served_book_quotes(seg) <= QUOTE_BUDGET_CHARS
                    and contains_cyrillic(seg)
                    and not leaks_internal_terms(seg)
                    and certify_outbound_safety(seg)
                    for seg in _segments
                ):
                    transport_segments = _segments
                    telemetry["transport_split"] = True
                    telemetry["transport_segments"] = len(_segments)
                    telemetry["answer_outcome"] = "served-split"
                    logger.info(
                        "v2 verified answer kept complete via transport split",
                        extra={"segments": len(_segments)},
                    )
            if transport_segments is None:
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
                # (Verified split answers above skip this entire
                # single-message path so no trailing supported substance is
                # lost; delivery splits deterministically.)
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
                    try:
                        final = compact_supported_to_envelope(units, result, text=final)
                    except ValueError:
                        telemetry["answer_outcome"] = "envelope-exceeded"
                        _finish_telemetry()
                        raise TurnFailed(  # noqa: B904
                            "envelope-exceeded", "no envelope fit", telemetry=dict(telemetry)
                        )
                    if not envelope_passes(final):
                        telemetry["answer_outcome"] = "envelope-exceeded"
                        _finish_telemetry()
                        raise TurnFailed(
                            "envelope-exceeded", "no envelope fit", telemetry=dict(telemetry)
                        )
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
                            try:
                                final = compact_supported_to_envelope(units, result, text=final)
                            except ValueError:
                                telemetry["answer_outcome"] = "envelope-exceeded"
                                _finish_telemetry()
                                raise TurnFailed(  # noqa: B904
                                    "envelope-exceeded",
                                    "no envelope fit",
                                    telemetry=dict(telemetry),
                                )
                            if not envelope_passes(final):
                                telemetry["answer_outcome"] = "envelope-exceeded"
                                _finish_telemetry()
                                raise TurnFailed(
                                    "envelope-exceeded",
                                    "no envelope fit",
                                    telemetry=dict(telemetry),
                                )
                        else:
                            (
                                second_units,
                                second_result,
                                second_passed,
                            ) = await _verify_with_telemetry(
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
                                        telemetry["answer_outcome"] = "envelope-exceeded"
                                        _finish_telemetry()
                                        raise TurnFailed(
                                            "envelope-exceeded",
                                            "no envelope fit",
                                            telemetry=dict(telemetry),
                                        )
                            else:
                                try:
                                    final = compact_supported_to_envelope(units, result, text=final)
                                except ValueError:
                                    telemetry["answer_outcome"] = "envelope-exceeded"
                                    _finish_telemetry()
                                    raise TurnFailed(  # noqa: B904
                                        "envelope-exceeded",
                                        "no envelope fit",
                                        telemetry=dict(telemetry),
                                    )
                                if not envelope_passes(final):
                                    telemetry["answer_outcome"] = "envelope-exceeded"
                                    _finish_telemetry()
                                    raise TurnFailed(
                                        "envelope-exceeded",
                                        "no envelope fit",
                                        telemetry=dict(telemetry),
                                    )
                    else:
                        try:
                            final = compact_supported_to_envelope(units, result, text=final)
                        except ValueError:
                            telemetry["answer_outcome"] = "envelope-exceeded"
                            _finish_telemetry()
                            raise TurnFailed(
                                "envelope-exceeded", "no envelope fit", telemetry=dict(telemetry)
                            ) from None
                        if not envelope_passes(final):
                            telemetry["answer_outcome"] = "envelope-exceeded"
                            _finish_telemetry()
                            raise TurnFailed(
                                "envelope-exceeded", "no envelope fit", telemetry=dict(telemetry)
                            ) from None
        # Quote-budget deterministic guard (origin-specific, #308).
        # (Verified split answers already satisfied the book budget
        # before splitting; splitting never bypasses it.)
        if _served_book_quotes(final) > QUOTE_BUDGET_CHARS:
            try:
                compacted = compact_text_to_envelope(final)
            except ValueError:
                telemetry["answer_outcome"] = "quote-budget-exceeded"
                _finish_telemetry()
                raise TurnFailed("quote-budget-exceeded", "quote budget", telemetry=dict(telemetry))  # noqa: B904
            if envelope_passes(compacted) and contains_cyrillic(compacted):
                final = compacted
            else:
                telemetry["answer_outcome"] = "quote-budget-exceeded"
                _finish_telemetry()
                raise TurnFailed("quote-budget-exceeded", "quote budget", telemetry=dict(telemetry))
        if not contains_cyrillic(final) or leaks_internal_terms(final):
            telemetry["answer_outcome"] = "language-guard"
            _finish_telemetry()
            raise TurnFailed("language-guard", "language or leak guard", telemetry=dict(telemetry))
        # Mandatory outbound safety gate (#252): independent of
        # book-grounding. An authentic book-supported draft that advises
        # drinking still fails here and is never delivered. A blocked
        # draft triggers one bounded diversified recovery: retrieve
        # other materially relevant RU passages whole-book, regenerate a
        # short safe book-supported answer, and verify units plus safety
        # again. Only a genuinely unavailable recovery falls back to the
        # transparent safe-unavailability reply (never generic glue).
        if not certify_outbound_safety(final):
            from aa.safety.outbound import (
                OUTBOUND_SAFETY_MAX_REPAIRS,
                SAFE_RECOVERY_INSTRUCTION,
            )

            blocked_cat = outbound_safety_category(final)
            logger.info(
                "v2 outbound safety blocked harmful draft",
                extra={"category": blocked_cat or "drink-test-advice"},
            )
            telemetry["outbound_safety"] = "blocked"
            telemetry["outbound_safety_category"] = blocked_cat
            telemetry["outbound_safety_recovery"] = "attempted"
            recovered: dict[str, Any] | None = None
            active_pack = list(pack)
            # Intent-preserving safe recovery (kodmial/aa#300): the
            # planner's resolved intent plus conversation history stays
            # the generation/retrieval request; the safety policy is a
            # separate control block, never concatenated to user content
            # or to a semantic query. The existing pack is tried first
            # (zero novel IDs allowed); targeted retrieval only adds
            # support for the genuine need afterwards. Every candidate
            # must pass outbound safety, grounding, relevance and
            # whole-turn adequacy before delivery. Bounded, with distinct
            # fingerprints and no domain keyword routing.
            _recovery_request = safety_recovery_request(
                resolved_intent=_resolved_intent, user_message=user_message
            )
            _recovery_policy = SAFE_RECOVERY_INSTRUCTION
            _seen_recovery: set[str] = {safety_candidate_fingerprint(final)}
            _recovery_recent_texts: list[str] = list(_recent_texts[-4:])
            _recovery_attempts = 0

            async def _certify_recovery_candidate(
                candidate: str, candidate_pack: list[dict[str, Any]]
            ) -> dict[str, Any] | None:
                fingerprint = safety_candidate_fingerprint(candidate)
                if fingerprint in _seen_recovery:
                    return None
                _seen_recovery.add(fingerprint)
                if not certify_outbound_safety(candidate):
                    return None
                if not contains_cyrillic(candidate) or leaks_internal_terms(candidate):
                    return None
                if not envelope_passes(candidate):
                    return None
                if aggregate_quote_chars(candidate) > QUOTE_BUDGET_CHARS:
                    return None
                rep_slice = _verifier_round_budget()
                if rep_slice is not None and rep_slice < TURN_VERIFIER_MIN_SLICE_S:
                    _mark_turn_budget_exceeded()
                    return None
                rep_units, rep_result, rep_passed = await _verify_with_telemetry(
                    candidate, candidate_pack, turn_budget_s=rep_slice
                )
                if not rep_passed or not rep_units or rep_result is None:
                    return None
                if not certify_outbound_safety(candidate):
                    return None
                rep_state = grounding_result_to_state(rep_result)
                rep_adequacy = _safe_assess_adequacy(
                    reply_text=candidate,
                    verification_state=rep_state,
                    verifier_outcome="passed",
                )
                if rep_adequacy.verdict != _ADEQ_PASS:
                    return None
                if not bool(getattr(rep_adequacy, "answers_request", False)):
                    return None
                return {
                    "text": candidate,
                    "units": rep_units,
                    "verification": rep_result,
                    "pack": list(candidate_pack),
                }

            # Phase 1: bounded same-pack regeneration (no novel IDs needed).
            # Works even when planner/index are unavailable: the already
            # retrieved pack may support a different safe helpful answer.
            if active_pack and _recovery_request.strip():
                for _ in range(OUTBOUND_SAFETY_MAX_REPAIRS):
                    if _end_to_end_elapsed_s() > _effective_turn_budget_s():
                        _mark_turn_budget_exceeded()
                        break
                    if _recovery_attempts >= OUTBOUND_SAFETY_MAX_REPAIRS:
                        break
                    candidate = await _draft_with_pack(
                        active_pack,
                        _recovery_request,
                        wider=True,
                        safety_policy=_recovery_policy,
                    )
                    _recovery_attempts += 1
                    if candidate is None:
                        break
                    certified = await _certify_recovery_candidate(candidate, active_pack)
                    if certified is not None:
                        recovered = certified
                        break
                    # Distinct unsafe/unverified candidates count as
                    # progress; identical fingerprints are already skipped
                    # inside the certifier, so keep trying within budget.
                    continue
            # Phase 2: bounded targeted retrieval for missing support,
            # aimed at the genuine user need (never at policy wording or
            # at the unsafe draft's premise).
            if recovered is None and planner_model is not None and retrieval_index is not None:
                for _ in range(OUTBOUND_SAFETY_MAX_REPAIRS):
                    if _end_to_end_elapsed_s() > _effective_turn_budget_s():
                        _mark_turn_budget_exceeded()
                        break
                    if _recovery_attempts >= 2 * OUTBOUND_SAFETY_MAX_REPAIRS:
                        break
                    try:
                        from aa.retrieval.evidence import (
                            RetrievalConfig,
                            retrieve_evidence,
                        )

                        active_config = (
                            retrieval_config if retrieval_config is not None else RetrievalConfig()
                        )
                        _safety_queries = build_safety_recovery_queries(
                            _recovery_request,
                            summary=summary,
                            recent_texts=_recovery_recent_texts,
                            max_queries=12,
                        )
                        if not _safety_queries:
                            break
                        # Safety invariant: policy wording never enters a
                        # semantic query (checked in tests, never logged).
                        fresh = retrieve_evidence(
                            retrieval_index,
                            list(_safety_queries),
                            config=active_config,
                            resolved_intent=_resolved_intent,
                            conversation_context=_conversation_context,
                        )
                        from aa.conversation.retrieval_node import (
                            pack_to_state as _pack_to_state,
                        )

                        _, fresh_dicts = _pack_to_state(fresh)
                    except Exception:
                        break
                    if fresh_dicts:
                        merged_pack = merge_pack_dicts(
                            active_pack,
                            fresh_dicts,
                            resolved_intent=_resolved_intent,
                            conversation_context=_conversation_context,
                        )
                        # Novel evidence is a bonus, never a requirement:
                        # regenerate even when nothing new arrived (the
                        # pack may already support a safe redraft) but
                        # track pack progress so identical retries stop.
                        if merged_pack:
                            active_pack = merged_pack
                    elif not active_pack:
                        break
                    candidate = await _draft_with_pack(
                        active_pack,
                        _recovery_request,
                        wider=True,
                        safety_policy=_recovery_policy,
                    )
                    _recovery_attempts += 1
                    if candidate is None:
                        break
                    certified = await _certify_recovery_candidate(candidate, active_pack)
                    if certified is not None:
                        recovered = certified
                        break
                    continue
            if recovered is not None:
                # Production delivery contract (kodmial/aa#257 recurrence 4):
                # the diversified safe recovery is a certified book delivery,
                # not a fallback. Prior repairs (predicate-count relaxation,
                # outbound-safety clause rescoping, narrowed-adequacy
                # allowlist) never touched this return path, so a
                # verifier-passing safe recovery still carried stale
                # adequacy/qualification telemetry (adequacy "unknown",
                # answers_request/technically_grounded/qualified False) and
                # Gate C counted the drinking turn as ungrounded while
                # relevance passed. The same turn may also carry one
                # targeted/adequacy repair round (repair_rounds=1), so the
                # aggregate signature changed while the category stayed
                # stable. Strategy change at this delivery boundary (not
                # another predicate/clause retune): certify the recovered
                # text exactly like a normal served candidate before
                # returning it.
                try:
                    _safe_pack = list(recovered["pack"])
                except Exception:
                    _safe_pack = list(active_pack)
                if _safe_pack:
                    pack = _safe_pack
                    telemetry["retrieval_passages"] = len(pack)
                    telemetry["retrieval_outcome"] = "repaired"
                    telemetry["answer_generation_window"] = len(pack)
                try:
                    _safe_result = recovered["verification"]
                    _safe_unavailable = len(getattr(_safe_result, "unavailable_unit_ids", ()) or ())
                except Exception:
                    _safe_result = None
                    _safe_unavailable = 0
                telemetry["verifier_unavailable_units"] = int(_safe_unavailable)
                if str(telemetry.get("verifier_outcome", "") or "").strip() not in (
                    "passed",
                    "passed-after-repair",
                    "passed-after-repair-existing-pack",
                    "passed-after-adequacy-repair",
                ):
                    telemetry["verifier_outcome"] = "passed"
                _safe_state = grounding_result_to_state(_safe_result)
                # The recovered delivery supersedes earlier history: its own
                # adequacy determines the turn's failure category, so a stale
                # mark must not poison a certified safe delivery.
                telemetry["failure_category"] = ""
                _record_adequacy(
                    reply_text=str(recovered["text"]),
                    verification_state=_safe_state,
                )
                if not _proven_glue and telemetry.get("adequacy_verdict") == _ADEQ_FAIL:
                    # A safe but inadequate recovery is no certified help:
                    # fall through to the transparent safety-blocked reply
                    # instead of serving it as success.
                    pass
                else:
                    telemetry["outbound_safety"] = "repaired"
                    telemetry["answer_outcome"] = "served"
                    _finish_telemetry()
                    return {
                        "text": str(recovered["text"]),
                        "units": recovered["units"],
                        "verification": _safe_state,
                        "rounds": rounds,
                        "recent_quote_ranges": merge_recent_ranges(
                            recent_ranges, ranges_from_pack(pack)
                        ),
                        "telemetry": dict(telemetry),
                        "evidence_pack": list(pack),
                    }
            telemetry["answer_outcome"] = "safety-blocked"
            if not str(telemetry.get("failure_category", "") or "").strip():
                telemetry["failure_category"] = _FAIL_REPAIR
            telemetry["adequacy_verdict"] = _ADEQ_FAIL
            telemetry["answers_request"] = False
            telemetry["technically_grounded"] = False
            telemetry["qualified"] = False
            _finish_telemetry()
            raise TurnFailed("safety-blocked", "no safe recovery", telemetry=dict(telemetry))
        telemetry["outbound_safety"] = "pass"
        # Whole-turn adequacy on the served candidate (kodmial/aa#251):
        # identifiers alone never prove relevance. A substantive turn that
        # collapsed to all-glue fails explicitly instead of serving glue
        # as a helpful success.
        _served_state = grounding_result_to_state(result)
        _record_adequacy(reply_text=final, verification_state=_served_state)
        if not _proven_glue and telemetry.get("adequacy_verdict") == _ADEQ_FAIL:
            telemetry["answer_outcome"] = "adequacy-failed"
            if not str(telemetry.get("failure_category", "") or "").strip():
                telemetry["failure_category"] = _FAIL_REPAIR
            telemetry["adequacy_verdict"] = _ADEQ_FAIL
            telemetry["answers_request"] = False
            telemetry["technically_grounded"] = False
            telemetry["qualified"] = False
            _finish_telemetry()
            raise TurnFailed(
                "adequacy-failed", "served candidate inadequate", telemetry=dict(telemetry)
            )
        if transport_segments is not None:
            # Preserve the split verdict: the complete verified answer is
            # served across sequential transport segments (delivery splits
            # deterministically); adequacy below judges the full text so no
            # trailing supported substance escapes relevance/grounding.
            telemetry["answer_outcome"] = "served-split"
            telemetry["transport_split"] = True
            telemetry["transport_segments"] = len(transport_segments)
        else:
            telemetry["answer_outcome"] = "served"
            if transport_segments is None:
                telemetry.setdefault("transport_split", False)
                telemetry.setdefault("transport_segments", 1)
        _finish_telemetry()
        return {
            "text": final,
            "units": units,
            "verification": grounding_result_to_state(result),
            "rounds": rounds,
            "recent_quote_ranges": merge_recent_ranges(recent_ranges, ranges_from_pack(pack)),
            "telemetry": dict(telemetry),
            "evidence_pack": list(pack),
            "segments": list(transport_segments) if transport_segments is not None else [final],
        }

    # Conversational delivery (issue #301): a planner-certified
    # conversational turn (model-resolved conversational mode with zero
    # queries, legitimate-glue reason and an empty Evidence Pack) whose
    # single draft/verify attempt produced no verifier verdict is served
    # with one model-composed dialogue reply (empty evidence, live
    # request verbatim), never a fixed string. The candidate still
    # passes the envelope, language, leak, quote-budget and
    # outbound-safety gates inside generate_dialogue_reply. Substantive
    # turns never enter this boundary and fail as typed outcomes.
    # A verifier that reports any book-required unit proves the planner
    # misclassified a book-dependent request: such a turn must not be
    # marked successful via this fallback even when the pack is empty.
    # When a separately instantiated whole-turn judge is supplied, it
    # additionally guards the result-None boundary (validation failure
    # or outage carries no scope signal): a substantive-claim verdict
    # serves honest retry as failure instead of qualified fallback.
    # Turn-independent, never an exact-question special case.
    if (
        _proven_glue
        and not pack
        and not passed
        and not _verifier_claims_book
        and _effective_reason == "legitimate-glue"
    ):
        if whole_turn_judge_model is not None:
            try:
                from aa.conversation.whole_turn_judge import judge_whole_turn as _fb_judge

                _fb_slice = _remaining_budget_s()
                if _fb_slice >= TURN_VERIFIER_MIN_SLICE_S:
                    _fb_probe = (
                        current_draft
                        if isinstance(current_draft, str) and current_draft.strip()
                        else user_message
                    )
                    _fb_verdict = await _fb_judge(
                        resolved_intent=_resolved_request,
                        reply=_fb_probe,
                        context=_conversation_context,
                        model=whole_turn_judge_model,
                    )
                    telemetry["independent_judge_substantive"] = bool(
                        _fb_verdict.contains_substantive_claim
                    )
                    if bool(_fb_verdict.contains_substantive_claim):
                        logger.info(
                            "v2 fallback withheld by whole-turn judge",
                            extra={"category": "fallback-substantive-overturn"},
                        )
                        telemetry["adequacy_verdict"] = _ADEQ_FAIL
                        if not str(telemetry.get("failure_category", "") or "").strip():
                            telemetry["failure_category"] = _FAIL_REPAIR
                        telemetry["answers_request"] = False
                        telemetry["technically_grounded"] = False
                        telemetry["qualified"] = False
                        _finish_telemetry()
                        raise TurnFailed(
                            "fallback-substantive-overturn",
                            "judge withheld fallback",
                            telemetry=dict(telemetry),
                        )
            except TurnFailed:
                raise
            except Exception as exc:
                logger.info(
                    "v2 fallback judge failed closed",
                    extra={"category": type(exc).__name__},
                )
        _dialogue_fallback = await generate_dialogue_reply(
            user_message=user_message,
            summary=summary,
            recent=list(recent or []),
            answer_model=answer_model,
        )
        if _dialogue_fallback is None:
            telemetry["answer_outcome"] = "conversational-generation-failed"
            if not str(telemetry.get("failure_category", "") or "").strip():
                telemetry["failure_category"] = _FAIL_REPAIR
            telemetry["adequacy_verdict"] = _ADEQ_FAIL
            telemetry["answers_request"] = False
            telemetry["technically_grounded"] = False
            telemetry["qualified"] = False
            _finish_telemetry()
            raise TurnFailed(
                "conversational-generation-failed",
                "no dialogue candidate",
                telemetry=dict(telemetry),
            )
        telemetry["outbound_safety"] = "pass"
        telemetry["answer_outcome"] = "conversational-generated"
        telemetry["adequacy_verdict"] = _ADEQ_PASS
        telemetry["answers_request"] = True
        telemetry["technically_grounded"] = False
        telemetry["qualified"] = True
        _finish_telemetry()
        return {
            "text": _dialogue_fallback,
            "units": [],
            "verification": grounding_result_to_state(None),
            "rounds": rounds,
            "recent_quote_ranges": list(recent_ranges),
            "telemetry": dict(telemetry),
            "evidence_pack": list(pack),
        }

    # Repair budget exhausted: narrow to supported material, serve honest
    # unavailability for a substantive request without book evidence, and
    # keep clarification only for positively proven glue (kodmial/aa#251).
    # A substantive turn never serves plausible generic help as success.
    narrowed = keep_supported_text(units, result) if units else ""
    substantive_narrowing_ok = (
        initial_query_count is None or initial_query_count == 0 or has_supported_book_unit(result)
    )
    # Origin-specific book quota for narrowed delivery (#308).
    _tail_user_spans = certified_user_span_texts(
        draft=current_draft,
        units=units if units else [],
        result=result,
        pack=pack,
        recent=list(recent or []),
        user_message=user_message,
    )

    def _tail_book_quotes(candidate: str) -> int:
        return final_book_quote_chars(
            candidate,
            user_span_texts=_tail_user_spans,
            pack=pack,
            recent=list(recent or []),
            user_message=user_message,
        )

    if (
        narrowed
        and substantive_narrowing_ok
        and contains_cyrillic(narrowed)
        and not leaks_internal_terms(narrowed)
        and envelope_passes(narrowed)
        and _tail_book_quotes(narrowed) <= QUOTE_BUDGET_CHARS
    ):
        # Narrowed material also certifies through the outbound gate:
        # repair budget is already exhausted here, so an unsafe narrowing
        # falls back to the transparent safe-unavailability reply.
        if not certify_outbound_safety(narrowed):
            telemetry["outbound_safety"] = "blocked"
            telemetry["outbound_safety_category"] = outbound_safety_category(narrowed)
            telemetry["answer_outcome"] = "safety-blocked"
            if not str(telemetry.get("failure_category", "") or "").strip():
                telemetry["failure_category"] = _FAIL_REPAIR
            telemetry["adequacy_verdict"] = _ADEQ_FAIL
            telemetry["answers_request"] = False
            telemetry["technically_grounded"] = False
            telemetry["qualified"] = False
            _finish_telemetry()
            raise TurnFailed("safety-blocked", "narrowed unsafe", telemetry=dict(telemetry))
        telemetry["outbound_safety"] = "pass"
        telemetry["answer_outcome"] = "narrowed-supported"
        # Narrowing preserves verified supported material when part of the
        # draft is unavailable/unsupported (existing partial-failure
        # contract). Whole-turn adequacy is recorded for qualification,
        # but the narrowed supported subset is still served rather than
        # collapsing to retry: unavailable units are excluded from the
        # served text, so per-claim grounding holds for what is delivered.
        # A narrowed subset that is irrelevant or all-glue must not be
        # served as success: fall back to honest failure when the
        # whole-turn relevance invariant fails. Other failures (for
        # example partial verifier unavailability, where excluded units
        # preserve per-claim grounding) keep the narrowing contract.
        _record_adequacy(
            reply_text=narrowed,
            verification_state=grounding_result_to_state(result),
        )
        if (
            not _proven_glue
            and telemetry.get("adequacy_verdict") == _ADEQ_FAIL
            and str(telemetry.get("failure_category", "") or "").strip()
            in ("irrelevant-citation", "all-glue-for-substantive")
        ):
            telemetry["answer_outcome"] = "adequacy-failed"
            if not str(telemetry.get("failure_category", "") or "").strip():
                telemetry["failure_category"] = _FAIL_REPAIR
            telemetry["adequacy_verdict"] = _ADEQ_FAIL
            telemetry["answers_request"] = False
            telemetry["technically_grounded"] = False
            telemetry["qualified"] = False
            _finish_telemetry()
            raise TurnFailed("adequacy-failed", "narrowed inadequate", telemetry=dict(telemetry))
        _finish_telemetry()
        return {
            "text": narrowed,
            "units": units,
            "verification": grounding_result_to_state(result),
            "rounds": rounds,
            "recent_quote_ranges": merge_recent_ranges(recent_ranges, ranges_from_pack(pack)),
            "telemetry": dict(telemetry),
            "evidence_pack": list(pack),
        }
    if narrowed and (not envelope_passes(narrowed)):
        try:
            compacted = compact_text_to_envelope(narrowed)
        except ValueError:
            telemetry["answer_outcome"] = "envelope-exceeded"
            _finish_telemetry()
            raise TurnFailed("envelope-exceeded", "narrowed overflow", telemetry=dict(telemetry))  # noqa: B904
        if (
            envelope_passes(compacted)
            and contains_cyrillic(compacted)
            and not leaks_internal_terms(compacted)
            and _tail_book_quotes(compacted) <= QUOTE_BUDGET_CHARS
        ):
            if not certify_outbound_safety(compacted):
                telemetry["outbound_safety"] = "blocked"
                telemetry["outbound_safety_category"] = outbound_safety_category(compacted)
                telemetry["answer_outcome"] = "safety-blocked"
                if not str(telemetry.get("failure_category", "") or "").strip():
                    telemetry["failure_category"] = _FAIL_REPAIR
                telemetry["adequacy_verdict"] = _ADEQ_FAIL
                telemetry["answers_request"] = False
                telemetry["technically_grounded"] = False
                telemetry["qualified"] = False
                _finish_telemetry()
                raise TurnFailed("safety-blocked", "compacted unsafe", telemetry=dict(telemetry))
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
                "evidence_pack": list(pack),
            }
    if _end_to_end_elapsed_s() > _effective_turn_budget_s():
        # End-to-end guard: no verified supported material exists and
        # the turn already spent its whole budget. Fail as a typed
        # unsuccessful outcome (no further model call, no canned text).
        logger.info(
            "v2 slow turn fails as typed outcome",
            extra={"category": "turn-budget-fallback"},
        )
        _mark_turn_budget_exceeded()
        telemetry["answer_outcome"] = "retry-turn-budget"
        _finish_telemetry()
        raise TurnFailed(
            "turn-budget-exceeded", "end-to-end budget spent", telemetry=dict(telemetry)
        )
    if not _proven_glue and not pack:
        # Substantive request without any book evidence: typed
        # unsuccessful outcome, never plausible generic help.
        telemetry["answer_outcome"] = "unavailable-substantive-no-evidence"
        if not str(telemetry.get("failure_category", "") or "").strip():
            telemetry["failure_category"] = _FAIL_REPAIR
        telemetry["adequacy_verdict"] = _ADEQ_FAIL
        telemetry["answers_request"] = False
        telemetry["technically_grounded"] = False
        telemetry["qualified"] = False
        _finish_telemetry()
        raise TurnFailed(
            "unavailable-substantive-no-evidence", "no book evidence", telemetry=dict(telemetry)
        )
    if _proven_glue and not _verifier_claims_book:
        _dialogue_tail = await generate_dialogue_reply(
            user_message=user_message,
            summary=summary,
            recent=list(recent or []),
            answer_model=answer_model,
        )
        if _dialogue_tail is not None:
            telemetry["answer_outcome"] = "conversational-generated"
            _finish_telemetry()
            return {
                "text": _dialogue_tail,
                "units": units,
                "verification": grounding_result_to_state(result),
                "rounds": rounds,
                "recent_quote_ranges": recent_ranges,
                "telemetry": dict(telemetry),
                "evidence_pack": list(pack),
            }
    telemetry["answer_outcome"] = "clarification-unavailable"
    if not str(telemetry.get("failure_category", "") or "").strip():
        telemetry["failure_category"] = _FAIL_REPAIR
    telemetry["adequacy_verdict"] = _ADEQ_FAIL
    telemetry["answers_request"] = False
    telemetry["technically_grounded"] = False
    telemetry["qualified"] = False
    _finish_telemetry()
    raise TurnFailed("clarification-unavailable", "no verified answer", telemetry=dict(telemetry))


async def answer_pipeline_node(
    state: Any,
    *,
    answer_model: Any,
    verifier_model: Any,
    planner_model: Any | None = None,
    retrieval_index: Any | None = None,
    retrieval_config: Any | None = None,
    whole_turn_judge_model: Any | None = None,
) -> dict[str, Any]:
    """LangGraph answer node: state in, grounded Russian reply out."""
    from aa.conversation.graph_state import NORMAL_ROUTE

    if str(state.get("route", NORMAL_ROUTE)) != NORMAL_ROUTE:
        return {}
    user_message = str(state.get("current_user_message", ""))
    if not user_message.strip():
        from aa.conversation.graph_runtime import GraphRuntimeError as _EmptyTurnError

        raise _EmptyTurnError("empty-turn", "refusing an empty turn")
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
    try:
        _prior_retry_for_reason = state.get("retry_state", {})
        _prior_retry_for_reason_d = (
            dict(_prior_retry_for_reason) if isinstance(_prior_retry_for_reason, dict) else {}
        )
        _prior_reason = str(_prior_retry_for_reason_d.get("planner_reason", ""))
        _prior_outcome_for_reason = str(_prior_retry_for_reason_d.get("planner_outcome", ""))
        _prior_trace = str(_prior_retry_for_reason_d.get("turn_trace_id", ""))
        _prior_mode = str(
            _prior_retry_for_reason_d.get("planner_mode", state.get("planner_mode", "retrieval"))
            or "retrieval"
        )
        _prior_intent = str(
            _prior_retry_for_reason_d.get("resolved_intent", state.get("resolved_intent", "")) or ""
        )
    except Exception:
        _prior_reason = ""
        _prior_outcome_for_reason = ""
        _prior_trace = ""
        _prior_mode = str(state.get("planner_mode", "retrieval") or "retrieval")
        _prior_intent = str(state.get("resolved_intent", "") or "")
    try:
        _canonical_ctx = state.get("conversation_context", None)
        _canonical_ctx_d = dict(_canonical_ctx) if isinstance(_canonical_ctx, dict) else None
    except Exception:
        _canonical_ctx_d = None
    try:
        _resolved_turn_raw = state.get("resolved_turn", None)
        if isinstance(_resolved_turn_raw, dict):
            _resolved_turn_d = dict(_resolved_turn_raw)
        else:
            _resolved_turn_d = None
    except Exception:
        _resolved_turn_d = None
    try:
        _ctx_digest = str(state.get("context_digest", "") or "")
    except Exception:
        _ctx_digest = ""
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
        planner_reason=_prior_reason or None,
        planner_outcome_raw=_prior_outcome_for_reason or None,
        turn_trace_id=_prior_trace or None,
        planner_mode=_prior_mode or None,
        resolved_intent=_prior_intent or None,
        whole_turn_judge_model=whole_turn_judge_model,
        conversation_context=_canonical_ctx_d,
        resolved_turn=_resolved_turn_d,
        context_digest=_ctx_digest or None,
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
        try:
            from aa.conversation.answer_adequacy import planner_reason_for as _reason_enrich

            prior_reason = str(prior_retry_d.get("planner_reason", "") or "")
            if prior_reason:
                telemetry["planner_reason"] = prior_reason
            elif str(telemetry.get("planner_reason", "") or "").strip() in ("", "unknown"):
                telemetry["planner_reason"] = _reason_enrich(
                    int(telemetry.get("planner_query_count", 0) or 0),
                    str(telemetry.get("planner_outcome", "") or ""),
                )
        except Exception:
            pass
        if not str(telemetry.get("turn_trace_id", "") or "").strip():
            try:
                from aa.conversation.answer_adequacy import new_turn_trace_id as _new_trace

                telemetry["turn_trace_id"] = _new_trace()
            except Exception:
                pass
        if not str(telemetry.get("runtime_sha", "") or "").strip():
            try:
                from aa.conversation.answer_adequacy import runtime_sha as _rt_sha

                telemetry["runtime_sha"] = _rt_sha()
            except Exception:
                pass
    except Exception:
        pass
    retry_state: dict[str, Any] = {"answer_rounds": int(outcome["rounds"])}
    if telemetry:
        # Privacy-safe stage telemetry travels in orchestration state only;
        # it carries counts/latencies/outcomes, never user or evidence text.
        retry_state["turn_telemetry"] = telemetry
    # The exact evidence bundle actually used (including repair/recovery)
    # travels back into LangGraph state here, not merely inside a local
    # return structure. The finalizer digests this same pack (kodmial/aa#304).
    served_text = str(outcome["text"])
    served_pack = [
        dict(item) for item in (outcome.get("evidence_pack", []) or []) if isinstance(item, dict)
    ]
    # Never reuse the pre-repair state pack when the outcome carries no
    # evidence: a repair that acquired new evidence but did not populate
    # the outcome must fail closed on the empty pack downstream instead
    # of certifying an outdated bundle.
    # Serialized unit texts let the finalizer bind the exact delivered
    # text to stored claim verdicts by verbatim text (renumbering-safe),
    # so narrowing/compaction subsets verify without borrowed verdicts.
    served_unit_texts: list[dict[str, Any]] = []
    try:
        for unit in outcome.get("units", []) or []:
            unit_id = str(getattr(unit, "unit_id", ""))
            unit_text = str(getattr(unit, "text", ""))
            if unit_id and unit_text:
                served_unit_texts.append({"unit_id": unit_id, "text": unit_text})
    except Exception:
        served_unit_texts = []
    return {
        "draft_response": served_text,
        "final_response": served_text,
        "evidence_pack": served_pack,
        "grounding_result": dict(outcome["verification"]),
        "retry_state": retry_state,
        "recent_quote_ranges": list(outcome["recent_quote_ranges"]),
        "response_unit_texts": served_unit_texts,
    }


__all__ = [
    "ANSWER_DRAFT_ATTEMPT_BUDGET_S",
    "ANSWER_FAST_RETRY_MAX_HISTORY",
    "ANSWER_FAST_RETRY_MAX_PASSAGES",
    "ANSWER_GENERATION_MAX_PASSAGES",
    "MAX_PACK_PASSAGES",
    "MAX_TARGETED_REPAIR_ROUNDS",
    "OUTBOUND_RECOVERY_QUERIES",
    "TURN_ANSWER_MIN_SLICE_S",
    "TURN_END_TO_END_BUDGET_S",
    "TURN_REPAIR_TIME_BUDGET_S",
    "TURN_VERIFIER_MIN_SLICE_S",
    "TurnFailed",
    "anchored_adequacy_regen_prompt",
    "anchored_repair_focus",
    "build_safety_recovery_queries",
    "certified_user_span_texts",
    "final_book_quote_chars",
    "generate_dialogue_reply",
    "safety_candidate_fingerprint",
    "safety_recovery_request",
    "answer_pipeline_node",
    "apply_answer_generation_window",
    "certify_outbound_safety",
    "compact_supported_to_envelope",
    "contains_cyrillic",
    "grounding_result_to_state",
    "has_relevant_supported_book_unit",
    "has_supported_book_unit",
    "keep_relevant_supported_text",
    "keep_relevant_supported_units",
    "keep_supported_text",
    "keep_supported_units",
    "leaks_internal_terms",
    "merge_pack_dicts",
    "narrowed_grounding_state",
    "outbound_safety_category",
    "run_v2_answer_turn",
    "strip_adjacent_quotes",
    "unsupported_unit_texts",
]
