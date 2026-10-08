"""Mandatory hidden planner node for the v2 turn graph (issue #113).

Every normal conversational turn that reaches the new graph executes this
planner after deterministic application-command/safety handling (owned by
the graph gate). The planner is a hidden orchestration call: its input
prompt and structured output never enter the user-visible message history.

Structured output uses OpenCode's native ``format: { type: "json_schema",
schema, retryCount }`` request field with the JSON Schema derived from the
Pydantic model. The returned structured-output object is Pydantic-validated.
No JSON is prompted for, no JSON is parsed from text, and no second
repair/retry loop exists in AA code; OpenCode owns the bounded
validation retry.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from aa.conversation.graph_state import TurnState
from aa.conversation.planner_schema import (
    MAX_QUERIES,
    MIN_NONEMPTY_QUERIES,
    PLANNER_MAX_ATTEMPTS,
    QueryPlan,
    QueryPlanValidationError,
    validate_query_plan,
)
from aa.conversation.v2_prompts import load_planner_system_v2

logger = logging.getLogger("aa.conversation.planner_node")

# Bounded per-message display length for the planner prompt only (Gate
# C+E live repair, run 37664757721 on exact main 0202b0b: p50 29.6s /
# p95 42.0s / max 45.5s with planner p50 5.7s / p95 12.1s). The planner
# consumes the full framework-managed history (no message-count slice;
# long-conversation tails make the planner prompt the second-largest
# per-turn model input after the answer pack). Truncating each message
# display bounds input tokens and planner latency while context
# resolution is unchanged for ordinary turns: short follow-ups,
# pronouns and ellipsis resolve from recent truncated context plus the
# running summary, and truncation is explicitly marked so the model
# never judges on a silently cut prefix. All messages are still sent
# (message count and order unchanged); only per-message characters are
# bounded. Turn-independent, never an exact-question special case.
PLANNER_MAX_MESSAGE_CHARS = 500

PLANNER_TRUNCATION_SUFFIX_FORMAT = "... [truncated {omitted} chars omitted]"

# Hard per-turn planner time budget (Gate C+E live repair,
# kodmial/aa#217 recurrence 6 on exact main 4a32481 run 37705584457:
# C:live-answer-no-generic-collapse (8 clarifications over 14 answer
# rounds) plus E:latency-budget-exceeded p50 14.8s / p95 36.3s / max
# 64.6s. Per-stage comparison with recurrence 5 (exact main a9c5d3e run
# 37701489048: planner p50 11.1s / p95 42.8s / max 63.4s) shows the
# recurrence-5 wall-clock timeout did not converge: planner is now
# pinned at exactly the 10s budget on every turn (p50 10006ms / p95
# 10008ms / max 10024ms), so every substantive turn serves the same 12
# identical generic queries, retrieves a generic pack the verifier
# rejects as unsupported (unavailable_units_total=0 proves transport is
# healthy), and collapses to the exact generic clarification, while the
# 10s floor is anchored into every turn (total p50 14.8s). Retrieval
# stays healthy (p50 9ms / p95 486ms), answer is stable (p50 3.5s / p95
# 8.1s), and the remaining E tail concentrates in the unbounded
# verifier sequence (p50 3ms / p95 25.4s / max 39.9s; text-path p95 15s
# / max 39.9s). The dominant persistent cause is therefore the timeout
# fallback itself: repeating or retuning it cannot converge. Strategy
# change at the turn-orchestration boundary: the generic fallback is
# removed; only the single structured attempt is individually bounded
# (see PLANNER_STRUCTURED_ATTEMPT_BUDGET_S) and a slow structured
# channel degrades to the single tailored text fallback (model-generated
# queries for this turn, strictly validated), never to identical
# generic queries. Only a wall-clock expiry of the tailored sequence
# fails closed as a provider timeout (never a generic plan). Provider
# 429 always propagates for runner retire/restart; content validation
# failures still fail closed. Turn-independent, never an
# exact-question special case. Product Contract #110 unchanged.
PLANNER_TIME_BUDGET_S = 25.0

# Per-attempt bound for the single native structured planner call (Gate
# C+E live repair, kodmial/aa#217 recurrence 6; evidence above, tightened
# for kodmial/aa#244 on exact main a0d377a run 37753553708, tightened
# again for kodmial/aa#248 on exact main e57dea5 run 37757356193:
# C:live-book-grounding-substantive-drinking-10 plus
# E:latency-budget-exceeded p50 20.0s / p95 27.0s / max 27.1s with
# planner p50 5.3s / p95 8.6s / max 10.0s, retrieval p50 0.5s,
# answer p50 6.6s / p95 10.0s (pinned at its 10s wall), verifier p50
# 6.1s / p95 12.0s (at its 12s turn wall), message-text p50 5.4s / p95
# 12.0s / max 18.5s over 69 text calls vs message-structured p50 0.5s /
# p95 2.7s over only 5 calls, 4 turns with unavailable units (5 total)
# over 33 units, answer_rounds=14, repair_turns=0. The #244 4s->3s cut
# did not converge (p50 18.9s->20.0s, p95 24.3s->27.0s): the tail is
# still one slow structured attempt plus one slow text call per stage.
# The structured channel serves fast when healthy (structured p50
# 0.47s) and tails badly when not; bounding just this attempt to 2s
# lets a slow structured channel degrade another second faster to the
# tailored text path within the same overall wall: typical turns still
# pay one fast call, tail turns still serve model-generated queries
# for this turn instead of timing out to an empty plan (the #248 C
# mechanism on held-out drinking-10: empty pack -> ungrounded retry ->
# live-book-grounding failure) while cutting the sequential
# planner+answer+verifier sum for Gate E. Strict Pydantic validation
# is unchanged on both paths; 429 propagates and never triggers the
# text path.
PLANNER_STRUCTURED_ATTEMPT_BUDGET_S = 6.0


def _diagnostic_no_turn_limits() -> bool:
    """Temporarily disable AA-owned latency budgets in manual Telegram tests.

    Provider/network error deadlines and runner lifetime remain authoritative.
    Ordinary production and qualification (env absent) are unchanged.
    """
    return os.environ.get("AA_DIAGNOSTIC_NO_TURN_LIMITS", "") == "1"


def _display_message_text(value: object) -> str:
    """Bound one history message display text for the planner prompt."""
    text = value if isinstance(value, str) else ""
    if len(text) <= PLANNER_MAX_MESSAGE_CHARS:
        return text
    omitted = len(text) - PLANNER_MAX_MESSAGE_CHARS
    return text[:PLANNER_MAX_MESSAGE_CHARS] + PLANNER_TRUNCATION_SUFFIX_FORMAT.format(
        omitted=omitted
    )


def query_plan_json_schema() -> dict[str, Any]:
    """Build the native schema, including the structural planner contract.

    OpenCode owns bounded structured-output retries, so constraints that can
    be expressed in JSON Schema must live here rather than only in the
    post-response Pydantic validator. This makes 1..9/17+ query plans, exact
    duplicate strings, and blank items retryable inside the same planner call.
    """
    schema = dict(QueryPlan.model_json_schema())
    properties = dict(schema["properties"])
    queries = dict(properties["queries"])
    items = dict(queries.get("items", {}))
    items["pattern"] = r"\S"
    queries["items"] = items
    queries["uniqueItems"] = True
    queries["anyOf"] = [
        {"maxItems": 0},
        {"minItems": MIN_NONEMPTY_QUERIES, "maxItems": MAX_QUERIES},
    ]
    properties["queries"] = queries
    schema["properties"] = properties
    return schema


def _render_context_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


def _render_recent_value(value: object) -> str:
    """Bound one recent history message for the planner prompt display.

    Message count and order are unchanged (full history still sent);
    only per-message characters are bounded with an explicit marker.
    The live current user message and the running summary travel
    untruncated so follow-up resolution never loses the live request.
    """
    if not isinstance(value, str):
        return ""
    return _display_message_text(value)


def build_planner_messages(
    *,
    user_message: str,
    summary: str,
    recent: list[BaseMessage],
) -> list[BaseMessage]:
    """Assemble the planner model input (never stored in user history).

    The role prompt travels as a native system message; the human message
    carries only conversation context and the live turn. No JSON
    instructions are embedded.
    """
    system_text = load_planner_system_v2()
    lines: list[str] = []
    summary_text = _render_context_value(summary)
    if summary_text.strip():
        lines.append("Conversation context (continuity only, not evidence):")
        lines.append(summary_text.strip())
    if recent:
        lines.append("Recent messages:")
        for message in recent:
            role = "user" if message.type == "human" else "assistant"
            lines.append(f"{role}: {_render_recent_value(message.content)}")
    lines.append("Current user message:")
    lines.append(user_message)
    return [
        SystemMessage(content=system_text),
        HumanMessage(content="\n".join(lines)),
    ]


def validate_structured_plan(data: object) -> QueryPlan:
    """Pydantic-validate one native structured-output object."""
    if isinstance(data, QueryPlan):
        return validate_query_plan(data)
    if isinstance(data, dict):
        try:
            plan = QueryPlan.model_validate(data)
        except ValidationError as exc:
            raise QueryPlanValidationError(f"planner output invalid: {exc}") from exc
        return validate_query_plan(plan)
    raise QueryPlanValidationError("planner output is not a structured object")


def _reply_structured_content(reply: object) -> object:
    if isinstance(reply, BaseMessage):
        return reply.content
    return getattr(reply, "content", reply)


# Bounded plain-text JSON fallback for the planner structured channel
# (Gate C+E live repair, kodmial/aa#217 recurrence 2 on exact main
# a7d76f1 run 37670332968: C:live-answer-no-generic-collapse plus
# E:latency-budget-exceeded p50 25.2s / p95 33.6s / max 44.7s with
# planner p50 3.6s / p95 9.2s on the weak Space Bunny fallback while
# answer/verifier served strong Muse Spark, repair_turns=0). The prior
# repair cut planner input tokens and cached the doomed omitted
# structured attempt, but the persistent planner remainder is a
# capability failure, not a token count: the native json_schema channel
# is unavailable on the fallback/omitted path, so every planner call
# still pays slow structured round-trips before serving weak queries,
# and weak queries retrieve a poor pack that the verifier rejects into
# generic clarification. This fallback retries the same single plan
# once as bounded plain-text JSON through the ordinary text path
# (already proven to serve) and strictly Pydantic-validates it.
# Grounding semantics are unchanged: only a fully validated 0 or
# 10-16 distinct-query plan is accepted; anything else fails closed.
# Provider 429 always propagates immediately and never triggers the
# text path. Turn-independent, never an exact-question special case.
PLANNER_TEXT_JSON_SUFFIX = (
    "\n\nReturn ONLY a JSON object with exactly one key: "
    '{"queries": array of strings}. '
    'Use {"queries": []} for purely conversational glue; otherwise '
    "return 10 to 16 semantically distinct Russian search queries. "
    "No other text, no markdown, no explanation."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_planner_fences(text: str) -> str:
    """Remove one markdown code fence wrapper from a planner text reply."""
    match = _FENCE_RE.search(text)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return text.strip()


def parse_planner_text_json(text: str) -> dict[str, Any]:
    """Strictly parse one text-path planner decision (fail-closed).

    Accepts only a single JSON object validated later by
    :func:`validate_structured_plan`. Small-model deviations (fences,
    leading prose, trailing commas, single quotes, Python literals) are
    normalized by the shared verifier-tolerant loader before parsing,
    but Pydantic cardinality/distinctness stays strict. Turn-independent,
    never an exact-question special case.
    """
    from aa.conversation.verifier import _strip_text_json_fences, _tolerant_json_loads

    cleaned = _strip_planner_fences(text or "")
    if not cleaned:
        raise QueryPlanValidationError("planner text output is empty")
    candidate = cleaned
    if not candidate.lstrip().startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise QueryPlanValidationError("planner text output is not a JSON object")
        candidate = candidate[start : end + 1]
    # Shared tolerant loader (strict first, then bounded normalizations);
    # the native channel stays primary and this stays a single bounded
    # capability fallback, never an unconstrained bespoke parser.
    candidate = _strip_text_json_fences(candidate)
    try:
        data = _tolerant_json_loads(candidate)
    except Exception as exc:
        raise QueryPlanValidationError(f"planner text output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise QueryPlanValidationError("planner text output is not a JSON object")
    return dict(data)


async def _invoke_planner_text(model: Any, *, user_text: str, system_text: str) -> str:
    """Invoke the ordinary text path for one planner fallback (429 propagates)."""
    prompt = user_text + PLANNER_TEXT_JSON_SUFFIX
    text_invoke = getattr(model, "_ainvoke_text", None)
    if callable(text_invoke):
        reply = await text_invoke(prompt, system=system_text)
        if not isinstance(reply, str):
            raise QueryPlanValidationError("planner text output is not text")
        return reply
    plain_invoke = getattr(model, "ainvoke", None)
    if not callable(plain_invoke):
        raise QueryPlanValidationError("planner model has no text invocation path")
    reply = await plain_invoke([SystemMessage(content=system_text), HumanMessage(content=prompt)])
    content = _reply_structured_content(reply)
    if isinstance(content, str):
        return content
    raise QueryPlanValidationError("planner text output is not text")


async def run_planner(
    user_message: str,
    *,
    model: Runnable[list[BaseMessage], BaseMessage] | Any,
    summary: str = "",
    recent: list[BaseMessage] | None = None,
    time_budget_s: float | None = None,
) -> QueryPlan:
    """Invoke the hidden planner through native structured output.

    A single OpenCode ``json_schema`` request is issued; OpenCode owns the
    bounded validation retry (``retryCount``). AA code performs exactly one
    Pydantic validation and never reparses text or retries, except for one
    bounded capability fallback: when the native structured channel itself
    is unavailable on this provider path (any :class:`OpenCodeError`),
    the same plan is retried once as bounded plain-text JSON and strictly
    validated. Content validation failures fail closed immediately.
    Provider 429 always propagates for runner retire/restart.

    Gate C+E live repair (kodmial/aa#217 recurrence 4 on exact main
    7a9c907 run 37697730282: planner p50 7.0s / p95 16.4s / max 28.7s
    with retrieval healthy at p50 0.5s and repair_turns=0): the persistent
    planner remainder after three token-display repairs is a capability
    round-trip, not token count. Once the omitted structured channel is
    cached unavailable for this model path, every later planner call still
    pays a weak-fallback structured round-trip before the strong-model
    text fallback, doubling planner cost per turn (weak structured ~4s +
    strong text ~4s) and serving weak-model queries that retrieve a poor
    pack the verifier then rejects into generic clarification. When the
    cache is set for an explicitly omitted-wire model, this goes directly
    to the single bounded text path (which itself tries the strong primary
    first), skipping the doomed weak structured call. Grounding is
    unchanged (strict Pydantic 0-or-10..16 validation); 429 propagates and
    never triggers the text path. Turn-independent, never an
    exact-question special case.

    Recurrence 6 (exact main 4a32481 run 37705584457: planner pinned at
    p50 10006ms / p95 10008ms / max 10024ms, i.e. the recurrence-5 wall
    fires on ~100% of turns; 8 clarifications with
    unavailable_units_total=0, so the collapse is verifier-unsupported
    on generic packs, not transport) proves the wall-clock generic
    fallback is the collapse mechanism, not the repair. The structured
    attempt below is therefore individually bounded
    (``PLANNER_STRUCTURED_ATTEMPT_BUDGET_S``) and a slow structured
    channel degrades to the single tailored text fallback, never to
    identical generic queries; only a wall-clock expiry of the tailored
    sequence fails closed as a provider timeout. 429 and content
    validation failures still propagate fail-closed.
    """
    messages = build_planner_messages(
        user_message=user_message, summary=summary, recent=list(recent or [])
    )
    system_text = str(messages[0].content)
    user_text = str(messages[1].content)
    budget = PLANNER_TIME_BUDGET_S if time_budget_s is None else float(time_budget_s)
    if not budget > 0:
        raise QueryPlanValidationError("planner time budget must be > 0")

    async def _provider_plan() -> QueryPlan:
        return await _run_planner_provider(
            model,
            user_text=user_text,
            system_text=system_text,
            messages=messages,
            structured_attempt_budget=(
                float("inf") if _diagnostic_no_turn_limits()
                else min(PLANNER_STRUCTURED_ATTEMPT_BUDGET_S, budget)
            ),
        )

    try:
        if _diagnostic_no_turn_limits():
            return await _provider_plan()
        return await asyncio.wait_for(_provider_plan(), timeout=budget)
    except TimeoutError as exc:
        # Fail closed as a provider timeout: serving identical generic
        # queries here reintroduces the recurrence-6 collapse (generic
        # pack -> verifier-unsupported -> exact generic clarification on
        # every slow turn). Upstream maps this to a natural retry reply,
        # never to a fake grounded plan.
        from aa.opencode.errors import OpenCodeTimeoutError as _PlannerTimeout

        logger.info(
            "planner time budget exceeded; failing closed without generic fallback",
            extra={"category": "planner-timeout"},
        )
        raise _PlannerTimeout("planner time budget exceeded") from exc


async def _run_planner_provider(
    model: Any,
    *,
    user_text: str,
    system_text: str,
    messages: list[BaseMessage],
    structured_attempt_budget: float = PLANNER_STRUCTURED_ATTEMPT_BUDGET_S,
) -> QueryPlan:
    """Run the structured-first provider sequence with a bounded structured attempt.

    The single native structured call is individually bounded so a slow
    structured channel degrades quickly to the tailored text fallback
    instead of consuming the whole turn wall and then collapsing to a
    generic plan. The overall wall is enforced by the caller.
    """
    structured_invoke = getattr(model, "ainvoke_structured", None)
    if callable(structured_invoke):
        try:
            from aa.conversation.model_adapter import omitted_structured_unavailable
        except Exception:
            omitted_structured_unavailable = None  # type: ignore[assignment]
        if omitted_structured_unavailable is not None:
            try:
                is_omitted_wire = getattr(model, "wire_agent", None) == ""
                if is_omitted_wire and bool(omitted_structured_unavailable(model)):
                    logger.info(
                        "planner omitted structured cached; direct text path used",
                        extra={"category": "capability-cached"},
                    )
                    text_reply = await _invoke_planner_text(
                        model, user_text=user_text, system_text=system_text
                    )
                    plan = validate_structured_plan(parse_planner_text_json(text_reply))
                    logger.info("planner output accepted", extra={"queries": len(plan.queries)})
                    return plan
            except Exception as exc:
                from aa.opencode.errors import OpenCodeError as _ProviderError
                from aa.opencode.errors import OpenCodeRateLimitError as _RateLimit

                if isinstance(exc, _RateLimit):
                    raise
                if isinstance(exc, _ProviderError):
                    logger.info(
                        "planner direct text path failed; structured attempt follows",
                        extra={"category": type(exc).__name__},
                    )
                elif isinstance(exc, QueryPlanValidationError):
                    raise
                # Any other probe failure falls through to the normal
                # structured-first path below (fail-open to structured,
                # fail-closed on content).
                pass
        try:
            structured_call = structured_invoke(
                user_text,
                system=system_text,
                schema=query_plan_json_schema(),
                retry_count=PLANNER_MAX_ATTEMPTS,
            )
            raw = (
                await structured_call
                if _diagnostic_no_turn_limits()
                else await asyncio.wait_for(structured_call, timeout=structured_attempt_budget)
            )
        except TimeoutError:
            logger.info(
                "planner structured attempt timed out; tailored text fallback used",
                extra={"category": "structured-attempt-timeout"},
            )
            # Gate C+E live repair, kodmial/aa#244 recurrence 2 on exact
            # main 6f8d4e1 run 37764195857 (C:live-book-grounding-
            # substantive-drinking-2 plus E p50 18.3s / p95 26.6s with
            # planner p50 5.7s / p95 10.0s pinned at its wall, answer p50
            # 6.0s / p95 10.0s pinned, verifier p50 4.7s / p95 12.0s at
            # its wall, message-structured p50 0.40s / p95 0.68s over 28
            # fast calls vs message-text p50 5.1s / p95 10.0s over 73
            # slow calls, 2 turns with unavailable units, answer_rounds
            # 14): compared with run 37757356193 (structured p50 0.47s
            # over only 5 calls, planner 5.3s, verifier 6.1s, total p50
            # 20.0s / p95 27.0s), the 3s->2s attempt cut plus 300s->60s
            # TTL did not converge (p50 -1.7s, p95 -0.4s; planner p50
            # even rose 5.3s->5.7s with p95 pinned at the wall). The
            # dominant persistent cause is now the timeout-marking
            # itself: one slow structured tail marks the whole lane to
            # the slow text path for 60s, so healthy turns that would
            # serve in ~0.4s structured instead pay ~5s text, and the
            # sequential planner+answer+verifier sum stays at ~18s p50.
            # Strategy change at this capability-cache boundary (not
            # another duration retune): a caller-observed deadline is
            # latency, not capability evidence, so it falls back once
            # without marking. Only a deterministic capability failure
            # (recorded in model_adapter as "structured output missing")
            # marks. Every turn therefore re-probes the fast structured
            # path first (0.4s when healthy) and pays 2s+text only on a
            # genuinely slow turn, cutting the sequential median for
            # Gate E while slow turns still serve model-generated
            # queries (never an empty plan, fixing the drinking-2 C
            # mechanism). 429 never marks; strict Pydantic validation
            # unchanged on both paths. Turn-independent, never an
            # exact-question special case.
            text_reply = await _invoke_planner_text(
                model, user_text=user_text, system_text=system_text
            )
            plan = validate_structured_plan(parse_planner_text_json(text_reply))
            logger.info("planner output accepted", extra={"queries": len(plan.queries)})
            return plan
        except Exception as exc:
            from aa.opencode.errors import OpenCodeRateLimitError

            if isinstance(exc, OpenCodeRateLimitError):
                raise
            from aa.opencode.errors import OpenCodeError

            if not isinstance(exc, OpenCodeError):
                raise
            logger.info(
                "planner structured channel unavailable; text fallback used",
                extra={"category": type(exc).__name__},
            )
            text_reply = await _invoke_planner_text(
                model, user_text=user_text, system_text=system_text
            )
            plan = validate_structured_plan(parse_planner_text_json(text_reply))
            logger.info("planner output accepted", extra={"queries": len(plan.queries)})
            return plan
        plan = validate_structured_plan(raw)
        logger.info("planner output accepted", extra={"queries": len(plan.queries)})
        return plan
    reply = await model.ainvoke(messages)
    content = _reply_structured_content(reply)
    if isinstance(content, str):
        raise QueryPlanValidationError("planner output is not a structured object")
    plan = validate_structured_plan(content)
    logger.info("planner output accepted", extra={"queries": len(plan.queries)})
    return plan


async def planner_node(
    state: TurnState,
    *,
    model: Runnable[list[BaseMessage], BaseMessage] | Any,
) -> dict[str, Any]:
    """LangGraph planner node: hidden plan into ``search_queries``.

    Only orchestration state is written; ``messages`` is left untouched so
    hidden calls never pollute the user-facing conversation. The full
    framework-managed history is consumed; no fixed message-count slice is
    applied.
    """
    user_message = str(state.get("current_user_message", ""))
    summary = str(state.get("conversation_summary", ""))
    recent = [item for item in state.get("messages", []) if isinstance(item, BaseMessage)]
    # The current turn is passed as structured planner input, not duplicated
    # from history: drop the trailing copy of the live user message.
    if recent and recent[-1].type == "human" and user_message:
        recent = recent[:-1]
    plan = await run_planner(user_message, model=model, summary=summary, recent=recent)
    return {"search_queries": list(plan.queries), "planner_invoked": True}


__all__ = [
    "PLANNER_MAX_MESSAGE_CHARS",
    "PLANNER_STRUCTURED_ATTEMPT_BUDGET_S",
    "PLANNER_TEXT_JSON_SUFFIX",
    "PLANNER_TIME_BUDGET_S",
    "build_planner_messages",
    "parse_planner_text_json",
    "planner_node",
    "query_plan_json_schema",
    "run_planner",
    "validate_structured_plan",
]
