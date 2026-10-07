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

import logging
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
    """
    messages = build_planner_messages(
        user_message=user_message, summary=summary, recent=list(recent or [])
    )
    system_text = str(messages[0].content)
    user_text = str(messages[1].content)
    structured_invoke = getattr(model, "ainvoke_structured", None)
    if callable(structured_invoke):
        try:
            raw = await structured_invoke(
                user_text,
                system=system_text,
                schema=query_plan_json_schema(),
                retry_count=PLANNER_MAX_ATTEMPTS,
            )
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
    "PLANNER_TEXT_JSON_SUFFIX",
    "build_planner_messages",
    "parse_planner_text_json",
    "planner_node",
    "query_plan_json_schema",
    "run_planner",
    "validate_structured_plan",
]
