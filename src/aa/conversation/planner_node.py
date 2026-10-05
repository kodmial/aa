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
            lines.append(f"{role}: {_render_context_value(message.content)}")
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
    Pydantic validation and never reparses text or retries.
    """
    messages = build_planner_messages(
        user_message=user_message, summary=summary, recent=list(recent or [])
    )
    system_text = str(messages[0].content)
    user_text = str(messages[1].content)
    structured_invoke = getattr(model, "ainvoke_structured", None)
    if callable(structured_invoke):
        raw = await structured_invoke(
            user_text,
            system=system_text,
            schema=query_plan_json_schema(),
            retry_count=PLANNER_MAX_ATTEMPTS,
        )
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
    "build_planner_messages",
    "planner_node",
    "query_plan_json_schema",
    "run_planner",
    "validate_structured_plan",
]
