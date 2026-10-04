"""Mandatory hidden planner node for the v2 turn graph (issue #113).

Every normal conversational turn that reaches the new graph executes this
planner after deterministic application-command/safety handling (owned by
the graph gate). The planner is a hidden orchestration call: its input
prompt and structured output never enter the user-visible message history.

Structured output uses LangChain's standard
:class:`~langchain_core.output_parsers.PydanticOutputParser` around the
text model boundary with one bounded repair retry that feeds the parser
error back to the model. No ad-hoc JSON parsing or bespoke retry
framework is introduced; no semantic fields beyond ``queries`` exist.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable

from aa.conversation.graph_state import TurnState
from aa.conversation.planner_schema import (
    PLANNER_MAX_ATTEMPTS,
    QueryPlan,
    QueryPlanValidationError,
    validate_query_plan,
)
from aa.conversation.v2_prompts import load_planner_system_v2

logger = logging.getLogger("aa.conversation.planner_node")


def _render_context_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


def build_planner_messages(
    *,
    user_message: str,
    summary: str,
    recent: list[BaseMessage],
    parser: PydanticOutputParser[QueryPlan],
) -> list[BaseMessage]:
    """Assemble the planner model input (never stored in user history)."""
    system_text = load_planner_system_v2() + "\n\n" + parser.get_format_instructions()
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


def parse_plan_text(text: str, *, parser: PydanticOutputParser[QueryPlan]) -> QueryPlan:
    """Parse and structurally validate one raw planner reply."""
    try:
        parsed = parser.parse(text)
    except (OutputParserException, ValueError) as exc:
        raise QueryPlanValidationError(f"planner output did not parse: {exc}") from exc
    return validate_query_plan(parsed)


async def run_planner(
    user_message: str,
    *,
    model: Runnable[list[BaseMessage], BaseMessage],
    summary: str = "",
    recent: list[BaseMessage] | None = None,
) -> QueryPlan:
    """Invoke the hidden planner with one bounded repair retry."""
    parser: PydanticOutputParser[QueryPlan] = PydanticOutputParser(pydantic_object=QueryPlan)
    messages = build_planner_messages(
        user_message=user_message, summary=summary, recent=list(recent or []), parser=parser
    )
    last_error = "unknown planner failure"
    for attempt in range(1, PLANNER_MAX_ATTEMPTS + 1):
        if attempt > 1:
            messages = [
                *messages,
                HumanMessage(
                    content=f"The previous output was invalid: {last_error}. "
                    "Return only the corrected structured output."
                ),
            ]
        reply = await model.ainvoke(messages)
        content = reply.content if isinstance(reply, BaseMessage) else getattr(reply, "content", "")
        text = content if isinstance(content, str) else str(content)
        try:
            plan = parse_plan_text(text, parser=parser)
        except QueryPlanValidationError as exc:
            last_error = str(exc)
            continue
        logger.info(
            "planner output accepted", extra={"attempt": attempt, "queries": len(plan.queries)}
        )
        return plan
    raise QueryPlanValidationError(last_error)


async def planner_node(
    state: TurnState,
    *,
    model: Runnable[list[BaseMessage], BaseMessage],
) -> dict[str, Any]:
    """LangGraph planner node: hidden plan into ``search_queries``.

    Only orchestration state is written; ``messages`` is left untouched so
    hidden calls never pollute the user-facing conversation.
    """
    user_message = str(state.get("current_user_message", ""))
    summary = str(state.get("conversation_summary", ""))
    recent = [item for item in state.get("messages", []) if isinstance(item, BaseMessage)][-10:]
    # The current turn is passed as structured planner input, not duplicated
    # from history: drop the trailing copy of the live user message.
    if recent and recent[-1].type == "human" and user_message:
        recent = recent[:-1]
    plan = await run_planner(user_message, model=model, summary=summary, recent=recent)
    return {"search_queries": list(plan.queries), "planner_invoked": True}


__all__ = ["build_planner_messages", "parse_plan_text", "planner_node", "run_planner"]
