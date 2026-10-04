"""Typed LangGraph state contract for the v2 conversational turn (issue #113).

The graph stores raw typed data; prompt text is rendered only at
model-call boundaries by the planner node and the prompt builder. Planner
and search data live in state fields, never as fake conversational
messages. ``conversation_summary`` is continuity memory only and can never
serve as authoritative AA evidence.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TurnState(TypedDict, total=False):
    """Canonical internal representation of one v2 conversational turn."""

    messages: Annotated[list[BaseMessage], add_messages]
    conversation_summary: str
    context: dict[str, Any]
    current_user_message: str
    search_queries: list[str]
    retrieval_hits: list[dict[str, Any]]
    evidence_pack: list[dict[str, Any]]
    draft_response: str
    grounding_result: dict[str, Any]
    retry_state: dict[str, Any]
    planner_invoked: bool
    route: str


NORMAL_ROUTE = "normal"
COMMAND_ROUTE = "command"
BLOCKED_ROUTE = "blocked"
EMERGENCY_ROUTE = "emergency"


def initial_state(user_message: str, *, summary: str = "") -> TurnState:
    """Build the initial raw state for one inbound user message."""
    from langchain_core.messages import HumanMessage

    return TurnState(
        messages=[HumanMessage(content=user_message)],
        conversation_summary=summary,
        current_user_message=user_message,
        search_queries=[],
        retrieval_hits=[],
        evidence_pack=[],
        draft_response="",
        grounding_result={},
        retry_state={},
        planner_invoked=False,
        route=NORMAL_ROUTE,
    )


__all__ = [
    "BLOCKED_ROUTE",
    "COMMAND_ROUTE",
    "EMERGENCY_ROUTE",
    "NORMAL_ROUTE",
    "TurnState",
    "initial_state",
]
