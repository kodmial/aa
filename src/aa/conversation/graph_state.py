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
    planner_mode: str
    resolved_intent: str
    retrieval_hits: list[dict[str, Any]]
    evidence_pack: list[dict[str, Any]]
    retrieval_latency_ms: float
    retrieval_over_budget: bool
    draft_response: str
    final_response: str
    grounding_result: dict[str, Any]
    retry_state: dict[str, Any]
    planner_invoked: bool
    route: str
    recent_quote_ranges: list[dict[str, Any]]
    answer_candidate: dict[str, Any]
    verification_certificate: dict[str, Any]
    delivery_status: str
    delivery_receipts: list[dict[str, Any]]
    response_unit_texts: list[dict[str, Any]]


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
        planner_mode="retrieval",
        resolved_intent="",
        retrieval_hits=[],
        evidence_pack=[],
        retrieval_latency_ms=0.0,
        retrieval_over_budget=False,
        draft_response="",
        final_response="",
        grounding_result={},
        retry_state={},
        planner_invoked=False,
        route=NORMAL_ROUTE,
        recent_quote_ranges=[],
    )


__all__ = [
    "BLOCKED_ROUTE",
    "COMMAND_ROUTE",
    "EMERGENCY_ROUTE",
    "NORMAL_ROUTE",
    "TurnState",
    "initial_state",
]
