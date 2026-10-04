"""Typed LangGraph state contract for the new turn foundation (issue #113).

Raw typed data is stored in the graph; prompt text is rendered only at
model-call boundaries. Recent user/assistant turns keep LangChain message
roles in ``messages``. Planner/search data lives in dedicated state fields
and is never faked as conversational messages. ``conversation_summary`` is
continuity memory only and can never authorize AA evidence.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict, cast

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TurnState(TypedDict, total=False):
    """Canonical internal representation of one conversational turn."""

    messages: Annotated[list[BaseMessage], add_messages]
    conversation_summary: str
    current_user_message: str
    search_queries: list[str]
    retrieval_hits: list[dict[str, Any]]
    evidence_pack: list[dict[str, Any]]
    draft_response: str
    grounding_result: dict[str, Any] | None
    retry_state: dict[str, Any]


GRAPH_STATE_FIELDS: tuple[str, ...] = (
    "messages",
    "conversation_summary",
    "current_user_message",
    "search_queries",
    "retrieval_hits",
    "evidence_pack",
    "draft_response",
    "grounding_result",
    "retry_state",
)


def initial_state(
    *,
    current_user_message: str,
    messages: list[BaseMessage] | None = None,
    conversation_summary: str = "",
) -> TurnState:
    """Build the initial raw state for one turn (no prompt rendering)."""
    history: list[BaseMessage] = list(messages) if messages else []
    return TurnState(
        messages=history,
        conversation_summary=conversation_summary,
        current_user_message=current_user_message,
        search_queries=[],
        retrieval_hits=[],
        evidence_pack=[],
        draft_response="",
        grounding_result=None,
        retry_state={},
    )


def append_user_message(state: TurnState) -> TurnState:
    """Return a copy with the current user message appended with its role."""
    from langchain_core.messages import HumanMessage

    current = str(state.get("current_user_message", ""))
    history = list(state.get("messages", []))
    if current.strip():
        history = history + [HumanMessage(content=current)]
    updated = cast(TurnState, dict(state))
    updated["messages"] = history
    return updated


__all__ = ["GRAPH_STATE_FIELDS", "TurnState", "append_user_message", "initial_state"]
