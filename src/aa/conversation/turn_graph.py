"""LangGraph turn foundation for the new conversation architecture (issue #113).

Flow for this foundation task:

``Telegram turn boundary -> ingest -> safety/command gate -> mandatory
hidden planner -> retrieval stub -> answer stub``

The legacy ``aa.conversation.orchestrator`` remains available; production
traffic still uses it. Retrieval/answer nodes are interfaces only here and
perform no semantic work. No lexical routing lives in this graph.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from aa.conversation.graph_memory import (
    CheckpointerConfig,
    MemoryConfig,
    create_checkpointer,
    thread_id_for_chat,
)
from aa.conversation.graph_planner import planner_node
from aa.conversation.graph_state import TurnState
from aa.safety.router import SafetyDecision, SafetyRouter

logger = logging.getLogger("aa.conversation.turn_graph")

COMMAND_PREFIX = "/"


def is_application_command(text: str) -> bool:
    """Return whether ``text`` is a deterministic application command."""
    return text.strip().startswith(COMMAND_PREFIX)


def ingest_turn(state: TurnState) -> dict[str, Any]:
    """Append the current user message with its real role (no logging)."""
    current = str(state.get("current_user_message", "") or "")
    if not current.strip():
        return {}
    return {"messages": [HumanMessage(content=current)]}


def safety_gate(
    state: TurnState,
    *,
    safety: SafetyRouter | None = None,
) -> dict[str, Any]:
    """Deterministic command/safety handling before the mandatory planner."""
    current = str(state.get("current_user_message", "") or "")
    router = safety or SafetyRouter()
    if not current.strip():
        logger.info("turn gate decided block")
        return {"retry_state": {"route": "blocked", "reason": "empty-message"}}
    if is_application_command(current):
        logger.info("turn gate decided command")
        return {"retry_state": {"route": "command"}}
    result = router.check(current)
    if result.decision is SafetyDecision.BLOCK:
        logger.info("turn gate decided block")
        return {"retry_state": {"route": "blocked", "reason": result.reason}}
    if result.decision is SafetyDecision.EMERGENCY:
        logger.info("turn gate decided emergency")
        return {"retry_state": {"route": "emergency", "reason": result.reason}}
    logger.info("turn gate decided normal")
    return {"retry_state": {"route": "normal"}}


def route_after_safety(state: TurnState) -> str:
    """Route normal turns to the planner; anything else ends the turn."""
    retry_state = state.get("retry_state", {}) or {}
    if retry_state.get("route") == "normal":
        return "planner"
    return END


def retrieval_stub(_state: TurnState) -> dict[str, Any]:
    """Interface placeholder for the future hybrid retrieval pipeline."""
    logger.info("retrieval stub reached")
    return {}


def answer_stub(_state: TurnState) -> dict[str, Any]:
    """Interface placeholder for the future AA answer node."""
    logger.info("answer stub reached")
    return {}


def build_turn_graph(
    model: BaseChatModel,
    *,
    checkpointer_config: CheckpointerConfig | None = None,
    memory_config: MemoryConfig | None = None,
    safety: SafetyRouter | None = None,
) -> Any:
    """Build and compile the foundation turn graph with a local checkpointer."""
    _ = memory_config

    async def _planner_step(state: TurnState) -> dict[str, Any]:
        return await planner_node(state, model=model)

    async def _safety_step(state: TurnState) -> dict[str, Any]:
        return safety_gate(state, safety=safety)

    graph: Any = StateGraph(TurnState)
    graph.add_node("ingest", ingest_turn)
    graph.add_node("safety_gate", _safety_step)
    graph.add_node("planner", _planner_step)
    graph.add_node("retrieval", retrieval_stub)
    graph.add_node("answer", answer_stub)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "safety_gate")
    graph.add_conditional_edges("safety_gate", route_after_safety, ["planner", END])
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "answer")
    graph.add_edge("answer", END)
    checkpointer = create_checkpointer(checkpointer_config)
    return graph.compile(checkpointer=checkpointer)


async def run_turn(
    compiled: Any,
    *,
    chat_id: int,
    user_message: str,
    conversation_summary: str = "",
) -> dict[str, Any]:
    """Run one turn for ``chat_id`` on its deterministic thread identity."""
    thread_id = thread_id_for_chat(chat_id)
    result = await compiled.ainvoke(
        {
            "current_user_message": user_message,
            "conversation_summary": conversation_summary,
            "search_queries": [],
            "retrieval_hits": [],
            "evidence_pack": [],
            "draft_response": "",
            "grounding_result": None,
            "retry_state": {},
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    logger.info("turn graph completed")
    return dict(result)


__all__ = [
    "COMMAND_PREFIX",
    "build_turn_graph",
    "ingest_turn",
    "is_application_command",
    "retrieval_stub",
    "answer_stub",
    "route_after_safety",
    "run_turn",
    "safety_gate",
]
