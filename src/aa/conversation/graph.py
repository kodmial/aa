"""LangGraph turn foundation for the v2 conversation architecture (issue #113).

Turn boundary: Telegram turn -> typed state -> deterministic
application-command/safety handling -> mandatory hidden planner -> managed
conversation memory -> downstream retrieval/answer interfaces (stubs owned
by later tasks #116/#117).

The legacy ``aa.conversation.orchestrator`` path is untouched and remains
the production path until the later cutover task. No legacy lexical
semantic-routing behavior (substantive/trivial classifiers, keyword, slang
or theme tables, legacy aspect-taxonomy planners) is imported or
reproduced here.

Privacy: logs carry only routes, counts and token lengths, never prompts,
user text, summaries, model outputs or raw chat identifiers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aa.conversation.graph_state import (
    BLOCKED_ROUTE,
    COMMAND_ROUTE,
    EMERGENCY_ROUTE,
    NORMAL_ROUTE,
    TurnState,
)
from aa.conversation.memory import (
    MemoryConfig,
    default_memory_config,
    maybe_compact_state,
)
from aa.conversation.planner_node import planner_node
from aa.conversation.planner_schema import QueryPlanValidationError
from aa.safety.router import SafetyDecision, SafetyResult, SafetyRouter

logger = logging.getLogger("aa.conversation.graph")

SafetyCheck = Callable[[str], SafetyResult]


def is_application_command(text: str) -> bool:
    """Deterministic application-command detection (``/start``, ``/new``...)."""
    return text.strip().startswith("/")


def _default_safety_check(text: str) -> SafetyResult:
    return SafetyRouter().check(text)


async def gate_node(state: TurnState, *, safety_check: SafetyCheck | None = None) -> dict[str, Any]:
    """Deterministic command/safety gate; planner runs only for normal turns."""
    text = str(state.get("current_user_message", ""))
    check = safety_check or _default_safety_check
    if not text.strip():
        logger.info("v2 turn gated", extra={"route": BLOCKED_ROUTE})
        return {"route": BLOCKED_ROUTE}
    if is_application_command(text):
        logger.info("v2 turn gated", extra={"route": COMMAND_ROUTE})
        return {"route": COMMAND_ROUTE}
    result = check(text)
    if result.decision is SafetyDecision.EMERGENCY:
        logger.info("v2 turn gated", extra={"route": EMERGENCY_ROUTE})
        return {"route": EMERGENCY_ROUTE}
    if result.decision is SafetyDecision.BLOCK:
        logger.info("v2 turn gated", extra={"route": BLOCKED_ROUTE})
        return {"route": BLOCKED_ROUTE}
    return {"route": NORMAL_ROUTE}


def _route_after_gate(state: TurnState) -> str:
    if state.get("route", NORMAL_ROUTE) == NORMAL_ROUTE:
        return NORMAL_ROUTE
    return END


def make_memory_node(
    *,
    summary_model: Runnable[list[BaseMessage], BaseMessage],
    memory_config: MemoryConfig,
) -> Any:
    """Build the managed-memory node (token compaction via framework)."""

    async def ensure_memory(state: TurnState) -> dict[str, Any]:
        messages = [item for item in state.get("messages", []) if isinstance(item, BaseMessage)]
        previous = str(state.get("conversation_summary", ""))
        merged, retained = await maybe_compact_state(
            messages=messages,
            previous_summary=previous,
            model=summary_model,
            config=memory_config,
        )
        update: dict[str, Any] = {}
        if merged != previous:
            update["conversation_summary"] = merged
        retained_ids = {id(item) for item in retained}
        dropped = [item for item in messages if id(item) not in retained_ids]
        removals: list[RemoveMessage] = []
        for item in dropped:
            if item.id:
                removals.append(RemoveMessage(id=str(item.id)))
        if removals:
            update["messages"] = removals
        logger.info(
            "v2 memory ensured",
            extra={"dropped_messages": len(removals), "compacted": merged != previous},
        )
        return update

    return ensure_memory


def make_planner_node(*, planner_model: Runnable[list[BaseMessage], BaseMessage]) -> Any:
    """Build the mandatory hidden planner node bound to one model."""

    async def run_hidden_planner(state: TurnState) -> dict[str, Any]:
        try:
            return await planner_node(state, model=planner_model)
        except QueryPlanValidationError as exc:
            logger.warning("v2 planner failed closed", extra={"category": "planner-invalid"})
            return {
                "search_queries": [],
                "planner_invoked": True,
                "retry_state": {"planner_error": str(exc)[:120]},
            }

    return run_hidden_planner


async def retrieval_stub_node(state: TurnState) -> dict[str, Any]:
    """Interface placeholder for the #116 retrieval pipeline (no-op).

    Keeps graph semantics stable while retrieval, reranking and Evidence
    Pack selection land in the next task. Emits counts only.
    """
    _ = state
    logger.info("v2 retrieval interface reached")
    return {"retrieval_hits": [], "evidence_pack": []}


def build_turn_graph(
    *,
    planner_model: Runnable[list[BaseMessage], BaseMessage],
    summary_model: Runnable[list[BaseMessage], BaseMessage] | None = None,
    memory_config: MemoryConfig | None = None,
    checkpointer: Any = None,
    safety_check: SafetyCheck | None = None,
) -> CompiledStateGraph[TurnState, None, TurnState, TurnState]:
    """Compile the v2 turn graph with managed checkpointing.

    ``checkpointer`` is the framework checkpointer (or ``None`` for
    stateless unit tests); production passes the factory-backed
    ``SqliteSaver`` so one Telegram chat maps to one persisted thread.
    """
    resolved_config = memory_config or default_memory_config()
    resolved_summary = summary_model if summary_model is not None else planner_model

    async def _gate(state: TurnState) -> dict[str, Any]:
        return await gate_node(state, safety_check=safety_check)

    builder: StateGraph[TurnState] = StateGraph(TurnState)
    builder.add_node("gate", _gate)
    builder.add_node(
        "ensure_memory",
        make_memory_node(summary_model=resolved_summary, memory_config=resolved_config),
    )
    builder.add_node("planner", make_planner_node(planner_model=planner_model))
    builder.add_node("retrieval_stub", retrieval_stub_node)
    builder.add_edge(START, "gate")
    builder.add_conditional_edges(
        "gate", _route_after_gate, {NORMAL_ROUTE: "ensure_memory", END: END}
    )
    builder.add_edge("ensure_memory", "planner")
    builder.add_edge("planner", "retrieval_stub")
    builder.add_edge("retrieval_stub", END)
    return builder.compile(checkpointer=checkpointer)


def turn_input(user_message: str, *, summary: str | None = None) -> TurnState:
    """Build one turn input payload (raw state; prompts render later)."""
    payload = TurnState(
        messages=[HumanMessage(content=user_message)],
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
    if summary is not None:
        payload["conversation_summary"] = summary
    return payload


__all__ = [
    "SafetyCheck",
    "build_turn_graph",
    "gate_node",
    "is_application_command",
    "make_memory_node",
    "make_planner_node",
    "retrieval_stub_node",
    "turn_input",
]
