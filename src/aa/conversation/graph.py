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

from langchain_core.messages import BaseMessage, HumanMessage
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
    build_summarization_node,
    default_memory_config,
    ensure_message_ids,
    running_summary_from_state,
    summary_text_from_running,
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
    """Build the managed-memory node (LangMem graph-native compaction).

    The LangMem ``SummarizationNode`` is an actual node delegate on the
    execution path before the planner. Budgets come from ``memory_config``;
    no project-owned split/running-summary logic exists here.
    """
    summarizer = build_summarization_node(summary_model, config=memory_config)

    async def ensure_memory(state: TurnState) -> dict[str, Any]:
        messages = [item for item in state.get("messages", []) if isinstance(item, BaseMessage)]
        ensure_message_ids(messages)
        previous = str(state.get("conversation_summary", ""))
        context = dict(state.get("context", {}) or {})
        running = running_summary_from_state(summary_text=previous, context=context)
        node_input: dict[str, Any] = {"messages": messages}
        if running is not None:
            node_input["context"] = {"running_summary": running}
        else:
            node_input["context"] = {}
        result = await summarizer.ainvoke(node_input)
        new_context = dict(result.get("context", {}) or {})
        new_running = new_context.get("running_summary")
        update: dict[str, Any] = {}
        if new_running is not None:
            new_summary = summary_text_from_running(new_running)
            if new_summary and new_summary != previous:
                update["conversation_summary"] = new_summary
            if new_context != context:
                update["context"] = new_context
            new_messages = result.get("messages", messages)
            # LangMem overwrite mode returns RemoveMessage + compacted view;
            # propagate it so history stays bounded by the framework.
            if new_messages is not messages:
                update["messages"] = new_messages
        logger.info(
            "v2 memory ensured",
            extra={"compacted": bool(update)},
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

    Keeps graph semantics stable when no RAM-resident index is bound.
    Pass ``retrieval_index`` plus explicit ``performance_accepted=True``
    to :func:`build_turn_graph` to run the real target pipeline
    (multi-query hybrid plus BGE rerank plus Evidence Pack selection).
    The explicit flag records performance acceptance for the ~20-22s
    warm p50/p95 BGE path against the 5s interactive budget; without it
    cutover stays blocked and binding the index fails closed. Emits
    counts only.
    """
    _ = state
    logger.info("v2 retrieval interface reached")
    return {
        "retrieval_hits": [],
        "evidence_pack": [],
        "retrieval_latency_ms": 0.0,
        "retrieval_over_budget": False,
    }


def build_turn_graph(
    *,
    planner_model: Runnable[list[BaseMessage], BaseMessage],
    summary_model: Runnable[list[BaseMessage], BaseMessage] | None = None,
    memory_config: MemoryConfig | None = None,
    checkpointer: Any = None,
    safety_check: SafetyCheck | None = None,
    retrieval_index: Any = None,
    reranker: Any = None,
    retrieval_config: Any = None,
    performance_accepted: bool = False,
) -> CompiledStateGraph[TurnState, None, TurnState, TurnState]:
    """Compile the v2 turn graph with managed checkpointing.

    ``checkpointer`` is the framework checkpointer (or ``None`` for
    stateless unit tests); production passes the factory-backed
    ``SqliteSaver`` so one Telegram chat maps to one persisted thread.

    ``retrieval_index`` binds the #115 RAM-resident canonical index; when
    given, the ``retrieval`` node runs the #116 target pipeline
    (multi-query hybrid plus pinned local BGE rerank plus small-to-big
    Evidence Pack selection) instead of the ``retrieval_stub`` no-op.
    ``reranker`` (one long-lived worker instance) and
    ``retrieval_config`` are forwarded when provided. The legacy
    production path stays untouched until the later cutover task:
    the v2 BGE path exceeds the 5s interactive budget (~20-22s warm
    p50/p95) so production cutover stays blocked pending explicit
    performance acceptance or optimization. Binding ``retrieval_index``
    therefore requires ``performance_accepted=True`` and fails closed
    otherwise via
    ``aa.qualification.v2_retrieval.require_v2_cutover_acceptance``.
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
    if retrieval_index is None:
        builder.add_node("retrieval_stub", retrieval_stub_node)
        retrieval_node_name = "retrieval_stub"
    else:
        from aa.conversation.retrieval_node import make_retrieval_node
        from aa.qualification.v2_retrieval import require_v2_cutover_acceptance
        from aa.retrieval.evidence import INTERACTIVE_LATENCY_BUDGET_MS

        require_v2_cutover_acceptance(performance_accepted=performance_accepted)
        logger.warning(
            "v2 retrieval graph wired with explicit performance acceptance",
            extra={
                "budget_ms": INTERACTIVE_LATENCY_BUDGET_MS,
                "expected_warm_p95_ms": 22472,
            },
        )
        builder.add_node(
            "retrieval",
            make_retrieval_node(
                index=retrieval_index,
                reranker=reranker,
                config=retrieval_config,
                performance_accepted=performance_accepted,
            ),
        )
        retrieval_node_name = "retrieval"
    builder.add_edge(START, "gate")
    builder.add_conditional_edges(
        "gate", _route_after_gate, {NORMAL_ROUTE: "ensure_memory", END: END}
    )
    builder.add_edge("ensure_memory", "planner")
    builder.add_edge("planner", retrieval_node_name)
    builder.add_edge(retrieval_node_name, END)
    return builder.compile(checkpointer=checkpointer)


def turn_input(user_message: str, *, summary: str | None = None) -> TurnState:
    """Build one turn input payload (raw state; prompts render later)."""
    payload = TurnState(
        messages=[HumanMessage(content=user_message)],
        current_user_message=user_message,
        search_queries=[],
        retrieval_hits=[],
        evidence_pack=[],
        retrieval_latency_ms=0.0,
        retrieval_over_budget=False,
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
