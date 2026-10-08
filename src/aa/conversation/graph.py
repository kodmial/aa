"""LangGraph turn foundation for the v2 conversation architecture.

Turn boundary: Telegram turn -> typed state -> deterministic
application-command/safety handling -> mandatory hidden planner -> managed
conversation memory -> retrieval/Evidence Pack -> natural AA answer with
claim-level verification and bounded repair.

This graph is the production conversational path (issue #118 cutover).
The retired ``aa.conversation.orchestrator`` module is never imported
here and is not consulted by any node. No legacy lexical
semantic-routing behavior (substantive/trivial classifiers, keyword, slang
or theme tables, legacy aspect-taxonomy planners) is imported or
reproduced here.

Privacy: logs carry only routes, counts and token lengths, never prompts,
user text, summaries, model outputs or raw chat identifiers.
"""

from __future__ import annotations

import asyncio
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
        # Fast path for ordinary turns: token-driven compaction needs no
        # model call when history is far below the trigger. Skipping the
        # summarizer here removes one sequential model round-trip per
        # ordinary turn (part of the 30-60s live pathology) without
        # changing semantics: under-trigger history compacts to itself.
        from aa.conversation.memory import needs_compaction

        try:
            if not needs_compaction(messages, trigger_tokens=memory_config.trigger_tokens):
                logger.info(
                    "v2 memory ensured",
                    extra={"compacted": False, "fast_path": True},
                )
                return {}
        except Exception:
            pass
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
        import time as _time

        from aa.conversation.answer_adequacy import planner_reason_for as _reason_for
        from aa.conversation.planner_node import build_generic_fallback_queries

        started = _time.perf_counter()
        try:
            result = await planner_node(state, model=planner_model)
            elapsed_ms = (_time.perf_counter() - started) * 1000.0
            queries = result.get("search_queries", [])
            count = len(queries) if isinstance(queries, list) else 0
            mode = str(result.get("planner_mode", "retrieval") or "retrieval")
            logger.info(
                "v2 planner done",
                extra={
                    "queries": count,
                    "latency_ms": round(elapsed_ms, 1),
                },
            )
            update = dict(result)
            retry = dict(update.get("retry_state", {}) or {})
            retry["planner_latency_ms"] = round(elapsed_ms, 1)
            retry["planner_query_count"] = count
            retry["planner_mode"] = mode
            retry["resolved_intent"] = str(result.get("resolved_intent", "") or "")
            if mode == "conversational":
                outcome = "empty"
                retry["planner_outcome"] = outcome
                retry["planner_reason"] = "legitimate-glue"
            else:
                outcome = "ok" if count else "empty"
                retry["planner_outcome"] = outcome
                retry["planner_reason"] = _reason_for(count, outcome)
            update["retry_state"] = retry
            return update
        except QueryPlanValidationError as exc:
            elapsed_ms = (_time.perf_counter() - started) * 1000.0
            logger.warning("v2 planner failed closed", extra={"category": "planner-invalid"})
            return {
                "search_queries": [],
                "planner_invoked": True,
                "planner_mode": "retrieval",
                "resolved_intent": str(state.get("current_user_message", "")),
                "retry_state": {
                    "planner_error": str(exc)[:120],
                    "planner_latency_ms": round(elapsed_ms, 1),
                    "planner_query_count": 0,
                    "planner_outcome": "invalid",
                    "planner_reason": "invalid",
                    "planner_mode": "retrieval",
                },
            }
        except Exception as exc:
            from aa.opencode.errors import OpenCodeRateLimitError as _PlannerRateLimit

            if isinstance(exc, _PlannerRateLimit):
                raise
            if isinstance(exc, asyncio.CancelledError):
                raise
            elapsed_ms = (_time.perf_counter() - started) * 1000.0
            try:
                from aa.opencode.errors import OpenCodeTimeoutError as _PlannerTimeout

                outcome = "timeout" if isinstance(exc, _PlannerTimeout | TimeoutError) else "failed"
            except Exception:
                outcome = "failed"
            logger.warning("v2 planner failed closed", extra={"category": outcome})
            from aa.conversation.answer_adequacy import planner_reason_for as _error_reason_for

            # Generic semantic retrieval fallback: raw turn plus bounded
            # recent conversation; never reinterpreted as glue.
            try:
                _summary = str(state.get("conversation_summary", "") or "")
                _recent_raw: list[str] = []
                for _msg in list(state.get("messages", []) or [])[-4:]:
                    _content = getattr(_msg, "content", "")
                    if isinstance(_content, str) and _content.strip():
                        _recent_raw.append(_content.strip()[:200])
                _fallback = build_generic_fallback_queries(
                    str(state.get("current_user_message", "")),
                    summary=_summary,
                    recent=_recent_raw,
                )
            except Exception:
                _fallback = []
            return {
                "search_queries": list(_fallback),
                "planner_invoked": True,
                "planner_mode": "retrieval",
                "resolved_intent": str(state.get("current_user_message", "")),
                "retry_state": {
                    "planner_error": type(exc).__name__[:120],
                    "planner_latency_ms": round(elapsed_ms, 1),
                    "planner_query_count": len(_fallback),
                    "planner_outcome": outcome,
                    "planner_reason": _error_reason_for(len(_fallback), outcome),
                    "planner_mode": "retrieval",
                },
            }

    return run_hidden_planner


async def retrieval_stub_node(state: TurnState) -> dict[str, Any]:
    """Interface placeholder for the #116 retrieval pipeline (no-op).

    Keeps graph semantics stable when no RAM-resident index is bound.
    Pass ``retrieval_index`` to :func:`build_turn_graph` to run the
    real RRF-only target pipeline (multi-query hybrid plus Evidence
    Pack selection). Emits counts only.
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
    retrieval_config: Any = None,
    answer_model: Runnable[list[BaseMessage], BaseMessage] | Any | None = None,
    verifier_model: Any | None = None,
) -> CompiledStateGraph[TurnState, None, TurnState, TurnState]:
    """Compile the v2 turn graph with managed checkpointing.

    ``checkpointer`` is the framework checkpointer (or ``None`` for
    stateless unit tests); production passes the factory-backed
    ``SqliteSaver`` so one Telegram chat maps to one persisted thread.

    ``retrieval_index`` binds the #115 RAM-resident canonical index; when
    given, the ``retrieval`` node runs the #116 RRF-only target
    pipeline (multi-query hybrid plus small-to-big Evidence Pack
    selection: ``QueryPlan -> BM25+E5 -> RRF -> dedup/diversity ->
    small-to-big``) instead of the ``retrieval_stub`` no-op.
    ``retrieval_config`` is forwarded when provided. The retired
    orchestrator module is never consulted by this graph.

    ``answer_model``/``verifier_model`` bind the natural answer pipeline:
    when both are given, an ``answer`` node runs AA generation,
    claim-level verification, bounded targeted repair, and the #83
    envelope after retrieval. When omitted, the graph keeps its
    planner-to-retrieval semantics for unit tests.
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

        builder.add_node(
            "retrieval",
            make_retrieval_node(
                index=retrieval_index,
                config=retrieval_config,
            ),
        )
        retrieval_node_name = "retrieval"
    builder.add_edge(START, "gate")
    builder.add_conditional_edges(
        "gate", _route_after_gate, {NORMAL_ROUTE: "ensure_memory", END: END}
    )
    builder.add_edge("ensure_memory", "planner")
    builder.add_edge("planner", retrieval_node_name)
    if answer_model is not None and verifier_model is not None:
        from aa.conversation.turn_pipeline import answer_pipeline_node

        async def _answer(state: TurnState) -> dict[str, Any]:
            return await answer_pipeline_node(
                state,
                answer_model=answer_model,
                verifier_model=verifier_model,
                planner_model=planner_model,
                retrieval_index=retrieval_index,
                retrieval_config=retrieval_config,
            )

        builder.add_node("answer", _answer)
        builder.add_edge(retrieval_node_name, "answer")
        builder.add_edge("answer", END)
    else:
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
        final_response="",
        grounding_result={},
        retry_state={},
        planner_invoked=False,
        route=NORMAL_ROUTE,
        recent_quote_ranges=[],
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
