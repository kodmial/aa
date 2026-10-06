"""LangGraph AA runtime for the Telegram production boundary (issue #118).

This module is the only ordinary conversational path after the cutover::

    Telegram -> safety/commands -> typing heartbeat -> LangGraph thread
      -> planner/retrieval/evidence/AA Agent/grounding -> Telegram delivery

Design:

- one Telegram private chat maps deterministically to one LangGraph
  ``thread_id`` via :func:`thread_id_for_chat` (one-way digest; the raw
  chat identifier is never logged or stored);
- the graph/checkpointer is the authoritative conversation-memory layer;
  accumulated OpenCode session history is never a second memory (hidden
  planner/summarizer/verifier/final-generation calls use the thin #113
  :class:`OpenCodeChatModel` adapter with fresh ephemeral OpenCode
  sessions created, invoked and deleted per call);
- one local ``opencode serve`` process per worker/runtime is kept (the
  runtime is owned by :class:`Application`, never duplicated here);
- no TUI/Electron, no second LLM/provider client;
- ``/new`` clears only that chat's LangGraph thread state via the
  checkpointer (no cross-chat leakage);
- text and voice share one conversation state: voice transcripts enter
  this exact boundary as normal user turns after local ASR.

The runtime is transport-independent so the production-boundary suite can
exercise the exact boundary Telegram uses. A lightweight in-memory
delegate mode exists for fast unit tests; production uses the compiled
LangGraph turn graph with the RAM-resident retrieval index.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("aa.conversation.graph_runtime")

TurnDelegate = Callable[[str, str], Awaitable[str]]
"""Test delegate: ``(thread_id, user_message) -> reply``."""


class GraphRuntimeError(ValueError):
    """Deterministic runtime failure (mapped to a natural reply upstream)."""

    def __init__(self, category: str, detail: str = "") -> None:
        super().__init__(f"graph runtime failed [{category}]" + (f": {detail}" if detail else ""))
        self.category = category
        self.detail = detail


class GraphTurnRuntime:
    """Owns the LangGraph turn graph and per-chat thread lifecycle.

    Production mode binds a compiled graph plus a live checkpointer;
    delegate mode (tests) runs an injected async function with an
    in-memory per-thread history so continuity/isolation semantics can be
    proven without models or a database.
    """

    def __init__(
        self,
        *,
        delegate: TurnDelegate | None = None,
        checkpointer_factory: Any | None = None,
        graph: Any | None = None,
    ) -> None:
        self._delegate = delegate
        self._factory = checkpointer_factory
        self._graph = graph
        self._checkpointer: Any | None = None
        self._checkpointer_ctx: Any | None = None
        self._histories: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()
        self._running = False

    @property
    def running(self) -> bool:
        """Whether the runtime is started."""
        return self._running

    @property
    def compiled_graph(self) -> Any | None:
        """The bound compiled graph (``None`` in delegate mode)."""
        return self._graph

    def thread_id(self, chat_id: int) -> str:
        """Map one Telegram chat deterministically to one thread id."""
        from aa.conversation.memory import thread_id_for_chat

        return thread_id_for_chat(chat_id)

    async def start(self) -> None:
        """Enter the checkpointer context and mark the runtime running."""
        if self._running:
            return
        if self._delegate is not None:
            self._running = True
            return
        if self._factory is not None and self._graph is None:
            ctx = self._factory.checkpointer()
            saver = await ctx.__aenter__()
            self._checkpointer_ctx = ctx
            self._checkpointer = saver
            self._graph = self._build_graph(saver)
        self._running = True

    async def stop(self) -> None:
        """Exit the checkpointer context (idempotent)."""
        self._running = False
        ctx, self._checkpointer_ctx = self._checkpointer_ctx, None
        self._checkpointer = None
        if self._factory is not None:
            self._graph = None
        if ctx is not None:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                logger.info("graph runtime checkpointer close failed")

    def attach_graph(self, graph: Any, *, checkpointer: Any | None = None) -> None:
        """Bind an already-compiled graph (tests/tooling)."""
        self._graph = graph
        if checkpointer is not None:
            self._checkpointer = checkpointer

    def _build_graph(self, saver: Any) -> Any:
        """Compile the production turn graph lazily (deferred imports)."""
        raise GraphRuntimeError("no-graph-factory", "production graph factory is not bound")

    async def run_turn(self, chat_id: int, text: str) -> str:
        """Run one ordinary user turn in that chat's thread and return text."""
        cleaned = text.strip() if isinstance(text, str) else ""
        if not cleaned:
            raise GraphRuntimeError("empty-turn", "refusing an empty turn")
        thread = self.thread_id(chat_id)
        if self._delegate is not None:
            async with self._lock:
                history = self._histories.setdefault(thread, [])
            try:
                reply = await self._delegate(thread, cleaned)
            except GraphRuntimeError:
                raise
            except Exception as exc:
                raise GraphRuntimeError("delegate-failed", type(exc).__name__) from exc
            if not isinstance(reply, str) or not reply.strip():
                raise GraphRuntimeError("empty-reply", "delegate returned no text")
            async with self._lock:
                history.append(cleaned)
                history.append(reply.strip())
            logger.info("graph turn completed", extra={"reply_len": len(reply.strip())})
            return reply.strip()
        graph = self._graph
        if graph is None:
            raise GraphRuntimeError("not-started", "graph runtime has no compiled graph")
        try:
            result = await self._invoke_graph(graph, thread, cleaned)
        except GraphRuntimeError:
            raise
        except Exception as exc:
            raise GraphRuntimeError("graph-failed", type(exc).__name__) from exc
        final = str(result.get("final_response", "") or result.get("draft_response", ""))
        if not final.strip():
            raise GraphRuntimeError("empty-reply", "graph returned no text")
        logger.info("graph turn completed", extra={"reply_len": len(final.strip())})
        return final.strip()

    async def _invoke_graph(self, graph: Any, thread: str, text: str) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage

        config = {"configurable": {"thread_id": thread}}
        payload: dict[str, Any] = {
            "messages": [HumanMessage(content=text)],
            "current_user_message": text,
        }
        result = await graph.ainvoke(payload, config=config)
        if not isinstance(result, dict):
            raise GraphRuntimeError("graph-failed", "graph returned no state")
        return dict(result)

    async def clear_chat(self, chat_id: int) -> None:
        """Clear only that chat's LangGraph conversation state (``/new``)."""
        thread = self.thread_id(chat_id)
        if self._delegate is not None:
            async with self._lock:
                self._histories.pop(thread, None)
            logger.info("graph thread cleared")
            return
        saver = self._checkpointer
        if saver is None:
            return
        try:
            deleter = getattr(saver, "adelete_thread", None)
            if callable(deleter):
                await deleter(thread)
            else:
                sync = getattr(saver, "delete_thread", None)
                if callable(sync):
                    await asyncio.to_thread(sync, thread)
            logger.info("graph thread cleared")
        except Exception:
            logger.info("graph thread clear failed")

    def history_for_thread(self, thread: str) -> list[str]:
        """Return the delegate-mode history for ``thread`` (tests only)."""
        return list(self._histories.get(thread, []))


def build_production_runtime(
    *,
    client: Any,
    settings: Any,
    index: Any | None,
    checkpoint_dir: Path | None = None,
) -> GraphTurnRuntime:
    """Build the production runtime bound to OpenCode models + index.

    Hidden planner/summarizer/verifier/answer calls use the thin #113
    model adapter (ephemeral OpenCode sessions per call, never polluting
    the user conversation). The caller owns ``start()``/``stop()`` around
    the checkpointer context. The compiled graph is built at ``start()``
    time inside :class:`_ProductionGraphRuntime`.
    """
    return _ProductionGraphRuntime(client=client, settings=settings, index=index)


class _ProductionGraphRuntime(GraphTurnRuntime):
    """Production specialization compiling the real turn graph on start."""

    def __init__(self, *, client: Any, settings: Any, index: Any | None) -> None:
        super().__init__()
        self._client = client
        self._settings = settings
        self._index = index

    def _build_graph(self, saver: Any) -> Any:  # pragma: no cover - production wiring
        from aa.conversation.graph import build_turn_graph
        from aa.conversation.memory import default_memory_config
        from aa.conversation.model_adapter import (
            ANSWER_AGENT_V2,
            PLANNER_AGENT_V2,
            SUMMARIZER_AGENT_V2,
            VERIFIER_AGENT_V2,
            OpenCodeChatModel,
        )
        from aa.retrieval.evidence import RetrievalConfig

        primary = str(getattr(self._settings, "opencode_model", ""))
        fallback = str(getattr(self._settings, "opencode_fallback_model", ""))
        planner = OpenCodeChatModel(
            self._client,
            agent=PLANNER_AGENT_V2,
            primary_model=primary,
            fallback_model=fallback,
        )
        summarizer = planner.with_agent(SUMMARIZER_AGENT_V2)
        answer = planner.with_agent(ANSWER_AGENT_V2)
        verifier = planner.with_agent(VERIFIER_AGENT_V2)
        return build_turn_graph(
            planner_model=planner,
            summary_model=summarizer,
            memory_config=default_memory_config(),
            checkpointer=saver,
            retrieval_index=self._index,
            retrieval_config=RetrievalConfig(),
            answer_model=answer,
            verifier_model=verifier,
        )

    async def start(self) -> None:
        if self.running:
            return
        from aa.conversation.memory import SqliteCheckpointerFactory, default_memory_config

        factory = SqliteCheckpointerFactory(default_memory_config())
        self._factory = factory
        await super().start()
        # super().start() calls _build_graph(saver) via the factory hook.
        # Keep the factory for cleanup visibility.
        self._factory = factory

    async def stop(self) -> None:
        await super().stop()
        factory = self._factory
        self._factory = None
        if factory is not None:
            cleanup = getattr(factory, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    logger.info("graph runtime checkpoint cleanup failed")


__all__ = [
    "GraphRuntimeError",
    "GraphTurnRuntime",
    "TurnDelegate",
    "build_production_runtime",
]
