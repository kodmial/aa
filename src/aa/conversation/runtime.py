"""Production LangGraph conversation boundary for Telegram turns.

This module is the only ordinary conversational runtime after the #118
cutover. LangGraph owns conversational state; OpenCode sessions are ephemeral
transport for hidden planner/summarizer/answer/verifier calls.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage

from aa.conversation.graph import build_turn_graph, turn_input
from aa.conversation.memory import (
    SqliteCheckpointerFactory,
    default_memory_config,
    thread_id_for_chat,
)
from aa.conversation.model_adapter import (
    ANSWER_AGENT_V2,
    PLANNER_AGENT_V2,
    SUMMARIZER_AGENT_V2,
    VERIFIER_AGENT_V2,
    OpenCodeChatModel,
)
from aa.opencode.client import OpenCodeClient

logger = logging.getLogger("aa.conversation.runtime")


class ProductConversationRuntime:
    """Lifecycle wrapper around the production LangGraph turn graph."""

    def __init__(
        self,
        *,
        client: OpenCodeClient,
        primary_model: str,
        fallback_model: str,
        index_loader: Callable[[], Any],
        safety_check: Callable[[str], Any] | None = None,
        request_timeout: float = 120.0,
    ) -> None:
        self._client = client
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._index_loader = index_loader
        self._safety_check = safety_check
        self._request_timeout = request_timeout
        self._factory = SqliteCheckpointerFactory(default_memory_config())
        self._checkpointer_cm: Any = None
        self._checkpointer: Any = None
        self._graph: Any = None

    @property
    def running(self) -> bool:
        return self._graph is not None

    async def start(self) -> None:
        if self._graph is not None:
            return
        index = self._index_loader()
        planner = OpenCodeChatModel(
            self._client,
            agent=PLANNER_AGENT_V2,
            primary_model=self._primary_model,
            fallback_model=self._fallback_model,
            request_timeout=self._request_timeout,
        )
        summarizer = planner.with_agent(SUMMARIZER_AGENT_V2)
        answer = planner.with_agent(ANSWER_AGENT_V2)
        verifier = planner.with_agent(VERIFIER_AGENT_V2)

        self._checkpointer_cm = self._factory.checkpointer()
        self._checkpointer = await self._checkpointer_cm.__aenter__()
        self._graph = build_turn_graph(
            planner_model=planner,
            summary_model=summarizer,
            checkpointer=self._checkpointer,
            safety_check=self._safety_check,
            retrieval_index=index,
            answer_model=answer,
            verifier_model=verifier,
        )
        logger.info("v2 conversation runtime started")

    async def stop(self) -> None:
        graph = self._graph
        _ = graph
        self._graph = None
        self._checkpointer = None
        cm = self._checkpointer_cm
        self._checkpointer_cm = None
        if cm is not None:
            await cm.__aexit__(None, None, None)
        self._factory.cleanup()
        logger.info("v2 conversation runtime stopped")

    async def respond(self, chat_id: int, text: str) -> str:
        graph = self._graph
        if graph is None:
            raise RuntimeError("v2 conversation runtime is not started")
        thread_id = thread_id_for_chat(chat_id)
        config = {"configurable": {"thread_id": thread_id}}
        result = await graph.ainvoke(turn_input(text), config=config)
        reply = str(result.get("final_response", "")).strip()
        if not reply:
            raise ValueError("v2 conversation graph produced no final response")
        # Persist the visible assistant reply in the same LangGraph thread so
        # later turns can resolve references to what AA actually said.
        await graph.aupdate_state(config, {"messages": [AIMessage(content=reply)]})
        logger.info("v2 turn completed", extra={"v2_thread": thread_id[:12]})
        return reply

    async def reset(self, chat_id: int) -> None:
        if self._checkpointer is None:
            return
        thread_id = thread_id_for_chat(chat_id)
        delete = getattr(self._checkpointer, "adelete_thread", None)
        if callable(delete):
            await delete(thread_id)
        else:  # pragma: no cover - backend compatibility fail-closed guard
            raise RuntimeError("configured LangGraph checkpointer cannot reset a thread")
        logger.info("v2 conversation thread reset", extra={"v2_thread": thread_id[:12]})


__all__ = ["ProductConversationRuntime"]
