"""OpenCode -> LangChain transport adapter for the v2 graph (issue #113).

This module is deliberately narrow: it contains transport and model
invocation glue only. There is no semantic routing, retrieval logic, AA
policy, keyword dictionary or domain classification here.

Hidden orchestration calls (planner, summarizer, future answer calls) run
in fresh ephemeral OpenCode sessions created, invoked and deleted inside
this adapter, so they never inherit accumulated OpenCode conversation
history and never pollute the user-facing conversation. LangGraph owns
conversational state; OpenCode sessions are pure transport.

The adapter reuses the configured AA runtime model/fallback policy: the
primary model is tried first and the configured fallback model serves
transient/timeout failures. No new product model is chosen here.

Nothing in this module logs prompts, user text, summaries or model
outputs; only operation counts and error categories are emitted.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, PrivateAttr

from aa.opencode.client import OpenCodeClient
from aa.opencode.errors import OpenCodeError, OpenCodeTimeoutError, OpenCodeTransientError

logger = logging.getLogger("aa.conversation.model_adapter")

PLANNER_AGENT_V2 = "aa-planner-v2"
SUMMARIZER_AGENT_V2 = "aa-summarizer-v2"
ANSWER_AGENT_V2 = "aa-v2"


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def render_messages_text(messages: list[BaseMessage]) -> str:
    """Render LangChain messages to one plain-text OpenCode prompt."""
    lines: list[str] = []
    for message in messages:
        kind = message.type
        role = {"human": "user", "ai": "assistant", "system": "system"}.get(kind, kind)
        lines.append(f"{role}: {_message_text(message)}")
    return "\n\n".join(lines)


def _run_coro_sync(coro: Any) -> Any:
    """Drive one coroutine from sync code without touching a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive
            error["exc"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.join()
    if error:
        raise error["exc"]
    return result.get("value")


class OpenCodeChatModel(BaseChatModel):
    """LangChain chat model backed by ephemeral OpenCode sessions.

    Each invocation creates a fresh session, sends one rendered prompt and
    deletes the session in ``finally``. ``agent`` selects the named OpenCode
    agent (planner/summarizer/answer); ``primary_model``/``fallback_model``
    reuse the configured AA runtime model policy.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    agent: str
    primary_model: str
    fallback_model: str = ""
    request_timeout: float = 120.0

    _client: OpenCodeClient = PrivateAttr()

    def __init__(
        self,
        client: OpenCodeClient,
        *,
        agent: str,
        primary_model: str,
        fallback_model: str = "",
        request_timeout: float = 120.0,
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            agent=agent,
            primary_model=primary_model,
            fallback_model=fallback_model,
            request_timeout=request_timeout,
        )
        self._client = client

    @property
    def _llm_type(self) -> str:
        return "opencode-ephemeral"

    @property
    def opencode_client(self) -> OpenCodeClient:
        """The underlying transport client (framework boundary)."""
        return self._client

    def with_agent(self, agent: str) -> OpenCodeChatModel:
        """Return a copy of this model bound to another named agent."""
        return OpenCodeChatModel(
            self._client,
            agent=agent,
            primary_model=self.primary_model,
            fallback_model=self.fallback_model,
            request_timeout=self.request_timeout,
        )

    async def _invoke_ephemeral(self, prompt: str, *, model: str, agent: str) -> str:
        client = self._client
        session = await client.create_session(title="")
        try:
            return await client.send_message(
                session.id,
                prompt,
                timeout=self.request_timeout,
                agent=agent,
                model=model,
            )
        finally:
            try:
                await client.delete_session(session.id)
            except OpenCodeError:
                logger.warning("ephemeral session cleanup failed")

    async def _ainvoke_text(self, prompt: str) -> str:
        if not self.primary_model.strip():
            raise ValueError("primary model must be pinned")
        try:
            return await self._invoke_ephemeral(prompt, model=self.primary_model, agent=self.agent)
        except (OpenCodeTransientError, OpenCodeTimeoutError):
            if self.fallback_model.strip() and self.fallback_model != self.primary_model:
                logger.info("opencode model fallback used")
                return await self._invoke_ephemeral(
                    prompt, model=self.fallback_model, agent=self.agent
                )
            raise

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = (stop, run_manager, kwargs)
        prompt = render_messages_text(messages)
        if not prompt.strip():
            raise ValueError("refusing to send an empty prompt")
        reply = _run_coro_sync(self._ainvoke_text(prompt))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(reply)))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = (stop, run_manager, kwargs)
        prompt = render_messages_text(messages)
        if not prompt.strip():
            raise ValueError("refusing to send an empty prompt")
        reply = await self._ainvoke_text(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(reply)))])


__all__ = [
    "ANSWER_AGENT_V2",
    "PLANNER_AGENT_V2",
    "SUMMARIZER_AGENT_V2",
    "OpenCodeChatModel",
    "render_messages_text",
]
