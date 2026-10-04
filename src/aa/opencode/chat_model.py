"""Thin LangChain chat-model adapter over the local OpenCode runtime.

The adapter is transport/model-invocation glue only: it renders LangChain
chat messages to the single-text OpenCode session API and returns the
assistant text. It contains no semantic routing, retrieval logic, AA policy,
keyword dictionaries, or domain classification.

Hidden orchestration calls (planner, summarizer, future answer calls) use an
isolated ephemeral OpenCode session per invocation, so they never depend on
-- and never pollute -- the user-facing conversation history. The configured
AA runtime model/fallback policy is reused; no new product model is chosen
here. Prompts, user text, summaries, and model outputs are never logged.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from collections.abc import Coroutine
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from aa.opencode.client import OpenCodeClient
from aa.opencode.errors import (
    OpenCodeDeterministicError,
    OpenCodeError,
    OpenCodeTimeoutError,
    OpenCodeTransientError,
)

logger = logging.getLogger("aa.opencode.chat_model")

HIDDEN_SESSION_TITLE = "aa-hidden-orchestration"


class OpenCodeChatModel(BaseChatModel):
    """LangChain chat model backed by an :class:`OpenCodeClient`.

    Each invocation creates an isolated ephemeral OpenCode session, sends one
    rendered text prompt, and best-effort deletes the session. The caller may
    pass a fake :class:`OpenCodeClient` (for example
    :class:`FakeOpenCodeClient`) so the adapter is fakeable in tests.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: OpenCodeClient
    agent: str = "aa"
    primary_model: str = ""
    fallback_model: str = ""
    request_timeout: float | None = None

    def __init__(
        self,
        client: OpenCodeClient,
        *,
        agent: str = "aa",
        primary_model: str = "",
        fallback_model: str = "",
        request_timeout: float | None = None,
    ) -> None:
        if not agent.strip():
            raise ValueError("agent must not be empty")
        if not primary_model.strip():
            raise ValueError("primary_model must not be empty")
        if not fallback_model.strip():
            raise ValueError("fallback_model must not be empty")
        if primary_model == fallback_model:
            raise ValueError("primary and fallback models must differ")
        super().__init__(
            client=client,
            agent=agent,
            primary_model=primary_model,
            fallback_model=fallback_model,
            request_timeout=request_timeout,
        )  # type: ignore[call-arg]

    @property
    def _llm_type(self) -> str:
        """Short model type identifier."""
        return "opencode-chat"

    def _render_prompt(self, messages: list[BaseMessage]) -> str:
        """Render chat messages to the single-text OpenCode prompt shape."""
        # Framework primitive: stable role-prefixed buffer string.
        return get_buffer_string(messages)

    async def _complete_once(self, prompt: str, *, model: str) -> str:
        """Send one prompt on an isolated session and return the reply."""
        if not prompt.strip():
            raise OpenCodeDeterministicError("refusing to send an empty prompt")
        info = await self.client.create_session(HIDDEN_SESSION_TITLE)
        try:
            return await self.client.send_message(
                info.id,
                prompt,
                timeout=self.request_timeout,
                agent=self.agent,
                model=model,
            )
        finally:
            try:
                await self.client.delete_session(info.id)
            except OpenCodeError:
                # Best-effort cleanup only; the turn result stands.
                logger.info("hidden session cleanup skipped")

    async def _complete_with_fallback(self, prompt: str) -> tuple[str, bool]:
        """Send one prompt with the configured primary/fallback policy."""
        try:
            reply = await self._complete_once(prompt, model=self.primary_model)
            return reply, False
        except (OpenCodeTransientError, OpenCodeTimeoutError):
            logger.info("hidden call will try fallback model")
            reply = await self._complete_once(prompt, model=self.fallback_model)
            return reply, True

    async def _agenerate_async(
        self,
        messages: list[BaseMessage],
    ) -> ChatResult:
        """Shared async generation path (no content is logged)."""
        prompt = self._render_prompt(messages)
        text, _fallback_used = await self._complete_with_fallback(prompt)
        if not text.strip():
            raise OpenCodeDeterministicError("opencode returned an empty response")
        # Never log the prompt or the reply; only the fact of completion.
        logger.info("hidden model call completed")
        return ChatResult(generations=[ChatGeneration(message=_assistant_message(text))])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation (bridges to the async OpenCode client)."""
        _ = stop
        _ = run_manager
        _ = kwargs
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._agenerate_async(messages))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_coro_in_new_loop, self._agenerate_async(messages))
            return future.result()

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronous generation used by ``ainvoke``."""
        _ = stop
        _ = run_manager
        _ = kwargs
        return await self._agenerate_async(messages)


def _run_coro_in_new_loop(coro: Coroutine[Any, Any, ChatResult]) -> ChatResult:
    """Run ``coro`` in a fresh event loop (sync-bridge helper)."""
    return asyncio.run(coro)


def _assistant_message(text: str) -> BaseMessage:
    """Build the assistant result message without importing agent logic."""
    from langchain_core.messages import AIMessage

    return AIMessage(content=text)


__all__ = ["HIDDEN_SESSION_TITLE", "OpenCodeChatModel"]
