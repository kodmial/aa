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
from aa.opencode.errors import (
    OpenCodeError,
    OpenCodeProviderAccessError,
    OpenCodeRateLimitError,
    OpenCodeTimeoutError,
    OpenCodeTransientError,
)

logger = logging.getLogger("aa.conversation.model_adapter")

PLANNER_AGENT_V2 = "aa-planner-v2"
SUMMARIZER_AGENT_V2 = "aa-summarizer-v2"
ANSWER_AGENT_V2 = "aa-v2"
VERIFIER_AGENT_V2 = "aa-verifier-v2"
RUNTIME_AGENT_V2 = "aa-runtime-v2"
STRUCTURED_RETRY_COUNT = 2
MODEL_TRANSIENT_RETRY_DELAYS = (1.0, 4.0)
# Live SLO guard (Gate C/E): a persistent provider-access (403) rejection
# for the pinned primary must fail over fast. Retrying the same rejected
# primary three times burns 35s of pure sleep per provider call before the
# configured fallback is tried (the 50-86s ordinary-turn pathology in Gate C
# run 37498373507: 3 delivery timeouts, p50 50s/p95 86s over the 30s budget).
# A single bounded 5s retry preserves the required >=5s base and the exact
# primary-then-fallback policy while keeping ordinary turns within budget.
MODEL_ACCESS_RETRY_DELAYS = (5.0,)


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
    """Render non-system LangChain messages to one plain-text OpenCode prompt.

    System messages are never flattened into user text; they travel through
    OpenCode's native ``system`` field via :func:`split_system_and_user`.
    """
    lines: list[str] = []
    for message in messages:
        kind = message.type
        if kind == "system":
            continue
        role = {"human": "user", "ai": "assistant"}.get(kind, kind)
        lines.append(f"{role}: {_message_text(message)}")
    return "\n\n".join(lines)


def split_system_and_user(messages: list[BaseMessage]) -> tuple[str, str]:
    """Split native system text from the plain-text user prompt."""
    system_parts = [_message_text(item) for item in messages if item.type == "system"]
    system_text = "\n\n".join(part for part in system_parts if part.strip())
    prompt = render_messages_text(messages)
    return system_text, prompt


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
    thread.start()
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

    async def _invoke_ephemeral(
        self,
        prompt: str,
        *,
        model: str,
        agent: str,
        system: str = "",
        format: dict[str, object] | None = None,
    ) -> str:
        client = self._client
        session = await client.create_session(title="")
        try:
            return await client.send_message(
                session.id,
                prompt,
                timeout=self.request_timeout,
                agent=agent,
                model=model,
                system=system,
                format=format,
            )
        finally:
            try:
                await client.delete_session(session.id)
            except OpenCodeError:
                logger.warning("ephemeral session cleanup failed")

    async def _invoke_ephemeral_structured(
        self,
        prompt: str,
        *,
        system: str,
        schema: dict[str, object],
        model: str,
        agent: str,
        retry_count: int = STRUCTURED_RETRY_COUNT,
    ) -> dict[str, object]:
        client = self._client
        session = await client.create_session(title="")
        try:
            return await client.send_structured_message(
                session.id,
                prompt,
                timeout=self.request_timeout,
                agent=agent,
                model=model,
                system=system,
                schema=schema,
                retry_count=retry_count,
            )
        finally:
            try:
                await client.delete_session(session.id)
            except OpenCodeError:
                logger.warning("ephemeral session cleanup failed")

    async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
        if not self.primary_model.strip():
            raise ValueError("primary model must be pinned")
        last_transient: BaseException | None = None
        access_attempt = 0
        transient_attempt = 0
        while True:
            try:
                return await self._invoke_ephemeral(
                    prompt, model=self.primary_model, agent=self.agent, system=system
                )
            except OpenCodeProviderAccessError as exc:
                last_transient = exc
                if access_attempt < len(MODEL_ACCESS_RETRY_DELAYS):
                    delay = MODEL_ACCESS_RETRY_DELAYS[access_attempt]
                    access_attempt += 1
                    logger.info("opencode primary 403 retry scheduled")
                    await asyncio.sleep(delay)
                    continue
                break
            except OpenCodeRateLimitError:
                raise
            except (OpenCodeTransientError, OpenCodeTimeoutError) as exc:
                last_transient = exc
                if transient_attempt < len(MODEL_TRANSIENT_RETRY_DELAYS):
                    delay = MODEL_TRANSIENT_RETRY_DELAYS[transient_attempt]
                    transient_attempt += 1
                    logger.info("opencode primary transient retry scheduled")
                    await asyncio.sleep(delay)
                    continue
                break
        if self.fallback_model.strip() and self.fallback_model != self.primary_model:
            logger.info("opencode model fallback used")
            return await self._invoke_ephemeral(
                prompt, model=self.fallback_model, agent=self.agent, system=system
            )
        if last_transient is not None:
            raise last_transient
        raise OpenCodeProviderAccessError("opencode primary model access rejected")

    async def ainvoke_structured(
        self,
        prompt: str,
        *,
        system: str,
        schema: dict[str, object],
        retry_count: int = STRUCTURED_RETRY_COUNT,
    ) -> dict[str, object]:
        """Invoke one native ``json_schema`` request with model fallback.

        No JSON text prompting/parsing happens here; OpenCode validates
        against ``schema`` with its own bounded ``retryCount`` and returns
        the structured object.
        """
        if not self.primary_model.strip():
            raise ValueError("primary model must be pinned")
        last_transient: BaseException | None = None
        access_attempt = 0
        transient_attempt = 0
        while True:
            try:
                return await self._invoke_ephemeral_structured(
                    prompt,
                    system=system,
                    schema=schema,
                    model=self.primary_model,
                    agent=self.agent,
                    retry_count=retry_count,
                )
            except OpenCodeProviderAccessError as exc:
                last_transient = exc
                if access_attempt < len(MODEL_ACCESS_RETRY_DELAYS):
                    delay = MODEL_ACCESS_RETRY_DELAYS[access_attempt]
                    access_attempt += 1
                    logger.info("opencode primary 403 retry scheduled")
                    await asyncio.sleep(delay)
                    continue
                break
            except OpenCodeRateLimitError:
                raise
            except (OpenCodeTransientError, OpenCodeTimeoutError) as exc:
                last_transient = exc
                if transient_attempt < len(MODEL_TRANSIENT_RETRY_DELAYS):
                    delay = MODEL_TRANSIENT_RETRY_DELAYS[transient_attempt]
                    transient_attempt += 1
                    logger.info("opencode primary transient retry scheduled")
                    await asyncio.sleep(delay)
                    continue
                break
        if self.fallback_model.strip() and self.fallback_model != self.primary_model:
            logger.info("opencode model fallback used")
            return await self._invoke_ephemeral_structured(
                prompt,
                system=system,
                schema=schema,
                model=self.fallback_model,
                agent=self.agent,
                retry_count=retry_count,
            )
        if last_transient is not None:
            raise last_transient
        raise OpenCodeProviderAccessError("opencode primary model access rejected")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = (stop, run_manager, kwargs)
        system_text, prompt = split_system_and_user(messages)
        if not prompt.strip():
            raise ValueError("refusing to send an empty prompt")
        reply = _run_coro_sync(self._ainvoke_text(prompt, system=system_text))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(reply)))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = (stop, run_manager, kwargs)
        system_text, prompt = split_system_and_user(messages)
        if not prompt.strip():
            raise ValueError("refusing to send an empty prompt")
        reply = await self._ainvoke_text(prompt, system=system_text)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(reply)))])


__all__ = [
    "ANSWER_AGENT_V2",
    "PLANNER_AGENT_V2",
    "RUNTIME_AGENT_V2",
    "STRUCTURED_RETRY_COUNT",
    "SUMMARIZER_AGENT_V2",
    "VERIFIER_AGENT_V2",
    "OpenCodeChatModel",
    "render_messages_text",
    "split_system_and_user",
]
