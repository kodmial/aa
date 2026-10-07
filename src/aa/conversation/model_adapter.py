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
import time
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
# Verifier transport selector (kodmial/aa#202): the logical audit identity
# stays ``aa-verifier-v2`` while the OpenCode transport agent selector is
# omitted (server default applies). The verifier system prompt still
# travels through the native ``system`` field and the requested model stays
# pinned to Muse Spark. An empty transport agent means no ``agent`` key is
# sent on the wire.
VERIFIER_TRANSPORT_AGENT_V2 = ""
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

# Persistent-403 circuit breaker (Gate C run 37504648482: p50 35s/p95 50s/max
# 59s with 14 generic clarifications, all heartbeat checks failing on 2%
# scheduling jitter, verifier never served; Gate C run 37519360307: p50 17s/
# p95 36s/max 45s with 14 clarifications and verifier never served). When the
# pinned primary is rejected, every subsequent provider call in the same
# process would otherwise burn another 5s sleep plus slow primary attempts
# before reaching the working fallback. Remember the rejection briefly and
# fail over directly to the configured fallback on later calls. The first
# failure still performs the required >=5s retry; the circuit re-closes after
# the TTL so a recovered primary is retried.
#
# The circuit is process-wide per primary model (not per agent instance):
# planner/answer/verifier/summarizer share one OpenCode client in the live
# graph, and a per-instance circuit would make every agent burn its own
# primary retry on every turn (verifier never reaching fallback fast enough
# to serve within the live SLO, answers collapsing to generic clarification).
PRIMARY_ACCESS_CIRCUIT_TTL_S = 300.0

_PRIMARY_CIRCUIT: dict[str, float] = {}
_CIRCUIT_LOCK = threading.Lock()


def _circuit_key(primary_model: str) -> str:
    """Return the process-wide circuit key for one pinned primary model."""
    return (primary_model or "").strip()


def _global_circuit_open(key: str) -> bool:
    """Whether the shared circuit for ``key`` is currently open."""
    if not key:
        return False
    with _CIRCUIT_LOCK:
        rejected_at = _PRIMARY_CIRCUIT.get(key)
    if rejected_at is None:
        return False
    try:
        return (time.monotonic() - float(rejected_at)) < PRIMARY_ACCESS_CIRCUIT_TTL_S
    except (TypeError, ValueError):
        return False


def _global_circuit_record(key: str) -> float:
    """Open the shared circuit for ``key`` and return the timestamp."""
    now = time.monotonic()
    if key:
        with _CIRCUIT_LOCK:
            _PRIMARY_CIRCUIT[key] = now
    return now


def _global_circuit_clear(key: str) -> None:
    """Close the shared circuit for ``key`` (primary recovered)."""
    if not key:
        return
    with _CIRCUIT_LOCK:
        _PRIMARY_CIRCUIT.pop(key, None)


def clear_primary_circuit() -> None:
    """Close all shared primary circuits (tests only)."""
    with _CIRCUIT_LOCK:
        _PRIMARY_CIRCUIT.clear()


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
    deletes the session in ``finally``. ``agent`` is the logical audit
    identity (planner/summarizer/answer/verifier); ``transport_agent``
    optionally overrides the OpenCode transport agent selector sent on the
    wire (``None`` means the wire uses ``agent``; an empty string omits the
    selector and lets the server default apply). ``primary_model``/
    ``fallback_model`` reuse the configured AA runtime model policy.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    agent: str
    primary_model: str
    fallback_model: str = ""
    request_timeout: float = 120.0
    transport_agent: str | None = None

    _client: OpenCodeClient = PrivateAttr()
    _primary_access_rejected_at: float | None = PrivateAttr(default=None)

    def __init__(
        self,
        client: OpenCodeClient,
        *,
        agent: str,
        primary_model: str,
        fallback_model: str = "",
        request_timeout: float = 120.0,
        transport_agent: str | None = None,
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            agent=agent,
            primary_model=primary_model,
            fallback_model=fallback_model,
            request_timeout=request_timeout,
            transport_agent=transport_agent,
        )
        self._client = client

    @property
    def wire_agent(self) -> str:
        """The transport agent selector actually sent on the wire."""
        if self.transport_agent is None:
            return self.agent
        return self.transport_agent

    @property
    def _llm_type(self) -> str:
        return "opencode-ephemeral"

    @property
    def opencode_client(self) -> OpenCodeClient:
        """The underlying transport client (framework boundary)."""
        return self._client

    def with_agent(self, agent: str) -> OpenCodeChatModel:
        """Return a copy of this model bound to another named agent.

        The process-wide primary circuit is shared: a new agent bound after
        the primary was rejected fast-fallbacks immediately instead of
        burning its own primary retry per turn. Derived agents always use
        the wire-equals-logical policy; the decoupled verifier is built
        via :func:`build_verifier_model`, never via ``with_agent``.
        """
        nxt = OpenCodeChatModel(
            self._client,
            agent=agent,
            primary_model=self.primary_model,
            fallback_model=self.fallback_model,
            request_timeout=self.request_timeout,
        )
        try:
            key = _circuit_key(self.primary_model)
            with _CIRCUIT_LOCK:
                shared_at = _PRIMARY_CIRCUIT.get(key)
            if shared_at is not None:
                nxt._primary_access_rejected_at = float(shared_at)
        except Exception:
            pass
        return nxt

    def _primary_circuit_open(self) -> bool:
        """Whether the primary was recently rejected (fast fallback allowed)."""
        if _global_circuit_open(_circuit_key(self.primary_model)):
            return True
        rejected_at = self._primary_access_rejected_at
        if rejected_at is None:
            return False
        try:
            return (time.monotonic() - float(rejected_at)) < PRIMARY_ACCESS_CIRCUIT_TTL_S
        except (TypeError, ValueError):
            return False

    def _record_primary_rejection(self) -> None:
        """Remember a persistent primary 403 to skip redundant retries."""
        try:
            now = _global_circuit_record(_circuit_key(self.primary_model))
        except Exception:
            now = 0.0
        try:
            self._primary_access_rejected_at = now
        except Exception:
            self._primary_access_rejected_at = 0.0

    def _clear_primary_rejection(self) -> None:
        """Clear the circuit when the primary serves again."""
        try:
            _global_circuit_clear(_circuit_key(self.primary_model))
        except Exception:
            pass
        self._primary_access_rejected_at = None

    def _fast_fallback_available(self) -> bool:
        """Whether a direct fallback is allowed under an open circuit."""
        return (
            bool(self.fallback_model.strip())
            and self.fallback_model != self.primary_model
            and self._primary_circuit_open()
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
        # Decoupled verifier (kodmial/aa#202): the wire carries the
        # transport selector while the audit keeps the logical identity.
        wire_agent = self.wire_agent if agent == self.agent else agent
        audit_agent = agent if wire_agent != agent else ""
        try:
            return await client.send_message(
                session.id,
                prompt,
                timeout=self.request_timeout,
                agent=wire_agent,
                model=model,
                system=system,
                format=format,
                audit_agent=audit_agent,
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
        wire_agent = self.wire_agent if agent == self.agent else agent
        audit_agent = agent if wire_agent != agent else ""
        try:
            return await client.send_structured_message(
                session.id,
                prompt,
                timeout=self.request_timeout,
                agent=wire_agent,
                model=model,
                system=system,
                schema=schema,
                retry_count=retry_count,
                audit_agent=audit_agent,
            )
        finally:
            try:
                await client.delete_session(session.id)
            except OpenCodeError:
                logger.warning("ephemeral session cleanup failed")

    async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
        if not self.primary_model.strip():
            raise ValueError("primary model must be pinned")
        # Fast path for a persistently rejected primary: the first failure
        # already performed the required >=5s retry. Later calls in the same
        # process go directly to the configured fallback instead of burning
        # another 5s sleep plus two slow primary attempts per call.
        if self._fast_fallback_available():
            logger.info("opencode primary circuit open, fast fallback used")
            try:
                reply = await self._invoke_ephemeral(
                    prompt, model=self.fallback_model, agent=self.agent, system=system
                )
            except OpenCodeRateLimitError:
                raise
            except (
                OpenCodeTransientError,
                OpenCodeTimeoutError,
                OpenCodeProviderAccessError,
            ):
                # Fallback failure: close the circuit so the next call
                # re-probes the primary instead of sticking to a bad fallback.
                self._clear_primary_rejection()
                raise
            return reply
        last_transient: BaseException | None = None
        access_attempt = 0
        transient_attempt = 0
        while True:
            try:
                reply = await self._invoke_ephemeral(
                    prompt, model=self.primary_model, agent=self.agent, system=system
                )
                self._clear_primary_rejection()
                return reply
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
            primary_access_rejected = isinstance(last_transient, OpenCodeProviderAccessError)
            if primary_access_rejected:
                self._record_primary_rejection()
            logger.info("opencode model fallback used")
            try:
                return await self._invoke_ephemeral(
                    prompt, model=self.fallback_model, agent=self.agent, system=system
                )
            except OpenCodeRateLimitError:
                raise
            except (
                OpenCodeProviderAccessError,
                OpenCodeTransientError,
                OpenCodeTimeoutError,
            ):
                if primary_access_rejected:
                    self._clear_primary_rejection()
                raise
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
        if self._fast_fallback_available():
            logger.info("opencode primary circuit open, fast fallback used")
            try:
                return await self._invoke_ephemeral_structured(
                    prompt,
                    system=system,
                    schema=schema,
                    model=self.fallback_model,
                    agent=self.agent,
                    retry_count=retry_count,
                )
            except OpenCodeRateLimitError:
                raise
            except (
                OpenCodeTransientError,
                OpenCodeTimeoutError,
                OpenCodeProviderAccessError,
            ):
                self._clear_primary_rejection()
                raise
        last_transient: BaseException | None = None
        access_attempt = 0
        transient_attempt = 0
        while True:
            try:
                result = await self._invoke_ephemeral_structured(
                    prompt,
                    system=system,
                    schema=schema,
                    model=self.primary_model,
                    agent=self.agent,
                    retry_count=retry_count,
                )
                self._clear_primary_rejection()
                return result
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
            primary_access_rejected = isinstance(last_transient, OpenCodeProviderAccessError)
            if primary_access_rejected:
                self._record_primary_rejection()
            logger.info("opencode model fallback used")
            try:
                return await self._invoke_ephemeral_structured(
                    prompt,
                    system=system,
                    schema=schema,
                    model=self.fallback_model,
                    agent=self.agent,
                    retry_count=retry_count,
                )
            except OpenCodeRateLimitError:
                raise
            except (
                OpenCodeProviderAccessError,
                OpenCodeTransientError,
                OpenCodeTimeoutError,
            ):
                if primary_access_rejected:
                    self._clear_primary_rejection()
                raise
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


def build_verifier_model(
    client: OpenCodeClient,
    *,
    primary_model: str,
    request_timeout: float = 120.0,
) -> OpenCodeChatModel:
    """Build the Muse-only grounding verifier (kodmial/aa#202).

    The logical audit identity stays ``aa-verifier-v2`` while the
    transport agent selector is omitted so a custom-agent rejection can
    never strand the Muse verifier. The fallback stays empty: Space Bunny
    must never serve the verifier. The verifier system prompt still
    travels through the native ``system`` field and the requested model
    stays pinned to Muse Spark.
    """
    from aa.config import DEFAULT_PRIMARY_MODEL

    pinned = (primary_model or "").strip() or DEFAULT_PRIMARY_MODEL
    return OpenCodeChatModel(
        client,
        agent=VERIFIER_AGENT_V2,
        primary_model=pinned,
        fallback_model="",
        request_timeout=request_timeout,
        transport_agent=VERIFIER_TRANSPORT_AGENT_V2,
    )


__all__ = [
    "ANSWER_AGENT_V2",
    "PLANNER_AGENT_V2",
    "PRIMARY_ACCESS_CIRCUIT_TTL_S",
    "RUNTIME_AGENT_V2",
    "STRUCTURED_RETRY_COUNT",
    "SUMMARIZER_AGENT_V2",
    "VERIFIER_AGENT_V2",
    "VERIFIER_TRANSPORT_AGENT_V2",
    "OpenCodeChatModel",
    "build_verifier_model",
    "clear_primary_circuit",
    "render_messages_text",
    "split_system_and_user",
]
