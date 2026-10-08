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
    OpenCodeDeterministicError,
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
# All hidden conversation stages use their own system prompts and logical
# audit identities. The custom OpenCode agent selector is not part of the
# product contract and has repeatedly produced provider 403s while the same
# pinned Muse model serves with the selector omitted. Omit it for the hidden
# planner family too; derived answer/summarizer models preserve this wire
# policy while retaining their distinct logical audit identities.
PLANNER_TRANSPORT_AGENT_V2 = ""
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

# Omitted-structured capability cache (Gate C live repair, run 37615447071
# on exact main e92385f): the live lane served the planner through the weak
# fallback while answer/verifier served the strong primary, with p50 34s /
# p95 59s / max 64s over the 30s budget and one generic clarification.
# The planner pays a doomed omitted-structured round-trip per invocation
# (custom 403 x2 + 5s sleep + omitted deterministic failure + fallback)
# before reaching the weak fallback, and each repair re-plan repeats it.
# A deterministic omitted-structured failure proves that provider path does
# not serve native structured output with the selector omitted; later
# planner turns then go directly to the configured fallback instead of
# burning the slow omitted attempt per call. Entries expire via TTL so a
# recovered provider is retried. Only deterministic capability failures
# mark the cache; transient/timeout fall back once without poisoning it.
# Provider 429 always propagates and never marks the cache. Turn-
# independent, never an exact-question special case.
#
# Gate C+E live repair, kodmial/aa#248 on exact main e57dea5 run
# 37757356193 (C:live-book-grounding-substantive-drinking-10 plus E
# p50 20.0s / p95 27.0s with message-structured p50 0.47s over 5 calls
# vs message-text p50 5.4s over 69 calls, all Muse Spark): the 300s TTL
# pins the whole lane to the slow path after one slow structured
# attempt. A 60s TTL still skips the doomed attempt within a slow burst
# while re-probing the fast structured path mid-lane for Gate E.
# Turn-independent, 429 never marks.
OMITTED_STRUCTURED_CAPABILITY_TTL_S = 60.0

_OMITTED_STRUCTURED_UNAVAILABLE: dict[str, float] = {}
_OMITTED_STRUCTURED_LOCK = threading.Lock()

# Persistent-rejection pinning (Gate C+E live repair, kodmial/aa#244
# recurrence 3; same run evidence as the verifier persistent-rejection
# threshold: structured attempts never time out but rejections persist,
# so waste is call count, not duration). Two consecutive
# capability-suggestive omitted-structured rejections pin the path for
# the TTL; isolated blips still re-probe. Timeout/transient, content
# validation and provider 429 never count; any structured success
# resets the streak. Turn-independent, never an exact-question special
# case. 429 always propagates and never marks.
OMITTED_STRUCTURED_PERSISTENT_THRESHOLD = 2

_OMITTED_STRUCTURED_CONSECUTIVE: dict[str, tuple[int, float]] = {}


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
    # Test isolation: the omitted-structured capability cache is also
    # process-wide per model path, so reset it here as well. Existing
    # fixtures call this helper between tests; without this, a deterministic
    # omitted failure cached in one test would skip the omitted attempt in
    # the next test with the same model path and break its call-count
    # assertions. Production never calls this helper.
    with _OMITTED_STRUCTURED_LOCK:
        _OMITTED_STRUCTURED_UNAVAILABLE.clear()
        _OMITTED_STRUCTURED_CONSECUTIVE.clear()


def _omitted_structured_key(model: Any | None) -> str:
    """Return the capability-cache key for one structured model path."""
    try:
        primary = getattr(model, "primary_model", "")
        fallback = getattr(model, "fallback_model", "")
        agent = getattr(model, "agent", "")
        if (
            (isinstance(primary, str) and primary.strip())
            or (isinstance(fallback, str) and fallback.strip())
            or (isinstance(agent, str) and agent.strip())
        ):
            return (
                f"{primary.strip() if isinstance(primary, str) else ''}"
                f"|{fallback.strip() if isinstance(fallback, str) else ''}"
                f"|{agent.strip() if isinstance(agent, str) else ''}"
            )
    except Exception:
        pass
    return "default"


def _omitted_structured_entry_fresh(recorded_at: float) -> bool:
    try:
        return (time.monotonic() - float(recorded_at)) < OMITTED_STRUCTURED_CAPABILITY_TTL_S
    except (TypeError, ValueError):
        return False


def _omitted_consecutive_fresh(recorded_at: float) -> bool:
    try:
        return (time.monotonic() - float(recorded_at)) < OMITTED_STRUCTURED_CAPABILITY_TTL_S
    except (TypeError, ValueError):
        return False


def _omitted_streak_counts(key: str) -> bool:
    """Whether a fresh persistent-rejection streak pins ``key``."""
    entry = _OMITTED_STRUCTURED_CONSECUTIVE.get(key)
    if entry is None:
        return False
    count, recorded_at = entry
    if not _omitted_consecutive_fresh(recorded_at):
        _OMITTED_STRUCTURED_CONSECUTIVE.pop(key, None)
        return False
    return int(count) >= OMITTED_STRUCTURED_PERSISTENT_THRESHOLD


def omitted_structured_unavailable(model: Any | None = None) -> bool:
    """Whether omitted structured output is cached unavailable for this path.

    A fresh persistent-rejection streak (recurrence 3) counts exactly
    like a mark, so persistently rejecting paths skip the doomed
    structured attempt without waiting for another slow failure.
    """
    with _OMITTED_STRUCTURED_LOCK:
        if model is None:
            fresh: list[str] = []
            for key, recorded_at in _OMITTED_STRUCTURED_UNAVAILABLE.items():
                if _omitted_structured_entry_fresh(recorded_at):
                    fresh.append(key)
            for key in list(_OMITTED_STRUCTURED_UNAVAILABLE):
                if key not in fresh:
                    _OMITTED_STRUCTURED_UNAVAILABLE.pop(key, None)
            for key in list(_OMITTED_STRUCTURED_CONSECUTIVE):
                if _omitted_streak_counts(key) and key not in fresh:
                    fresh.append(key)
            return bool(fresh)
        key = _omitted_structured_key(model)
        if key == "":
            return any(
                _omitted_structured_entry_fresh(stamp)
                for stamp in _OMITTED_STRUCTURED_UNAVAILABLE.values()
            )
        cached_at: float | None = _OMITTED_STRUCTURED_UNAVAILABLE.get(key)
        if cached_at is not None:
            if not _omitted_structured_entry_fresh(cached_at):
                _OMITTED_STRUCTURED_UNAVAILABLE.pop(key, None)
            else:
                return True
        if _omitted_streak_counts(key):
            return True
        return False


def mark_omitted_structured_unavailable(model: Any | None = None) -> None:
    """Remember that omitted structured output is unavailable for this path."""
    key = _omitted_structured_key(model)
    if not key:
        key = "default"
    with _OMITTED_STRUCTURED_LOCK:
        _OMITTED_STRUCTURED_UNAVAILABLE[key] = time.monotonic()


def record_omitted_structured_rejection(
    model: Any | None = None, *, immediate: bool = False
) -> None:
    """Record one capability-suggestive omitted-structured rejection.

    Deterministic failures pass ``immediate=True`` and mark at once
    (existing behavior). Generic provider errors pass ``immediate=False``:
    an isolated rejection only starts a streak, while
    ``OMITTED_STRUCTURED_PERSISTENT_THRESHOLD`` consecutive rejections
    mark the path for the TTL. Timeout/transient, content validation
    and provider 429 must never call this. Any structured success
    clears the streak via :func:`record_omitted_structured_success`.
    """
    key = _omitted_structured_key(model)
    if not key:
        key = "default"
    with _OMITTED_STRUCTURED_LOCK:
        now = time.monotonic()
        if immediate:
            _OMITTED_STRUCTURED_UNAVAILABLE[key] = now
            _OMITTED_STRUCTURED_CONSECUTIVE[key] = (
                OMITTED_STRUCTURED_PERSISTENT_THRESHOLD,
                now,
            )
            return
        entry = _OMITTED_STRUCTURED_CONSECUTIVE.get(key)
        if entry is not None and _omitted_consecutive_fresh(entry[1]):
            count = int(entry[0]) + 1
        else:
            count = 1
        _OMITTED_STRUCTURED_CONSECUTIVE[key] = (count, now)
        if count >= OMITTED_STRUCTURED_PERSISTENT_THRESHOLD:
            _OMITTED_STRUCTURED_UNAVAILABLE[key] = now


def record_omitted_structured_success(model: Any | None = None) -> None:
    """Clear the consecutive-rejection streak after a structured success."""
    key = _omitted_structured_key(model)
    if not key:
        key = "default"
    with _OMITTED_STRUCTURED_LOCK:
        _OMITTED_STRUCTURED_CONSECUTIVE.pop(key, None)


def clear_omitted_structured_cache() -> None:
    """Reset the omitted-structured capability cache (tests only)."""
    with _OMITTED_STRUCTURED_LOCK:
        _OMITTED_STRUCTURED_UNAVAILABLE.clear()
        _OMITTED_STRUCTURED_CONSECUTIVE.clear()


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
        burning its own primary retry per turn. Derived agents preserve
        this model's transport-agent policy, so hidden planner/answer/
        summarizer stages can omit a provider-rejected selector while
        retaining distinct logical audit identities. The verifier is built
        separately via :func:`build_verifier_model` because it has no fallback.
        """
        nxt = OpenCodeChatModel(
            self._client,
            agent=agent,
            primary_model=self.primary_model,
            fallback_model=self.fallback_model,
            request_timeout=self.request_timeout,
            transport_agent=self.transport_agent,
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

    async def _invoke_omitted_primary_text(self, prompt: str, *, system: str = "") -> str:
        """Invoke the pinned primary with the transport selector omitted.

        Gate C live repair (run 37608207212 on exact main 8c57bba): the
        live lane served planner/answer/summarizer through the weak
        fallback while the decoupled verifier (omitted selector) served
        Muse Spark, with 13 generic clarifications, diversity collapse and
        max latency over budget while verifier stayed ``unavailable``. The
        verifier probe already isolates this as an agent-selector
        rejection (custom selector 403, omitted serves). Retrying the same
        strong primary with the selector omitted before falling back to
        the weak model keeps grounding strong without any exact-question
        special case. Provider 429 always propagates and never triggers
        this path. The logical audit identity stays ``self.agent`` while
        the wire omits the selector.
        """
        if not self.wire_agent:
            raise ValueError("omitted retry is only for custom wire agents")
        client = self._client
        session = await client.create_session(title="")
        try:
            return await client.send_message(
                session.id,
                prompt,
                timeout=self.request_timeout,
                agent="",
                model=self.primary_model,
                system=system,
                audit_agent=self.agent,
            )
        finally:
            try:
                await client.delete_session(session.id)
            except OpenCodeError:
                logger.warning("ephemeral session cleanup failed")

    async def _invoke_omitted_primary_structured(
        self,
        prompt: str,
        *,
        system: str,
        schema: dict[str, object],
        retry_count: int = STRUCTURED_RETRY_COUNT,
    ) -> dict[str, object]:
        """Invoke pinned primary structured output with selector omitted.

        Same Gate C agent-selector repair as
        :meth:`_invoke_omitted_primary_text` for the native ``json_schema``
        path (planner). Only provider-access (403) callers use this; 429
        propagates and deterministic failures stay fail-closed.
        """
        if not self.wire_agent:
            raise ValueError("omitted retry is only for custom wire agents")
        client = self._client
        session = await client.create_session(title="")
        try:
            return await client.send_structured_message(
                session.id,
                prompt,
                timeout=self.request_timeout,
                agent="",
                model=self.primary_model,
                system=system,
                schema=schema,
                retry_count=retry_count,
                audit_agent=self.agent,
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
        # Gate C repair: when the custom selector is rejected but the same
        # strong primary serves with the selector omitted (verifier probe),
        # try the omitted primary first so later turns stay strong instead
        # of sticking to the weak fallback and collapsing to generic
        # clarification. Any omitted failure except 429 falls through to the
        # weak fallback (kodmial/aa#206): a deterministic omitted failure
        # (for example structured-output-missing or served-model-mismatch)
        # must never strand the configured fallback and collapse the graph
        # to natural retry replies with no telemetry. 429 always propagates
        # for runner retire/restart; deterministic primary failures still
        # propagate without fallback.
        if self._fast_fallback_available():
            if self.wire_agent:
                try:
                    omitted = await self._invoke_omitted_primary_text(prompt, system=system)
                    logger.info("opencode omitted primary served while circuit open")
                    return omitted
                except OpenCodeRateLimitError:
                    raise
                except OpenCodeError:
                    pass
            else:
                # Omitted-wire path (planner/answer/summarizer with an empty
                # transport selector, Gate C run 37725833818): the circuit
                # would otherwise pin every later turn to the weak fallback
                # for the full TTL even after the pinned primary recovers,
                # serving weak drafts that fail the language/leak guard and
                # collapse to generic clarification with the verifier never
                # served. Re-probe the already-omitted primary once with no
                # sleep before falling back; a recovered primary clears the
                # circuit and restores strong serves, a still-rejected
                # primary falls through fast to the fallback.
                try:
                    reply = await self._invoke_ephemeral(
                        prompt, model=self.primary_model, agent=self.agent, system=system
                    )
                    self._clear_primary_rejection()
                    logger.info("opencode omitted-wire primary re-probed while circuit open")
                    return reply
                except OpenCodeRateLimitError:
                    raise
                except OpenCodeError:
                    pass
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
                if self.wire_agent:
                    try:
                        omitted = await self._invoke_omitted_primary_text(prompt, system=system)
                        logger.info("opencode omitted primary served after custom 403")
                        return omitted
                    except OpenCodeRateLimitError:
                        raise
                    except OpenCodeError:
                        pass
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

        # A selector-omitted planner may discover that the pinned primary
        # serves text but not native json_schema for this schema/provider.
        # Cache that deterministic capability result and use the configured
        # fallback directly on subsequent planner turns.
        if (
            not self.wire_agent
            and self.fallback_model.strip()
            and self.fallback_model != self.primary_model
            and omitted_structured_unavailable(self)
        ):
            logger.info("opencode omitted structured capability cached unavailable; fallback used")
            return await self._invoke_ephemeral_structured(
                prompt,
                system=system,
                schema=schema,
                model=self.fallback_model,
                agent=self.agent,
                retry_count=retry_count,
            )

        if self._fast_fallback_available():
            if self.wire_agent and not omitted_structured_unavailable(self):
                try:
                    omitted = await self._invoke_omitted_primary_structured(
                        prompt,
                        system=system,
                        schema=schema,
                        retry_count=retry_count,
                    )
                    logger.info("opencode omitted primary served while circuit open")
                    return omitted
                except OpenCodeRateLimitError:
                    raise
                except OpenCodeDeterministicError as exc:
                    if "structured output missing" in str(exc).lower():
                        mark_omitted_structured_unavailable(self)
                except OpenCodeError:
                    pass
            elif not self.wire_agent and not omitted_structured_unavailable(self):
                # Omitted-wire structured path (same Gate C recovery as the
                # text path above): re-probe the already-omitted primary
                # once with no sleep when the capability is not known to be
                # missing. A recovered primary clears the circuit and
                # restores strong structured plans; a still-rejected primary
                # falls through fast to the fallback. A cached deterministic
                # capability failure keeps the direct fallback above.
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
                    logger.info("opencode omitted-wire primary re-probed while circuit open")
                    return result
                except OpenCodeRateLimitError:
                    raise
                except OpenCodeDeterministicError as exc:
                    if "structured output missing" in str(exc).lower():
                        mark_omitted_structured_unavailable(self)
                except OpenCodeError:
                    pass
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
            except OpenCodeDeterministicError as exc:
                if (
                    not self.wire_agent
                    and self.fallback_model.strip()
                    and self.fallback_model != self.primary_model
                    and "structured output missing" in str(exc).lower()
                ):
                    mark_omitted_structured_unavailable(self)
                    last_transient = exc
                    break
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
                if self.wire_agent and not omitted_structured_unavailable(self):
                    try:
                        omitted = await self._invoke_omitted_primary_structured(
                            prompt,
                            system=system,
                            schema=schema,
                            retry_count=retry_count,
                        )
                        logger.info("opencode omitted primary served after custom 403")
                        return omitted
                    except OpenCodeRateLimitError:
                        raise
                    except OpenCodeDeterministicError as exc:
                        if "structured output missing" in str(exc).lower():
                            mark_omitted_structured_unavailable(self)
                    except OpenCodeError:
                        pass
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


def build_planner_model(
    client: OpenCodeClient,
    *,
    primary_model: str,
    fallback_model: str,
    request_timeout: float = 120.0,
) -> OpenCodeChatModel:
    """Build the hidden planner family with an omitted transport selector.

    Planner/answer/summarizer keep their configured primary/fallback model
    policy. Only the provider-specific transport selector is omitted; the
    logical agent identity and native system prompt remain intact.
    """
    from aa.config import DEFAULT_FALLBACK_MODEL, DEFAULT_PRIMARY_MODEL

    pinned_primary = (primary_model or "").strip() or DEFAULT_PRIMARY_MODEL
    pinned_fallback = (fallback_model or "").strip() or DEFAULT_FALLBACK_MODEL
    return OpenCodeChatModel(
        client,
        agent=PLANNER_AGENT_V2,
        primary_model=pinned_primary,
        fallback_model=pinned_fallback,
        request_timeout=request_timeout,
        transport_agent=PLANNER_TRANSPORT_AGENT_V2,
    )


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
    "OMITTED_STRUCTURED_CAPABILITY_TTL_S",
    "OMITTED_STRUCTURED_PERSISTENT_THRESHOLD",
    "PLANNER_AGENT_V2",
    "PLANNER_TRANSPORT_AGENT_V2",
    "PRIMARY_ACCESS_CIRCUIT_TTL_S",
    "RUNTIME_AGENT_V2",
    "STRUCTURED_RETRY_COUNT",
    "SUMMARIZER_AGENT_V2",
    "VERIFIER_AGENT_V2",
    "VERIFIER_TRANSPORT_AGENT_V2",
    "OpenCodeChatModel",
    "build_planner_model",
    "build_verifier_model",
    "clear_omitted_structured_cache",
    "clear_primary_circuit",
    "mark_omitted_structured_unavailable",
    "omitted_structured_unavailable",
    "record_omitted_structured_rejection",
    "record_omitted_structured_success",
    "render_messages_text",
    "split_system_and_user",
]
