"""Async application lifecycle for the AA Telegram worker.

Concurrency architecture (issue #5): one authoritative Telegram poller and
one local ``opencode serve`` process exist per worker. Accepted updates are
dispatched through a keyed per-chat dispatcher: turns for the same chat are
strict FIFO with at most one active turn, turns for different chats may run
concurrently under a global ``MAX_CONCURRENT_TURNS`` bound, and each
per-chat pending queue is bounded. ``/new`` travels through the same per-chat
queue as ordinary turns, so it is ordered relative to them and never resets
a session while an older turn is still mutating it.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from aa.config import Settings
from aa.control.runtime_control import RuntimeController
from aa.conversation.orchestrator import (
    FAIL_CLOSED_REPLY,
    TurnFailed,
    TurnRunner,
    is_substantive,
    meets_russian_only,
    run_trivial_turn,
)
from aa.conversation.output_limits import (
    compact_text_to_envelope,
    envelope_passes,
    resolve_generation_budget,
)
from aa.corpus.context import CorpusContext
from aa.grounding import GroundingGate
from aa.opencode.errors import OpenCodeError, OpenCodeSessionNotFoundError
from aa.opencode.runtime import LocalOpenCodeRuntime, OpenCodeConfig, OpenCodeRuntime
from aa.retrieval.index import HybridIndex, open_hybrid_index
from aa.safety.response import build_emergency_response
from aa.safety.router import SafetyDecision, SafetyRouter
from aa.sessions.coordinator import SessionCoordinator
from aa.telegram.dispatcher import ChatQueueFullError, ChatTurnDispatcher
from aa.telegram.transport import (
    PollingTelegramTransport,
    StubTelegramTransport,
    TelegramApiError,
    TelegramEnvelopeError,
    TelegramIncoming,
    TelegramReply,
    TelegramTransport,
)

logger = logging.getLogger("aa.app")

_START_REPLY = "Бот готов. Напишите сообщение."
_NEW_REPLY = "Новая беседа начата."
_TEMPORARY_ERROR_REPLY = "Не удалось обработать сообщение. Попробуйте ещё раз."
_BUSY_REPLY = "Сейчас много сообщений. Попробуйте ещё раз через минуту."


class Application:
    """Composes module boundaries and owns startup/shutdown order."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: TelegramTransport | None = None,
        opencode_runtime: OpenCodeRuntime | None = None,
        corpus: CorpusContext | None = None,
        sessions: SessionCoordinator | None = None,
        safety: SafetyRouter | None = None,
        controller: RuntimeController | None = None,
        grounding: GroundingGate | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport or StubTelegramTransport()
        self.opencode_runtime = opencode_runtime or LocalOpenCodeRuntime(
            OpenCodeConfig(
                base_url=settings.opencode_base_url,
                command=settings.opencode_command,
                workdir=settings.opencode_workdir,
                model=settings.opencode_model,
                context_limit_tokens=settings.opencode_context_limit_tokens,
                max_output_tokens=settings.opencode_max_output_tokens,
            )
        )
        self.corpus = corpus or CorpusContext(
            path=settings.aa_corpus_path, version=settings.aa_corpus_version
        )
        self.sessions = sessions or SessionCoordinator()
        self.safety = safety or SafetyRouter()
        self.controller = controller or RuntimeController(
            session_duration_seconds=settings.bot_session_duration_seconds
        )
        # Deterministic Russian quotation/grounding policy owned by the
        # Python orchestrator (aa.grounding). Production fails closed:
        # exact Russian quotations require the version-pinned Russian
        # corpus (issue #50); translation fallback stays disabled unless
        # a caller explicitly allows it. The agent prompt states the
        # user-facing duties and defers enforcement to this gate.
        self.grounding = grounding or GroundingGate(corpus_version=settings.aa_corpus_version)
        self._turn_runner: TurnRunner | None = None
        self._index: HybridIndex | None = None
        self._index_error: str | None = None
        self._running = False
        # Transport/orchestration only: the dispatcher calls the production
        # #9 turn orchestrator via ``respond`` and never creates another
        # LLM/provider client or knowledge pipeline.
        self.dispatcher = ChatTurnDispatcher(
            self._process_dispatched_update,
            max_concurrent_turns=settings.max_concurrent_turns,
            per_chat_queue_size=settings.per_chat_queue_size,
        )
        self._wire_transport_handlers()

    @property
    def running(self) -> bool:
        """Whether the application is running."""
        return self._running

    async def start(self) -> None:
        """Start all components in dependency order.

        The bounded runtime clock is armed only after OpenCode and Telegram
        are ready, so bootstrap time never consumes the requested live window.
        """
        if self._running:
            return
        self.settings.validate(require_bot_token=False)
        logger.info("starting worker", extra={"config": self.settings.to_safe_dict()})
        await self.corpus.load()
        try:
            await self.opencode_runtime.start()
            # Readiness gate: no Telegram traffic is accepted until the
            # local OpenCode runtime has proven healthy.
            await self.opencode_runtime.ensure_ready()
            await self.sessions.start()
            await self.safety.start()
            await self.dispatcher.start()
            await self.transport.start()
            # Start the requested 15m/1h/2h/3h window only after the poller
            # is live and all dependencies have completed bootstrap.
            await self.controller.start()
        except Exception:
            await self.transport.stop()
            await self.dispatcher.stop()
            await self.safety.stop()
            await self.sessions.stop()
            await self.opencode_runtime.stop()
            await self.corpus.unload()
            await self.controller.stop()
            raise
        self._running = True
        logger.info("worker started")

    async def stop(self) -> None:
        """Stop all components in reverse order (idempotent)."""
        if not self._running:
            # Still ensure subcomponents are stopped for partial startups.
            await self.transport.stop()
            await self.dispatcher.stop()
            await self.safety.stop()
            await self.sessions.stop()
            await self.opencode_runtime.stop()
            await self.corpus.unload()
            await self.controller.stop()
            return
        logger.info("stopping worker")
        self._running = False
        # Stop accepting new Telegram updates first, then drain dispatched
        # turns (bounded) before releasing OpenCode/corpus resources. This
        # keeps clean shutdown/handoff safe; the remaining crash window
        # (accepted but unprocessed updates lost on crash) is documented in
        # ``aa.telegram.dispatcher`` and ``docs/opencode-runtime.md``.
        await self.transport.stop()
        await self.dispatcher.stop()
        await self.safety.stop()
        await self.sessions.stop()
        await self.opencode_runtime.stop()
        await self.corpus.unload()
        await self.controller.stop()
        logger.info("worker stopped")

    async def run(self) -> None:
        """Run until the controller requests shutdown."""
        await self.start()
        try:
            while not self.controller.should_stop():
                await asyncio.sleep(0.05)
        finally:
            await self.stop()

    def _wire_transport_handlers(self) -> None:
        """Connect the concrete polling transport to application behavior.

        Every accepted update (ordinary text, ``/start`` and ``/new``) is
        enqueued into the same per-chat dispatcher queue. The poller callback
        stays fast and never awaits a full OpenCode turn, so one slow chat
        cannot head-of-line block unrelated chats. ``/new`` is ordered
        relative to ordinary turns for that chat because it shares the same
        FIFO queue and serialized worker.
        """
        if not isinstance(self.transport, PollingTelegramTransport):
            return
        self.transport.on_update(self._enqueue_telegram_update)
        self.transport.on_command("start", self._enqueue_telegram_update)
        self.transport.on_command("new", self._enqueue_telegram_update)

    async def _enqueue_telegram_update(self, incoming: TelegramIncoming) -> None:
        """Accept one update into the per-chat dispatcher (poller-fast).

        Transport acknowledgement happens after this returns, so this must
        never await a full OpenCode turn. Overflow backpressures with a
        single bounded reply instead of unbounded queue growth.
        """
        if not self.dispatcher.running:
            await self._process_dispatched_update(incoming)
            return
        try:
            await self.dispatcher.submit(incoming)
        except ChatQueueFullError:
            logger.warning(
                "chat queue full; backpressure reply",
                extra={"chat_id": incoming.chat_id, "update_id": incoming.update_id},
            )
            try:
                await self.transport.send(TelegramReply(chat_id=incoming.chat_id, text=_BUSY_REPLY))
            except (TelegramApiError, TelegramEnvelopeError):
                logger.warning(
                    "backpressure reply delivery failed",
                    extra={"chat_id": incoming.chat_id, "update_id": incoming.update_id},
                )

    async def _process_dispatched_update(self, incoming: TelegramIncoming) -> None:
        """Run one dispatched turn inside that chat's serialized worker.

        Safety routing precedes normal AA handling; substantive turns call
        the production #9 orchestrator via :meth:`respond` as the single
        substantive-turn API. Session create/reset happens here, inside the
        per-chat serialization, so concurrent first messages cannot create
        competing sessions and ``/new`` cannot interleave with an older
        turn for the same chat.
        """
        if incoming.command == "start":
            await self._handle_start_command(incoming)
            return
        if incoming.command == "new":
            await self._handle_new_command(incoming)
            return
        await self._handle_telegram_update(incoming)

    async def _handle_start_command(self, incoming: TelegramIncoming) -> None:
        await self.transport.send(TelegramReply(chat_id=incoming.chat_id, text=_START_REPLY))

    async def _handle_new_command(self, incoming: TelegramIncoming) -> None:
        try:
            await self.sessions.reset_opencode_session(
                incoming.chat_id, self.opencode_runtime.client
            )
            reply = _NEW_REPLY
        except OpenCodeError:
            logger.warning("telegram new-session reset failed", extra={"chat_id": incoming.chat_id})
            reply = _TEMPORARY_ERROR_REPLY
        await self.transport.send(TelegramReply(chat_id=incoming.chat_id, text=reply))

    async def _handle_telegram_update(self, incoming: TelegramIncoming) -> None:
        """Process one private text update and always emit a bounded reply.

        Exactly one Telegram message is delivered per update: overflow is
        never split into multiple messages. When the final transport guard
        blocks an escaped overlong payload, a single bounded fallback is
        delivered instead. Substantive work goes through :meth:`respond`,
        the single production #9 orchestrator entry point; this transport
        layer never sends a substantive user message to OpenCode directly.
        """
        self.sessions.record_message(incoming.chat_id)
        try:
            reply = await self.respond(incoming.chat_id, incoming.text)
            if not reply.strip():
                raise OpenCodeError("opencode returned an empty response")
            if not meets_russian_only(reply):
                logger.warning(
                    "telegram reply failed closed on RU-only contract",
                    extra={"chat_id": incoming.chat_id, "update_id": incoming.update_id},
                )
                reply = FAIL_CLOSED_REPLY
        except (OpenCodeError, ValueError):
            logger.warning(
                "telegram message processing failed",
                extra={"chat_id": incoming.chat_id, "update_id": incoming.update_id},
            )
            reply = _TEMPORARY_ERROR_REPLY
        try:
            await self.transport.send(TelegramReply(chat_id=incoming.chat_id, text=reply))
        except TelegramEnvelopeError:
            logger.warning(
                "telegram outbound reply blocked by envelope guard",
                extra={"chat_id": incoming.chat_id, "update_id": incoming.update_id},
            )
            try:
                await self.transport.send(
                    TelegramReply(chat_id=incoming.chat_id, text=_TEMPORARY_ERROR_REPLY)
                )
            except TelegramApiError:
                logger.warning(
                    "telegram fallback reply delivery failed",
                    extra={"chat_id": incoming.chat_id, "update_id": incoming.update_id},
                )
        except TelegramApiError:
            logger.warning(
                "telegram outbound send failed",
                extra={"chat_id": incoming.chat_id, "update_id": incoming.update_id},
            )

    def _turn_index(self) -> HybridIndex:
        """Open (once) the RU-first hybrid index or fail closed.

        The opened index is reused across ordinary turns; provider
        failures never invalidate or rebuild it here.
        """
        if self._index is not None:
            return self._index
        if self._index_error is not None:
            raise TurnFailed("corpus-unavailable", self._index_error)
        try:
            corpus_root = Path(self.settings.aa_corpus_path)
            index_dir = corpus_root / "generated" / "retrieval"
            self._index = open_hybrid_index(
                index_dir,
                ru_manifest_path=corpus_root / "canonical.ru.manifest.json",
                en_manifest_path=corpus_root / "canonical.manifest.json",
                lock_path=corpus_root / "embedding.lock.json",
            )
            return self._index
        except (ValueError, OSError, RuntimeError) as exc:
            self._index_error = str(exc)
            raise TurnFailed("corpus-unavailable", self._index_error) from exc

    def _get_turn_runner(self) -> TurnRunner:
        """Build the deterministic turn runner for the pinned runtime."""
        if self._turn_runner is not None and self._turn_runner.index is not None:
            return self._turn_runner
        self._index_error = None
        try:
            index: HybridIndex | None = self._turn_index()
        except TurnFailed:
            index = None
        runner = TurnRunner(
            index=index,
            ru_corpus_version=self.settings.aa_corpus_version,
            agent=self.settings.opencode_agent,
            primary_model=self.settings.opencode_model,
            fallback_model=self.settings.opencode_fallback_model,
            generation_budget_tokens=resolve_generation_budget(
                self.settings.opencode_max_output_tokens
            ),
        )
        if runner.index is not None:
            self._turn_runner = runner
        return runner

    async def _run_trivial_turn(self, session_id: str, text: str) -> str:
        """Execute the direct bounded agent path for one non-substantive turn."""
        client = self.opencode_runtime.client

        async def _trivial_send(
            sid: str,
            prompt: str,
            *,
            agent: str = "",
            model: str = "",
            timeout: float | None = None,
        ) -> str:
            return await client.send_message(sid, prompt, timeout=timeout, agent=agent, model=model)

        synthesis = await run_trivial_turn(
            text,
            session_id=session_id,
            send=_trivial_send,
            agent=self.settings.opencode_agent,
            primary_model=self.settings.opencode_model,
            fallback_model=self.settings.opencode_fallback_model,
        )
        return synthesis.text

    async def _run_grounded_turn(self, session_id: str, text: str) -> str:
        """Execute the production grounded pipeline for one substantive turn."""
        runner = self._get_turn_runner()
        if runner.index is None:
            raise TurnFailed("corpus-unavailable", "RU corpus/index is unavailable")
        client = self.opencode_runtime.client

        async def _send(
            sid: str,
            prompt: str,
            *,
            agent: str = "",
            model: str = "",
            timeout: float | None = None,
        ) -> str:
            return await client.send_message(sid, prompt, timeout=timeout, agent=agent, model=model)

        try:
            response = await runner.run_grounded_turn(text, session_id=session_id, send=_send)
        except TurnFailed as exc:
            if exc.category == "session-not-found":
                raise
            raise
        return response.text

    async def respond(self, chat_id: int, text: str) -> str:
        """Answer one inbound message with emergency precedence.

        The deterministic safety layer runs first: when it takes the
        emergency route, the bounded safe reply is returned immediately
        and no OpenCode work is scheduled (the LLM never decides whether
        the emergency route is taken). Substantive turns run the full
        Russian-first grounded pipeline (issue #9); non-substantive
        greetings take a direct bounded agent path. Retrieval/grounding
        failures fail closed with a fixed message instead of an invented
        answer.

        Only message lengths and routing decisions are logged, never the
        message body.

        Every returned reply is deterministically confined to the #83
        hard Telegram envelope (``<= 900`` graphemes / ``<= 130`` words,
        verbatim quote aggregate ``<= 300`` chars). The orchestrator owns
        compact regeneration for grounded turns; this boundary applies a
        final complete-unit compaction so emergency, trivial, grounded,
        and fail-closed paths all satisfy the same cap. Replies are
        returned as a single message; overflow is never split.
        """
        result, _emergency_reply = self.safety.route(text)
        if result.decision is SafetyDecision.EMERGENCY and result.classification is not None:
            # Production Telegram runtime is RU-only: the emergency reply
            # is always the deterministic Russian template, regardless of
            # the detected input language. No English fallback may leak.
            logger.info(
                "emergency response served",
                extra={"chat_id": chat_id, "reason": result.reason},
            )
            return self._fit_envelope(
                build_emergency_response(result.classification, language="ru")
            )
        if result.decision is SafetyDecision.BLOCK:
            logger.info("blocked message refused", extra={"chat_id": chat_id})
            raise ValueError("refusing to answer an empty message")
        session_id = await self.sessions.ensure_opencode_session(
            chat_id, self.opencode_runtime.client
        )
        try:
            if not is_substantive(text):
                try:
                    trivial_reply = await self._run_trivial_turn(session_id, text)
                except TurnFailed as exc:
                    if exc.category == "session-not-found":
                        try:
                            session_id = await self.sessions.reset_opencode_session(
                                chat_id, self.opencode_runtime.client, delete_remote=False
                            )
                            trivial_reply = await self._run_trivial_turn(session_id, text)
                        except OpenCodeSessionNotFoundError as exc2:
                            raise TurnFailed(
                                "session-not-found", "opencode session is gone"
                            ) from exc2
                    else:
                        raise
                if not meets_russian_only(trivial_reply):
                    logger.warning(
                        "trivial turn failed closed on RU-only contract",
                        extra={"chat_id": chat_id},
                    )
                    return FAIL_CLOSED_REPLY
                logger.info("trivial response served", extra={"chat_id": chat_id})
                return self._fit_envelope(trivial_reply)
            try:
                reply = await self._run_grounded_turn(session_id, text)
            except TurnFailed as exc:
                if exc.category == "session-not-found":
                    try:
                        session_id = await self.sessions.reset_opencode_session(
                            chat_id, self.opencode_runtime.client, delete_remote=False
                        )
                        reply = await self._run_grounded_turn(session_id, text)
                    except OpenCodeSessionNotFoundError as exc2:
                        raise TurnFailed("session-not-found", "opencode session is gone") from exc2
                else:
                    raise
        except TurnFailed as exc:
            logger.warning(
                "grounded turn failed closed",
                extra={"chat_id": chat_id, "category": exc.category},
            )
            return FAIL_CLOSED_REPLY
        except OpenCodeSessionNotFoundError:
            # A local chat mapping can outlive an OpenCode session after a
            # runtime restart. Rebind once and retry against a fresh session,
            # preserving the original routing: non-substantive greetings retry
            # through the direct trivial path (never the RU grounded pipeline).
            try:
                session_id = await self.sessions.reset_opencode_session(
                    chat_id, self.opencode_runtime.client, delete_remote=False
                )
            except OpenCodeSessionNotFoundError:
                logger.warning(
                    "grounded turn failed closed after rebind",
                    extra={"chat_id": chat_id, "category": "session-not-found"},
                )
                return FAIL_CLOSED_REPLY
            if not is_substantive(text):
                try:
                    trivial_retry = await self._run_trivial_turn(session_id, text)
                except TurnFailed as exc:
                    logger.warning(
                        "trivial turn failed closed after rebind",
                        extra={"chat_id": chat_id, "category": exc.category},
                    )
                    return FAIL_CLOSED_REPLY
                except OpenCodeSessionNotFoundError:
                    logger.warning(
                        "trivial turn failed closed after rebind",
                        extra={"chat_id": chat_id, "category": "session-not-found"},
                    )
                    return FAIL_CLOSED_REPLY
                if not meets_russian_only(trivial_retry):
                    logger.warning(
                        "trivial turn failed closed on RU-only contract after rebind",
                        extra={"chat_id": chat_id},
                    )
                    return FAIL_CLOSED_REPLY
                logger.info("trivial response served", extra={"chat_id": chat_id})
                return self._fit_envelope(trivial_retry)
            try:
                reply = await self._run_grounded_turn(session_id, text)
            except TurnFailed as exc:
                logger.warning(
                    "grounded turn failed closed after rebind",
                    extra={"chat_id": chat_id, "category": exc.category},
                )
                return FAIL_CLOSED_REPLY
            except OpenCodeSessionNotFoundError:
                logger.warning(
                    "grounded turn failed closed after rebind",
                    extra={"chat_id": chat_id, "category": "session-not-found"},
                )
                return FAIL_CLOSED_REPLY
        if not meets_russian_only(reply):
            logger.warning(
                "grounded turn failed closed on RU-only contract",
                extra={"chat_id": chat_id},
            )
            return FAIL_CLOSED_REPLY
        logger.info("normal response served", extra={"chat_id": chat_id})
        return self._fit_envelope(reply)

    @staticmethod
    def _fit_envelope(reply: str) -> str:
        """Confine ``reply`` to the hard envelope (complete-unit safe).

        The orchestrator already applies compact regeneration upstream;
        this is the final deterministic guard so every path served here
        satisfies the same cap. Only lengths are logged on compaction,
        never message text.
        """
        if envelope_passes(reply):
            return reply
        compacted = compact_text_to_envelope(reply)
        logger.info("application reply compacted to envelope")
        return compacted

    async def __aenter__(self) -> Application:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()


def create_application(settings: Settings) -> Application:
    """Factory used by the entrypoint and tests."""
    transport: TelegramTransport
    if settings.has_bot_token:
        transport = PollingTelegramTransport(token=settings.telegram_bot_token)
    else:
        transport = StubTelegramTransport()
    return Application(settings, transport=transport)
