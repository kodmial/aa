"""Async application lifecycle for the AA Telegram worker."""

from __future__ import annotations

import asyncio
import logging

from aa.config import Settings
from aa.control.runtime_control import RuntimeController
from aa.corpus.context import CorpusContext
from aa.grounding import GroundingGate
from aa.opencode.errors import OpenCodeError, OpenCodeSessionNotFoundError
from aa.opencode.runtime import LocalOpenCodeRuntime, OpenCodeConfig, OpenCodeRuntime
from aa.output_limits import (
    OVERLONG_FALLBACK_TEXT,
    assess,
    bulk_export_summary_reply,
    compact_to_hard_cap,
    detect_language,
    enforce_async,
    is_bulk_export_request,
    is_continuation_request,
    resolve_generation_budget,
)
from aa.safety.router import SafetyDecision, SafetyRouter
from aa.sessions.coordinator import SessionCoordinator
from aa.telegram.transport import (
    PollingTelegramTransport,
    StubTelegramTransport,
    TelegramIncoming,
    TelegramReply,
    TelegramReplyTooLongError,
    TelegramTransport,
)

logger = logging.getLogger("aa.app")

_START_REPLY = "Бот готов. Напишите сообщение. / Bot is ready. Send a message."
_NEW_REPLY = "Новая беседа начата. / New conversation started."
_TEMPORARY_ERROR_REPLY = (
    "Не удалось обработать сообщение. Попробуйте ещё раз. / "
    "Could not process the message. Please try again."
)


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
        self._running = False
        # Chats whose last turn was served in bulk-export summary mode.
        # A continuation/paging request while flagged stays in summary
        # mode so serial "continue/next part" turns cannot page through
        # canonical text across turns.
        self._bulk_export_chats: set[int] = set()
        self._wire_transport_handlers()

    @property
    def generation_budget_tokens(self) -> int:
        """Bounded output-token efficiency guard (not the product contract).

        The pinned OpenCode ``POST /session/{id}/message`` surface
        exposes no per-message max-token field, so this budget cannot be
        enforced as an API parameter without inventing an unsupported
        field or a second provider client. It sizes cost/latency
        expectations only; the deterministic character validator in
        :mod:`aa.output_limits` is the authoritative enforcement.
        """
        return resolve_generation_budget(self.settings.opencode_max_output_tokens)

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
            await self.transport.start()
            # Start the requested 15m/1h/2h/3h window only after the poller
            # is live and all dependencies have completed bootstrap.
            await self.controller.start()
        except Exception:
            await self.transport.stop()
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
            await self.safety.stop()
            await self.sessions.stop()
            await self.opencode_runtime.stop()
            await self.corpus.unload()
            await self.controller.stop()
            return
        logger.info("stopping worker")
        self._running = False
        await self.transport.stop()
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
        """Connect the concrete polling transport to application behavior."""
        if not isinstance(self.transport, PollingTelegramTransport):
            return
        self.transport.on_update(self._handle_telegram_update)
        self.transport.on_command("start", self._handle_start_command)
        self.transport.on_command("new", self._handle_new_command)

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

        Exactly one Telegram message is ever emitted per inbound update:
        overflow is compacted upstream and never auto-split here. A
        transport envelope rejection degrades to one bounded fallback
        message rather than a split or an oversized delivery.
        """
        self.sessions.record_message(incoming.chat_id)
        try:
            reply = await self.respond(incoming.chat_id, incoming.text)
            if not reply.strip():
                raise OpenCodeError("opencode returned an empty response")
        except (OpenCodeError, ValueError):
            logger.warning(
                "telegram message processing failed",
                extra={"chat_id": incoming.chat_id, "update_id": incoming.update_id},
            )
            reply = _TEMPORARY_ERROR_REPLY
        try:
            await self.transport.send(TelegramReply(chat_id=incoming.chat_id, text=reply))
        except TelegramReplyTooLongError:
            logger.warning(
                "telegram oversized reply replaced with fallback",
                extra={"chat_id": incoming.chat_id, "update_id": incoming.update_id},
            )
            await self.transport.send(
                TelegramReply(chat_id=incoming.chat_id, text=OVERLONG_FALLBACK_TEXT)
            )

    async def _send_grounded_message(self, session_id: str, text: str) -> str:
        """Send through the named AA agent with a technical model fallback only."""
        try:
            return await self.opencode_runtime.client.send_message(
                session_id,
                text,
                agent=self.settings.opencode_agent,
                model=self.settings.opencode_model,
            )
        except OpenCodeError as exc:
            if not exc.transient:
                raise
            logger.warning("primary AA model unavailable; trying fallback")
            return await self.opencode_runtime.client.send_message(
                session_id,
                text,
                agent=self.settings.opencode_agent,
                model=self.settings.opencode_fallback_model,
            )

    async def respond(self, chat_id: int, text: str) -> str:
        """Answer one inbound message with emergency precedence.

        The deterministic safety layer runs first: when it takes the
        emergency route, the bounded safe reply is returned immediately
        and no OpenCode work is scheduled (the LLM never decides whether
        the emergency route is taken). Bulk corpus-reproduction requests
        are served in summary mode without emitting source bulk text.
        Otherwise the message is forwarded to the OpenCode runtime bound
        to ``chat_id`` and the result passes the deterministic output
        envelope (at most one compact regeneration, then complete-unit
        compaction) before it is returned.

        Only message lengths and routing decisions are logged, never the
        message body.
        """
        result, emergency_reply = self.safety.route(text)
        if result.decision is SafetyDecision.EMERGENCY and emergency_reply is not None:
            logger.info(
                "emergency response served",
                extra={"chat_id": chat_id, "reason": result.reason},
            )
            if assess(emergency_reply).acceptable:
                return emergency_reply
            return compact_to_hard_cap(emergency_reply)
        if result.decision is SafetyDecision.BLOCK:
            logger.info("blocked message refused", extra={"chat_id": chat_id})
            raise ValueError("refusing to answer an empty message")
        if is_bulk_export_request(text):
            self._bulk_export_chats.add(chat_id)
            summary = bulk_export_summary_reply(detect_language(text))
            logger.info("bulk export request served in summary mode", extra={"chat_id": chat_id})
            return summary
        if chat_id in self._bulk_export_chats and is_continuation_request(text):
            summary = bulk_export_summary_reply(detect_language(text))
            logger.info("bulk export continuation kept in summary mode", extra={"chat_id": chat_id})
            return summary
        self._bulk_export_chats.discard(chat_id)
        _ = self.generation_budget_tokens
        session_id = await self.sessions.ensure_opencode_session(
            chat_id, self.opencode_runtime.client
        )
        try:
            first = await self._send_grounded_message(session_id, text)
        except OpenCodeSessionNotFoundError:
            # A local chat mapping can outlive an OpenCode session after a
            # runtime restart. Rebind once and retry against a fresh session.
            session_id = await self.sessions.reset_opencode_session(
                chat_id, self.opencode_runtime.client, delete_remote=False
            )
            first = await self._send_grounded_message(session_id, text)

        async def _regenerate(_budget_instruction: str) -> str:
            # Same validated evidence (same OpenCode session history) with
            # an explicit remaining size budget; grounding is not bypassed.
            return await self._send_grounded_message(session_id, _budget_instruction)

        outcome = await enforce_async(first, _regenerate)
        logger.info(
            "normal response served",
            extra={
                "chat_id": chat_id,
                "regenerations": outcome.regenerations,
                "compacted": outcome.compacted,
            },
        )
        if assess(outcome.text).acceptable and outcome.text.strip():
            return outcome.text
        fallback = compact_to_hard_cap(outcome.text) if outcome.text.strip() else ""
        if fallback and assess(fallback).acceptable:
            return fallback
        return OVERLONG_FALLBACK_TEXT

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
