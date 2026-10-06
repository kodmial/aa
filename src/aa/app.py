"""Async application lifecycle for the AA Telegram worker.

Production conversational path (issue #118 cutover)::

    Telegram -> safety/commands -> typing heartbeat -> LangGraph thread
      -> planner/retrieval/evidence/AA Agent/grounding -> Telegram delivery

Text and voice share one LangGraph conversation state after ASR. The
graph/checkpointer is the authoritative conversation-memory layer;
accumulated OpenCode session history is never a second memory (hidden
calls use ephemeral OpenCode sessions per call).

Concurrency architecture (issue #5): one authoritative Telegram poller and
one local ``opencode serve`` process exist per worker. Accepted updates are
dispatched through a keyed per-chat dispatcher: turns for the same chat are
strict FIFO with at most one active turn, turns for different chats may run
concurrently under a global ``MAX_CONCURRENT_TURNS`` bound, and each
per-chat pending queue is bounded. ``/new`` travels through the same per-chat
queue as ordinary turns, so it is ordered relative to them and never clears
a thread while an older turn is still using it.

Privacy: logs carry only counts, categories and lengths, never prompts,
user text, summaries, model outputs, corpus text or raw chat identifiers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from aa.config import Settings
from aa.control.readiness import (
    CONTROL_ISSUE_NUMBER,
    RuntimeIdentity,
    assert_marker_privacy_safe,
    current_timestamp_seconds,
    format_ready_marker,
    format_startup_failed_marker,
    resolve_runtime_identity,
    validate_category,
)
from aa.control.runtime_control import RuntimeController
from aa.conversation.graph_runtime import GraphRuntimeError, GraphTurnRuntime
from aa.conversation.output_limits import (
    compact_text_to_envelope,
    envelope_passes,
)
from aa.conversation.turn_pipeline import (
    NATURAL_CLARIFICATION_REPLY,
    NATURAL_RETRY_REPLY,
)
from aa.corpus.context import CorpusContext
from aa.opencode.errors import OpenCodeError, OpenCodeRateLimitError
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
    TelegramVoiceReply,
)
from aa.telegram.tts import (
    DEFAULT_VOICE,
    FfmpegOpusEncoder,
    SileroSynthesizer,
    TtsError,
    TtsPipeline,
    build_tts_pipeline,
    compact_voice_text_to_policy,
    resolve_tts_voice,
    voice_for_presentation,
    voice_policy_passes,
)
from aa.telegram.typing import TypingHeartbeat
from aa.telegram.voice import (
    FfmpegDecoder,
    GigaAMRecognizer,
    VoiceError,
    VoicePipeline,
    build_pipeline,
    voice_error_reply,
)
from aa.telegram.voice_presentation import (
    PresentationError,
    VoicePresentationClassifier,
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
        grounding: Any | None = None,
        voice_pipeline: VoicePipeline | None = None,
        tts_pipeline: TtsPipeline | None = None,
        presentation_classifier: VoicePresentationClassifier | None = None,
        graph_runtime: GraphTurnRuntime | None = None,
        readiness_publisher: Callable[[str], Awaitable[None]] | None = None,
        runtime_identity: RuntimeIdentity | None = None,
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
        # Kept for constructor compatibility; the LangGraph verifier owns
        # grounding after the cutover and this handle is never consulted.
        self.grounding = grounding
        self._index: HybridIndex | None = None
        self._index_error: str | None = None
        self._graph_runtime: GraphTurnRuntime | None = graph_runtime
        # Local voice recognition (issue #76): loaded once per worker and
        # reused for all turns. ``None`` means the voice capability is
        # unavailable; the text poller is unaffected.
        self._voice_pipeline: VoicePipeline | None = voice_pipeline
        # Local TTS replies (issue #77): Silero loaded once per worker and
        # reused. ``None`` means voice replies fall back to text; the
        # text poller is unaffected.
        self._tts_pipeline: TtsPipeline | None = tts_pipeline
        # Ephemeral acoustic routing (issue #78): loaded once per worker.
        # ``None`` (or any classifier error) deterministically defaults to
        # ``xenia``. The result lives only in a per-turn local variable,
        # never in sessions/history/profiles/logs.
        self._presentation_classifier: VoicePresentationClassifier | None = presentation_classifier
        if (
            self._presentation_classifier is not None
            and self._voice_pipeline is not None
            and getattr(self._voice_pipeline, "presentation_classifier", None) is None
        ):
            try:
                self._voice_pipeline.presentation_classifier = self._presentation_classifier
            except Exception:
                pass
        self._running = False
        self._fatal_error: BaseException | None = None
        # Authoritative readiness control plane (issue #144): READY is
        # published to control issue #31 only after long polling is live.
        # Tests inject an in-memory publisher; production uses the default
        # GitHub issue publisher (best-effort, privacy-safe).
        self._readiness_publisher = readiness_publisher
        self._runtime_identity = runtime_identity
        self.readiness_marker: str | None = None
        self.startup_failure_marker: str | None = None
        # Transport/orchestration only: the dispatcher calls the LangGraph
        # production boundary via ``respond`` and never creates another
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

    @property
    def graph_runtime(self) -> GraphTurnRuntime | None:
        """The bound LangGraph turn runtime (``None`` before start)."""
        return self._graph_runtime

    @property
    def voice_available(self) -> bool:
        """Whether local voice recognition is ready for turns."""
        pipeline = self._voice_pipeline
        if pipeline is None or pipeline.recognizer is None:
            return False
        return pipeline.recognizer.available

    @property
    def tts_available(self) -> bool:
        """Whether local TTS voice replies are ready for voice turns."""
        pipeline = self._tts_pipeline
        if pipeline is None:
            return False
        return pipeline.synthesizer.available

    async def _init_voice_capability(self) -> None:
        """Load the pinned GigaAM recognizer once; fail voice-only on error."""
        if self._voice_pipeline is not None:
            logger.info("voice capability ready", extra={"injected": True})
            return
        if not isinstance(self.transport, PollingTelegramTransport):
            logger.info("voice capability disabled without polling transport")
            return
        if not self.settings.telegram_bot_token:
            logger.info("voice capability disabled without bot token")
            return
        try:
            recognizer = GigaAMRecognizer(Path(self.settings.aa_voice_model_dir))
            await asyncio.to_thread(recognizer.ensure_loaded)
        except VoiceError as exc:
            logger.warning("voice capability disabled", extra={"category": exc.category})
            return
        except Exception:
            logger.warning("voice capability disabled", extra={"category": "asr-unavailable"})
            return
        transport = self.transport

        class _TransportFetcher:
            async def fetch(self, file_id: str) -> bytes:
                return await transport.fetch_voice_bytes(file_id)

        self._voice_pipeline = build_pipeline(
            fetcher=_TransportFetcher(),
            decoder=FfmpegDecoder(),
            recognizer=recognizer,
            presentation_classifier=self._presentation_classifier,
        )
        logger.info("voice capability ready", extra={"injected": False})

    async def _init_presentation_capability(self) -> None:
        """Load the pinned presentation classifier once; default to xenia on error."""
        if self._presentation_classifier is not None:
            logger.info("presentation capability ready", extra={"injected": True})
            self._attach_presentation_classifier()
            return
        if not isinstance(self.transport, PollingTelegramTransport):
            logger.info("presentation capability disabled without polling transport")
            return
        if not self.settings.telegram_bot_token:
            logger.info("presentation capability disabled without bot token")
            return
        try:
            classifier = VoicePresentationClassifier(
                Path(self.settings.aa_voice_presentation_model_path)
            )
            await asyncio.to_thread(classifier.ensure_loaded)
        except PresentationError as exc:
            logger.warning("presentation capability disabled", extra={"category": exc.category})
            return
        except Exception:
            logger.warning(
                "presentation capability disabled", extra={"category": "model-unavailable"}
            )
            return
        self._presentation_classifier = classifier
        self._attach_presentation_classifier()
        logger.info("presentation capability ready", extra={"injected": False})

    def _attach_presentation_classifier(self) -> None:
        """Attach the classifier to the voice pipeline without persisting turns."""
        pipeline = self._voice_pipeline
        if pipeline is None or self._presentation_classifier is None:
            return
        try:
            if getattr(pipeline, "presentation_classifier", None) is None:
                pipeline.presentation_classifier = self._presentation_classifier
        except Exception:
            pass

    async def _init_tts_capability(self) -> None:
        """Load the pinned Silero synthesizer once; fallback to text on error."""
        if self._tts_pipeline is not None:
            logger.info("tts capability ready", extra={"injected": True})
            return
        if not isinstance(self.transport, PollingTelegramTransport):
            logger.info("tts capability disabled without polling transport")
            return
        if not self.settings.telegram_bot_token:
            logger.info("tts capability disabled without bot token")
            return
        try:
            synthesizer = SileroSynthesizer(Path(self.settings.aa_tts_model_path))
            await asyncio.to_thread(synthesizer.ensure_loaded)
        except TtsError as exc:
            logger.warning("tts capability disabled", extra={"category": exc.category})
            return
        except Exception:
            logger.warning("tts capability disabled", extra={"category": "tts-unavailable"})
            return
        self._tts_pipeline = build_tts_pipeline(
            synthesizer=synthesizer,
            encoder=FfmpegOpusEncoder(),
        )
        logger.info("tts capability ready", extra={"injected": False})

    def _transport_polling_live(self) -> bool:
        """Whether Telegram long polling is actually live (not merely starting).

        READY must be emitted only after the polling task is live: the
        transport reports running and, for the polling transport, holds an
        active poll task. A GitHub Actions step that is merely
        ``in_progress`` never satisfies this.
        """
        try:
            if not self.transport.running:
                return False
        except Exception:
            return False
        poll_task = getattr(self.transport, "_poll_task", None)
        if poll_task is None:
            # Stub/offline transports have no background task; running is live.
            return True
        try:
            return not bool(poll_task.done())
        except Exception:
            return False

    @staticmethod
    def _categorize_startup_failure(exc: BaseException) -> str:
        """Map a startup exception to a bounded privacy-safe category."""
        from aa.opencode.errors import (
            OpenCodeNotReadyError,
            OpenCodeRateLimitError,
            OpenCodeStartupError,
            OpenCodeTimeoutError,
        )
        from aa.telegram.transport import TelegramAuthError

        if isinstance(exc, OpenCodeRateLimitError):
            return "opencode-429"
        if isinstance(exc, (OpenCodeNotReadyError, OpenCodeTimeoutError)):
            return "opencode-not-ready"
        if isinstance(exc, OpenCodeStartupError):
            return "opencode-startup"
        if isinstance(exc, TelegramAuthError):
            return "telegram-auth"
        if isinstance(exc, TelegramApiError):
            return "telegram-bootstrap"
        if isinstance(exc, ValueError):
            message = str(exc).lower()
            if "bot_session_duration_seconds" in message or "telegram" in message:
                return "config-invalid"
            if "corpus" in message:
                return "corpus-unavailable"
            return "config-invalid"
        if exc.__class__.__name__ == "GraphRuntimeError":
            if "corpus" in str(exc).lower():
                return "corpus-unavailable"
            return "unknown"
        if "controller" in exc.__class__.__name__.lower():
            return "controller-start"
        return "unknown"

    async def _publish_readiness_text(self, marker: str) -> None:
        """Publish one readiness marker via the injected or default publisher."""
        assert_marker_privacy_safe(marker)
        publisher = self._readiness_publisher
        if publisher is not None:
            await publisher(marker)
            return
        await self._publish_marker_to_control_issue(marker)

    @staticmethod
    async def _publish_marker_to_control_issue(marker: str) -> None:
        """Best-effort default publisher to control issue #31 (GitHub API).

        Hermetic by default: unit tests and local runs never touch the
        network. Publication is enabled only when the bounded runtime sets
        ``AA_ENABLE_READY_PUBLISH=1`` (aa-runtime.yml, issues:write), which
        posts the privacy-safe marker as an issue comment. Failures are
        logged by category only and never carry secrets or message text.
        """
        enabled = os.environ.get("AA_ENABLE_READY_PUBLISH", "").strip().lower()
        if enabled not in ("1", "true", "yes"):
            logger.info("readiness marker kept local-only (publish not enabled)")
            return
        token = (os.environ.get("GITHUB_TOKEN", "") or os.environ.get("GH_TOKEN", "")).strip()
        repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
        if not token or not repository or "/" not in repository:
            logger.info("readiness marker not published (no GitHub context)")
            return
        owner, repo = repository.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{CONTROL_ISSUE_NUMBER}/comments"
        payload = json.dumps({"body": marker}).encode("utf-8")

        def _post() -> None:
            request = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "User-Agent": "aa-runtime-readiness/1",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                response.read()

        for attempt in (1, 2, 3):
            try:
                await asyncio.to_thread(_post)
                logger.info("readiness marker published", extra={"attempt": attempt})
                return
            except Exception:
                logger.warning("readiness marker publish failed", extra={"attempt": attempt})
                if attempt < 3:
                    await asyncio.sleep(float(attempt))
        logger.warning("readiness marker publish gave up after retries")

    async def _publish_ready(self) -> None:
        """Publish the READY marker for the current run (best-effort)."""
        identity = self._runtime_identity or resolve_runtime_identity()
        if identity is None:
            logger.info("readiness identity unavailable; READY kept local-only")
            return
        marker = format_ready_marker(
            run_id=identity.run_id,
            sha=identity.sha,
            ready_at=current_timestamp_seconds(),
            ordinal=identity.ordinal,
        )
        assert_marker_privacy_safe(marker)
        self.readiness_marker = marker
        try:
            await self._publish_readiness_text(marker)
        except Exception:
            logger.warning("READY publish failed; poller stays live")
        logger.info(
            "worker ready",
            extra={"run_id": identity.run_id, "ordinal": identity.ordinal},
        )

    async def _publish_startup_failed(self, exc: BaseException) -> None:
        """Publish a STARTUP_FAILED marker (best-effort, never masks ``exc``)."""
        try:
            category = validate_category(self._categorize_startup_failure(exc))
        except ValueError:
            category = "unknown"
        identity = self._runtime_identity or resolve_runtime_identity()
        if identity is None:
            return
        try:
            marker = format_startup_failed_marker(
                run_id=identity.run_id,
                sha=identity.sha,
                failed_at=current_timestamp_seconds(),
                ordinal=identity.ordinal,
                category=category,
            )
        except ValueError:
            return
        self.startup_failure_marker = marker
        try:
            await self._publish_readiness_text(marker)
        except Exception:
            logger.warning("STARTUP_FAILED publish failed", extra={"category": category})
        logger.warning("worker startup failed", extra={"category": category})

    async def start(self) -> None:
        """Start all components in dependency order.

        The bounded runtime clock is armed only after OpenCode and Telegram
        are ready, so bootstrap time never consumes the requested live window.
        A privacy-safe READY marker is published to control issue #31 only
        after long polling is live and the controller has started; startup
        failures before READY publish STARTUP_FAILED and invoke the bounded
        recovery path (notably the 429 runner-restart exit 75 upstream).
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
            # Voice capability loads once here and is reused for all turns.
            # Initialization failure disables voice only, never the poller.
            await self._init_presentation_capability()
            await self._init_voice_capability()
            self._attach_presentation_classifier()
            # TTS capability loads once here and is reused for voice replies.
            # Initialization failure falls back to text, never the poller.
            await self._init_tts_capability()
            await self._ensure_graph_runtime()
            await self.transport.start()
            # Start the requested fixed 5h window only after the poller
            # is live and all dependencies have completed bootstrap.
            await self.controller.start()
            if not self._transport_polling_live():
                raise TelegramApiError("telegram polling is not live after start")
        except Exception as exc:
            try:
                await self._publish_startup_failed(exc)
            except Exception:
                logger.warning("STARTUP_FAILED publish failed", extra={"category": "unknown"})
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
        await self._publish_ready()

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
            if self._graph_runtime is not None:
                try:
                    await self._graph_runtime.stop()
                except Exception:
                    pass
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
        if self._graph_runtime is not None:
            try:
                await self._graph_runtime.stop()
            except Exception:
                pass
        logger.info("worker stopped")

    async def run(self) -> None:
        """Run until the controller requests shutdown."""
        await self.start()
        try:
            while not self.controller.should_stop():
                await asyncio.sleep(0.05)
        finally:
            await self.stop()
        if self._fatal_error is not None:
            raise self._fatal_error

    def _wire_transport_handlers(self) -> None:
        """Connect the concrete polling transport to application behavior.

        Every accepted update (ordinary text, ``/start`` and ``/new``) is
        enqueued into the same per-chat dispatcher queue. The poller callback
        stays fast and never awaits a full graph turn, so one slow chat
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
        never await a full graph turn. Overflow backpressures with a
        single bounded reply instead of unbounded queue growth.
        """
        if not self.dispatcher.running:
            await self._process_dispatched_update(incoming)
            return
        try:
            await self.dispatcher.submit(incoming)
        except ChatQueueFullError:
            logger.warning("chat queue full; backpressure reply")
            try:
                await self.transport.send(TelegramReply(chat_id=incoming.chat_id, text=_BUSY_REPLY))
            except (TelegramApiError, TelegramEnvelopeError):
                logger.warning("backpressure reply delivery failed")

    async def _process_dispatched_update(self, incoming: TelegramIncoming) -> None:
        """Run one dispatched turn inside that chat's serialized worker.

        Safety routing precedes normal AA handling; ordinary turns run the
        LangGraph runtime as the single conversational path. ``/new``
        clears only that chat's graph thread state inside the same
        serialization, so it cannot interleave with an older turn for the
        same chat.
        """
        try:
            if incoming.command == "start":
                await self._handle_start_command(incoming)
                return
            if incoming.command == "new":
                await self._handle_new_command(incoming)
                return
            if incoming.voice is not None:
                await self._handle_voice_update(incoming)
                return
            await self._handle_telegram_update(incoming)
        except OpenCodeRateLimitError as exc:
            # Provider 429 is a worker lifecycle failure: stop accepting work,
            # let the main run loop unwind, and surface the fatal exception so
            # the hosted workflow can replace this runner.
            if self._fatal_error is None:
                self._fatal_error = exc
            await self.controller.stop()
            raise

    async def _handle_start_command(self, incoming: TelegramIncoming) -> None:
        await self.transport.send(TelegramReply(chat_id=incoming.chat_id, text=_START_REPLY))

    async def _handle_new_command(self, incoming: TelegramIncoming) -> None:
        try:
            runtime = self._graph_runtime
            if runtime is not None:
                await runtime.clear_chat(incoming.chat_id)
            # Local metrics generation advances; the graph thread holds the
            # authoritative conversation memory.
            self.sessions.reset(incoming.chat_id)
            reply = _NEW_REPLY
        except OpenCodeError:
            logger.warning("telegram new-session reset failed")
            reply = _TEMPORARY_ERROR_REPLY
        except Exception:
            logger.warning("telegram new-session reset failed")
            reply = _TEMPORARY_ERROR_REPLY
        await self.transport.send(TelegramReply(chat_id=incoming.chat_id, text=reply))

    async def _handle_telegram_update(self, incoming: TelegramIncoming) -> None:
        """Process one private text update and always emit a bounded reply.

        Exactly one Telegram message is delivered per update: overflow is
        never split into multiple messages. The LangGraph runtime owns the
        ordinary conversational path; this transport layer never sends a
        user message to OpenCode directly.
        """
        await self._respond_and_deliver(incoming, text=incoming.text, voice_input=False)

    async def _handle_voice_update(self, incoming: TelegramIncoming) -> None:
        """Process one Telegram voice note through ASR into the text boundary.

        The transcript enters the exact same LangGraph conversational turn
        boundary used by text messages, marked ``voice_input=True``. Any
        download/decode/ASR failure sends one short Russian text error to
        that user and leaves the poller alive. Temporary audio files are
        removed by the pipeline in ``finally``. Acoustic presentation is
        classified ephemerally from the already-decoded audio for
        opposite-voice TTS routing only; it is never logged, persisted,
        or exposed to the user.
        """
        attachment = incoming.voice
        if attachment is None:
            return
        pipeline = self._voice_pipeline
        if pipeline is None or not self.voice_available:
            logger.warning("voice turn without recognizer")
            await self._send_text_reply(incoming, voice_error_reply("voice-disabled"))
            return
        try:
            transcribe_with_presentation = getattr(
                pipeline, "transcribe_voice_with_presentation", None
            )
            if callable(transcribe_with_presentation):
                transcript, presentation = await transcribe_with_presentation(
                    file_id=attachment.file_id,
                    file_size_bytes=attachment.file_size_bytes,
                    duration_seconds=attachment.duration_seconds,
                )
            else:
                transcript = await pipeline.transcribe_voice(
                    file_id=attachment.file_id,
                    file_size_bytes=attachment.file_size_bytes,
                    duration_seconds=attachment.duration_seconds,
                )
                presentation = "unknown"
        except VoiceError as exc:
            logger.warning("voice turn failed", extra={"category": exc.category})
            await self._send_text_reply(incoming, voice_error_reply(exc.category))
            return
        except Exception:
            logger.warning("voice turn failed", extra={"category": "voice-failed"})
            await self._send_text_reply(incoming, voice_error_reply("voice-failed"))
            return
        # Ephemeral only: a per-turn local, never stored in sessions/history.
        await self._respond_and_deliver(
            incoming, text=transcript, voice_input=True, voice_presentation=presentation
        )

    def _typing_heartbeat(self, chat_id: int) -> TypingHeartbeat:
        """Build one turn's typing heartbeat on the configured cadence."""
        interval = float(self.settings.typing_heartbeat_seconds)
        if interval <= 0:
            interval = 4.0
        return TypingHeartbeat(self.transport, chat_id, interval_seconds=interval)

    async def _respond_and_deliver(
        self,
        incoming: TelegramIncoming,
        *,
        text: str,
        voice_input: bool,
        voice_presentation: str | None = None,
    ) -> None:
        """Run :meth:`respond` for one turn and deliver exactly one reply.

        The typing heartbeat starts immediately after the accepted normal
        turn begins processing and stops only after Telegram confirms final
        outbound delivery (or the turn is definitively aborted with no
        outbound message). Retried delivery keeps the heartbeat alive;
        generation finishing never stops it early.

        Voice turns (``voice_input=True``) request voice output: the
        already-generated answer is synthesized locally and delivered via
        ``sendVoice``. Any TTS/encoding/delivery failure deterministically
        falls back to the same answer as text; the response is never
        dropped. Text turns always receive text output. ``voice_presentation``
        is an ephemeral acoustic routing signal for the current turn only
        (never persisted); ``None``/``unknown``/error defaults to ``xenia``.
        """
        self.sessions.record_message(incoming.chat_id)
        # Safety/commands bypass the heartbeat: only an accepted normal
        # turn starts typing. The safety check here mirrors respond() so an
        # emergency/blocked turn never starts a heartbeat task.
        decision = self.safety.check(text).decision
        if decision is SafetyDecision.EMERGENCY or decision is SafetyDecision.BLOCK:
            try:
                reply = await self.respond(incoming.chat_id, text, voice_input=voice_input)
            except OpenCodeRateLimitError:
                raise
            except Exception:
                reply = _TEMPORARY_ERROR_REPLY
            if not reply.strip():
                return
            if voice_input:
                delivered = await self._send_voice_reply(
                    incoming, reply, voice_presentation=voice_presentation
                )
                if delivered:
                    return
            await self._send_text_reply(incoming, reply)
            return
        if text.strip().startswith("/"):
            try:
                reply = await self.respond(incoming.chat_id, text, voice_input=voice_input)
            except OpenCodeRateLimitError:
                raise
            except Exception:
                reply = _TEMPORARY_ERROR_REPLY
            if not reply.strip():
                return
            await self._send_text_reply(incoming, reply)
            return
        heartbeat = self._typing_heartbeat(incoming.chat_id)
        await heartbeat.start()
        try:
            try:
                reply = await self.respond(incoming.chat_id, text, voice_input=voice_input)
            except OpenCodeRateLimitError:
                raise
            except Exception:
                logger.warning("telegram message processing failed")
                reply = _TEMPORARY_ERROR_REPLY
            if not reply.strip():
                return
            # Delivery keeps the heartbeat alive: it stops only after
            # confirmed delivery (or definitive abort with no message).
            if voice_input:
                delivered = await self._send_voice_reply(
                    incoming, reply, voice_presentation=voice_presentation
                )
                if delivered:
                    return
            await self._send_text_reply(incoming, reply)
        finally:
            await heartbeat.stop()

    def _resolve_voice_for_turn(self, presentation: str | None = None) -> str:
        """Resolve the TTS voice for one turn (issue #78 opposite-voice rule).

        ``male-presenting`` -> ``xenia``, ``female-presenting`` ->
        ``eugene``, and ``unknown``/``None``/error -> ``xenia``. Only
        ``xenia``/``eugene`` are ever returned. The presentation is an
        ephemeral acoustic signal for the current turn only and is never
        logged, persisted, or exposed to the user.
        """
        voice = voice_for_presentation(presentation)
        return resolve_tts_voice(voice)

    async def _send_voice_reply(
        self,
        incoming: TelegramIncoming,
        reply: str,
        *,
        voice_presentation: str | None = None,
    ) -> bool:
        """Synthesize and deliver one voice reply; ``False`` means fallback.

        Returns ``True`` when the voice was delivered via ``sendVoice``.
        Any TTS, encoding, or delivery failure logs only categories/sizes
        and returns ``False`` so the caller sends the same answer as text.
        The presentation routing signal is ephemeral and never logged.
        """
        pipeline = self._tts_pipeline
        if pipeline is None or not self.tts_available:
            logger.info("voice reply fallback to text")
            return False
        voice = self._resolve_voice_for_turn(voice_presentation)
        if voice not in (DEFAULT_VOICE, "eugene"):
            voice = DEFAULT_VOICE
        try:
            ogg_bytes = await pipeline.synthesize_voice_ogg(reply, voice)
        except TtsError as exc:
            logger.warning(
                "voice reply synthesis failed; fallback to text",
                extra={"category": exc.category},
            )
            return False
        except Exception:
            logger.warning(
                "voice reply synthesis failed; fallback to text",
                extra={"category": "tts-failed"},
            )
            return False
        try:
            await self.transport.send_voice(
                TelegramVoiceReply(chat_id=incoming.chat_id, voice_bytes=ogg_bytes)
            )
        except (TelegramApiError, TelegramEnvelopeError):
            logger.warning("voice reply delivery failed; fallback to text")
            return False
        except Exception:
            logger.warning("voice reply delivery failed; fallback to text")
            return False
        logger.info("voice reply delivered")
        return True

    async def _send_text_reply(self, incoming: TelegramIncoming, reply: str) -> None:
        """Deliver one bounded text reply with envelope fallback handling."""
        try:
            await self.transport.send(TelegramReply(chat_id=incoming.chat_id, text=reply))
        except TelegramEnvelopeError:
            logger.warning("telegram outbound reply blocked by envelope guard")
            try:
                await self.transport.send(
                    TelegramReply(chat_id=incoming.chat_id, text=_TEMPORARY_ERROR_REPLY)
                )
            except TelegramApiError:
                logger.warning("telegram fallback reply delivery failed")
        except TelegramApiError:
            logger.warning("telegram outbound send failed")

    def _turn_index(self) -> HybridIndex:
        """Open (once) the RU-first hybrid index or fail closed.

        The opened index is reused across ordinary turns; provider
        failures never invalidate or rebuild it here.
        """
        if self._index is not None:
            return self._index
        if self._index_error is not None:
            raise GraphRuntimeError("corpus-unavailable", self._index_error)
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
            raise GraphRuntimeError("corpus-unavailable", self._index_error) from exc

    async def _ensure_graph_runtime(self) -> None:
        """Bind and start the LangGraph turn runtime (authoritative memory)."""
        if self._graph_runtime is not None:
            if not self._graph_runtime.running:
                await self._graph_runtime.start()
            return
        try:
            index: HybridIndex | None = self._turn_index()
        except GraphRuntimeError:
            index = None
        from aa.conversation.graph_runtime import build_production_runtime

        runtime = build_production_runtime(
            client=self.opencode_runtime.client,
            settings=self.settings,
            index=index,
        )
        await runtime.start()
        self._graph_runtime = runtime

    async def respond(self, chat_id: int, text: str, *, voice_input: bool = False) -> str:
        """Answer one inbound message with emergency precedence.

        Ordinary turns use only the LangGraph runtime: the graph thread for
        ``chat_id`` owns conversation memory, and text/voice transcripts
        share that state. ``voice_input`` marks turns transcribed from
        Telegram voice notes and requests the existing voice-mode brevity
        (compact leading sentences to <=4/<=80) on the already-generated
        grounded text.

        The deterministic safety layer runs first: an emergency turn returns
        the bounded Russian safe reply immediately with no graph work (the
        LLM never decides the emergency route). Blocked (empty) turns raise
        ``ValueError``. Application commands (``/start``/``/new`` text) are
        answered as ordinary conversational turns here; the dispatcher owns
        the real ``/new`` control event that clears thread state.

        Internal failures stay internal: provider/retrieval failures yield a
        natural Russian continuation/clarification, never mechanics and
        never a technical fail-closed reply.

        Only lengths and routing decisions are logged, never message bodies.
        Every returned reply is confined to the #83 hard Telegram envelope.
        Replies are returned as a single message; overflow is never split.
        """
        logger.info("turn started", extra={"voice_input": voice_input, "text_len": len(text)})
        result, _emergency_reply = self.safety.route(text)
        if result.decision is SafetyDecision.EMERGENCY and result.classification is not None:
            # Production Telegram runtime is RU-only: the emergency reply
            # is always the deterministic Russian template, regardless of
            # the detected input language. No English fallback may leak.
            logger.info("emergency response served", extra={"reason": result.reason})
            return self._fit_envelope(
                build_emergency_response(result.classification, language="ru")
            )
        if result.decision is SafetyDecision.BLOCK:
            logger.info("blocked message refused")
            raise ValueError("refusing to answer an empty message")
        if not self._graph_runtime or not self._graph_runtime.running:
            await self._ensure_graph_runtime()
        assert self._graph_runtime is not None
        try:
            reply = await self._graph_runtime.run_turn(chat_id, text)
        except OpenCodeRateLimitError:
            raise
        except GraphRuntimeError as exc:
            logger.warning("graph turn used natural fallback", extra={"category": exc.category})
            reply = NATURAL_RETRY_REPLY
        except OpenCodeError:
            logger.warning("graph turn used natural fallback")
            reply = NATURAL_RETRY_REPLY
        except ValueError:
            raise
        except Exception:
            logger.warning("graph turn used natural fallback")
            reply = NATURAL_RETRY_REPLY
        if voice_input and reply.strip():
            reply = self._apply_voice_brevity(reply)
        if not reply.strip():
            reply = NATURAL_CLARIFICATION_REPLY
        logger.info("normal response served")
        return self._fit_envelope(reply)

    def _apply_voice_brevity(self, reply: str) -> str:
        """Apply the existing #77 voice-mode brevity to grounded text."""
        if voice_policy_passes(reply) and envelope_passes(reply):
            return reply
        compacted = compact_voice_text_to_policy(reply)
        if compacted.strip() and envelope_passes(compacted):
            logger.info("voice reply compacted to policy")
            return compacted
        return self._fit_envelope(compacted if compacted.strip() else reply)

    @staticmethod
    def _fit_envelope(reply: str) -> str:
        """Confine ``reply`` to the hard envelope (complete-unit safe).

        The turn pipeline already applies compact regeneration upstream;
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
