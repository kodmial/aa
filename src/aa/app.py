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
from aa.conversation.failures import SERVICE_ERROR_REPLY, is_service_error
from aa.conversation.graph_runtime import GraphRuntimeError, GraphTurnRuntime
from aa.conversation.output_limits import (
    compact_text_to_envelope,
    envelope_passes,
)
from aa.corpus.context import CorpusContext
from aa.meeting_invitation.service import MeetingService
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
    resolve_tts_voice,
    voice_for_presentation,
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
# Issue #301: typed unsuccessful outcomes surface here as a clearly
# identified service error, never as synthetic AA conversation. This
# text carries the service-error marker so qualifiers/tests never
# count it as a substantive answer.
_TEMPORARY_ERROR_REPLY = SERVICE_ERROR_REPLY
_BUSY_REPLY = "Сейчас много сообщений. Попробуйте ещё раз через минуту."
# Issue #327: the voluntary meeting offer text (allowed UI/protocol string).
# It is sent only through the model-gated offer_meeting() helper, never
# as an automatic footer on certified answers.
_MEETING_OFFER_TEXT = "Если хочешь, могу помочь найти собрание. Это добровольно."


def _is_allowed_control_reply(text: str) -> bool:
    """Whether ``text`` is a typed non-answer control reply (no certificate).

    Only explicitly classified control templates may bypass book
    certification: the bounded emergency responses and the bounded voice
    failure replies. Everything else without a certificate fails closed.
    """
    try:
        from aa.safety.response import EMERGENCY_RESPONSE_EN, EMERGENCY_RESPONSE_RU

        if text in (EMERGENCY_RESPONSE_RU, EMERGENCY_RESPONSE_EN):
            return True
    except Exception:
        pass
    try:
        from aa.telegram.voice import (
            VOICE_EMPTY_REPLY,
            VOICE_ERROR_REPLY,
            VOICE_TOO_LARGE_REPLY,
            VOICE_UNAVAILABLE_REPLY,
        )

        if text in (
            VOICE_ERROR_REPLY,
            VOICE_EMPTY_REPLY,
            VOICE_TOO_LARGE_REPLY,
            VOICE_UNAVAILABLE_REPLY,
        ):
            return True
    except Exception:
        pass
    return False


def _is_voiceable_control_reply(text: str) -> bool:
    """Whether ``text`` is a deterministic control template safe to voice.

    Only the bounded emergency responses may be synthesized: they carry
    no book certificate by design and are tamper-evident by exact
    allow-list match. Voice failure replies stay text-only.
    """
    try:
        from aa.safety.response import EMERGENCY_RESPONSE_EN, EMERGENCY_RESPONSE_RU

        if text in (EMERGENCY_RESPONSE_RU, EMERGENCY_RESPONSE_EN):
            return True
    except Exception:
        pass
    return False


# Explicit runtime lifecycle states (issue #145). READY is published only
# after OpenCode health + Telegram bootstrap + long polling are all live;
# /bot status distinguishes STARTING (bootstrap in progress) from READY.
RUNTIME_STARTING = "STARTING"
RUNTIME_READY = "READY"
RUNTIME_STOPPING = "STOPPING"
RUNTIME_STOPPED = "STOPPED"
RUNTIME_FAILED = "FAILED"


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
        # Explicit lifecycle state: STARTING during bootstrap, READY only
        # after OpenCode health + Telegram bootstrap + long polling live.
        self._runtime_state = RUNTIME_STOPPED
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
        # Meeting wizard flows (issue #327): one live flow per chat, kept
        # for the worker lifetime. A restart rotates callback secrets and
        # generations, so former buttons fail inert as stale.
        self.meetings = MeetingService()
        self._wire_transport_handlers()

    @property
    def running(self) -> bool:
        """Whether the application is running."""
        return self._running

    @property
    def runtime_state(self) -> str:
        """Explicit lifecycle state: STARTING/READY/STOPPING/STOPPED/FAILED."""
        return self._runtime_state

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
        READY is published only after OpenCode health, Telegram bootstrap
        (getMe/deleteWebhook/commands) and long polling are all live.
        A privacy-safe READY marker is published to control issue #31 only
        after long polling is live and the controller has started; startup
        failures before READY publish STARTUP_FAILED and invoke the bounded
        recovery path (notably the 429 runner-restart exit 75 upstream).
        """
        if self._running:
            return
        self._runtime_state = RUNTIME_STARTING
        self.settings.validate(require_bot_token=False)
        logger.info("starting worker", extra={"config": self.settings.to_safe_dict()})
        try:
            await self.corpus.load()
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
            # Long polling is live only after transport.start() resolves
            # its bootstrap (getMe -> deleteWebhook -> commands) and the
            # poll loop task is running.
            # Start the requested fixed 5h window only after the poller
            # is live and all dependencies have completed bootstrap.
            await self.controller.start()
            if not self._transport_polling_live():
                raise TelegramApiError("telegram polling is not live after start")
        except Exception as exc:
            self._runtime_state = RUNTIME_FAILED
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
        self._runtime_state = RUNTIME_READY
        logger.info("worker started", extra={"runtime_state": RUNTIME_READY})
        await self._publish_ready()

    async def stop(self) -> None:
        """Stop all components in reverse order (idempotent)."""
        if not self._running:
            # Still ensure subcomponents are stopped for partial startups.
            self._runtime_state = RUNTIME_STOPPING
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
            if self._runtime_state != RUNTIME_FAILED:
                self._runtime_state = RUNTIME_STOPPED
            return
        logger.info("stopping worker")
        self._running = False
        self._runtime_state = RUNTIME_STOPPING
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
        if self._runtime_state != RUNTIME_FAILED:
            self._runtime_state = RUNTIME_STOPPED
        logger.info("worker stopped", extra={"runtime_state": self._runtime_state})

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
            if incoming.is_callback and incoming.callback_id:
                # Callback-only busy/retry notice: no false consent or
                # decline state, keyboard stays active for another tap.
                try:
                    await self.transport.answer_callback(str(incoming.callback_id))
                except (TelegramApiError, TelegramEnvelopeError):
                    logger.warning("callback backpressure acknowledgement failed")
                except Exception:
                    logger.warning("callback backpressure acknowledgement failed")
                return
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
        same chat. Callback button presses never invoke the model.
        """
        try:
            if incoming.is_callback:
                await self._handle_callback_update(incoming)
                return
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

    async def offer_meeting(self, chat_id: int) -> bool:
        """Send one voluntary meeting offer with two buttons (model-gated).

        Callers invoke this only after a book-certified relevant turn and
        a positive policy decision; this method never decides relevance
        itself and never inspects message text. Delivery binds the offer:
        only a confirmed send activates the keyboard.
        """
        try:
            flow = self.meetings.begin_offer(int(chat_id))
        except Exception:
            return False
        try:
            sent_id = await self.transport.send(
                TelegramReply(chat_id=int(chat_id), text=_MEETING_OFFER_TEXT, reply_markup=None)
            )
        except Exception:
            try:
                self.meetings.confirm_offer_delivery(int(chat_id), -1, "failed")
            except Exception:
                pass
            _ = flow
            return False
        if sent_id is None:
            try:
                self.meetings.confirm_offer_delivery(int(chat_id), -1, "failed")
            except Exception:
                pass
            return False
        try:
            result = self.meetings.activate_offer(int(chat_id), int(sent_id))
        except Exception:
            return False
        try:
            await self.transport.edit_reply_markup(
                int(chat_id), int(sent_id), dict(result.keyboard)
            )
        except Exception:
            try:
                self.meetings.confirm_offer_delivery(int(chat_id), int(sent_id), "failed")
            except Exception:
                pass
            return False
        try:
            self.meetings.confirm_offer_delivery(int(chat_id), int(sent_id), "confirmed")
        except Exception:
            return False
        return True

    async def _handle_new_command(self, incoming: TelegramIncoming) -> None:
        try:
            runtime = self._graph_runtime
            if runtime is not None:
                await runtime.clear_chat(incoming.chat_id)
            # Local metrics generation advances; the graph thread holds the
            # authoritative conversation memory.
            self.sessions.reset(incoming.chat_id)
            try:
                self.meetings.handle_new(incoming.chat_id)
            except Exception:
                pass
            reply = _NEW_REPLY
        except OpenCodeError:
            logger.warning("telegram new-session reset failed")
            reply = _TEMPORARY_ERROR_REPLY
        except Exception:
            logger.warning("telegram new-session reset failed")
            reply = _TEMPORARY_ERROR_REPLY
        await self.transport.send(TelegramReply(chat_id=incoming.chat_id, text=reply))

    async def _handle_callback_update(self, incoming: TelegramIncoming) -> None:
        """Process one wizard button press with zero model calls."""
        if incoming.callback_inaccessible or not incoming.callback_data:
            return
        try:
            outcome = self.meetings.handle_callback(
                int(incoming.chat_id),
                int(incoming.sender_id if incoming.sender_id is not None else incoming.chat_id),
                int(incoming.callback_message_id)
                if incoming.callback_message_id is not None
                else int(incoming.message_id),
                str(incoming.callback_data),
            )
        except Exception:
            logger.warning("meeting callback handling failed")
            return
        if outcome.stale or outcome.duplicate or not outcome.accepted:
            return
        try:
            if outcome.remove_keyboard and incoming.callback_message_id is not None:
                await self.transport.edit_reply_markup(
                    int(incoming.chat_id), int(incoming.callback_message_id), None
                )
            elif outcome.edit_keyboard is not None and incoming.callback_message_id is not None:
                await self.transport.edit_reply_markup(
                    int(incoming.chat_id),
                    int(incoming.callback_message_id),
                    dict(outcome.edit_keyboard),
                )
            if outcome.send_text is not None and outcome.send_text.strip():
                sent_id = await self.transport.send(
                    TelegramReply(
                        chat_id=int(incoming.chat_id),
                        text=outcome.send_text,
                        reply_markup=dict(outcome.send_keyboard)
                        if outcome.send_keyboard is not None
                        else None,
                    )
                )
                if sent_id is not None:
                    try:
                        self.meetings.note_sent_message(int(incoming.chat_id), int(sent_id))
                    except Exception:
                        pass
        except (TelegramApiError, TelegramEnvelopeError):
            logger.warning("meeting callback UI delivery failed")
        except Exception:
            logger.warning("meeting callback UI delivery failed")

    async def _handle_telegram_update(self, incoming: TelegramIncoming) -> None:
        """Process one private text update and always emit a bounded reply.

        Exactly one Telegram message is delivered per update: overflow is
        never split into multiple messages. The LangGraph runtime owns the
        ordinary conversational path; this transport layer never sends a
        user message to OpenCode directly.
        """
        try:
            wizard = self.meetings.handle_text(int(incoming.chat_id), str(incoming.text))
        except Exception:
            wizard = None
        if wizard is not None and wizard.handled:
            try:
                if wizard.send_text is not None and wizard.send_text.strip():
                    sent_id = await self.transport.send(
                        TelegramReply(
                            chat_id=int(incoming.chat_id),
                            text=str(wizard.send_text),
                            reply_markup=dict(wizard.send_keyboard)
                            if wizard.send_keyboard is not None
                            else None,
                        )
                    )
                    if sent_id is not None:
                        try:
                            self.meetings.note_sent_message(int(incoming.chat_id), int(sent_id))
                        except Exception:
                            pass
                if wizard.edit_keyboard is not None:
                    flow = self.meetings.flow_for(int(incoming.chat_id))
                    bound = flow.bound_message_id if flow is not None else None
                    if bound is not None:
                        await self.transport.edit_reply_markup(
                            int(incoming.chat_id), int(bound), dict(wizard.edit_keyboard)
                        )
            except (TelegramApiError, TelegramEnvelopeError):
                logger.warning("meeting city UI delivery failed")
            except Exception:
                logger.warning("meeting city UI delivery failed")
            return
        await self._respond_and_deliver(incoming, text=incoming.text, want_voice=False)

    async def _handle_voice_update(self, incoming: TelegramIncoming) -> None:
        """Process one Telegram voice note through ASR into the text boundary.

        The transcript enters the exact same LangGraph conversational
        turn boundary used by text messages (issue #294: voice is
        transport only, never a conversational mode). No audio flag or
        acoustic feature reaches generation, retrieval, planning,
        memory, grounding, verification or safety decisions. Any
        download/decode/ASR failure sends one short Russian text error
        to that user and leaves the poller alive. Temporary audio files
        are removed by the pipeline in ``finally``. Acoustic
        presentation is classified ephemerally from the already-decoded
        audio for opposite-voice TTS routing only; it is never logged,
        persisted, or exposed to the user.
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
        # ``want_voice`` selects the delivery format only (sendVoice vs
        # sendMessage); the conversational computation is identical.
        # A voice transcript may answer a pending city question: route it
        # through the offline wizard first with zero model calls. Directory
        # results still arrive as text/buttons, never as spoken URLs.
        try:
            wizard = self.meetings.handle_text(int(incoming.chat_id), str(transcript))
        except Exception:
            wizard = None
        if wizard is not None and wizard.handled:
            try:
                if wizard.send_text is not None and wizard.send_text.strip():
                    sent_id = await self.transport.send(
                        TelegramReply(
                            chat_id=int(incoming.chat_id),
                            text=str(wizard.send_text),
                            reply_markup=dict(wizard.send_keyboard)
                            if wizard.send_keyboard is not None
                            else None,
                        )
                    )
                    if sent_id is not None:
                        try:
                            self.meetings.note_sent_message(int(incoming.chat_id), int(sent_id))
                        except Exception:
                            pass
            except (TelegramApiError, TelegramEnvelopeError):
                logger.warning("meeting city UI delivery failed")
            except Exception:
                logger.warning("meeting city UI delivery failed")
            return
        await self._respond_and_deliver(
            incoming, text=transcript, want_voice=True, voice_presentation=presentation
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
        want_voice: bool,
        voice_presentation: str | None = None,
    ) -> None:
        """Run :meth:`respond` for one turn and deliver exactly one reply.

        The typing heartbeat starts immediately after the accepted normal
        turn begins processing and stops only after Telegram confirms final
        outbound delivery (or the turn is definitively aborted with no
        outbound message). Retried delivery keeps the heartbeat alive;
        generation finishing never stops it early.

        ``want_voice`` is transport only (issue #294): voice turns
        request voice output, so the already-generated final approved
        answer is synthesized locally and delivered via ``sendVoice``
        with no rewriting. Any TTS/encoding/delivery failure
        deterministically falls back to the byte-identical answer as
        text; the response is never dropped. Text turns always receive
        text output. ``voice_presentation`` is an ephemeral acoustic
        routing signal for the current turn only (never persisted);
        ``None``/``unknown``/error defaults to ``xenia``.
        """
        self.sessions.record_message(incoming.chat_id)
        # Safety/commands bypass the heartbeat: only an accepted normal
        # turn starts typing. The safety check here mirrors respond() so an
        # emergency/blocked turn never starts a heartbeat task.
        decision = self.safety.check(text).decision
        if decision is SafetyDecision.EMERGENCY or decision is SafetyDecision.BLOCK:
            try:
                reply = await self.respond(incoming.chat_id, text)
            except OpenCodeRateLimitError:
                raise
            except Exception:
                reply = _TEMPORARY_ERROR_REPLY
            if not reply.strip():
                return
            if want_voice:
                delivered = await self._send_voice_reply(
                    incoming, reply, voice_presentation=voice_presentation
                )
                if delivered:
                    return
            await self._send_text_reply(incoming, reply)
            return
        if text.strip().startswith("/"):
            try:
                reply = await self.respond(incoming.chat_id, text)
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
                reply = await self.respond(incoming.chat_id, text)
            except OpenCodeRateLimitError:
                raise
            except Exception:
                logger.warning("telegram message processing failed")
                reply = _TEMPORARY_ERROR_REPLY
            if not reply.strip():
                return
            # Delivery keeps the heartbeat alive: it stops only after
            # confirmed delivery (or definitive abort with no message).
            if want_voice:
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

        TTS receives exactly the final approved string with no
        rewriting (issue #294). Returns ``True`` when the voice was
        delivered via ``sendVoice``. Any TTS, encoding, or delivery
        failure logs only categories/sizes and returns ``False`` so
        the caller sends the byte-identical answer as text.
        The presentation routing signal is ephemeral and never logged.

        Transport acknowledgment seam (#304 for #312): successful
        ``sendVoice`` records a confirmed ``sendVoice`` receipt while a
        TTS-fallback-to-text records the voice attempt as failed and
        leaves text delivery receipts to ``_send_text_reply``. Telegram
        failure/partial delivery never claims full voice acceptance.
        """
        import uuid as _voice_uuid

        def _record_voice(status: str, channel: str) -> None:
            try:
                if self._graph_runtime is None:
                    return
                from aa.conversation.finalization import sha256_text as _vsha

                thread = self._graph_runtime.thread_id(incoming.chat_id)
                cert_store = self._graph_runtime.last_certificate_for_thread(thread)
                stored_certificate = (
                    cert_store.get("certificate", {}) if isinstance(cert_store, dict) else {}
                )
                certificate_id = (
                    str(stored_certificate.get("certificate_id", ""))
                    if isinstance(stored_certificate, dict)
                    else ""
                )
                try:
                    digest = _vsha(reply)
                except Exception:
                    digest = ""
                self._graph_runtime.record_delivery_receipts(
                    thread,
                    [
                        {
                            "turn_id": thread,
                            "certificate_id": certificate_id,
                            "final_sha256": digest,
                            "segment_index": 0,
                            "segment_count": 1,
                            "char_start": 0,
                            "char_end": len(reply),
                            "utf8_start": 0,
                            "utf8_end": len(reply.encode("utf-8")),
                            "status": status,
                            "channel": channel,
                            "retry_id": _voice_uuid.uuid4().hex,
                        }
                    ],
                )
            except Exception:
                pass

        pipeline = self._tts_pipeline
        if pipeline is None or not self.tts_available:
            logger.info("voice reply fallback to text")
            _record_voice("failed", "sendVoice")
            return False
        # Delivery gate re-verification (same as the text path): voice
        # must never bypass the stale/missing-certificate block. Only
        # the exact certified text may be synthesized; anything else
        # falls back to text (which fails closed) instead of sending.
        # Deterministic control templates (bounded emergency replies)
        # carry no book certificate by design and are tamper-evident by
        # exact allow-list match, so they may be voiced identically.
        try:
            from aa.conversation.finalization import (
                AnswerCandidate as _VCandidate,
            )
            from aa.conversation.finalization import (
                VerificationCertificate as _VCertificate,
            )
            from aa.conversation.finalization import (
                normalize_answer_text as _vnormalize,
            )
            from aa.conversation.finalization import (
                sha256_text as _vsha_check,
            )
            from aa.conversation.finalization import (
                verify_certificate as _vverify,
            )

            if _vnormalize(reply) != reply:
                logger.warning("voice delivery blocked: post-cert mutation")
                _record_voice("failed", "sendVoice")
                return False
            if is_service_error(reply):
                _record_voice("failed", "sendVoice")
                return False
            if not _is_voiceable_control_reply(reply):
                cert_store_v: dict[str, Any] = {}
                if self._graph_runtime is not None:
                    try:
                        thread_v = self._graph_runtime.thread_id(incoming.chat_id)
                        cert_store_v = self._graph_runtime.last_certificate_for_thread(thread_v)
                    except Exception:
                        cert_store_v = {}
                stored_candidate_v = (
                    cert_store_v.get("candidate", {}) if isinstance(cert_store_v, dict) else {}
                )
                stored_certificate_v = (
                    cert_store_v.get("certificate", {}) if isinstance(cert_store_v, dict) else {}
                )
                if (
                    not isinstance(stored_candidate_v, dict)
                    or not isinstance(stored_certificate_v, dict)
                    or not stored_certificate_v.get("answer_sha256", "")
                ):
                    logger.warning("voice delivery blocked: missing certificate")
                    _record_voice("failed", "sendVoice")
                    return False
                if _vsha_check(reply) != str(stored_certificate_v.get("answer_sha256", "")):
                    logger.warning("voice delivery blocked: stale certificate")
                    _record_voice("failed", "sendVoice")
                    return False
                _vverify(
                    candidate=_VCandidate.model_validate(stored_candidate_v),
                    certificate=_VCertificate.model_validate(stored_certificate_v),
                )
                if _VCandidate.model_validate(stored_candidate_v).text != reply:
                    logger.warning("voice delivery blocked: stale certificate")
                    _record_voice("failed", "sendVoice")
                    return False
        except Exception:
            logger.warning("voice delivery blocked: certification error")
            try:
                _record_voice("failed", "sendVoice")
            except Exception:
                pass
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
            _record_voice("failed", "sendVoice")
            return False
        except Exception:
            logger.warning(
                "voice reply synthesis failed; fallback to text",
                extra={"category": "tts-failed"},
            )
            _record_voice("failed", "sendVoice")
            return False
        try:
            await self.transport.send_voice(
                TelegramVoiceReply(chat_id=incoming.chat_id, voice_bytes=ogg_bytes)
            )
        except (TelegramApiError, TelegramEnvelopeError):
            logger.warning("voice reply delivery failed; fallback to text")
            _record_voice("failed", "sendVoice")
            return False
        except Exception:
            logger.warning("voice reply delivery failed; fallback to text")
            _record_voice("failed", "sendVoice")
            return False
        logger.info("voice reply delivered")
        _record_voice("confirmed", "sendVoice")
        # Voice/text equivalence (#312): the certified spoken text counts
        # as delivered once via sendVoice; a later text fallback never
        # double-counts it.
        try:
            if self._graph_runtime is not None:
                thread = self._graph_runtime.thread_id(incoming.chat_id)
                await self._graph_runtime.apply_delivery_commit(thread)
        except Exception:
            pass
        return True

    async def _send_text_reply(
        self, incoming: TelegramIncoming, reply: str
    ) -> list[dict[str, Any]]:
        """Deliver one certified reply and return serializable receipts.

        Certification is not delivery (#304 seam for #312): the exact
        certified text travels byte-for-byte; transport splitting happens
        after semantic finalization with order/text preservation proof.
        Each segment yields a ``DeliveryReceipt`` dict (confirmed/failed/
        unknown, ``sendMessage`` channel, retry identity). Partial sends
        never resend the full reply as complete history: an explicit
        retry signal follows instead. Service-error signals and emergency
        control replies bypass certification with their own typed policy.
        """
        import time as _time
        import uuid as _uuid

        started = _time.perf_counter()
        receipts: list[dict[str, Any]] = []

        def _record(receipts_in: list[dict[str, Any]]) -> None:
            try:
                if self._graph_runtime is not None:
                    thread = self._graph_runtime.thread_id(incoming.chat_id)
                    self._graph_runtime.record_delivery_receipts(thread, receipts_in)
            except Exception:
                pass

        async def _commit_confirmed_quotes() -> None:
            """Persist receipt-confirmed book quotes (best-effort, never fails delivery)."""
            try:
                runtime = self._graph_runtime
                if runtime is None:
                    return
                thread = runtime.thread_id(incoming.chat_id)
                await runtime.apply_delivery_commit(thread)
            except Exception:
                pass

        async def _send_single(text: str) -> None:
            await self.transport.send(TelegramReply(chat_id=incoming.chat_id, text=text))

        # Service-error and control replies are typed non-answers with
        # their own policy; they never consume a book certificate.
        if is_service_error(reply):
            try:
                await _send_single(reply)
            except (TelegramApiError, TelegramEnvelopeError):
                logger.warning("service-error signal delivery failed")
            except Exception:
                logger.warning("service-error signal delivery failed")
            turn_id = ""
            try:
                if self._graph_runtime is not None:
                    turn_id = self._graph_runtime.thread_id(incoming.chat_id)
            except Exception:
                turn_id = ""
            receipt = {
                "turn_id": turn_id,
                "certificate_id": "",
                "final_sha256": "",
                "segment_index": 0,
                "segment_count": 1,
                "char_start": 0,
                "char_end": len(reply),
                "utf8_start": 0,
                "utf8_end": len(reply.encode("utf-8")),
                "status": "unknown",
                "channel": "sendMessage",
                "retry_id": _uuid.uuid4().hex,
            }
            receipts.append(receipt)
            _record(receipts)
            return receipts

        # Certified path: mechanically re-verify digests before send; any
        # post-certification mutation fails closed without delivery.
        certificate_id = ""
        turn_id = ""
        try:
            from aa.conversation.finalization import (
                build_delivery_receipts as _build_receipts,
            )
            from aa.conversation.finalization import (
                normalize_answer_text as _normalize,
            )
            from aa.conversation.finalization import (
                sha256_text as _sha,
            )
            from aa.conversation.finalization import (
                split_certified_text as _split_certified,
            )

            normalized = _normalize(reply)
            if normalized != reply:
                logger.warning("telegram delivery blocked: post-cert mutation")
                try:
                    await _send_single(SERVICE_ERROR_REPLY)
                except Exception:
                    pass
                return receipts
            stage: dict[str, object] = {}
            cert_store: dict[str, Any] = {}
            if self._graph_runtime is not None:
                try:
                    turn_id = self._graph_runtime.thread_id(incoming.chat_id)
                    stage = self._graph_runtime.last_telemetry_for_thread(turn_id)
                    cert_store = self._graph_runtime.last_certificate_for_thread(turn_id)
                except Exception:
                    stage = {}
                    cert_store = {}
            _ = stage
            stored_candidate = cert_store.get("candidate", {})
            stored_certificate = cert_store.get("certificate", {})
            has_certificate = (
                isinstance(stored_candidate, dict)
                and isinstance(stored_certificate, dict)
                and bool(stored_certificate.get("answer_sha256", ""))
            )
            if not has_certificate:
                # Only explicitly classified control templates bypass
                # book certification. Any other missing/stale certificate
                # fails closed to the service error instead of sending
                # with an empty certificate id.
                if not _is_allowed_control_reply(reply):
                    logger.warning("telegram delivery blocked: missing certificate")
                    try:
                        await _send_single(SERVICE_ERROR_REPLY)
                    except Exception:
                        pass
                    return receipts
                # Explicitly classified protocol/service/command/emergency
                # control path (no book certificate by design): typed
                # policy handling, never a certified book answer. Send
                # single-message with the transport envelope guard and a
                # control receipt (empty certificate id).
                try:
                    from aa.conversation.output_limits import envelope_passes as _ctrl_env

                    if not bool(_ctrl_env(reply)):
                        logger.warning("control reply blocked by envelope guard")
                        try:
                            await _send_single(SERVICE_ERROR_REPLY)
                        except Exception:
                            pass
                        return receipts
                    await _send_single(reply)
                    status = "confirmed"
                except (TelegramApiError, TelegramEnvelopeError):
                    logger.warning("control reply delivery failed")
                    status = "failed"
                except Exception:
                    logger.warning("control reply delivery failed")
                    status = "failed"
                receipts.append(
                    {
                        "turn_id": turn_id,
                        "certificate_id": "",
                        "final_sha256": "",
                        "segment_index": 0,
                        "segment_count": 1,
                        "char_start": 0,
                        "char_end": len(reply),
                        "utf8_start": 0,
                        "utf8_end": len(reply.encode("utf-8")),
                        "status": status,
                        "channel": "sendMessage",
                        "retry_id": _uuid.uuid4().hex,
                    }
                )
                _record(receipts)
                return receipts
            try:
                certificate_id = str(stored_certificate.get("certificate_id", ""))
                if _sha(normalized) != str(stored_certificate.get("answer_sha256", "")):
                    logger.warning("telegram delivery blocked: stale certificate")
                    try:
                        await _send_single(SERVICE_ERROR_REPLY)
                    except Exception:
                        pass
                    return receipts
                # Full digest re-verification before send: text, evidence
                # bundle, and context snapshot must all match the stored
                # certificate, which must also carry a positive verdict.
                from aa.conversation.finalization import (
                    AnswerCandidate as _SendCandidate,
                )
                from aa.conversation.finalization import (
                    VerificationCertificate as _SendCertificate,
                )
                from aa.conversation.finalization import (
                    verify_certificate as _send_verify,
                )

                _send_verify(
                    candidate=_SendCandidate.model_validate(stored_candidate),
                    certificate=_SendCertificate.model_validate(stored_certificate),
                )
                if _SendCandidate.model_validate(stored_candidate).text != normalized or _sha(
                    normalized
                ) != str(_SendCertificate.model_validate(stored_certificate).answer_sha256):
                    logger.warning("telegram delivery blocked: stale certificate")
                    try:
                        await _send_single(SERVICE_ERROR_REPLY)
                    except Exception:
                        pass
                    return receipts
            except Exception:
                logger.warning("telegram delivery blocked: certification error")
                try:
                    await _send_single(SERVICE_ERROR_REPLY)
                except Exception:
                    pass
                return receipts
            segments = _split_certified(normalized)
            pending = _build_receipts(
                certified_text=normalized,
                segments=segments,
                certificate_id=certificate_id,
                turn_id=turn_id,
                channel="sendMessage",
                retry_id=_uuid.uuid4().hex,
                status="unknown",
            )
            sent_count = 0
            try:
                for segment in segments:
                    await _send_single(segment)
                    sent_count += 1
                confirmed = [
                    {**item, "status": "confirmed"}
                    for item in [r.model_dump(mode="json") for r in pending]
                ]
                receipts.extend(confirmed)
                _record(receipts)
                await _commit_confirmed_quotes()
                elapsed_ms = (_time.perf_counter() - started) * 1000.0
                logger.info(
                    "telegram delivery done",
                    extra={
                        "delivery_outcome": "sent-split" if len(segments) > 1 else "sent",
                        "delivery_latency_ms": round(elapsed_ms, 1),
                        "reply_len": len(reply),
                        "segments": len(segments),
                    },
                )
                return receipts
            except (TelegramApiError, TelegramEnvelopeError) as exc:
                failed = [
                    {**item, "status": ("confirmed" if i < sent_count else "failed")}
                    for i, item in enumerate([r.model_dump(mode="json") for r in pending])
                ]
                receipts.extend(failed)
                _record(receipts)
                await _commit_confirmed_quotes()
                if sent_count > 0:
                    logger.warning("telegram split delivery partial; full resend skipped")
                    try:
                        await self.transport.send(
                            TelegramReply(chat_id=incoming.chat_id, text=_TEMPORARY_ERROR_REPLY)
                        )
                    except Exception:
                        logger.warning("telegram split partial error-signal delivery failed")
                    return receipts
                logger.warning("telegram delivery failed", extra={"category": type(exc).__name__})
                return receipts
            except Exception:
                failed = [
                    {**item, "status": ("confirmed" if i < sent_count else "unknown")}
                    for i, item in enumerate([r.model_dump(mode="json") for r in pending])
                ]
                receipts.extend(failed)
                _record(receipts)
                await _commit_confirmed_quotes()
                if sent_count > 0:
                    logger.warning("telegram split delivery partial; full resend skipped")
                    try:
                        await self.transport.send(
                            TelegramReply(chat_id=incoming.chat_id, text=_TEMPORARY_ERROR_REPLY)
                        )
                    except Exception:
                        logger.warning("telegram split partial error-signal delivery failed")
                    return receipts
                return receipts
        except Exception:
            logger.warning("telegram delivery blocked: certification error")
            try:
                await _send_single(SERVICE_ERROR_REPLY)
            except Exception:
                pass
            return receipts

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

    async def respond(self, chat_id: int, text: str) -> str:
        """Answer one inbound message with emergency precedence.

        Ordinary turns use only the LangGraph runtime: the graph thread
        for ``chat_id`` owns conversation memory, and text/voice
        transcripts share that state through this single boundary
        (issue #294: voice is transport only, never a conversational
        mode). No voice flag or acoustic feature may reach this method;
        format choice (sendVoice vs sendMessage) lives in the Telegram
        delivery adapter only, after this method returns the final
        approved text.

        The deterministic safety layer runs first: an emergency turn returns
        the bounded Russian safe reply immediately with no graph work (the
        LLM never decides the emergency route). Blocked (empty) turns raise
        ``ValueError``. Application commands (``/start``/``/new`` text) are
        answered as ordinary conversational turns here; the dispatcher owns
        the real ``/new`` control event that clears thread state.

        Internal failures stay internal and are typed unsuccessful
        outcomes: provider/retrieval/verifier/timeout failures return
        the clearly marked service-error signal (never synthetic AA
        conversation and never a substantive answer). Every normal
        user-facing chat output is composed by the AA model from the
        user message and conversational state (issue #301).

        Only lengths and routing decisions are logged, never message bodies.
        Ordinary replies are confined to the #83 hard Telegram envelope per
        message. Issue #295: a fully verified grounded complete answer that
        exceeds one envelope is returned in full (same identical content for
        text and voice per #294); text delivery sends it as sequential
        envelope-passing segments, voice synthesizes the identical full
        text. Bulk/attack, unverified or unsafe drafts stay single-message
        (compacted/fallback, never split).
        """
        logger.info("turn started", extra={"text_len": len(text)})
        import time as _time

        turn_started = _time.perf_counter()
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
        service_error = False
        try:
            reply = await self._graph_runtime.run_turn(chat_id, text)
        except OpenCodeRateLimitError:
            raise
        except GraphRuntimeError as exc:
            logger.warning("graph turn failed as typed outcome", extra={"category": exc.category})
            reply = SERVICE_ERROR_REPLY
            service_error = True
        except OpenCodeError:
            logger.warning("graph turn failed as typed outcome")
            reply = SERVICE_ERROR_REPLY
            service_error = True
        except ValueError:
            raise
        except Exception:
            logger.warning("graph turn failed as typed outcome")
            reply = SERVICE_ERROR_REPLY
            service_error = True
        if not reply.strip():
            reply = SERVICE_ERROR_REPLY
            service_error = True
        # Final outbound safety guard (#252, defense in depth): the turn
        # pipeline already certifies its drafts, but no text reaches
        # Telegram without passing the outbound gate here either. A
        # harmful reply is never delivered: the turn fails as a typed
        # service-error outcome (never a canned conversational reply).
        try:
            from aa.safety.outbound import is_outbound_safe

            if reply.strip() and not is_service_error(reply) and not is_outbound_safe(reply):
                logger.info("outbound safety blocked delivery", extra={"fallback_used": True})
                reply = SERVICE_ERROR_REPLY
                service_error = True
        except Exception:
            logger.warning("outbound safety gate error: failing closed", exc_info=True)
            reply = SERVICE_ERROR_REPLY
            service_error = True
        is_service = is_service_error(reply)
        # Privacy-safe stage snapshot first: a verified split answer must
        # bypass single-message compaction (delivery splits it below).
        try:
            thread = self._graph_runtime.thread_id(chat_id)
            stage = self._graph_runtime.last_telemetry_for_thread(thread)
        except Exception:
            stage = {}
            thread = ""
        # Delivery-gate re-verification (#304): the exact text returned by
        # the runtime must match its certificate. Never strip/clip/
        # substitute after certification; stale or missing certificates
        # fail closed to the typed service error (never a borrowed
        # verdict, never silent shortening).
        if not service_error and not is_service:
            try:
                from aa.conversation.finalization import sha256_text as _sha_text

                cert_store: dict[str, Any] = {}
                try:
                    if thread:
                        cert_store = self._graph_runtime.last_certificate_for_thread(thread)
                except Exception:
                    cert_store = {}
                stored_certificate = cert_store.get("certificate", {})
                if not isinstance(stored_certificate, dict) or not stored_certificate:
                    logger.warning("app delivery blocked: missing certificate")
                    reply = SERVICE_ERROR_REPLY
                    service_error = True
                    is_service = True
                elif _sha_text(reply) != str(stored_certificate.get("answer_sha256", "")):
                    logger.warning("app delivery blocked: stale certificate")
                    reply = SERVICE_ERROR_REPLY
                    service_error = True
                    is_service = True
                else:
                    whole = stored_certificate.get("whole_answer_verdict", {})
                    if isinstance(whole, dict):
                        if not (
                            bool(whole.get("supported", False))
                            and bool(whole.get("coverage_ok", False))
                            and bool(whole.get("conditions_preserved", False))
                            and bool(whole.get("quote_ok", False))
                        ):
                            logger.warning("app delivery blocked: negative certificate")
                            reply = SERVICE_ERROR_REPLY
                            service_error = True
                            is_service = True
            except Exception:
                logger.warning("app delivery blocked: certificate error")
                reply = SERVICE_ERROR_REPLY
                service_error = True
                is_service = True
        if is_service or service_error:
            fitted = reply
        else:
            # Certified text travels byte-for-byte. Single-message
            # envelope payloads pass through; multi-message verified
            # answers split deterministically at the transport boundary
            # with preservation proof. Anything else fails closed to the
            # typed service error: no post-verification shortening,
            # trimming, substitution or replay with a borrowed verdict.
            try:
                from aa.conversation.finalization import split_certified_text as _split_cert
                from aa.conversation.output_limits import envelope_passes as _env

                if bool(_env(reply)):
                    fitted = reply
                else:
                    segments = _split_cert(reply)
                    if len(segments) > 1:
                        fitted = reply
                    else:
                        logger.warning("app delivery blocked: envelope exceeded")
                        fitted = SERVICE_ERROR_REPLY
                        service_error = True
                        is_service = True
            except Exception:
                logger.warning("app delivery blocked: envelope/split guard")
                fitted = SERVICE_ERROR_REPLY
                service_error = True
                is_service = True
        total_ms = (_time.perf_counter() - turn_started) * 1000.0
        # Privacy-safe turn telemetry: stage outcomes come from the graph
        # runtime snapshot (counts/latencies/outcomes only); this log
        # carries no user text, reply text or identifiers.
        logger.info(
            "normal response served",
            extra={
                "latency_ms": round(total_ms, 1),
                "reply_len": len(fitted),
                "fallback_used": service_error or is_service,
                "is_clarification": False,
                "is_service_error": is_service,
                "planner_outcome": str(stage.get("planner_outcome", "unknown")),
                "retrieval_outcome": str(stage.get("retrieval_outcome", "unknown")),
                "answer_outcome": str(stage.get("answer_outcome", "unknown")),
                "verifier_outcome": str(stage.get("verifier_outcome", "unknown")),
                "repair_rounds": int(stage.get("repair_rounds", 0) or 0),
            },
        )
        return fitted

    @staticmethod
    def _fit_envelope(reply: str) -> str:
        """Confine ``reply`` to the hard envelope (complete-unit safe).

        The turn pipeline already applies compact regeneration upstream;
        this is the final deterministic guard so every path served here
        satisfies the same cap. Only lengths are logged on compaction,
        never message text. An un-fittable payload is a typed
        unsuccessful outcome surfaced as the marked service error.
        """
        if envelope_passes(reply):
            return reply
        try:
            compacted = compact_text_to_envelope(reply)
        except ValueError:
            logger.info("application reply un-fittable; service error used")
            return SERVICE_ERROR_REPLY
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
