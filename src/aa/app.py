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
from aa.conversation.memory import thread_id_for_chat
from aa.conversation.output_limits import compact_text_to_envelope, envelope_passes
from aa.conversation.runtime import ProductConversationRuntime
from aa.conversation.turn_pipeline import contains_cyrillic
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
    resolve_tts_voice,
    voice_for_presentation,
)
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
        grounding: object | None = None,
        conversation_runtime: ProductConversationRuntime | None = None,
        voice_pipeline: VoicePipeline | None = None,
        tts_pipeline: TtsPipeline | None = None,
        presentation_classifier: VoicePresentationClassifier | None = None,
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
        # Kept only as an inert compatibility attribute for old tests/callers;
        # production conversation semantics no longer invoke the old grounding
        # object or any legacy orchestrator.
        self.grounding = grounding
        self._index: HybridIndex | None = None
        self._index_error: str | None = None
        self._conversation_runtime = conversation_runtime or ProductConversationRuntime(
            client=self.opencode_runtime.client,
            primary_model=settings.opencode_model,
            fallback_model=settings.opencode_fallback_model,
            index_loader=self._load_v2_index,
            safety_check=self.safety.check,
        )
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
        # Transport only: every ordinary turn reaches the single production
        # LangGraph boundary through ``respond``; hidden model calls reuse the
        # one local OpenCode runtime via the thin v2 adapter.
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
            logger.info(
                "voice capability ready",
                extra={"injected": True},
            )
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
            await self._conversation_runtime.start()
            await self.dispatcher.start()
            # Voice capability loads once here and is reused for all turns.
            # Initialization failure disables voice only, never the poller.
            await self._init_presentation_capability()
            await self._init_voice_capability()
            self._attach_presentation_classifier()
            # TTS capability loads once here and is reused for voice replies.
            # Initialization failure falls back to text, never the poller.
            await self._init_tts_capability()
            await self.transport.start()
            # Start the requested fixed 5h window only after the poller
            # is live and all dependencies have completed bootstrap.
            await self.controller.start()
        except Exception:
            await self.transport.stop()
            await self.dispatcher.stop()
            await self._conversation_runtime.stop()
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
            await self._conversation_runtime.stop()
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
        await self._conversation_runtime.stop()
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
        if self._fatal_error is not None:
            raise self._fatal_error

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
        the production LangGraph runtime via :meth:`respond` as the single
        substantive-turn API. Session create/reset happens here, inside the
        per-chat serialization, so concurrent first messages cannot create
        competing sessions and ``/new`` cannot interleave with an older
        turn for the same chat.
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
            await self._conversation_runtime.reset(incoming.chat_id)
            self.sessions.reset(incoming.chat_id)
            reply = _NEW_REPLY
        except (RuntimeError, OpenCodeError):
            logger.warning(
                "telegram new-session reset failed",
                extra=self._thread_extra(incoming.chat_id),
            )
            reply = _TEMPORARY_ERROR_REPLY
        await self.transport.send(TelegramReply(chat_id=incoming.chat_id, text=reply))

    async def _handle_telegram_update(self, incoming: TelegramIncoming) -> None:
        """Process one private text update and always emit a bounded reply.

        Exactly one Telegram message is delivered per update: overflow is
        never split into multiple messages. When the final transport guard
        blocks an escaped overlong payload, a single bounded fallback is
        delivered instead. Substantive work goes through :meth:`respond`,
        the single production LangGraph runtime entry point; this transport
        layer never sends a substantive user message to OpenCode directly.
        """
        await self._respond_and_deliver(incoming, text=incoming.text, voice_input=False)

    async def _handle_voice_update(self, incoming: TelegramIncoming) -> None:
        """Process one Telegram voice note through ASR into the text boundary.

        The transcript enters the exact same production AA turn boundary
        used by text messages, marked ``voice_input=True``. Any
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
            logger.warning(
                "voice turn without recognizer",
                extra={"chat_id": incoming.chat_id, "update_id": incoming.update_id},
            )
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
            logger.warning(
                "voice turn failed",
                extra={
                    "chat_id": incoming.chat_id,
                    "update_id": incoming.update_id,
                    "category": exc.category,
                },
            )
            await self._send_text_reply(incoming, voice_error_reply(exc.category))
            return
        except Exception:
            logger.warning(
                "voice turn failed",
                extra={
                    "chat_id": incoming.chat_id,
                    "update_id": incoming.update_id,
                    "category": "voice-failed",
                },
            )
            await self._send_text_reply(incoming, voice_error_reply("voice-failed"))
            return
        # Ephemeral only: a per-turn local, never stored in sessions/history.
        await self._respond_and_deliver(
            incoming, text=transcript, voice_input=True, voice_presentation=presentation
        )

    async def _typing_heartbeat(self, chat_id: int) -> None:
        """Refresh Telegram typing until the owning turn finishes delivery."""
        interval = self.settings.telegram_typing_interval_seconds
        while True:
            try:
                await asyncio.sleep(interval)
                await self.transport.send_chat_action(chat_id, "typing")
            except asyncio.CancelledError:
                raise
            except NotImplementedError:
                return
            except (TelegramApiError, TimeoutError, OSError):
                logger.warning("telegram typing refresh failed", extra=self._thread_extra(chat_id))

    async def _begin_typing(self, chat_id: int) -> asyncio.Task[None]:
        """Send typing immediately, then start the independent refresh loop."""
        try:
            await self.transport.send_chat_action(chat_id, "typing")
        except NotImplementedError:
            pass
        except (TelegramApiError, TimeoutError, OSError):
            logger.warning("telegram typing start failed", extra=self._thread_extra(chat_id))
        return asyncio.create_task(self._typing_heartbeat(chat_id))

    @staticmethod
    def _thread_extra(chat_id: int) -> dict[str, str]:
        return {"v2_thread": thread_id_for_chat(chat_id)[:12]}

    async def _respond_and_deliver(
        self,
        incoming: TelegramIncoming,
        *,
        text: str,
        voice_input: bool,
        voice_presentation: str | None = None,
    ) -> None:
        """Run one v2 turn and keep typing alive through confirmed delivery."""
        self.sessions.record_message(incoming.chat_id)
        heartbeat = await self._begin_typing(incoming.chat_id)
        try:
            try:
                reply = await self.respond(incoming.chat_id, text, voice_input=voice_input)
                if not reply.strip() or not contains_cyrillic(reply):
                    raise ValueError("v2 reply violates Russian output boundary")
            except OpenCodeRateLimitError:
                raise
            except (OpenCodeError, RuntimeError, ValueError):
                logger.warning(
                    "telegram message processing failed",
                    extra=self._thread_extra(incoming.chat_id),
                )
                reply = _TEMPORARY_ERROR_REPLY

            if voice_input:
                delivered = await self._send_voice_reply(
                    incoming, reply, voice_presentation=voice_presentation
                )
                if delivered:
                    return
            await self._send_text_reply(incoming, reply)
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

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
            logger.info("voice reply fallback to text", extra={"chat_id": incoming.chat_id})
            return False
        voice = self._resolve_voice_for_turn(voice_presentation)
        if voice not in (DEFAULT_VOICE, "eugene"):
            voice = DEFAULT_VOICE
        try:
            ogg_bytes = await pipeline.synthesize_voice_ogg(reply, voice)
        except TtsError as exc:
            logger.warning(
                "voice reply synthesis failed; fallback to text",
                extra={"chat_id": incoming.chat_id, "category": exc.category},
            )
            return False
        except Exception:
            logger.warning(
                "voice reply synthesis failed; fallback to text",
                extra={"chat_id": incoming.chat_id, "category": "tts-failed"},
            )
            return False
        try:
            await self.transport.send_voice(
                TelegramVoiceReply(chat_id=incoming.chat_id, voice_bytes=ogg_bytes)
            )
        except (TelegramApiError, TelegramEnvelopeError):
            logger.warning(
                "voice reply delivery failed; fallback to text",
                extra={"chat_id": incoming.chat_id},
            )
            return False
        except Exception:
            logger.warning(
                "voice reply delivery failed; fallback to text",
                extra={"chat_id": incoming.chat_id},
            )
            return False
        logger.info("voice reply delivered", extra={"chat_id": incoming.chat_id})
        return True

    async def _send_text_reply(self, incoming: TelegramIncoming, reply: str) -> None:
        """Deliver one bounded text reply with envelope fallback handling."""
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

    def _load_v2_index(self) -> HybridIndex:
        """Open and cache the canonical RU-first retrieval index for LangGraph."""
        if self._index is not None:
            return self._index
        if self._index_error is not None:
            raise RuntimeError(self._index_error)
        try:
            corpus_root = Path(self.settings.aa_corpus_path)
            self._index = open_hybrid_index(
                corpus_root / "generated" / "retrieval",
                ru_manifest_path=corpus_root / "canonical.ru.manifest.json",
                en_manifest_path=corpus_root / "canonical.manifest.json",
                lock_path=corpus_root / "embedding.lock.json",
            )
            return self._index
        except (ValueError, OSError, RuntimeError) as exc:
            self._index_error = str(exc)
            raise RuntimeError(self._index_error) from exc

    async def respond(self, chat_id: int, text: str, *, voice_input: bool = False) -> str:
        """Answer one turn through deterministic safety then the v2 LangGraph."""
        logger.info(
            "v2 turn started",
            extra={**self._thread_extra(chat_id), "voice_input": voice_input, "text_len": len(text)},
        )
        result = self.safety.check(text)
        if result.decision is SafetyDecision.EMERGENCY and result.classification is not None:
            logger.info(
                "emergency response served",
                extra={**self._thread_extra(chat_id), "reason": result.reason},
            )
            return self._fit_envelope(
                build_emergency_response(result.classification, language="ru")
            )
        if result.decision is SafetyDecision.BLOCK:
            raise ValueError("refusing to answer an empty message")

        reply = await self._conversation_runtime.respond(chat_id, text)
        if not contains_cyrillic(reply):
            raise ValueError("v2 graph returned a non-Russian reply")
        logger.info("v2 response served", extra=self._thread_extra(chat_id))
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
