"""Voice/text parity regression suite (issue #294).

Voice is transport only, never a conversational mode: Telegram voice
notes pass through local ASR into the exact same ``respond(chat_id,
text)`` boundary as typed text, and TTS synthesizes exactly the final
approved string with no rewriting. For equivalent recognized and typed
text with the same starting dialogue state, the entire conversational
computation and final approved output must be identical; the only
differences are voice-to-text ASR beforehand and final-text-to-audio
TTS plus sendVoice afterward.

All coverage runs at the raw Telegram Update -> dispatcher/ASR ->
shared answer -> delivery boundary with deterministic model stubs.
"""

from __future__ import annotations

import inspect
import io
from collections.abc import Sequence
from pathlib import Path

from aa import logging as aa_logging
from aa.app import Application
from aa.config import Settings
from aa.conversation.graph_runtime import GraphTurnRuntime
from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
from aa.safety.response import EMERGENCY_RESPONSE_RU
from aa.telegram.transport import (
    StubTelegramTransport,
    TelegramApiError,
    TelegramIncoming,
    VoiceAttachment,
)
from aa.telegram.tts import TtsPipeline
from aa.telegram.voice import VoiceError


def _settings(**overrides: str) -> Settings:
    base: dict[str, str] = {"TYPING_HEARTBEAT_SECONDS": "0.001"}
    base.update(overrides)
    return Settings.from_env(base)


def _stub_runtime() -> StubOpenCodeRuntime:
    return StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )


class _FakeRecognizer:
    def __init__(self, transcript: str = "привет", *, fail: bool = False) -> None:
        self._transcript = transcript
        self.fail = fail
        self.calls = 0

    @property
    def available(self) -> bool:
        return True

    def transcribe(self, samples: Sequence[float]) -> str:
        _ = samples
        return self._transcript


class _FakeVoicePipeline:
    """Minimal ASR stand-in at the app boundary."""

    def __init__(self, transcript: str = "привет", *, fail: bool = False) -> None:
        self.recognizer = _FakeRecognizer(transcript)
        self._transcript = transcript
        self._fail = fail
        self.calls = 0

    async def transcribe_voice(
        self,
        *,
        file_id: str,
        file_size_bytes: int | None = None,
        duration_seconds: int | None = None,
    ) -> str:
        _ = (file_id, file_size_bytes, duration_seconds)
        self.calls += 1
        if self._fail:
            raise VoiceError("asr-failed", "injected ASR failure")
        return self._transcript


class _FakeSynthesizer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.seen_texts: list[str] = []
        self.seen_speakers: list[str] = []

    @property
    def available(self) -> bool:
        return True

    def synthesize(self, text: str, speaker: str) -> list[float]:
        self.calls += 1
        self.seen_texts.append(text)
        self.seen_speakers.append(speaker)
        if self.fail:
            from aa.telegram.tts import TtsError

            raise TtsError("tts-failed", "injected synthesis failure")
        return [0.1] * 480


class _FakeEncoder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def encode(self, samples: Sequence[float], *, workdir: Path) -> bytes:
        if self.fail:
            from aa.telegram.tts import TtsError

            raise TtsError("encode-failed", "injected encode failure")
        assert len(samples) > 0
        probe = workdir / "encode-probe.tmp"
        probe.write_bytes(b"staged")
        return b"OggS-fake-parity-payload"


def _tts_pipeline(
    tmp_path: Path, *, synth_fail: bool = False, encode_fail: bool = False
) -> tuple[TtsPipeline, _FakeSynthesizer, _FakeEncoder]:
    from aa.telegram.tts import build_tts_pipeline

    synth = _FakeSynthesizer(fail=synth_fail)
    encoder = _FakeEncoder(fail=encode_fail)
    pipeline: TtsPipeline = build_tts_pipeline(
        synthesizer=synth, encoder=encoder, work_parent=tmp_path
    )
    return pipeline, synth, encoder


def _voice_update(update_id: int, chat_id: int) -> TelegramIncoming:
    return TelegramIncoming(
        update_id=update_id,
        chat_id=chat_id,
        message_id=1,
        text="",
        command=None,
        voice=VoiceAttachment(file_id="f1", duration_seconds=5, file_size_bytes=100),
    )


def _text_update(update_id: int, chat_id: int, text: str) -> TelegramIncoming:
    return TelegramIncoming(
        update_id=update_id, chat_id=chat_id, message_id=1, text=text, command=None
    )


def _counting_runtime(reply: str, calls: list[tuple[str, str]]) -> GraphTurnRuntime:
    async def _delegate(thread: str, text: str) -> str:
        calls.append((thread, text))
        return reply

    return GraphTurnRuntime(delegate=_delegate)


# ---------------------------------------------------------------------------
# Equivalence: same question, same context, byte-identical approved string.
# ---------------------------------------------------------------------------


async def test_equivalent_voice_and_text_share_computation_and_output(
    tmp_path: Path,
) -> None:
    long_answer = (
        "Поддержка рядом помогает пережить тягу спокойно. "
        "Спокойный вечер и режим помогают отдыху. "
        "Расскажите, что сейчас важнее всего для трезвости. "
        "Финальное существенное предложение про поддержку."
    )
    transport = StubTelegramTransport()
    calls: list[tuple[str, str]] = []
    pipeline, synth, _ = _tts_pipeline(tmp_path)
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=_FakeVoicePipeline("тяга вечером, что делать?"),  # type: ignore[arg-type]
        tts_pipeline=pipeline,
        graph_runtime=_counting_runtime(long_answer, calls),
    )
    await app.start()
    try:
        await app._process_dispatched_update(_text_update(1, 901, "тяга вечером, что делать?"))
        assert len(transport.sent) == 1
        text_reply = transport.sent[0].text
        assert text_reply == long_answer
        transport.sent.clear()
        await app._process_dispatched_update(_voice_update(2, 902))
        assert len(transport.sent_voices) == 1
        assert len(transport.sent) == 0
        # TTS received exactly the final approved string: no rewriting.
        assert synth.calls == 1
        assert synth.seen_texts == [long_answer]
        assert synth.seen_texts[0] == text_reply
        # Same conversational computation: identical input text reached
        # the shared graph boundary for both formats.
        assert [text for _, text in calls] == [
            "тяга вечером, что делать?",
            "тяга вечером, что делать?",
        ]
    finally:
        await app.stop()


async def test_long_answer_final_sentence_intact_for_both_formats(tmp_path: Path) -> None:
    sentences = [f"Поддержка помогает спокойно{idx}." for idx in range(10)]
    long_answer = " ".join(sentences)
    transport = StubTelegramTransport()
    calls: list[tuple[str, str]] = []
    pipeline, synth, _ = _tts_pipeline(tmp_path)
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=_FakeVoicePipeline("тяга вечером"),  # type: ignore[arg-type]
        tts_pipeline=pipeline,
        graph_runtime=_counting_runtime(long_answer, calls),
    )
    await app.start()
    try:
        text_reply = await app.respond(911, "тяга вечером")
        assert text_reply == long_answer
        assert "спокойно9" in text_reply
        await app._process_dispatched_update(_voice_update(3, 912))
        assert synth.seen_texts == [long_answer]
        assert "спокойно9" in synth.seen_texts[0]
    finally:
        await app.stop()


async def test_mixed_text_voice_text_turns_share_one_fifo_state(tmp_path: Path) -> None:
    transport = StubTelegramTransport()
    calls: list[tuple[str, str]] = []
    pipeline, _, _ = _tts_pipeline(tmp_path)

    async def _delegate(thread: str, text: str) -> str:
        calls.append((thread, text))
        return f"Принято: {text}. Что сейчас важнее?"

    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=_FakeVoicePipeline("второе голосом"),  # type: ignore[arg-type]
        tts_pipeline=pipeline,
        graph_runtime=GraphTurnRuntime(delegate=_delegate),
    )
    await app.start()
    try:
        await app._process_dispatched_update(_text_update(10, 920, "первое текстом"))
        await app._process_dispatched_update(_voice_update(11, 920))
        await app._process_dispatched_update(_text_update(12, 920, "третье текстом"))
        assert [text for _, text in calls] == [
            "первое текстом",
            "второе голосом",
            "третье текстом",
        ]
        threads = {thread for thread, _ in calls}
        assert len(threads) == 1
        assert len(transport.sent) == 2
        assert len(transport.sent_voices) == 1
        assert transport.sent[0].text == "Принято: первое текстом. Что сейчас важнее?"
        assert transport.sent[1].text == "Принято: третье текстом. Что сейчас важнее?"
    finally:
        await app.stop()


# ---------------------------------------------------------------------------
# Failure parity: ASR failure invents no turn; TTS failure keeps bytes.
# ---------------------------------------------------------------------------


async def test_asr_failure_sends_transport_error_without_conversational_turn() -> None:
    transport = StubTelegramTransport()
    calls: list[tuple[str, str]] = []
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=_FakeVoicePipeline("привет", fail=True),  # type: ignore[arg-type]
        graph_runtime=_counting_runtime("Привет. Держись.", calls),
    )
    await app.start()
    try:
        await app._process_dispatched_update(_voice_update(20, 930))
        assert calls == []
        assert len(transport.sent) == 1
        assert len(transport.sent_voices) == 0
        from aa.conversation.orchestrator import meets_russian_only

        assert meets_russian_only(transport.sent[0].text)
        assert transport.sent[0].text.strip()
        assert app.running
    finally:
        await app.stop()


async def test_tts_synthesis_failure_falls_back_to_byte_identical_text(
    tmp_path: Path,
) -> None:
    answer = "Поддержка рядом помогает. Спокойный вечер важен. Что сейчас важнее?"
    transport = StubTelegramTransport()
    calls: list[tuple[str, str]] = []
    pipeline, _, _ = _tts_pipeline(tmp_path, synth_fail=True)
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=_FakeVoicePipeline("тяга вечером"),  # type: ignore[arg-type]
        tts_pipeline=pipeline,
        graph_runtime=_counting_runtime(answer, calls),
    )
    await app.start()
    try:
        await app._process_dispatched_update(_voice_update(21, 931))
        assert len(transport.sent_voices) == 0
        assert len(transport.sent) == 1
        assert transport.sent[0].text == answer
        expected = await app.respond(932, "тяга вечером")
        assert transport.sent[0].text == expected
    finally:
        await app.stop()


async def test_opus_encode_failure_falls_back_to_byte_identical_text(tmp_path: Path) -> None:
    answer = "Поддержка рядом помогает. Спокойный вечер важен. Что сейчас важнее?"
    transport = StubTelegramTransport()
    pipeline, _, _ = _tts_pipeline(tmp_path, encode_fail=True)
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=_FakeVoicePipeline("тяга вечером"),  # type: ignore[arg-type]
        tts_pipeline=pipeline,
        graph_runtime=_counting_runtime(answer, []),
    )
    await app.start()
    try:
        await app._process_dispatched_update(_voice_update(22, 933))
        assert len(transport.sent_voices) == 0
        assert len(transport.sent) == 1
        assert transport.sent[0].text == answer
    finally:
        await app.stop()


async def test_sendvoice_delivery_failure_falls_back_to_byte_identical_text(
    tmp_path: Path,
) -> None:
    answer = "Поддержка рядом помогает. Спокойный вечер важен. Что сейчас важнее?"

    class _FailingVoiceTransport(StubTelegramTransport):
        async def send_voice(self, reply) -> None:  # type: ignore[no-untyped-def]
            raise TelegramApiError("injected sendVoice failure")

    transport = _FailingVoiceTransport()
    pipeline, synth, _ = _tts_pipeline(tmp_path)
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=_FakeVoicePipeline("тяга вечером"),  # type: ignore[arg-type]
        tts_pipeline=pipeline,
        graph_runtime=_counting_runtime(answer, []),
    )
    await app.start()
    try:
        await app._process_dispatched_update(_voice_update(23, 934))
        assert synth.calls == 1
        assert synth.seen_texts == [answer]
        assert len(transport.sent) == 1
        assert transport.sent[0].text == answer
    finally:
        await app.stop()


# ---------------------------------------------------------------------------
# Safety parity: emergencies identical for voice and text.
# ---------------------------------------------------------------------------


async def test_emergency_identical_for_voice_and_text(tmp_path: Path) -> None:
    transport = StubTelegramTransport()
    calls: list[tuple[str, str]] = []
    pipeline, synth, _ = _tts_pipeline(tmp_path)
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=_FakeVoicePipeline("I want to kill myself tonight"),  # type: ignore[arg-type]
        tts_pipeline=pipeline,
        graph_runtime=_counting_runtime("must never be used", calls),
    )
    await app.start()
    try:
        text_reply = await app.respond(940, "I want to kill myself tonight")
        assert text_reply == EMERGENCY_RESPONSE_RU
        await app._process_dispatched_update(_voice_update(24, 941))
        # Emergency never reaches the graph for either format.
        assert calls == []
        assert synth.calls == 1
        assert synth.seen_texts == [EMERGENCY_RESPONSE_RU]
        assert len(transport.sent_voices) == 1
    finally:
        await app.stop()


# ---------------------------------------------------------------------------
# Isolation, privacy, and structural guards.
# ---------------------------------------------------------------------------


async def test_concurrent_chats_stay_isolated_with_identical_results(
    tmp_path: Path,
) -> None:
    import asyncio

    transport = StubTelegramTransport()
    pipeline, _, _ = _tts_pipeline(tmp_path)

    async def _delegate(thread: str, text: str) -> str:
        await asyncio.sleep(0.01)
        return f"Принято: {text}. Что сейчас важнее?"

    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=_FakeVoicePipeline("общий вопрос"),  # type: ignore[arg-type]
        tts_pipeline=pipeline,
        graph_runtime=GraphTurnRuntime(delegate=_delegate),
    )
    await app.start()
    try:
        await asyncio.gather(
            app._process_dispatched_update(_text_update(30, 950, "общий вопрос")),
            app._process_dispatched_update(_voice_update(31, 951)),
        )
        # Text chat gets text, voice chat gets voice; both carry the
        # same conversational result for the same question.
        assert len(transport.sent) == 1
        assert len(transport.sent_voices) == 1
        assert transport.sent[0].text == "Принято: общий вопрос. Что сейчас важнее?"
    finally:
        await app.stop()


async def test_voice_turn_leaks_no_speech_metadata_or_user_data(tmp_path: Path) -> None:
    stream = io.StringIO()
    aa_logging.configure_logging("INFO", stream=stream)
    secret = "секретный паритетный ответ трезвость"
    transport = StubTelegramTransport()
    pipeline, _, _ = _tts_pipeline(tmp_path)

    async def _reply(thread: str, text: str) -> str:
        return secret + ". Держись."

    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=_FakeVoicePipeline(secret),  # type: ignore[arg-type]
        tts_pipeline=pipeline,
        graph_runtime=GraphTurnRuntime(delegate=_reply),
    )
    await app.start()
    try:
        await app._process_dispatched_update(_voice_update(32, 952))
    finally:
        await app.stop()
    output = stream.getvalue()
    assert secret not in output
    assert "OggS" not in output


def test_structural_no_voice_flags_or_prompts_in_content_graph() -> None:
    import aa.conversation.orchestrator as orchestrator_mod
    from aa.conversation.graph_runtime import GraphTurnRuntime as _Runtime

    assert "voice_input" not in inspect.signature(Application.respond).parameters
    assert not hasattr(Application, "_apply_voice_brevity")
    assert "voice_mode" not in inspect.signature(orchestrator_mod.build_synthesis_prompt).parameters
    assert "voice_mode" not in inspect.signature(orchestrator_mod.build_trivial_prompt).parameters
    assert "voice_mode" not in inspect.signature(orchestrator_mod.run_trivial_turn).parameters
    assert (
        "voice_mode"
        not in inspect.signature(orchestrator_mod.TurnRunner.run_grounded_turn).parameters
    )
    assert "voice" not in inspect.signature(_Runtime.run_turn).parameters
    orchestrator_src = inspect.getsource(orchestrator_mod)
    assert "voice_mode" not in orchestrator_src
    assert "ГОЛОСОВОЙ РЕЖИМ" not in orchestrator_src
    app_src = inspect.getsource(Application)
    assert "_apply_voice_brevity" not in app_src
    assert "voice_policy_passes" not in app_src
    assert "compact_voice_text_to_policy" not in app_src
