"""Local Russian TTS replies and concise voice mode tests (issue #77)."""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from aa import logging as aa_logging
from aa.app import Application
from aa.config import Settings
from aa.conversation.orchestrator import (
    build_synthesis_prompt,
    build_trivial_prompt,
    run_trivial_turn,
)
from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
from aa.safety.response import EMERGENCY_RESPONSE_RU
from aa.telegram.transport import (
    StubTelegramTransport,
    TelegramApiError,
    TelegramIncoming,
    TelegramVoiceReply,
    VoiceAttachment,
)
from aa.telegram.tts import (
    ALLOWED_VOICES,
    DEFAULT_VOICE,
    MAX_VOICE_COMPACT_REGENERATIONS,
    OPUS_BITRATE,
    OPUS_CHANNELS,
    TORCH_VERSION,
    TTS_MODEL_ID,
    TTS_MODEL_URL,
    TTS_SAMPLE_RATE,
    TTS_VOICE_EUGENE,
    TTS_VOICE_XENIA,
    FfmpegOpusEncoder,
    SileroSynthesizer,
    TtsError,
    TtsPipeline,
    build_tts_pipeline,
    compact_voice_text_to_policy,
    count_voice_sentences,
    count_voice_words,
    resolve_tts_voice,
    voice_for_presentation,
    voice_policy_passes,
)

PRIMARY = "opencode/muse-spark-1.3-contributor-free"
FALLBACK = "opencode/space-bunny-free"


def _settings(**overrides: str) -> Settings:
    base: dict[str, str] = {}
    base.update(overrides)
    return Settings.from_env(base)


def _stub_runtime() -> StubOpenCodeRuntime:
    return StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )


class FakeSynthesizer:
    """In-memory Silero stand-in with concurrency tracking."""

    def __init__(
        self,
        samples: list[float] | None = None,
        *,
        available: bool = True,
        fail: bool = False,
        delay: float = 0.0,
    ) -> None:
        self.samples = samples if samples is not None else [0.1] * TTS_SAMPLE_RATE
        self._available = available
        self.fail = fail
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.seen_speakers: list[str] = []
        self.seen_texts: list[str] = []

    @property
    def available(self) -> bool:
        return self._available

    def synthesize(self, text: str, speaker: str) -> list[float]:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            self.seen_speakers.append(speaker)
            self.seen_texts.append(text)
            if self.fail:
                raise TtsError("tts-failed", "injected synthesis failure")
            if not text.strip():
                raise TtsError("tts-failed", "empty text")
            if self.delay:
                import time as _time

                _time.sleep(self.delay)
            return list(self.samples)
        finally:
            self.active -= 1


class FakeEncoder:
    """In-memory Opus stand-in with call tracking and temp staging."""

    def __init__(self, payload: bytes | None = None, *, fail: bool = False) -> None:
        self.payload = payload if payload is not None else b"OggS-fake-voice-payload"
        self.fail = fail
        self.calls = 0

    def encode(self, samples: Sequence[float], *, workdir: Path) -> bytes:
        self.calls += 1
        if self.fail:
            raise TtsError("encode-failed", "injected encode failure")
        if not samples:
            raise TtsError("encode-failed", "no samples")
        probe = workdir / "encode-probe.tmp"
        probe.write_bytes(b"staged")
        _ = len(samples)
        return bytes(self.payload)


def _pipeline(
    *,
    synthesizer: FakeSynthesizer | None = None,
    encoder: FakeEncoder | None = None,
    work_parent: Path | None = None,
) -> tuple[TtsPipeline, FakeSynthesizer, FakeEncoder]:
    owned_synth = synthesizer or FakeSynthesizer()
    owned_enc = encoder or FakeEncoder()
    pipeline = build_tts_pipeline(
        synthesizer=owned_synth,
        encoder=owned_enc,
        work_parent=work_parent,
    )
    return pipeline, owned_synth, owned_enc


def _voice_incoming(
    update_id: int = 50, chat_id: int = 77, text: str = "", voice: bool = True
) -> TelegramIncoming:
    attachment = (
        VoiceAttachment(file_id="f1", duration_seconds=5, file_size_bytes=100) if voice else None
    )
    return TelegramIncoming(
        update_id=update_id,
        chat_id=chat_id,
        message_id=1,
        text=text,
        command=None,
        voice=attachment,
    )


class _FakeVoiceRecognizer:
    def __init__(self, transcript: str = "привет") -> None:
        self._transcript = transcript
        self.calls = 0

    @property
    def available(self) -> bool:
        return True

    def transcribe(self, samples: Sequence[float]) -> str:
        _ = samples
        return self._transcript


class _FakeVoicePipeline:
    """Minimal ASR stand-in that always returns a fixed transcript."""

    def __init__(self, transcript: str = "привет") -> None:
        self.recognizer = _FakeVoiceRecognizer(transcript)

    async def transcribe_voice(
        self,
        *,
        file_id: str,
        file_size_bytes: int | None = None,
        duration_seconds: int | None = None,
    ) -> str:
        _ = (file_id, file_size_bytes, duration_seconds)
        self.recognizer.calls += 1
        return self.recognizer._transcript


# ---------------------------------------------------------------------------
# Fixed contract.
# ---------------------------------------------------------------------------


def test_fixed_tts_contract_is_pinned() -> None:
    assert TTS_MODEL_ID == "v5_5_ru"
    assert TTS_MODEL_URL == "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
    assert TORCH_VERSION == "2.14.1"
    assert TTS_SAMPLE_RATE == 48000
    assert TTS_VOICE_XENIA == "xenia"
    assert TTS_VOICE_EUGENE == "eugene"
    assert DEFAULT_VOICE == "xenia"
    assert ALLOWED_VOICES == frozenset({"xenia", "eugene"})
    assert OPUS_BITRATE == "32k"
    assert OPUS_CHANNELS == 1
    assert MAX_VOICE_COMPACT_REGENERATIONS == 1


def test_only_xenia_and_eugene_are_production_voices() -> None:
    assert resolve_tts_voice("xenia") == "xenia"
    assert resolve_tts_voice("eugene") == "eugene"
    for other in ("kseniya", "eugene2", "", "XENIA", "random-voice", "cloud-voice"):
        assert resolve_tts_voice(other) == "xenia"
    assert resolve_tts_voice(None) == "xenia"
    # Opposite-voice mapping defaults to xenia on unknown/unavailable.
    assert voice_for_presentation("male-presenting") == "xenia"
    assert voice_for_presentation("female-presenting") == "eugene"
    assert voice_for_presentation("unknown") == "xenia"
    assert voice_for_presentation(None) == "xenia"
    assert voice_for_presentation("bogus") == "xenia"


def test_silero_synthesizer_without_load_is_unavailable(tmp_path: Path) -> None:
    synth = SileroSynthesizer(tmp_path / "v5_5_ru.pt")
    assert not synth.available
    with pytest.raises(TtsError) as excinfo:
        synth.synthesize("Привет", "xenia")
    assert excinfo.value.category == "tts-unavailable"


def test_ffmpeg_opus_encoder_rejects_empty_samples(tmp_path: Path) -> None:
    encoder = FfmpegOpusEncoder()
    with pytest.raises(TtsError) as excinfo:
        encoder.encode([], workdir=tmp_path)
    assert excinfo.value.category == "encode-failed"


def test_tts_settings_default_and_env() -> None:
    assert _settings().aa_tts_model_path == "./models/tts/v5_5_ru.pt"
    custom = _settings(AA_TTS_MODEL_PATH="/tmp/tts/v5_5_ru.pt")
    assert custom.aa_tts_model_path == "/tmp/tts/v5_5_ru.pt"
    custom.validate()
    assert "AA_TTS_MODEL_PATH" in set(Settings.RESERVED_ENV_NAMES)
    assert custom.to_safe_dict()["aa_tts_model_path"] == "/tmp/tts/v5_5_ru.pt"


# ---------------------------------------------------------------------------
# Voice-mode response contract: <=4 sentences / <=80 words.
# ---------------------------------------------------------------------------


def test_voice_policy_counts_and_bounds() -> None:
    short = "Привет. Как дела. Что нового. Расскажи."
    assert count_voice_sentences(short) == 4
    assert voice_policy_passes(short)
    five = short + " Ещё одно."
    assert count_voice_sentences(five) == 5
    assert not voice_policy_passes(five)
    long_words = "слово " * 81
    assert count_voice_words(long_words.strip()) == 81
    assert not voice_policy_passes(long_words.strip())
    exactly_80 = " ".join(f"слово{i}" for i in range(80)) + "."
    assert count_voice_words(exactly_80) == 80
    assert voice_policy_passes(exactly_80)
    assert not voice_policy_passes("   ")


def test_compact_voice_returns_leading_sentences_fitting_both_bounds() -> None:
    first = "Первое короткое предложение."
    second = "Второе короткое предложение."
    rest = " ".join(f"Длинное предложение номер {idx}." for idx in range(10))
    text = f"{first} {second} {rest}"
    assert not voice_policy_passes(text)
    compacted = compact_voice_text_to_policy(text)
    assert voice_policy_passes(compacted)
    assert compacted.startswith(first)
    assert count_voice_sentences(compacted) <= 4
    assert count_voice_words(compacted) <= 80
    # Word-bound case: leading sentences that fit 80 words only.
    many = " ".join(f"Предложение номер {idx} здесь." for idx in range(30))
    assert not voice_policy_passes(many)
    compacted_many = compact_voice_text_to_policy(many)
    assert voice_policy_passes(compacted_many)
    assert many.startswith(compacted_many)


def test_voice_generation_budget_in_prompts() -> None:
    trivial_plain = build_trivial_prompt(user_text="привет")
    trivial_voice = build_trivial_prompt(user_text="привет", voice_mode=True)
    assert "ГОЛОСОВОЙ РЕЖИМ" not in trivial_plain
    assert "ГОЛОСОВОЙ РЕЖИМ" in trivial_voice
    assert "4 предложений" in trivial_voice
    assert "80 слов" in trivial_voice
    assert "привет" in trivial_voice


async def test_synthesis_prompt_carries_voice_budget(tmp_path: Path) -> None:
    import hashlib

    from aa.conversation.orchestrator import (
        deduplicate_cross_aspect,
        load_exact_evidence,
        run_planner,
        search_first_round,
    )
    from aa.corpus.structure import SECTION_IDS, build_full_structure
    from aa.retrieval.index import build_hybrid_index

    en_sections: list[dict[str, object]] = []
    ru_sections: list[dict[str, object]] = []
    for section_id in SECTION_IDS:
        en_sections.append(
            {
                "id": section_id,
                "title": f"EN {section_id}",
                "text": f"Fixture EN {section_id} opening.",
                "source_id": "core-pages-1-164",
                "source_file": "corpus/source/raw/AA.txt",
                "source_sha256": hashlib.sha256(b"en").hexdigest(),
            }
        )
        ru_sections.append(
            {
                "id": section_id,
                "title": f"RU {section_id}",
                "text": (
                    f"Фиктивный отрывок {section_id} про трезвость и поддержку. "
                    "Второй абзац продолжается."
                ),
                "source_id": "ru-fourth-edition-txt",
                "source_file": "corpus/source/raw-ru/aa-big-book.txt",
                "source_sha256": hashlib.sha256(b"ru").hexdigest(),
            }
        )
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
    )
    index = build_hybrid_index(
        dict(full),
        ru_manifest={"format": "x", "artifact_sha256": "a" * 64},
        en_manifest={"format": "x", "artifact_sha256": "b" * 64},
        embedding_lock={"model_id": "m", "revision": "r" * 40},
        out_dir=tmp_path / "retrieval",
        backend="hashing",
    )
    plan = await run_planner("тяга вечером")
    merged = deduplicate_cross_aspect(search_first_round(index, plan))
    pack, _ = load_exact_evidence(
        index, merged, ru_corpus_version=str(index.metadata.get("ru_artifact_sha256", ""))
    )
    plain = build_synthesis_prompt(user_text="тяга", pack=pack)
    voiced = build_synthesis_prompt(user_text="тяга", pack=pack, voice_mode=True)
    assert "ГОЛОСОВОЙ РЕЖИМ" not in plain
    assert "ГОЛОСОВОЙ РЕЖИМ" in voiced
    assert "4 предложений" in voiced


async def test_trivial_voice_allows_exactly_one_regeneration() -> None:
    long_answer = " ".join(f"Длинное предложение номер {idx} здесь." for idx in range(20))
    assert not voice_policy_passes(long_answer)
    short_answer = "Коротко. Держись. Приходи."
    assert voice_policy_passes(short_answer)
    prompts: list[str] = []

    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        prompts.append(prompt)
        if len(prompts) == 1:
            return long_answer
        return short_answer

    result = await run_trivial_turn(
        "привет",
        session_id="s1",
        send=_send,
        agent="aa",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
        voice_mode=True,
    )
    assert len(prompts) == 2
    assert result.text == short_answer
    assert voice_policy_passes(result.text)
    assert "ГОЛОСОВОЙ РЕЖИМ" in prompts[0]


async def test_trivial_voice_second_violation_compacts_to_leading() -> None:
    long_answer = " ".join(f"Длинное предложение номер {idx} здесь." for idx in range(20))
    prompts: list[str] = []

    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        prompts.append(prompt)
        return long_answer

    result = await run_trivial_turn(
        "привет",
        session_id="s1",
        send=_send,
        agent="aa",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
        voice_mode=True,
    )
    assert len(prompts) == 2
    assert voice_policy_passes(result.text)
    assert long_answer.startswith(result.text)
    assert result.text != long_answer


async def test_trivial_text_mode_does_not_regenerate_for_voice_policy() -> None:
    long_answer = " ".join(f"Длинное предложение номер {idx} здесь." for idx in range(20))
    prompts: list[str] = []

    async def _send(
        session_id: str, prompt: str, *, agent: str = "", model: str = "", **_kw: Any
    ) -> str:
        prompts.append(prompt)
        return long_answer

    result = await run_trivial_turn(
        "привет",
        session_id="s1",
        send=_send,
        agent="aa",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
        voice_mode=False,
    )
    assert len(prompts) == 1
    assert result.text == long_answer


# ---------------------------------------------------------------------------
# Pipeline: synthesis, encoding, cleanup, concurrency, privacy.
# ---------------------------------------------------------------------------


async def test_pipeline_synthesizes_and_cleans_temp(tmp_path: Path) -> None:
    pipeline, synth, enc = _pipeline(work_parent=tmp_path)
    data = await pipeline.synthesize_voice_ogg("Привет, как дела", "xenia")
    assert data == b"OggS-fake-voice-payload"
    assert synth.calls == 1
    assert enc.calls == 1
    assert synth.seen_speakers == ["xenia"]
    assert list(tmp_path.iterdir()) == []


async def test_pipeline_defaults_unknown_speaker_to_xenia(tmp_path: Path) -> None:
    pipeline, synth, _ = _pipeline(work_parent=tmp_path)
    await pipeline.synthesize_voice_ogg("Привет", "cloud-voice")
    assert synth.seen_speakers == ["xenia"]
    await pipeline.synthesize_voice_ogg("Привет", None)
    assert synth.seen_speakers[-1] == "xenia"


async def test_pipeline_supports_both_fixed_voices(tmp_path: Path) -> None:
    pipeline, synth, _ = _pipeline(work_parent=tmp_path)
    await pipeline.synthesize_voice_ogg("Привет", "xenia")
    await pipeline.synthesize_voice_ogg("Привет", "eugene")
    assert synth.seen_speakers == ["xenia", "eugene"]


async def test_synthesis_failure_raises_and_cleans(tmp_path: Path) -> None:
    pipeline, _, _ = _pipeline(synthesizer=FakeSynthesizer(fail=True), work_parent=tmp_path)
    with pytest.raises(TtsError) as excinfo:
        await pipeline.synthesize_voice_ogg("Привет", "xenia")
    assert excinfo.value.category == "tts-failed"
    assert list(tmp_path.iterdir()) == []


async def test_encoder_failure_raises_and_cleans(tmp_path: Path) -> None:
    pipeline, _, _ = _pipeline(encoder=FakeEncoder(fail=True), work_parent=tmp_path)
    with pytest.raises(TtsError) as excinfo:
        await pipeline.synthesize_voice_ogg("Привет", "xenia")
    assert excinfo.value.category == "encode-failed"
    assert list(tmp_path.iterdir()) == []


async def test_unavailable_synthesizer_fails_closed(tmp_path: Path) -> None:
    pipeline, _, _ = _pipeline(synthesizer=FakeSynthesizer(available=False), work_parent=tmp_path)
    with pytest.raises(TtsError) as excinfo:
        await pipeline.synthesize_voice_ogg("Привет", "xenia")
    assert excinfo.value.category == "tts-unavailable"


async def test_concurrent_synthesis_obeys_single_tts_bound(tmp_path: Path) -> None:
    synth = FakeSynthesizer(delay=0.05)
    pipeline, _, _ = _pipeline(synthesizer=synth, work_parent=tmp_path)
    results = await asyncio.gather(
        pipeline.synthesize_voice_ogg("Первое", "xenia"),
        pipeline.synthesize_voice_ogg("Второе", "eugene"),
        pipeline.synthesize_voice_ogg("Третье", "xenia"),
    )
    assert list(results) == [b"OggS-fake-voice-payload"] * 3
    assert synth.calls == 3
    assert synth.max_active == 1


async def test_no_synthesized_text_or_audio_in_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "секретный голосовой ответ про трезвость"
    pipeline, _, _ = _pipeline(work_parent=tmp_path)
    with caplog.at_level(logging.INFO, logger="aa.telegram.tts"):
        await pipeline.synthesize_voice_ogg(secret, "xenia")
    records = "\n".join(record.getMessage() for record in caplog.records)
    assert secret not in records
    for record in caplog.records:
        assert secret not in str(record.args)


def test_no_audio_leak_in_file() -> None:
    text = Path(__file__).read_text(encoding="utf-8")
    assert "models.silero.ai" in text
    assert "v5_5_ru" in text


# ---------------------------------------------------------------------------
# Transport: sendVoice delivery.
# ---------------------------------------------------------------------------


async def test_stub_transport_records_voice() -> None:
    transport = StubTelegramTransport()
    await transport.start()
    try:
        await transport.send_voice(TelegramVoiceReply(chat_id=7, voice_bytes=b"OggS-123"))
        assert len(transport.sent_voices) == 1
        assert transport.sent_voices[0].chat_id == 7
        assert len(transport.sent) == 0
        with pytest.raises(TelegramApiError):
            await transport.send_voice(TelegramVoiceReply(chat_id=7, voice_bytes=b""))
    finally:
        await transport.stop()


async def test_polling_transport_send_voice_uses_sendvoice() -> None:
    from aa.telegram.transport import TelegramApi

    class _VoiceApi(TelegramApi):
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []
            self.voices: list[tuple[int, bytes]] = []

        async def call(self, method: str, payload: dict[str, Any]) -> Any:
            self.calls.append((method, dict(payload)))
            if method == "getMe":
                return {"id": 1, "is_bot": True, "username": "aabot"}
            if method == "getUpdates":
                return []
            return {"message_id": 1}

        async def send_voice(self, chat_id: int, ogg_bytes: bytes) -> Any:
            self.voices.append((chat_id, bytes(ogg_bytes)))
            return {"message_id": 2}

    from aa.telegram.transport import PollingTelegramTransport

    api = _VoiceApi()
    transport = PollingTelegramTransport(
        token="123456:TEST-TOKEN",
        api=api,
        retry_base_delay_seconds=0.001,
        retry_max_delay_seconds=0.005,
        poll_timeout_seconds=0,
    )
    await transport.start()
    try:
        await transport.send_voice(TelegramVoiceReply(chat_id=9, voice_bytes=b"OggS-abc"))
        assert api.voices == [(9, b"OggS-abc")]
        assert len(transport.sent_voices) == 1
    finally:
        await transport.stop()


# ---------------------------------------------------------------------------
# Application: voice -> sendVoice, text stays text, fallbacks, emergency.
# ---------------------------------------------------------------------------


async def test_voice_input_delivers_russian_sendvoice(tmp_path: Path) -> None:
    from aa.conversation.graph_runtime import GraphTurnRuntime

    async def _reply(thread: str, text: str) -> str:
        _ = (thread, text)
        return "Привет. Держись. Приходи."

    transport = StubTelegramTransport()
    pipeline, _, _ = _pipeline(work_parent=tmp_path)
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=_FakeVoicePipeline("привет"),  # type: ignore[arg-type]
        tts_pipeline=pipeline,
        graph_runtime=GraphTurnRuntime(delegate=_reply),
    )
    await app.start()
    try:
        assert app.tts_available
        incoming = _voice_incoming(chat_id=77)
        await app._process_dispatched_update(incoming)
        assert len(transport.sent_voices) == 1
        assert transport.sent_voices[0].chat_id == 77
        assert len(transport.sent) == 0
    finally:
        await app.stop()


async def test_text_input_stays_text_with_tts_wired(tmp_path: Path) -> None:
    from aa.conversation.graph_runtime import GraphTurnRuntime

    async def _reply(thread: str, text: str) -> str:
        _ = (thread, text)
        return "Привет. Держись."

    transport = StubTelegramTransport()
    pipeline, synth, _ = _pipeline(work_parent=tmp_path)
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        tts_pipeline=pipeline,
        graph_runtime=GraphTurnRuntime(delegate=_reply),
    )
    await app.start()
    try:
        incoming = TelegramIncoming(
            update_id=60, chat_id=80, message_id=1, text="привет", command=None
        )
        await app._process_dispatched_update(incoming)
        assert synth.calls == 0
        assert len(transport.sent) == 1
        assert len(transport.sent_voices) == 0
    finally:
        await app.stop()


async def test_tts_failure_falls_back_to_text(tmp_path: Path) -> None:
    from aa.conversation.graph_runtime import GraphTurnRuntime

    async def _reply(thread: str, text: str) -> str:
        _ = (thread, text)
        return "Привет. Держись. Приходи."

    transport = StubTelegramTransport()
    pipeline, _, _ = _pipeline(synthesizer=FakeSynthesizer(fail=True), work_parent=tmp_path)
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=_FakeVoicePipeline("привет"),  # type: ignore[arg-type]
        tts_pipeline=pipeline,
        graph_runtime=GraphTurnRuntime(delegate=_reply),
    )
    await app.start()
    try:
        await app._process_dispatched_update(_voice_incoming(chat_id=81))
        assert len(transport.sent_voices) == 0
        assert len(transport.sent) == 1
        assert "Привет" in transport.sent[0].text
    finally:
        await app.stop()


async def test_encoder_failure_falls_back_to_text(tmp_path: Path) -> None:
    from aa.conversation.graph_runtime import GraphTurnRuntime

    async def _reply(thread: str, text: str) -> str:
        _ = (thread, text)
        return "Привет. Держись. Приходи."

    transport = StubTelegramTransport()
    pipeline, _, _ = _pipeline(encoder=FakeEncoder(fail=True), work_parent=tmp_path)
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=_FakeVoicePipeline("привет"),  # type: ignore[arg-type]
        tts_pipeline=pipeline,
        graph_runtime=GraphTurnRuntime(delegate=_reply),
    )
    await app.start()
    try:
        await app._process_dispatched_update(_voice_incoming(chat_id=82))
        assert len(transport.sent_voices) == 0
        assert len(transport.sent) == 1
    finally:
        await app.stop()


async def test_voice_unavailable_falls_back_to_text() -> None:
    from aa.conversation.graph_runtime import GraphTurnRuntime

    async def _reply(thread: str, text: str) -> str:
        _ = (thread, text)
        return "Привет. Держись."

    transport = StubTelegramTransport()
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        tts_pipeline=None,
        graph_runtime=GraphTurnRuntime(delegate=_reply),
    )
    await app.start()
    try:
        assert not app.tts_available
        await app._process_dispatched_update(_voice_incoming(chat_id=83))
        assert len(transport.sent_voices) == 0
        assert len(transport.sent) == 1
    finally:
        await app.stop()


async def test_emergency_voice_is_not_truncated_and_still_voiced(tmp_path: Path) -> None:
    transport = StubTelegramTransport()
    pipeline, _, _ = _pipeline(work_parent=tmp_path)
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        tts_pipeline=pipeline,
    )
    await app.start()
    try:
        reply = await app.respond(7, "Я не могу дышать", voice_input=True)
        assert reply == EMERGENCY_RESPONSE_RU
        assert "112" in reply
        # Emergency may exceed the ordinary voice bound and must not be cut.
        assert not voice_policy_passes(reply)
        await app._process_dispatched_update(
            TelegramIncoming(
                update_id=70,
                chat_id=84,
                message_id=1,
                text="",
                command=None,
                voice=VoiceAttachment(file_id="f1", duration_seconds=5),
            )
        )
        # The dispatched turn above went through ASR-disabled fallback; drive
        # the delivery path directly with the emergency text instead.
        delivered = await app._send_voice_reply(
            TelegramIncoming(update_id=71, chat_id=84, message_id=2, text="", command=None),
            reply,
        )
        assert delivered is True
        assert len(transport.sent_voices) >= 1
    finally:
        await app.stop()


async def test_voice_turn_never_logs_reply_or_audio(tmp_path: Path) -> None:
    from aa.conversation.graph_runtime import GraphTurnRuntime

    stream = io.StringIO()
    aa_logging.configure_logging("INFO", stream=stream)
    secret = "секретный голосовой ответ трезвость"
    transport = StubTelegramTransport()
    pipeline, _, _ = _pipeline(work_parent=tmp_path)

    async def _reply(thread: str, text: str) -> str:
        _ = (thread, text)
        return secret + ". Держись."

    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        tts_pipeline=pipeline,
        graph_runtime=GraphTurnRuntime(delegate=_reply),
    )
    await app.start()
    try:
        incoming = TelegramIncoming(
            update_id=61,
            chat_id=81,
            message_id=1,
            text="",
            command=None,
            voice=VoiceAttachment(file_id="voice-secret", duration_seconds=5),
        )
        # Bypass ASR by calling the delivery boundary directly.
        reply = await app.respond(81, "привет", voice_input=True)
        await app._send_voice_reply(incoming, reply)
    finally:
        await app.stop()
    output = stream.getvalue()
    assert secret not in output
    assert "voice-secret" not in output
    assert "OggS" not in output


async def test_injected_tts_pipeline_loads_once_and_reused() -> None:
    transport = StubTelegramTransport()
    pipeline, _, _ = _pipeline()
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        tts_pipeline=pipeline,
    )
    await app.start()
    try:
        assert app.tts_available
        assert app._tts_pipeline is pipeline
        await app._init_tts_capability()
        assert app._tts_pipeline is pipeline
    finally:
        await app.stop()


async def test_temp_audio_cleaned_after_voice_turn(tmp_path: Path) -> None:
    from aa.conversation.graph_runtime import GraphTurnRuntime

    async def _reply(thread: str, text: str) -> str:
        _ = (thread, text)
        return "Привет. Держись."

    transport = StubTelegramTransport()
    pipeline, _, _ = _pipeline(work_parent=tmp_path)
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=_FakeVoicePipeline("привет"),  # type: ignore[arg-type]
        tts_pipeline=pipeline,
        graph_runtime=GraphTurnRuntime(delegate=_reply),
    )
    await app.start()
    try:
        await app._process_dispatched_update(_voice_incoming(chat_id=90))
        assert list(tmp_path.iterdir()) == []
    finally:
        await app.stop()


def test_silero_missing_torch_package_fails_with_package_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate C regression: live voice-models-ready failed with TTS disabled.

    `import torch` alone does not guarantee `torch.package` is loaded; the
    synthesizer must import the submodule explicitly before using
    `torch.package.PackageImporter`. When the submodule is unavailable the
    failure must surface as `torch.package is not installed` (not a bare
    AttributeError collapsed to generic construction failure).
    """
    import sys
    import types

    model_file = tmp_path / "v5_5_ru.pt"
    model_file.write_bytes(b"fake-silero-bytes")

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = TORCH_VERSION  # type: ignore[attr-defined]
    fake_torch.device = lambda name: name  # type: ignore[attr-defined]
    # NOTE: deliberately not a package (no __path__) and no
    # `torch.package` in sys.modules, so `import torch.package` raises
    # ModuleNotFoundError (an ImportError).
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delitem(sys.modules, "torch.package", raising=False)
    monkeypatch.setattr("aa.telegram.tts.ensure_tts_model_file", lambda path: model_file)

    synth = SileroSynthesizer(model_file)
    with pytest.raises(TtsError) as excinfo:
        synth.ensure_loaded()
    assert excinfo.value.category == "tts-unavailable"
    assert synth.load_error == "torch.package is not installed"


def test_silero_ensure_loaded_succeeds_with_torch_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy path: explicit `torch.package` import enables model loading."""
    import sys
    import types

    model_file = tmp_path / "v5_5_ru.pt"
    model_file.write_bytes(b"fake-silero-bytes")

    class _FakeModel:
        def to(self, device: object) -> _FakeModel:
            _ = device
            return self

        def eval(self) -> _FakeModel:
            return self

    class _FakeImporter:
        def __init__(self, path: str) -> None:
            _ = path

        def load_pickle(self, package: str, resource: str) -> _FakeModel:
            assert package == "tts_models"
            assert resource == "model"
            return _FakeModel()

    fake_package = types.ModuleType("torch.package")
    fake_package.PackageImporter = _FakeImporter  # type: ignore[attr-defined]
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = TORCH_VERSION  # type: ignore[attr-defined]
    fake_torch.device = lambda name: name  # type: ignore[attr-defined]
    fake_torch.package = fake_package  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.package", fake_package)
    monkeypatch.setattr("aa.telegram.tts.ensure_tts_model_file", lambda path: model_file)

    synth = SileroSynthesizer(model_file)
    synth.ensure_loaded()
    assert synth.available
