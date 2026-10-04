"""Local Russian Telegram voice recognition tests (issue #76)."""

from __future__ import annotations

import asyncio
import io
import logging
import pathlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from aa import logging as aa_logging
from aa.app import Application
from aa.config import Settings
from aa.conversation.orchestrator import meets_russian_only
from aa.conversation.output_limits import envelope_passes
from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
from aa.telegram.transport import (
    TelegramIncoming,
    VoiceAttachment,
    parse_update,
)
from aa.telegram.voice import (
    FEATURE_DIM,
    MAX_VOICE_DURATION_SECONDS,
    MAX_VOICE_FILE_BYTES,
    MODEL_FILE,
    MODEL_REPO,
    MODEL_REVISION,
    NUM_THREADS,
    SAMPLE_RATE,
    SHERPA_ONNX_VERSION,
    TOKENS_FILE,
    FfmpegDecoder,
    GigaAMRecognizer,
    VoiceError,
    VoicePipeline,
    build_pipeline,
    check_voice_bounds,
    model_file_urls,
    voice_error_reply,
)


def _settings(**overrides: str) -> Settings:
    base: dict[str, str] = {}
    base.update(overrides)
    return Settings.from_env(base)


def _stub_runtime() -> StubOpenCodeRuntime:
    return StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )


def _voice_raw(
    update_id: int,
    chat_id: int,
    message_id: int,
    file_id: str = "voice-file-1",
    duration: int = 12,
    file_size: int | None = 1024,
) -> dict[str, Any]:
    voice: dict[str, Any] = {"file_id": file_id, "duration": duration}
    if file_size is not None:
        voice["file_size"] = file_size
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "chat": {"id": chat_id, "type": "private"},
            "voice": voice,
        },
    }


class FakeFetcher:
    """In-memory voice downloader with call tracking."""

    def __init__(self, payload: bytes = b"fake-ogg", *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail
        self.calls = 0

    async def fetch(self, file_id: str) -> bytes:
        self.calls += 1
        if self.fail or not file_id:
            raise VoiceError("download-failed", "injected download failure")
        return self.payload


class FakeDecoder:
    """In-memory decoder with call tracking."""

    def __init__(
        self,
        samples: list[float] | None = None,
        *,
        fail: bool = False,
        delay: float = 0.0,
    ) -> None:
        self.samples = samples if samples is not None else [0.1] * SAMPLE_RATE
        self.fail = fail
        self.delay = delay
        self.calls = 0

    def decode(self, ogg_bytes: bytes, *, workdir: Path) -> list[float]:
        self.calls += 1
        if self.fail:
            raise VoiceError("decode-failed", "injected decode failure")
        if not ogg_bytes:
            raise VoiceError("decode-failed", "empty payload")
        # Stage a temp artifact to prove per-turn cleanup in finally.
        probe = workdir / "probe.tmp"
        probe.write_bytes(b"staged")
        if self.delay:
            import time as _time

            _time.sleep(self.delay)
        return list(self.samples)


class FakeRecognizer:
    """In-memory recognizer with concurrency tracking."""

    def __init__(
        self,
        transcript: str = "привет как дела",
        *,
        available: bool = True,
        fail: bool = False,
        delay: float = 0.0,
    ) -> None:
        self.transcript = transcript
        self._available = available
        self.fail = fail
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.max_active = 0

    @property
    def available(self) -> bool:
        return self._available

    def transcribe(self, samples: Sequence[float]) -> str:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.fail:
                raise VoiceError("asr-failed", "injected ASR failure")
            if self.delay:
                import time as _time

                _time.sleep(self.delay)
            _ = len(samples)
            return self.transcript
        finally:
            self.active -= 1


def _pipeline(
    *,
    fetcher: FakeFetcher | None = None,
    decoder: FakeDecoder | None = None,
    recognizer: FakeRecognizer | None = None,
    work_parent: Path | None = None,
) -> tuple[VoicePipeline, FakeFetcher, FakeDecoder, FakeRecognizer]:
    owned_fetcher = fetcher or FakeFetcher()
    owned_decoder = decoder or FakeDecoder()
    owned_recognizer = recognizer if recognizer is not None else FakeRecognizer()
    pipeline = build_pipeline(
        fetcher=owned_fetcher,
        decoder=owned_decoder,
        recognizer=owned_recognizer,
        work_parent=work_parent,
    )
    return pipeline, owned_fetcher, owned_decoder, owned_recognizer


# ---------------------------------------------------------------------------
# Fixed contract.
# ---------------------------------------------------------------------------


def test_fixed_contract_constants_are_pinned() -> None:
    assert MODEL_REPO == "fussraider/GigaAM-Multilingual-sherpa-onnx-ctc"
    assert MODEL_REVISION == "9f5a77e8975211abe8511693accd3a63ee1e9f43"
    assert MODEL_FILE == "large/model.int8.onnx"
    assert TOKENS_FILE == "large/tokens.txt"
    assert SHERPA_ONNX_VERSION == "1.13.8"
    assert SAMPLE_RATE == 16000
    assert FEATURE_DIM == 64
    assert MAX_VOICE_FILE_BYTES == 20 * 1024 * 1024
    assert MAX_VOICE_DURATION_SECONDS == 600
    model_url, tokens_url = model_file_urls()
    assert MODEL_REVISION in model_url
    assert model_url.endswith(MODEL_FILE)
    assert tokens_url.endswith(TOKENS_FILE)
    assert NUM_THREADS >= 1


def test_voice_error_replies_are_russian_and_bounded() -> None:
    for category in (
        "download-failed",
        "decode-failed",
        "asr-failed",
        "asr-unavailable",
        "voice-disabled",
        "empty-transcript",
        "too-large",
        "too-long",
        "unknown",
    ):
        reply = voice_error_reply(category)
        assert reply.strip()
        assert meets_russian_only(reply)
        assert envelope_passes(reply)


def test_bounds_reject_before_asr() -> None:
    with pytest.raises(VoiceError) as excinfo:
        check_voice_bounds(file_size_bytes=MAX_VOICE_FILE_BYTES + 1, duration_seconds=5)
    assert excinfo.value.category == "too-large"
    with pytest.raises(VoiceError) as excinfo2:
        check_voice_bounds(file_size_bytes=10, duration_seconds=601)
    assert excinfo2.value.category == "too-long"
    check_voice_bounds(file_size_bytes=MAX_VOICE_FILE_BYTES, duration_seconds=600)


# ---------------------------------------------------------------------------
# Transport parsing: text unchanged, voice newly accepted.
# ---------------------------------------------------------------------------


def test_parse_update_text_behavior_unchanged() -> None:
    raw = {
        "update_id": 1,
        "message": {
            "message_id": 1,
            "chat": {"id": 7, "type": "private"},
            "text": "hello",
        },
    }
    parsed = parse_update(raw)
    assert parsed is not None
    assert parsed.text == "hello"
    assert parsed.voice is None
    assert parsed.command is None


def test_parse_update_accepts_voice_note() -> None:
    parsed = parse_update(_voice_raw(11, 7, 1))
    assert parsed is not None
    assert parsed.text == ""
    assert parsed.command is None
    assert parsed.voice is not None
    assert parsed.voice.file_id == "voice-file-1"
    assert parsed.voice.duration_seconds == 12
    assert parsed.voice.file_size_bytes == 1024


def test_parse_update_ignores_non_voice_without_text() -> None:
    raw = {
        "update_id": 12,
        "message": {
            "message_id": 2,
            "chat": {"id": 7, "type": "private"},
        },
    }
    assert parse_update(raw) is None


# ---------------------------------------------------------------------------
# Pipeline: valid voice, empty, bounds, corrupt, download/ASR failures.
# ---------------------------------------------------------------------------


async def test_valid_voice_transcribes_and_trims(tmp_path: Path) -> None:
    pipeline, fetcher, decoder, recognizer = _pipeline(
        recognizer=FakeRecognizer(transcript="  привет мир  "),
        work_parent=tmp_path,
    )
    text = await pipeline.transcribe_voice(file_id="f1", file_size_bytes=100, duration_seconds=5)
    assert text == "привет мир"
    assert fetcher.calls == 1
    assert decoder.calls == 1
    assert recognizer.calls == 1
    # Temporary audio is always removed.
    assert list(tmp_path.iterdir()) == []


async def test_empty_transcript_is_bounded_failure(tmp_path: Path) -> None:
    pipeline, _, _, _ = _pipeline(
        recognizer=FakeRecognizer(transcript="   \n  "),
        work_parent=tmp_path,
    )
    with pytest.raises(VoiceError) as excinfo:
        await pipeline.transcribe_voice(file_id="f1", duration_seconds=5)
    assert excinfo.value.category == "empty-transcript"
    assert list(tmp_path.iterdir()) == []


async def test_long_duration_rejected_before_asr(tmp_path: Path) -> None:
    pipeline, fetcher, _, recognizer = _pipeline(work_parent=tmp_path)
    with pytest.raises(VoiceError) as excinfo:
        await pipeline.transcribe_voice(file_id="f1", duration_seconds=601)
    assert excinfo.value.category == "too-long"
    assert fetcher.calls == 0
    assert recognizer.calls == 0


async def test_large_file_rejected_before_asr(tmp_path: Path) -> None:
    pipeline, fetcher, _, recognizer = _pipeline(work_parent=tmp_path)
    with pytest.raises(VoiceError) as excinfo:
        await pipeline.transcribe_voice(
            file_id="f1",
            file_size_bytes=MAX_VOICE_FILE_BYTES + 1,
            duration_seconds=5,
        )
    assert excinfo.value.category == "too-large"
    assert fetcher.calls == 0
    assert recognizer.calls == 0


async def test_large_download_rejected_before_asr(tmp_path: Path) -> None:
    big = b"x" * (MAX_VOICE_FILE_BYTES + 1)
    pipeline, _, _, recognizer = _pipeline(
        fetcher=FakeFetcher(payload=big),
        work_parent=tmp_path,
    )
    with pytest.raises(VoiceError) as excinfo:
        await pipeline.transcribe_voice(file_id="f1", duration_seconds=5)
    assert excinfo.value.category == "too-large"
    assert recognizer.calls == 0


async def test_corrupt_ogg_is_decode_failure(tmp_path: Path) -> None:
    pipeline, _, _, recognizer = _pipeline(
        decoder=FakeDecoder(fail=True),
        work_parent=tmp_path,
    )
    with pytest.raises(VoiceError) as excinfo:
        await pipeline.transcribe_voice(file_id="f1", duration_seconds=5)
    assert excinfo.value.category == "decode-failed"
    assert recognizer.calls == 0
    assert list(tmp_path.iterdir()) == []


async def test_download_failure_is_bounded(tmp_path: Path) -> None:
    pipeline, _, _, recognizer = _pipeline(
        fetcher=FakeFetcher(fail=True),
        work_parent=tmp_path,
    )
    with pytest.raises(VoiceError) as excinfo:
        await pipeline.transcribe_voice(file_id="f1", duration_seconds=5)
    assert excinfo.value.category == "download-failed"
    assert recognizer.calls == 0


async def test_asr_unavailable_fails_voice_only(tmp_path: Path) -> None:
    pipeline, fetcher, _, _ = _pipeline(
        recognizer=FakeRecognizer(available=False),
        work_parent=tmp_path,
    )
    with pytest.raises(VoiceError) as excinfo:
        await pipeline.transcribe_voice(file_id="f1", duration_seconds=5)
    assert excinfo.value.category == "asr-unavailable"
    assert fetcher.calls == 0


async def test_asr_inference_failure_is_bounded(tmp_path: Path) -> None:
    pipeline, _, _, _ = _pipeline(
        recognizer=FakeRecognizer(fail=True),
        work_parent=tmp_path,
    )
    with pytest.raises(VoiceError) as excinfo:
        await pipeline.transcribe_voice(file_id="f1", duration_seconds=5)
    assert excinfo.value.category == "asr-failed"
    assert list(tmp_path.iterdir()) == []


async def test_missing_recognizer_means_disabled(tmp_path: Path) -> None:
    pipeline = build_pipeline(
        fetcher=FakeFetcher(),
        decoder=FakeDecoder(),
        recognizer=None,
        work_parent=tmp_path,
    )
    with pytest.raises(VoiceError) as excinfo:
        await pipeline.transcribe_voice(file_id="f1", duration_seconds=5)
    assert excinfo.value.category == "asr-unavailable"


async def test_concurrent_voice_turns_obey_single_asr_bound(tmp_path: Path) -> None:
    recognizer = FakeRecognizer(transcript="привет", delay=0.05)
    pipeline, _, _, _ = _pipeline(recognizer=recognizer, work_parent=tmp_path)
    results = await asyncio.gather(
        pipeline.transcribe_voice(file_id="a", duration_seconds=5),
        pipeline.transcribe_voice(file_id="b", duration_seconds=5),
        pipeline.transcribe_voice(file_id="c", duration_seconds=5),
    )
    assert list(results) == ["привет", "привет", "привет"]
    assert recognizer.calls == 3
    assert recognizer.max_active == 1


async def test_temp_files_cleaned_on_failure(tmp_path: Path) -> None:
    pipeline, _, _, _ = _pipeline(
        decoder=FakeDecoder(fail=True),
        work_parent=tmp_path,
    )
    with pytest.raises(VoiceError):
        await pipeline.transcribe_voice(file_id="f1", duration_seconds=5)
    assert list(tmp_path.iterdir()) == []


async def test_no_transcript_or_audio_in_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    secret_transcript = "секретная фраза про трезвость"
    pipeline, _, _, _ = _pipeline(
        recognizer=FakeRecognizer(transcript=secret_transcript),
        work_parent=tmp_path,
    )
    with caplog.at_level(logging.INFO, logger="aa.telegram.voice"):
        await pipeline.transcribe_voice(file_id="f1", duration_seconds=5)
    records = "\n".join(record.getMessage() for record in caplog.records)
    assert secret_transcript not in records
    assert "f1" not in records


def test_ffmpeg_decoder_rejects_empty_payload(tmp_path: Path) -> None:
    decoder = FfmpegDecoder()
    with pytest.raises(VoiceError) as excinfo:
        decoder.decode(b"", workdir=tmp_path)
    assert excinfo.value.category == "decode-failed"


def test_gigaam_recognizer_without_load_is_unavailable(tmp_path: Path) -> None:
    recognizer = GigaAMRecognizer(tmp_path / "models")
    assert not recognizer.available
    with pytest.raises(VoiceError) as excinfo:
        recognizer.transcribe([0.0] * 160)
    assert excinfo.value.category == "asr-unavailable"


# ---------------------------------------------------------------------------
# Application: voice reaches the production text boundary as voice_input=True.
# ---------------------------------------------------------------------------


async def test_voice_reaches_production_turn_boundary(tmp_path: Path) -> None:
    pipeline, _, _, _ = _pipeline(
        recognizer=FakeRecognizer(transcript="привет"),
        work_parent=tmp_path,
    )
    app = Application(_settings(), opencode_runtime=_stub_runtime(), voice_pipeline=pipeline)
    await app.start()
    try:
        assert app.voice_available
        seen: list[tuple[int, str, bool]] = []
        original_respond = app.respond

        async def _spy(chat_id: int, text: str, *, voice_input: bool = False) -> str:
            seen.append((chat_id, text, voice_input))
            return await original_respond(chat_id, text, voice_input=voice_input)

        app.respond = _spy  # type: ignore[method-assign]
        incoming = TelegramIncoming(
            update_id=50,
            chat_id=77,
            message_id=1,
            text="",
            command=None,
            voice=VoiceAttachment(file_id="f1", duration_seconds=5, file_size_bytes=100),
        )
        await app._process_dispatched_update(incoming)
        assert len(seen) == 1
        assert seen[0] == (77, "привет", True)
    finally:
        await app.stop()


async def test_voice_failures_send_russian_error_and_keep_poller(tmp_path: Path) -> None:
    from aa.telegram.transport import StubTelegramTransport

    for category_recognizer, _ in (
        (FakeRecognizer(transcript="   "), "empty"),
        (FakeRecognizer(fail=True), "asr"),
    ):
        transport = StubTelegramTransport()
        pipeline, _, _, _ = _pipeline(
            recognizer=category_recognizer,
            work_parent=tmp_path,
        )
        app = Application(
            _settings(),
            transport=transport,
            opencode_runtime=_stub_runtime(),
            voice_pipeline=pipeline,
        )
        await app.start()
        try:
            incoming = TelegramIncoming(
                update_id=51,
                chat_id=78,
                message_id=1,
                text="",
                command=None,
                voice=VoiceAttachment(file_id="f1", duration_seconds=5, file_size_bytes=100),
            )
            await app._process_dispatched_update(incoming)
            assert len(transport.sent) == 1
            assert meets_russian_only(transport.sent[0].text)
            assert envelope_passes(transport.sent[0].text)
            assert app.running
        finally:
            await app.stop()


async def test_voice_disabled_sends_unavailable_reply() -> None:
    from aa.telegram.transport import StubTelegramTransport

    transport = StubTelegramTransport()
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=None,
    )
    await app.start()
    try:
        assert not app.voice_available
        incoming = TelegramIncoming(
            update_id=52,
            chat_id=79,
            message_id=1,
            text="",
            command=None,
            voice=VoiceAttachment(file_id="f1", duration_seconds=5, file_size_bytes=100),
        )
        await app._process_dispatched_update(incoming)
        assert len(transport.sent) == 1
        assert meets_russian_only(transport.sent[0].text)
    finally:
        await app.stop()


async def test_text_turns_unchanged_with_voice_wired(tmp_path: Path) -> None:
    from aa.telegram.transport import StubTelegramTransport

    transport = StubTelegramTransport()
    pipeline, _, _, recognizer = _pipeline(work_parent=tmp_path)
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=pipeline,
    )
    await app.start()
    try:
        incoming = TelegramIncoming(
            update_id=60, chat_id=80, message_id=1, text="привет", command=None
        )
        await app._process_dispatched_update(incoming)
        assert recognizer.calls == 0
        assert len(transport.sent) == 1
    finally:
        await app.stop()


async def test_voice_turn_never_logs_transcript(tmp_path: Path) -> None:
    from aa.telegram.transport import StubTelegramTransport

    stream = io.StringIO()
    aa_logging.configure_logging("INFO", stream=stream)
    secret = "секретный голосовой текст"
    transport = StubTelegramTransport()
    pipeline, _, _, _ = _pipeline(
        recognizer=FakeRecognizer(transcript=secret),
        work_parent=tmp_path,
    )
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=pipeline,
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
        await app._process_dispatched_update(incoming)
    finally:
        await app.stop()
    output = stream.getvalue()
    assert secret not in output
    assert "voice-secret" not in output


async def test_asr_init_failure_keeps_text_poller_alive() -> None:
    from aa.telegram.transport import StubTelegramTransport

    transport = StubTelegramTransport()
    failing = build_pipeline(
        fetcher=FakeFetcher(fail=True),
        decoder=FakeDecoder(),
        recognizer=FakeRecognizer(available=False),
    )
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=failing,
    )
    await app.start()
    try:
        assert not app.voice_available
        # Ordinary text still works when voice is unavailable.
        incoming = TelegramIncoming(
            update_id=62, chat_id=82, message_id=1, text="привет", command=None
        )
        await app._process_dispatched_update(incoming)
        assert len(transport.sent) == 1
        assert app.running
    finally:
        await app.stop()


def test_voice_settings_default_and_env() -> None:
    assert _settings().aa_voice_model_dir == "./models/gigaam"
    custom = _settings(AA_VOICE_MODEL_DIR="/tmp/voice-models")
    assert custom.aa_voice_model_dir == "/tmp/voice-models"
    custom.validate()
    assert "AA_VOICE_MODEL_DIR" in set(Settings.RESERVED_ENV_NAMES)
    assert custom.to_safe_dict()["aa_voice_model_dir"] == "/tmp/voice-models"


def test_model_urls_use_pinned_revision() -> None:
    model_url, tokens_url = model_file_urls()
    assert "9f5a77e8975211abe8511693accd3a63ee1e9f43" in model_url
    assert "9f5a77e8975211abe8511693accd3a63ee1e9f43" in tokens_url


def _check_no_leak_in_file(path: pathlib.Path, *secrets: str) -> None:
    text = path.read_text(encoding="utf-8")
    for secret in secrets:
        assert secret not in text


def test_no_audio_or_transcript_persisted_after_turn(tmp_path: Path) -> None:
    # The pipeline stages audio only under a TemporaryDirectory that is
    # removed in finally; nothing may remain under the parent afterwards.
    async def _run() -> None:
        pipeline, _, _, _ = _pipeline(work_parent=tmp_path)
        await pipeline.transcribe_voice(file_id="f1", duration_seconds=5)

    asyncio.run(_run())
    assert list(tmp_path.iterdir()) == []
    _check_no_leak_in_file(pathlib.Path(__file__))
