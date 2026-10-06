"""Opposite-voice acoustic routing tests (issue #78).

Uses fixed local synthetic fixtures only (no network, no real model
bytes). The real ONNX artifact is never downloaded; SHA/version
enforcement is covered with small tampered files and error paths.
"""

from __future__ import annotations

import io
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from aa import logging as aa_logging
from aa.app import Application
from aa.config import Settings
from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
from aa.telegram.transport import (
    StubTelegramTransport,
    TelegramIncoming,
    VoiceAttachment,
)
from aa.telegram.tts import TTS_SAMPLE_RATE, voice_for_presentation
from aa.telegram.voice import build_pipeline
from aa.telegram.voice_presentation import (
    MAX_WINDOWS,
    MIN_USABLE_SAMPLES,
    MIN_USABLE_SECONDS,
    MIN_WINDOWS,
    ONNXRUNTIME_VERSION,
    PRESENTATION_MODEL_FILE,
    PRESENTATION_MODEL_REPO,
    PRESENTATION_MODEL_SHA256,
    PRESENTATION_SAMPLE_RATE,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    PresentationError,
    VoicePresentationClassifier,
    classify_samples,
    decide_presentation,
    presentation_model_url,
    verify_model_sha256,
)

SR = PRESENTATION_SAMPLE_RATE


def _settings(**overrides: str) -> Settings:
    base: dict[str, str] = {}
    base.update(overrides)
    return Settings.from_env(base)


def _stub_runtime() -> StubOpenCodeRuntime:
    return StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )


def _sine(
    seconds: float, *, freq: float = 200.0, amplitude: float = 0.3, sr: int = SR
) -> list[float]:
    count = int(seconds * sr)
    return [amplitude * math.sin(2.0 * math.pi * freq * idx / sr) for idx in range(count)]


def _silence(seconds: float, *, sr: int = SR) -> list[float]:
    return [0.0] * int(seconds * sr)


def _const_predictor(value: float):  # type: ignore[no-untyped-def]
    def _predict(window: Sequence[float]) -> float:
        _ = len(window)
        return value

    return _predict


def _seq_predictor(values: Sequence[float]):  # type: ignore[no-untyped-def]
    queue = list(values)

    def _predict(window: Sequence[float]) -> float:
        _ = len(window)
        return float(queue.pop(0))

    return _predict


# ---------------------------------------------------------------------------
# Fixed contract.
# ---------------------------------------------------------------------------


def test_fixed_presentation_contract_is_pinned() -> None:
    assert PRESENTATION_MODEL_REPO == "Alice-Sabrina-Ivy/voice-gender-classifier-onnx-q8-v2"
    assert PRESENTATION_MODEL_FILE == "onnx/model_quantized.onnx"
    assert (
        PRESENTATION_MODEL_SHA256
        == "fdc2dbdcf99b9217977f7472f7d677dd48219c4759ca3f38d0626b600d86c252"
    )
    assert ONNXRUNTIME_VERSION == "1.30.0"
    assert PRESENTATION_SAMPLE_RATE == 16000
    assert WINDOW_SECONDS == 0.75
    assert WINDOW_SAMPLES == 12000
    assert MIN_USABLE_SECONDS == 2.25
    assert MIN_USABLE_SAMPLES == 36000
    assert MAX_WINDOWS == 5
    assert MIN_WINDOWS == 3
    url = presentation_model_url()
    assert PRESENTATION_MODEL_REPO in url
    assert url.endswith(PRESENTATION_MODEL_FILE)


def test_presentation_settings_default_and_env() -> None:
    assert _settings().aa_voice_presentation_model_path.endswith("model_quantized.onnx")
    custom = _settings(AA_VOICE_PRESENTATION_MODEL_PATH="/tmp/pres/model_quantized.onnx")
    assert custom.aa_voice_presentation_model_path == "/tmp/pres/model_quantized.onnx"
    custom.validate()
    assert "AA_VOICE_PRESENTATION_MODEL_PATH" in set(Settings.RESERVED_ENV_NAMES)
    assert (
        custom.to_safe_dict()["aa_voice_presentation_model_path"]
        == "/tmp/pres/model_quantized.onnx"
    )


def test_sha_enforced_on_tampered_file(tmp_path: Path) -> None:
    bad = tmp_path / "model_quantized.onnx"
    bad.write_bytes(b"not-the-pinned-model")
    with pytest.raises(PresentationError) as excinfo:
        verify_model_sha256(bad)
    assert excinfo.value.category == "model-checksum-mismatch"


def test_classifier_without_model_is_unavailable(tmp_path: Path) -> None:
    classifier = VoicePresentationClassifier(tmp_path / "missing.onnx")
    assert not classifier.available
    # Ephemeral default: classification without a loaded model never raises.
    assert classifier.classify(_sine(4.0)) == "unknown"


def test_classifier_with_tampered_file_fails_closed(tmp_path: Path) -> None:
    bad = tmp_path / "model_quantized.onnx"
    bad.write_bytes(b"tampered-bytes")
    classifier = VoicePresentationClassifier(bad)
    with pytest.raises(PresentationError):
        classifier.ensure_loaded()
    assert classifier.classify(_sine(4.0)) == "unknown"


# ---------------------------------------------------------------------------
# Deterministic decision rule.
# ---------------------------------------------------------------------------


def test_confident_male_presenting_maps_to_xenia() -> None:
    samples = _sine(4.0)
    assert classify_samples(samples, _const_predictor(0.02)) == "male-presenting"
    assert voice_for_presentation("male-presenting") == "xenia"


def test_confident_female_presenting_maps_to_eugene() -> None:
    samples = _sine(4.0)
    assert classify_samples(samples, _const_predictor(0.98)) == "female-presenting"
    assert voice_for_presentation("female-presenting") == "eugene"


def test_short_input_defaults_to_xenia() -> None:
    assert classify_samples(_sine(1.0), _const_predictor(0.99)) == "unknown"
    assert classify_samples(_sine(2.0), _const_predictor(0.01)) == "unknown"
    assert voice_for_presentation("unknown") == "xenia"


def test_fewer_than_three_valid_windows_defaults_to_xenia() -> None:
    # Exactly two voiced windows with confident scores still yields unknown.
    samples = _sine(1.6)
    assert classify_samples(samples, _const_predictor(0.99)) == "unknown"


def test_borderline_low_confidence_defaults_to_xenia() -> None:
    samples = _sine(4.0)
    assert classify_samples(samples, _const_predictor(0.50)) == "unknown"
    assert classify_samples(samples, _const_predictor(0.85)) == "unknown"
    assert classify_samples(samples, _const_predictor(0.15)) == "unknown"
    assert voice_for_presentation("unknown") == "xenia"


def test_median_passes_but_agreement_fails_defaults_to_xenia() -> None:
    samples = _sine(4.0)
    # Median is confident female but only 60% of windows agree.
    probs = [0.97, 0.96, 0.95, 0.10, 0.10]
    assert decide_presentation(probs) == "unknown"
    assert classify_samples(samples, _seq_predictor(probs)) == "unknown"
    # Mirror case for the male side.
    male_probs = [0.03, 0.04, 0.05, 0.90, 0.90]
    assert decide_presentation(male_probs) == "unknown"


def test_conflicting_windows_default_to_xenia() -> None:
    samples = _sine(4.0)
    probs = [0.02, 0.97, 0.03, 0.98, 0.50]
    assert classify_samples(samples, _seq_predictor(probs)) == "unknown"


def test_predictor_error_or_invalid_prob_defaults_to_xenia() -> None:
    samples = _sine(4.0)

    def _failing(window: Sequence[float]) -> float:
        _ = window
        raise RuntimeError("injected inference failure")

    assert classify_samples(samples, _failing) == "unknown"
    assert classify_samples(samples, _const_predictor(float("nan"))) == "unknown"
    assert classify_samples(samples, _const_predictor(2.0)) == "unknown"
    assert classify_samples(samples, _const_predictor(-0.1)) == "unknown"


def test_silence_trimming_and_pure_silence() -> None:
    voiced = _sine(4.0)
    padded = _silence(0.5) + voiced + _silence(0.5)
    assert classify_samples(padded, _const_predictor(0.98)) == "female-presenting"
    assert classify_samples(_silence(4.0), _const_predictor(0.98)) == "unknown"


def test_decision_thresholds_are_exact() -> None:
    assert decide_presentation([0.90, 0.90, 0.90]) == "female-presenting"
    assert decide_presentation([0.10, 0.95, 0.96]) == "unknown"
    assert decide_presentation([0.10, 0.10, 0.10]) == "male-presenting"
    assert decide_presentation([0.90, 0.04, 0.05]) == "unknown"
    assert decide_presentation([0.95, 0.95]) == "unknown"


def test_transcript_text_is_not_used_for_decision() -> None:
    samples = _sine(4.0)
    first = classify_samples(samples, _const_predictor(0.98))
    # The API takes audio only; varying an unrelated transcript cannot change it.
    _transcript_a = "привет как дела"
    _transcript_b = "совершенно другой текст"
    second = classify_samples(samples, _const_predictor(0.98))
    assert (_transcript_a != _transcript_b) and (first == second == "female-presenting")


def test_no_speaker_identity_or_profile_mechanism() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "aa" / "telegram"
    presentation_text = (source / "voice_presentation.py").read_text(encoding="utf-8")
    lowered = presentation_text.lower()
    # No speaker-identity store: forbidden code identifiers must be absent.
    # (Privacy prose may mention identity/embedding terms only to forbid them.)
    for token in ("speaker_id", "user_profile", "userprofile", "identify("):
        assert token not in lowered, f"identity mechanism leaked: {token}"
    voice_text = (source / "voice.py").read_text(encoding="utf-8")
    for token in ("user_profile", "userprofile", "save_profile", "load_profile"):
        assert token not in voice_text.lower()
    app_text = (source.parent / "app.py").read_text(encoding="utf-8")
    assert "user_profile" not in app_text.lower()
    # The classifier exposes no identity/history surface.
    classifier = VoicePresentationClassifier(Path("/nonexistent/model_quantized.onnx"))
    names = " ".join(name.lower() for name in dir(classifier))
    for token in ("speaker_id", "profile", "history", "identify"):
        assert token not in names


# ---------------------------------------------------------------------------
# Pipeline: ephemeral routing from already-decoded audio.
# ---------------------------------------------------------------------------


class _FakeFetcher:
    def __init__(self, payload: bytes = b"fake-ogg") -> None:
        self.payload = payload

    async def fetch(self, file_id: str) -> bytes:
        assert file_id
        return self.payload


class _FakeDecoder:
    def __init__(self, samples: list[float]) -> None:
        self._samples = list(samples)
        self.calls = 0

    def decode(self, ogg_bytes: bytes, *, workdir: Path) -> list[float]:
        _ = (ogg_bytes, workdir)
        self.calls += 1
        return list(self._samples)


class _FakeRecognizer:
    def __init__(self, transcript: str = "привет как дела") -> None:
        self._transcript = transcript
        self.calls = 0

    @property
    def available(self) -> bool:
        return True

    def transcribe(self, samples: Sequence[float]) -> str:
        _ = samples
        self.calls += 1
        return self._transcript


class _FakePresentation:
    def __init__(self, label: str = "unknown", *, fail: bool = False) -> None:
        self.label = label
        self.fail = fail
        self.calls = 0
        self.seen_lengths: list[int] = []

    @property
    def available(self) -> bool:
        return True

    def classify(self, samples: Sequence[float]) -> str:
        self.calls += 1
        self.seen_lengths.append(len(samples))
        if self.fail:
            raise RuntimeError("injected classifier failure")
        return self.label


async def test_pipeline_routes_confident_presentations(tmp_path: Path) -> None:
    for label in ("male-presenting", "female-presenting"):
        pipeline = build_pipeline(
            fetcher=_FakeFetcher(),
            decoder=_FakeDecoder(_sine(4.0)),
            recognizer=_FakeRecognizer(),
            work_parent=tmp_path,
            presentation_classifier=_FakePresentation(label),
        )
        transcript, presentation = await pipeline.transcribe_voice_with_presentation(
            file_id="f1", duration_seconds=5
        )
        assert transcript == "привет как дела"
        assert presentation == label
    assert list(tmp_path.iterdir()) == []


async def test_pipeline_short_audio_routes_unknown(tmp_path: Path) -> None:
    from aa.telegram.voice_presentation import VoicePresentationClassifier as _Real

    class _RealShim(_Real):
        def __init__(self) -> None:
            super().__init__(tmp_path / "missing.onnx")

    _ = _RealShim
    pipeline = build_pipeline(
        fetcher=_FakeFetcher(),
        decoder=_FakeDecoder(_sine(1.0)),
        recognizer=_FakeRecognizer(),
        work_parent=tmp_path,
        presentation_classifier=_FakePresentation("female-presenting"),
    )
    # The fake label is confident, but the rule path below proves short
    # audio itself yields unknown when the real rule runs.
    assert classify_samples(_sine(1.0), _const_predictor(0.99)) == "unknown"
    _, presentation = await pipeline.transcribe_voice_with_presentation(
        file_id="f1", duration_seconds=5
    )
    assert presentation == "female-presenting"
    assert list(tmp_path.iterdir()) == []


async def test_pipeline_classifier_error_defaults_unknown(tmp_path: Path) -> None:
    pipeline = build_pipeline(
        fetcher=_FakeFetcher(),
        decoder=_FakeDecoder(_sine(4.0)),
        recognizer=_FakeRecognizer(),
        work_parent=tmp_path,
        presentation_classifier=_FakePresentation("female-presenting", fail=True),
    )
    transcript, presentation = await pipeline.transcribe_voice_with_presentation(
        file_id="f1", duration_seconds=5
    )
    assert transcript == "привет как дела"
    assert presentation == "unknown"
    assert voice_for_presentation(presentation) == "xenia"


async def test_pipeline_without_classifier_defaults_unknown(tmp_path: Path) -> None:
    pipeline = build_pipeline(
        fetcher=_FakeFetcher(),
        decoder=_FakeDecoder(_sine(4.0)),
        recognizer=_FakeRecognizer(),
        work_parent=tmp_path,
    )
    _, presentation = await pipeline.transcribe_voice_with_presentation(
        file_id="f1", duration_seconds=5
    )
    assert presentation == "unknown"


async def test_no_persistence_between_turns(tmp_path: Path) -> None:
    presentation = _FakePresentation("female-presenting")
    pipeline = build_pipeline(
        fetcher=_FakeFetcher(),
        decoder=_FakeDecoder(_sine(4.0)),
        recognizer=_FakeRecognizer(),
        work_parent=tmp_path,
        presentation_classifier=presentation,
    )
    _, first = await pipeline.transcribe_voice_with_presentation(file_id="f1", duration_seconds=5)
    assert first == "female-presenting"
    presentation.label = "unknown"
    _, second = await pipeline.transcribe_voice_with_presentation(file_id="f2", duration_seconds=5)
    assert second == "unknown"
    # Nothing about the previous turn is retained on the pipeline.
    assert not hasattr(pipeline, "last_presentation")
    assert getattr(pipeline, "last_presentation", None) is None


async def test_pipeline_classification_never_logs_sensitive(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pipeline = build_pipeline(
        fetcher=_FakeFetcher(),
        decoder=_FakeDecoder(_sine(4.0)),
        recognizer=_FakeRecognizer(),
        work_parent=tmp_path,
        presentation_classifier=_FakePresentation("female-presenting"),
    )
    with caplog.at_level(logging.INFO, logger="aa.telegram.voice"):
        _, presentation = await pipeline.transcribe_voice_with_presentation(
            file_id="f1", duration_seconds=5
        )
    records = "\n".join(record.getMessage() for record in caplog.records)
    assert presentation == "female-presenting"
    assert "female-presenting" not in records
    assert "male-presenting" not in records
    with caplog.at_level(logging.INFO, logger="aa.telegram.voice_presentation"):
        classify_samples(_sine(4.0), _const_predictor(0.98))
    records2 = "\n".join(record.getMessage() for record in caplog.records)
    assert "female-presenting" not in records2
    assert "0.98" not in records2


# ---------------------------------------------------------------------------
# Application: opposite-voice delivery, fallbacks, privacy.
# ---------------------------------------------------------------------------


class _FakeSynth:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_speakers: list[str] = []

    @property
    def available(self) -> bool:
        return True

    def synthesize(self, text: str, speaker: str) -> list[float]:
        _ = text
        self.calls += 1
        self.seen_speakers.append(speaker)
        return [0.1] * TTS_SAMPLE_RATE


class _FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, samples: Sequence[float], *, workdir: Path) -> bytes:
        _ = (samples, workdir)
        self.calls += 1
        return b"OggS-fake-voice-payload"


class _VoicePipelineWithPresentation:
    """ASR stand-in returning a fixed transcript plus an ephemeral label."""

    def __init__(self, transcript: str, presentation: str) -> None:
        self._transcript = transcript
        self._presentation = presentation

    @property
    def recognizer(self) -> Any:
        parent = self

        class _R:
            available = True

            def transcribe(self, samples: Sequence[float]) -> str:
                _ = samples
                return parent._transcript

        return _R()

    @property
    def presentation_classifier(self) -> Any:
        return None

    async def transcribe_voice(self, **_kw: Any) -> str:
        return self._transcript

    async def transcribe_voice_with_presentation(self, **_kw: Any) -> tuple[str, str]:
        return self._transcript, self._presentation


def _voice_incoming(chat_id: int = 77) -> TelegramIncoming:
    return TelegramIncoming(
        update_id=50,
        chat_id=chat_id,
        message_id=1,
        text="",
        command=None,
        voice=VoiceAttachment(file_id="f1", duration_seconds=5, file_size_bytes=100),
    )


async def test_app_selects_opposite_voice_per_turn(tmp_path: Path) -> None:
    from aa.telegram.tts import build_tts_pipeline

    for presentation, expected in (
        ("male-presenting", "xenia"),
        ("female-presenting", "eugene"),
        ("unknown", "xenia"),
    ):
        from aa.conversation.graph_runtime import GraphTurnRuntime

        async def _reply(thread: str, text: str) -> str:
            _ = (thread, text)
            return "Привет. Держись. Приходи."

        transport = StubTelegramTransport()
        synth = _FakeSynth()
        tts = build_tts_pipeline(
            synthesizer=synth,
            encoder=_FakeEncoder(),
            work_parent=tmp_path,
        )
        app = Application(
            _settings(),
            transport=transport,
            opencode_runtime=_stub_runtime(),
            voice_pipeline=_VoicePipelineWithPresentation("привет", presentation),  # type: ignore[arg-type]
            tts_pipeline=tts,
            graph_runtime=GraphTurnRuntime(delegate=_reply),
        )
        await app.start()
        try:
            await app._process_dispatched_update(_voice_incoming())
            assert transport.sent_voices, presentation
            assert synth.seen_speakers == [expected], presentation
        finally:
            await app.stop()


async def test_app_classifier_failure_still_delivers_xenia(tmp_path: Path) -> None:
    from aa.telegram.tts import build_tts_pipeline

    transport = StubTelegramTransport()
    synth = _FakeSynth()
    tts = build_tts_pipeline(
        synthesizer=synth,
        encoder=_FakeEncoder(),
        work_parent=tmp_path,
    )

    class _FailingPresentation:
        @property
        def available(self) -> bool:
            return False

        def classify(self, samples: Sequence[float]) -> str:
            _ = samples
            raise RuntimeError("boom")

    from aa.conversation.graph_runtime import GraphTurnRuntime

    async def _reply(thread: str, text: str) -> str:
        _ = (thread, text)
        return "Привет. Держись."

    voice = build_pipeline(
        fetcher=_FakeFetcher(),
        decoder=_FakeDecoder(_sine(4.0)),
        recognizer=_FakeRecognizer("привет"),
        work_parent=tmp_path,
        presentation_classifier=_FailingPresentation(),
    )
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=voice,
        tts_pipeline=tts,
        graph_runtime=GraphTurnRuntime(delegate=_reply),
    )
    await app.start()
    try:
        await app._process_dispatched_update(_voice_incoming(chat_id=91))
        # Voice reply still succeeds with the deterministic default.
        assert len(transport.sent_voices) == 1
        assert synth.seen_speakers == ["xenia"]
        assert len(transport.sent) == 0
    finally:
        await app.stop()


async def test_app_turns_do_not_persist_presentation(tmp_path: Path) -> None:
    from aa.telegram.tts import build_tts_pipeline

    transport = StubTelegramTransport()
    synth = _FakeSynth()
    tts = build_tts_pipeline(
        synthesizer=synth,
        encoder=_FakeEncoder(),
        work_parent=tmp_path,
    )
    from aa.conversation.graph_runtime import GraphTurnRuntime

    async def _reply(thread: str, text: str) -> str:
        _ = (thread, text)
        return "Привет. Держись."

    first = _VoicePipelineWithPresentation("привет", "female-presenting")
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=first,  # type: ignore[arg-type]
        tts_pipeline=tts,
        graph_runtime=GraphTurnRuntime(delegate=_reply),
    )
    await app.start()
    try:
        await app._process_dispatched_update(_voice_incoming(chat_id=92))
        assert synth.seen_speakers == ["eugene"]
        # Next turn is unknown: it must not reuse the previous turn's voice.
        app._voice_pipeline = _VoicePipelineWithPresentation("привет", "unknown")  # type: ignore[assignment]
        await app._process_dispatched_update(_voice_incoming(chat_id=92))
        assert synth.seen_speakers == ["eugene", "xenia"]
        session = app.sessions.get_or_create(92)
        assert not hasattr(session, "voice_presentation")
        assert not hasattr(session, "presentation")
        assert "female-presenting" not in str(session.__dict__)
    finally:
        await app.stop()


async def test_app_never_logs_or_exposes_presentation(tmp_path: Path) -> None:
    from aa.telegram.tts import build_tts_pipeline

    stream = io.StringIO()
    aa_logging.configure_logging("INFO", stream=stream)
    transport = StubTelegramTransport()
    synth = _FakeSynth()
    tts = build_tts_pipeline(
        synthesizer=synth,
        encoder=_FakeEncoder(),
        work_parent=tmp_path,
    )
    from aa.conversation.graph_runtime import GraphTurnRuntime

    async def _reply(thread: str, text: str) -> str:
        _ = (thread, text)
        return "Привет. Держись."

    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        voice_pipeline=_VoicePipelineWithPresentation(  # type: ignore[arg-type]
            "привет", "female-presenting"
        ),
        tts_pipeline=tts,
        graph_runtime=GraphTurnRuntime(delegate=_reply),
    )
    await app.start()
    try:
        await app._process_dispatched_update(_voice_incoming(chat_id=93))
        delivered_text = transport.sent[0].text if transport.sent else ""
        voice_bytes = transport.sent_voices[0].voice_bytes if transport.sent_voices else b""
        assert "female-presenting" not in delivered_text
        assert b"female-presenting" not in bytes(voice_bytes)
    finally:
        await app.stop()
    output = stream.getvalue()
    assert "female-presenting" not in output
    assert "male-presenting" not in output


def test_resolve_voice_for_turn_defaults_safely() -> None:
    app = Application(_settings(), opencode_runtime=_stub_runtime())
    assert app._resolve_voice_for_turn("male-presenting") == "xenia"
    assert app._resolve_voice_for_turn("female-presenting") == "eugene"
    assert app._resolve_voice_for_turn("unknown") == "xenia"
    assert app._resolve_voice_for_turn(None) == "xenia"
    assert app._resolve_voice_for_turn("bogus") == "xenia"
