"""Local Russian Telegram voice replies with Silero TTS (issue #77).

Fixed implementation contract (no model-selection research):

- TTS model: Silero ``v5_5_ru``;
- model URL: ``https://models.silero.ai/models/tts/ru/v5_5_ru.pt``;
- runtime: ``torch==2.14.1``, CPU only;
- synthesis sample rate: 48,000 Hz;
- female-presenting production voice: ``xenia``;
- male-presenting production voice: ``eugene``;
- default voice when #78 reports unknown/unavailable: ``xenia``;
- Telegram encoding: ``ffmpeg`` -> OGG/Opus, mono, 48 kHz, ``libopus``,
  32 kbit/s;
- delivery through Telegram ``sendVoice``.

Only ``xenia`` and ``eugene`` are adaptive production voices. Voice
selection for #78 (opposite-voice routing) is represented here as a
deterministic resolver that defaults to ``xenia``; the acoustic
classifier itself lives in #78.

Voice-mode response contract (ordinary non-emergency turns):

- generation target of 2-4 short sentences;
- hard contract of at most 4 sentences and at most 80 Russian words;
- enforced in the LLM generation/output budget, not by blind truncation;
- exactly one compact-regeneration attempt on violation;
- deterministic fallback to the longest complete leading sentences that
  fit both bounds.

Safety/emergency responses may exceed the brevity bound when required
for correctness.

Privacy: this module never logs synthesized text, generated audio, or
temporary paths containing user identifiers. Only sizes, durations,
voice names from the fixed set, and failure categories are logged.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger("aa.telegram.tts")

TTS_MODEL_ID = "v5_5_ru"
TTS_MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
TORCH_VERSION = "2.14.1"
TTS_SAMPLE_RATE = 48000
TTS_VOICE_XENIA = "xenia"
TTS_VOICE_EUGENE = "eugene"
DEFAULT_VOICE = TTS_VOICE_XENIA
ALLOWED_VOICES = frozenset({TTS_VOICE_XENIA, TTS_VOICE_EUGENE})
OPUS_BITRATE = "32k"
OPUS_CHANNELS = 1

VOICE_MAX_SENTENCES = 4
VOICE_MAX_WORDS = 80
VOICE_TARGET_MIN_SENTENCES = 2
VOICE_TARGET_MAX_SENTENCES = 4
MAX_VOICE_COMPACT_REGENERATIONS = 1

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")
_CITATION_RE = re.compile(r"\[[A-Za-z0-9_.\-/]+(?:#[A-Za-z0-9_:.\-]+)?\]")


class TtsError(Exception):
    """Bounded TTS/encoding failure (fallback to text, never drop)."""

    def __init__(self, category: str, detail: str = "") -> None:
        super().__init__(f"tts failed [{category}]" + (f": {detail}" if detail else ""))
        self.category = category
        self.detail = detail


def resolve_tts_voice(speaker: str | None) -> str:
    """Resolve an explicit speaker name to the fixed production set.

    Only ``xenia`` and ``eugene`` are allowed; anything else (including
    ``None``/empty/unknown) deterministically maps to ``xenia``.
    """
    if speaker in ALLOWED_VOICES:
        return str(speaker)
    return DEFAULT_VOICE


def voice_for_presentation(presentation: str | None) -> str:
    """Map an acoustic presentation label to the opposite TTS voice.

    This is the #77 side of the #78 opposite-voice rule. The acoustic
    classifier itself lives in #78; here only the deterministic mapping
    is owned:

    - ``male-presenting`` -> ``xenia``;
    - ``female-presenting`` -> ``eugene``;
    - ``unknown``/``None``/classifier error -> ``xenia``.
    """
    if presentation == "male-presenting":
        return TTS_VOICE_XENIA
    if presentation == "female-presenting":
        return TTS_VOICE_EUGENE
    return DEFAULT_VOICE


def split_voice_sentences(text: str) -> list[str]:
    """Split ``text`` into complete sentences for the voice policy."""
    parts = [item.strip() for item in _SENTENCE_SPLIT_RE.split(text.strip()) if item.strip()]
    return parts


def count_voice_sentences(text: str) -> int:
    """Count complete sentences in ``text`` under the voice policy."""
    if not text.strip():
        return 0
    return len(split_voice_sentences(text))


def count_voice_words(text: str) -> int:
    """Count whitespace-separated words (citations excluded)."""
    cleaned = _CITATION_RE.sub(" ", text)
    return len(cleaned.split())


def voice_policy_passes(text: str) -> bool:
    """Whether ``text`` fits the hard voice contract (<=4 / <=80)."""
    if not text.strip():
        return False
    return count_voice_sentences(text) <= VOICE_MAX_SENTENCES and (
        count_voice_words(text) <= VOICE_MAX_WORDS
    )


def compact_voice_text_to_policy(text: str) -> str:
    """Keep the longest leading sentences fitting <=4 sentences / <=80 words.

    Cuts happen only at complete-sentence boundaries. When even the
    first sentence cannot fit both bounds, the first sentence alone is
    returned (shortest complete unit) so the caller never emits a
    truncated fragment or an empty reply.
    """
    sentences = split_voice_sentences(text)
    if not sentences:
        return text.strip()
    kept: list[str] = []
    kept_words = 0
    for sentence in sentences:
        if len(kept) + 1 > VOICE_MAX_SENTENCES:
            break
        words = len(_CITATION_RE.sub(" ", sentence).split())
        if kept_words + words > VOICE_MAX_WORDS:
            break
        kept.append(sentence)
        kept_words += words
    if kept:
        compacted = " ".join(kept)
        logger.info(
            "voice reply compacted to policy",
            extra={
                "kept_sentences": len(kept),
                "total_sentences": len(sentences),
                "words": kept_words,
            },
        )
        return compacted
    logger.info("voice reply kept first sentence only")
    return sentences[0]


def voice_generation_instruction() -> str:
    """Build the voice-mode budget hint embedded in synthesis prompts."""
    return (
        "ГОЛОСОВОЙ РЕЖИМ (обязательно): отвечай кратко для голосового "
        "сообщения, обычно 2-4 коротких предложения. Жёсткий предел "
        "голосового ответа: не более 4 предложений и не более 80 слов "
        "всего. Один главный смысл, без списков и длинных цитат."
    )


def voice_compact_retry_instruction(
    *, remaining_sentences: int = VOICE_MAX_SENTENCES, remaining_words: int = VOICE_MAX_WORDS
) -> str:
    """Build the explicit remaining-budget instruction for one voice regen."""
    return (
        "Перепиши ответ короче для голосового сообщения, используя ТОЛЬКО "
        "те же проверенные отрывки. "
        f"Остаток голосового бюджета: не более {remaining_sentences} предложений и "
        f"{remaining_words} слов всего. Сохрани 2-4 коротких предложения и "
        "один главный смысл. Не добавляй новых утверждений без опоры."
    )


def ensure_tts_model_file(model_path: Path) -> Path:
    """Ensure the pinned Silero model file exists locally (download if needed).

    Downloads exactly ``TTS_MODEL_URL`` over HTTPS with the standard
    library only. Raises :class:`TtsError` on any failure so the caller
    can disable voice replies without failing the poller.
    """
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_suffix(model_path.suffix + ".part")
    try:
        with urllib.request.urlopen(TTS_MODEL_URL, timeout=300) as resp:  # noqa: S310
            with open(tmp, "wb") as handle:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    handle.write(chunk)
        tmp.replace(model_path)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise TtsError("tts-unavailable", "could not provision Silero model file") from exc
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise TtsError("tts-unavailable", "Silero model file is missing after provisioning")
    return model_path


class SpeechSynthesizer(Protocol):
    """Abstract Russian TTS synthesizer to mono 48 kHz float32 PCM."""

    @property
    def available(self) -> bool:
        """Whether the synthesizer is loaded and ready."""
        raise NotImplementedError

    def synthesize(self, text: str, speaker: str) -> list[float]:
        """Synthesize ``text`` with ``speaker`` into 48 kHz float32 samples."""
        raise NotImplementedError


class VoiceEncoder(Protocol):
    """Abstract PCM to OGG/Opus encoder."""

    def encode(self, samples: Sequence[float], *, workdir: Path) -> bytes:
        """Encode mono 48 kHz float32 ``samples`` into OGG/Opus bytes."""
        raise NotImplementedError


class SileroSynthesizer:
    """Pinned Silero ``v5_5_ru`` synthesizer over ``torch==2.14.1`` CPU.

    The model is loaded once and reused for all turns. ``torch`` is
    imported lazily so text-only workers and unit tests never require
    it. Any other installed torch version fails voice initialization
    closed. Only ``xenia``/``eugene`` speakers are accepted.
    """

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._model: object | None = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        """Whether the Silero model loaded successfully."""
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        """Initialization failure detail (category only, no paths)."""
        return self._load_error

    def ensure_loaded(self) -> None:
        """Load the pinned model once; raise :class:`TtsError` if unusable."""
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            self._load_error = "torch is not installed"
            raise TtsError("tts-unavailable", "torch is not installed") from exc
        installed = str(getattr(torch, "__version__", ""))
        if not (installed == TORCH_VERSION or installed.startswith(TORCH_VERSION + "+")):
            self._load_error = "unexpected torch version"
            raise TtsError("tts-unavailable", f"torch must be {TORCH_VERSION}")
        try:
            model_file = ensure_tts_model_file(self._model_path)
        except TtsError as exc:
            self._load_error = exc.category
            raise
        except OSError as exc:
            self._load_error = "model provisioning failed"
            raise TtsError("tts-unavailable", "model provisioning failed") from exc
        try:
            # torch.package is a separate submodule: `import torch` alone
            # does not guarantee `torch.package` is loaded (Gate C live
            # voice-models-ready failed with TTS disabled while GigaAM and
            # presentation loaded). Import it explicitly before use.
            import torch.package  # noqa: PLC0415,F401
        except ImportError as exc:
            self._load_error = "torch.package is not installed"
            raise TtsError("tts-unavailable", "torch.package is not installed") from exc
        try:
            device = torch.device("cpu")
            importer = torch.package.PackageImporter(str(model_file))  # type: ignore[attr-defined]
            model = importer.load_pickle("tts_models", "model")
            model.to(device)
            model.eval()
        except Exception as exc:
            self._load_error = "synthesizer construction failed"
            raise TtsError("tts-unavailable", "synthesizer construction failed") from exc
        self._model = model
        logger.info("tts synthesizer loaded", extra={"model": TTS_MODEL_ID})

    def synthesize(self, text: str, speaker: str) -> list[float]:
        """Run one CPU synthesis for ``text`` with a fixed production voice."""
        model = self._model
        if model is None:
            raise TtsError("tts-unavailable", "synthesizer is not loaded")
        voice = resolve_tts_voice(speaker)
        if not text.strip():
            raise TtsError("tts-failed", "no text to synthesize")
        try:
            apply_tts = getattr(model, "apply_tts", None)
            if callable(apply_tts):
                audio = apply_tts(text=text, speaker=voice, sample_rate=TTS_SAMPLE_RATE)
            else:
                raise TtsError("tts-unavailable", "synthesizer has no apply_tts")
            import torch  # noqa: PLC0415

            if isinstance(audio, torch.Tensor):
                values = audio.detach().to("cpu").flatten().tolist()
            elif isinstance(audio, (list, tuple)):
                values = [float(item) for item in audio]
            else:
                values = [float(item) for item in list(audio)]
        except TtsError:
            raise
        except Exception as exc:
            raise TtsError("tts-failed", "TTS synthesis failed") from exc
        if not values:
            raise TtsError("tts-failed", "synthesis produced no audio")
        logger.info("tts synthesized", extra={"sample_count": len(values), "voice": voice})
        return [float(item) for item in values]


class FfmpegOpusEncoder:
    """Production PCM to OGG/Opus encoder using system ``ffmpeg``."""

    def __init__(self, *, ffmpeg_binary: str = "ffmpeg", timeout_seconds: float = 60.0) -> None:
        self._ffmpeg = ffmpeg_binary
        self._timeout_seconds = timeout_seconds

    def encode(self, samples: Sequence[float], *, workdir: Path) -> bytes:
        """Encode mono 48 kHz float32 PCM to OGG/Opus (32 kbit/s)."""
        if not samples:
            raise TtsError("encode-failed", "no PCM samples to encode")
        if shutil.which(self._ffmpeg) is None:
            raise TtsError("encode-failed", "ffmpeg is not available")
        workdir.mkdir(parents=True, exist_ok=True)
        stamp = time.monotonic_ns()
        src = workdir / f"tts-{stamp}.pcm"
        dst = workdir / f"tts-{stamp}.ogg"
        try:
            try:
                import array as _array

                raw = _array.array("f", [float(item) for item in samples]).tobytes()
                src.write_bytes(raw)
            except OSError as exc:
                raise TtsError("encode-failed", "could not stage PCM") from exc
            try:
                subprocess.run(
                    [
                        self._ffmpeg,
                        "-y",
                        "-v",
                        "error",
                        "-f",
                        "f32le",
                        "-ar",
                        str(TTS_SAMPLE_RATE),
                        "-ac",
                        "1",
                        "-i",
                        str(src),
                        "-ac",
                        str(OPUS_CHANNELS),
                        "-ar",
                        str(TTS_SAMPLE_RATE),
                        "-c:a",
                        "libopus",
                        "-b:a",
                        OPUS_BITRATE,
                        str(dst),
                    ],
                    timeout=self._timeout_seconds,
                    check=True,
                    capture_output=True,
                )
            except FileNotFoundError as exc:
                raise TtsError("encode-failed", "ffmpeg is not available") from exc
            except subprocess.CalledProcessError as exc:
                raise TtsError("encode-failed", "ffmpeg could not encode voice") from exc
            except subprocess.TimeoutExpired as exc:
                raise TtsError("encode-failed", "ffmpeg encode timed out") from exc
            try:
                data = dst.read_bytes()
            except OSError as exc:
                raise TtsError("encode-failed", "encoded voice is unreadable") from exc
            if not data:
                raise TtsError("encode-failed", "encoded voice is empty")
            logger.info("tts encoded", extra={"byte_len": len(data)})
            return data
        finally:
            for path in (src, dst):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass


@dataclass
class TtsPipeline:
    """Single-worker TTS pipeline with a single-synthesis bound.

    Synthesis is serialized by ``tts_gate`` so at most one TTS inference
    runs at a time per worker. Encoded artifacts live under one
    per-turn temporary directory that is always removed in ``finally``.
    """

    synthesizer: SpeechSynthesizer
    encoder: VoiceEncoder
    tts_gate: asyncio.Semaphore
    work_parent: Path | None = None

    async def synthesize_voice_ogg(self, text: str, speaker: str | None = None) -> bytes:
        """Synthesize ``text`` and encode it to OGG/Opus bytes."""
        voice = resolve_tts_voice(speaker)
        if self.synthesizer is None or not self.synthesizer.available:
            raise TtsError("tts-unavailable", "voice synthesizer is not ready")
        if not text.strip():
            raise TtsError("tts-failed", "no text to synthesize")
        parent = self.work_parent
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="aa-tts-", dir=str(parent) if parent else None
        ) as tmp:
            workdir = Path(tmp)
            try:
                async with self.tts_gate:
                    try:
                        samples = await asyncio.to_thread(self.synthesizer.synthesize, text, voice)
                    except TtsError:
                        raise
                    except Exception as exc:
                        raise TtsError("tts-failed", "TTS synthesis failed") from exc
            finally:
                pass
            try:
                data = await asyncio.to_thread(self.encoder.encode, samples, workdir=workdir)
            except TtsError:
                raise
            except Exception as exc:
                raise TtsError("encode-failed", "voice encoding failed") from exc
            finally:
                del samples
            if not data:
                raise TtsError("encode-failed", "encoded voice is empty")
            logger.info("voice reply synthesized", extra={"byte_len": len(data)})
            return data
        # ``TemporaryDirectory`` removes all staged audio on exit.


def build_tts_pipeline(
    *,
    synthesizer: SpeechSynthesizer,
    encoder: VoiceEncoder | None = None,
    work_parent: Path | None = None,
) -> TtsPipeline:
    """Build a TTS pipeline with a per-worker single-synthesis gate."""
    return TtsPipeline(
        synthesizer=synthesizer,
        encoder=encoder or FfmpegOpusEncoder(),
        tts_gate=asyncio.Semaphore(1),
        work_parent=work_parent,
    )


__all__ = [
    "ALLOWED_VOICES",
    "DEFAULT_VOICE",
    "MAX_VOICE_COMPACT_REGENERATIONS",
    "OPUS_BITRATE",
    "OPUS_CHANNELS",
    "TORCH_VERSION",
    "TTS_MODEL_ID",
    "TTS_MODEL_URL",
    "TTS_SAMPLE_RATE",
    "TTS_VOICE_EUGENE",
    "TTS_VOICE_XENIA",
    "VOICE_MAX_SENTENCES",
    "VOICE_MAX_WORDS",
    "VOICE_TARGET_MAX_SENTENCES",
    "VOICE_TARGET_MIN_SENTENCES",
    "FfmpegOpusEncoder",
    "SileroSynthesizer",
    "SpeechSynthesizer",
    "TtsError",
    "TtsPipeline",
    "VoiceEncoder",
    "build_tts_pipeline",
    "compact_voice_text_to_policy",
    "count_voice_sentences",
    "count_voice_words",
    "ensure_tts_model_file",
    "resolve_tts_voice",
    "split_voice_sentences",
    "voice_compact_retry_instruction",
    "voice_for_presentation",
    "voice_generation_instruction",
    "voice_policy_passes",
]
