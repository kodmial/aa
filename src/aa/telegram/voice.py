"""Local Russian Telegram voice recognition with GigaAM/sherpa-onnx (issue #76).

Fixed implementation contract (no model-selection research):

- ASR model repository: ``fussraider/GigaAM-Multilingual-sherpa-onnx-ctc``;
- repository revision: ``9f5a77e8975211abe8511693accd3a63ee1e9f43``;
- model file: ``large/model.int8.onnx`` (600M CTC INT8);
- tokens: ``large/tokens.txt``;
- runtime: ``sherpa-onnx==1.13.8``;
- input to ASR: mono PCM float32, 16 kHz;
- sherpa feature dimension: 64;
- CPU inference only.

Only the standard library is used besides the pinned ``sherpa-onnx``
runtime (lazy import so unit tests and text-only workers never require
it). Model files are fetched from Hugging Face with the pinned revision
over HTTPS (``urllib`` in a worker thread). OGG/Opus decoding uses the
system ``ffmpeg`` binary to mono 16 kHz float32 PCM.

Privacy: this module never logs raw audio, transcripts, user text, or
file identifiers. Only sizes, durations, latencies and failure
categories are logged.
"""

from __future__ import annotations

import array
import asyncio
import logging
import shutil
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("aa.telegram.voice")

MODEL_REPO = "fussraider/GigaAM-Multilingual-sherpa-onnx-ctc"
MODEL_REVISION = "9f5a77e8975211abe8511693accd3a63ee1e9f43"
MODEL_FILE = "large/model.int8.onnx"
TOKENS_FILE = "large/tokens.txt"
SHERPA_ONNX_VERSION = "1.13.8"
SAMPLE_RATE = 16000
FEATURE_DIM = 64
NUM_THREADS = 4
MAX_VOICE_FILE_BYTES = 20 * 1024 * 1024
MAX_VOICE_DURATION_SECONDS = 600

_HF_BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}"

# Short deterministic Russian user-safe replies (RU-only, envelope-safe).
# Each contains Cyrillic and no Latin prose so ``meets_russian_only``
# holds for every voice failure path.
VOICE_ERROR_REPLY = "Не удалось распознать голосовое сообщение. Попробуйте ещё раз."
VOICE_EMPTY_REPLY = "Не удалось распознать речь. Скажите громче и чётче."
VOICE_TOO_LARGE_REPLY = "Голосовое сообщение слишком большое. Отправьте запись короче."
VOICE_UNAVAILABLE_REPLY = "Голосовые сообщения временно недоступны. Напишите текстом."


class VoiceError(Exception):
    """Bounded voice-turn failure (user-safe, poller stays alive)."""

    def __init__(self, category: str, detail: str = "") -> None:
        super().__init__(f"voice failed [{category}]" + (f": {detail}" if detail else ""))
        self.category = category
        self.detail = detail


def voice_error_reply(category: str) -> str:
    """Map a voice failure category to one short Russian text reply."""
    if category in ("too-large", "too-long"):
        return VOICE_TOO_LARGE_REPLY
    if category == "empty-transcript":
        return VOICE_EMPTY_REPLY
    if category in ("asr-unavailable", "voice-disabled"):
        return VOICE_UNAVAILABLE_REPLY
    return VOICE_ERROR_REPLY


def check_voice_bounds(*, file_size_bytes: int | None, duration_seconds: int | None) -> None:
    """Reject oversized/overlong voice input before any ASR work."""
    if duration_seconds is not None and duration_seconds > MAX_VOICE_DURATION_SECONDS:
        raise VoiceError("too-long", "voice duration exceeds 10 minutes")
    if file_size_bytes is not None and file_size_bytes > MAX_VOICE_FILE_BYTES:
        raise VoiceError("too-large", "voice file exceeds 20 MiB")


def model_file_urls() -> tuple[str, str]:
    """Return the pinned Hugging Face URLs for the model and tokens."""
    return f"{_HF_BASE}/{MODEL_FILE}", f"{_HF_BASE}/{TOKENS_FILE}"


def ensure_model_files(model_dir: Path) -> tuple[Path, Path]:
    """Ensure the pinned model/token files exist locally (download if needed).

    Downloads use the pinned repository revision over HTTPS with the
    standard library only. Raises :class:`VoiceError` on any failure so
    the caller can fail the voice capability without failing the poller.
    """
    model_path = model_dir / "model.int8.onnx"
    tokens_path = model_dir / "tokens.txt"
    if model_path.is_file() and tokens_path.is_file():
        return model_path, tokens_path
    model_dir.mkdir(parents=True, exist_ok=True)
    model_url, tokens_url = model_file_urls()
    try:
        _download_to_file(tokens_url, tokens_path)
        _download_to_file(model_url, model_path)
    except (OSError, VoiceError) as exc:
        raise VoiceError("asr-unavailable", "could not provision GigaAM model files") from exc
    if not model_path.is_file() or not tokens_path.is_file():
        raise VoiceError("asr-unavailable", "GigaAM model files are missing after provisioning")
    if model_path.stat().st_size == 0 or tokens_path.stat().st_size == 0:
        raise VoiceError("asr-unavailable", "GigaAM model files are empty")
    return model_path, tokens_path


def _download_to_file(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310
            with open(tmp, "wb") as handle:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    handle.write(chunk)
        tmp.replace(dest)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise VoiceError("asr-unavailable", "model download failed") from exc


class VoiceFileFetcher(Protocol):
    """Abstract Telegram voice-file downloader (bytes only)."""

    async def fetch(self, file_id: str) -> bytes:
        """Download one voice file by Telegram ``file_id``."""
        raise NotImplementedError


class WaveformDecoder(Protocol):
    """Abstract OGG/Opus decoder to mono 16 kHz float32 PCM."""

    def decode(self, ogg_bytes: bytes, *, workdir: Path) -> list[float]:
        """Decode ``ogg_bytes`` into mono float32 samples at 16 kHz."""
        raise NotImplementedError


class TranscriptRecognizer(Protocol):
    """Abstract ASR recognizer over mono 16 kHz float32 samples."""

    @property
    def available(self) -> bool:
        """Whether the recognizer is ready for inference."""
        raise NotImplementedError

    def transcribe(self, samples: Sequence[float]) -> str:
        """Transcribe PCM samples into raw text (may be empty)."""
        raise NotImplementedError


class TelegramVoiceFetcher:
    """Production Telegram voice downloader via ``getFile`` + file HTTPS.

    Uses the existing Bot API surface for ``getFile`` and plain HTTPS for
    the file content. Enforces the 20 MiB bound while streaming so an
    oversized file is rejected before full buffering.
    """

    def __init__(
        self, *, api: Any, token: str, file_base_url: str = "https://api.telegram.org"
    ) -> None:
        if not token:
            raise ValueError("Telegram bot token is required for voice downloads")
        self._api = api
        self._token = token
        self._file_base_url = file_base_url.rstrip("/")

    async def fetch(self, file_id: str) -> bytes:
        """Download one voice file, enforcing the 20 MiB bound."""
        if not file_id:
            raise VoiceError("download-failed", "missing voice file id")
        try:
            result = await self._api.call("getFile", {"file_id": file_id})
        except Exception as exc:
            raise VoiceError("download-failed", "telegram getFile failed") from exc
        if not isinstance(result, dict):
            raise VoiceError("download-failed", "telegram getFile returned no file")
        file_path = result.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            raise VoiceError("download-failed", "telegram getFile returned no path")
        if ".." in file_path or file_path.startswith("/"):
            raise VoiceError("download-failed", "telegram file path is unsafe")
        url = f"{self._file_base_url}/file/bot{self._token}/{file_path}"
        try:
            data = await asyncio.to_thread(self._download_sync, url)
        except VoiceError:
            raise
        except Exception as exc:
            raise VoiceError("download-failed", "voice download failed") from exc
        if len(data) > MAX_VOICE_FILE_BYTES:
            raise VoiceError("too-large", "voice file exceeds 20 MiB")
        if not data:
            raise VoiceError("download-failed", "voice download was empty")
        logger.info("voice file downloaded", extra={"byte_len": len(data)})
        return data

    def _download_sync(self, url: str) -> bytes:
        limit = MAX_VOICE_FILE_BYTES + 1
        chunks: list[bytes] = []
        total = 0
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise VoiceError("too-large", "voice file exceeds 20 MiB")
                chunks.append(chunk)
        return b"".join(chunks)


class FfmpegDecoder:
    """Production OGG/Opus decoder using the system ``ffmpeg`` binary."""

    def __init__(self, *, ffmpeg_binary: str = "ffmpeg", timeout_seconds: float = 60.0) -> None:
        self._ffmpeg = ffmpeg_binary
        self._timeout_seconds = timeout_seconds

    def decode(self, ogg_bytes: bytes, *, workdir: Path) -> list[float]:
        """Decode OGG bytes to mono 16 kHz float32 samples via ffmpeg."""
        if not ogg_bytes:
            raise VoiceError("decode-failed", "voice payload is empty")
        if len(ogg_bytes) > MAX_VOICE_FILE_BYTES:
            raise VoiceError("too-large", "voice file exceeds 20 MiB")
        if shutil.which(self._ffmpeg) is None:
            raise VoiceError("decode-failed", "ffmpeg is not available")
        workdir.mkdir(parents=True, exist_ok=True)
        stamp = time.monotonic_ns()
        src = workdir / f"voice-{stamp}.ogg"
        dst = workdir / f"voice-{stamp}.pcm"
        try:
            src.write_bytes(ogg_bytes)
        except OSError as exc:
            raise VoiceError("decode-failed", "could not stage voice payload") from exc
        try:
            try:
                subprocess.run(
                    [
                        self._ffmpeg,
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(src),
                        "-ac",
                        "1",
                        "-ar",
                        str(SAMPLE_RATE),
                        "-f",
                        "f32le",
                        "-acodec",
                        "pcm_f32le",
                        str(dst),
                    ],
                    timeout=self._timeout_seconds,
                    check=True,
                    capture_output=True,
                )
            except FileNotFoundError as exc:
                raise VoiceError("decode-failed", "ffmpeg is not available") from exc
            except subprocess.CalledProcessError as exc:
                raise VoiceError("decode-failed", "ffmpeg could not decode voice") from exc
            except subprocess.TimeoutExpired as exc:
                raise VoiceError("decode-failed", "ffmpeg decode timed out") from exc
            try:
                raw = dst.read_bytes()
            except OSError as exc:
                raise VoiceError("decode-failed", "decoded voice is unreadable") from exc
            if len(raw) % 4 != 0 or not raw:
                raise VoiceError("decode-failed", "decoded voice has no PCM frames")
            samples = array.array("f")
            samples.frombytes(raw)
            try:
                values = [float(item) for item in samples]
            except (ValueError, OverflowError) as exc:
                raise VoiceError("decode-failed", "decoded PCM is invalid") from exc
            duration_seconds = len(values) / float(SAMPLE_RATE)
            if duration_seconds > float(MAX_VOICE_DURATION_SECONDS) + 1.0:
                raise VoiceError("too-long", "decoded voice exceeds 10 minutes")
            if not values:
                raise VoiceError("decode-failed", "decoded voice is empty")
            logger.info(
                "voice decoded",
                extra={"sample_count": len(values), "byte_len": len(ogg_bytes)},
            )
            return values
        finally:
            for path in (src, dst):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass


class GigaAMRecognizer:
    """Pinned GigaAM 600M INT8 recognizer over ``sherpa-onnx==1.13.8``.

    The model is loaded once and reused for all turns. ``sherpa-onnx``
    is imported lazily so text-only workers and unit tests never require
    it. The pinned runtime version is enforced: any other installed
    version fails voice initialization closed.
    """

    def __init__(self, model_dir: Path, *, num_threads: int = NUM_THREADS) -> None:
        self._model_dir = model_dir
        self._num_threads = num_threads
        self._recognizer: Any | None = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        """Whether the recognizer loaded successfully."""
        return self._recognizer is not None

    @property
    def load_error(self) -> str | None:
        """Initialization failure detail (category only, no paths)."""
        return self._load_error

    def ensure_loaded(self) -> None:
        """Load the pinned model once; raise :class:`VoiceError` if unusable."""
        if self._recognizer is not None:
            return
        try:
            import sherpa_onnx  # type: ignore[import-untyped]  # noqa: PLC0415
        except ImportError as exc:
            self._load_error = "sherpa-onnx is not installed"
            raise VoiceError("asr-unavailable", "sherpa-onnx is not installed") from exc
        installed = str(getattr(sherpa_onnx, "__version__", ""))
        if installed != SHERPA_ONNX_VERSION:
            self._load_error = "unexpected sherpa-onnx version"
            raise VoiceError(
                "asr-unavailable",
                f"sherpa-onnx must be {SHERPA_ONNX_VERSION}",
            )
        try:
            model_path, tokens_path = ensure_model_files(self._model_dir)
        except VoiceError as exc:
            self._load_error = exc.category
            raise
        except OSError as exc:
            self._load_error = "model provisioning failed"
            raise VoiceError("asr-unavailable", "model provisioning failed") from exc
        try:
            recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(model_path),
                tokens=str(tokens_path),
                num_threads=self._num_threads,
                sample_rate=SAMPLE_RATE,
                feature_dim=FEATURE_DIM,
            )
        except Exception as exc:
            self._load_error = "recognizer construction failed"
            raise VoiceError("asr-unavailable", "recognizer construction failed") from exc
        self._recognizer = recognizer
        logger.info("voice recognizer loaded", extra={"num_threads": self._num_threads})

    def transcribe(self, samples: Sequence[float]) -> str:
        """Run one CPU inference over mono 16 kHz float32 samples."""
        recognizer = self._recognizer
        if recognizer is None:
            raise VoiceError("asr-unavailable", "recognizer is not loaded")
        if not samples:
            raise VoiceError("decode-failed", "no PCM samples to transcribe")
        try:
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, list(samples))
            recognizer.decode_stream(stream)
            text = str(stream.result.text or "")
        except VoiceError:
            raise
        except Exception as exc:
            raise VoiceError("asr-failed", "ASR inference failed") from exc
        logger.info("voice transcribed", extra={"sample_count": len(samples)})
        return text


@dataclass
class VoicePipeline:
    """Single-worker voice pipeline with a single-ASR inference bound.

    Download and decode run concurrently across chats; only the actual
    ASR inference is serialized by ``asr_gate`` so other chats continue
    through non-ASR stages while one voice turn is being recognized.
    Temporary audio files live under one per-turn directory that is
    always removed in ``finally``.
    """

    fetcher: VoiceFileFetcher
    decoder: WaveformDecoder
    recognizer: TranscriptRecognizer | None
    asr_gate: asyncio.Semaphore
    work_parent: Path | None = None

    async def transcribe_voice(
        self,
        *,
        file_id: str,
        file_size_bytes: int | None = None,
        duration_seconds: int | None = None,
    ) -> str:
        """Download, decode and transcribe one voice message to trimmed text."""
        check_voice_bounds(file_size_bytes=file_size_bytes, duration_seconds=duration_seconds)
        if self.recognizer is None or not self.recognizer.available:
            raise VoiceError("asr-unavailable", "voice recognizer is not ready")
        raw = await self._fetch_guarded(file_id)
        check_voice_bounds(file_size_bytes=len(raw), duration_seconds=duration_seconds)
        parent = self.work_parent
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="aa-voice-", dir=str(parent) if parent else None
        ) as tmp:
            workdir = Path(tmp)
            try:
                samples = await asyncio.to_thread(self.decoder.decode, raw, workdir=workdir)
            except VoiceError:
                raise
            except Exception as exc:
                raise VoiceError("decode-failed", "voice decode failed") from exc
            if not samples:
                raise VoiceError("decode-failed", "decoded voice is empty")
            # Only ASR inference is serialized; download/decode already ran
            # concurrently for other chats before reaching this gate.
            try:
                async with self.asr_gate:
                    try:
                        text = await asyncio.to_thread(self.recognizer.transcribe, samples)
                    except VoiceError:
                        raise
                    except Exception as exc:
                        raise VoiceError("asr-failed", "ASR inference failed") from exc
            finally:
                # Never retain PCM after the turn, even when inference fails.
                del samples
            transcript = text.strip()
            if not transcript:
                raise VoiceError("empty-transcript", "ASR returned no speech")
            logger.info("voice turn transcribed", extra={"text_len": len(transcript)})
            return transcript
        # ``TemporaryDirectory`` removes all staged audio on exit.

    async def _fetch_guarded(self, file_id: str) -> bytes:
        try:
            return await self.fetcher.fetch(file_id)
        except VoiceError:
            raise
        except Exception as exc:
            raise VoiceError("download-failed", "voice download failed") from exc


def build_pipeline(
    *,
    fetcher: VoiceFileFetcher,
    decoder: WaveformDecoder | None = None,
    recognizer: TranscriptRecognizer | None = None,
    work_parent: Path | None = None,
) -> VoicePipeline:
    """Build a voice pipeline with a per-worker single-ASR gate."""
    return VoicePipeline(
        fetcher=fetcher,
        decoder=decoder or FfmpegDecoder(),
        recognizer=recognizer,
        asr_gate=asyncio.Semaphore(1),
        work_parent=work_parent,
    )


def decode_ogg_bytes(ogg_bytes: bytes, *, workdir: Path) -> list[float]:
    """Decode OGG/Opus bytes with ffmpeg (convenience wrapper)."""
    return FfmpegDecoder().decode(ogg_bytes, workdir=workdir)


__all__ = [
    "FEATURE_DIM",
    "MAX_VOICE_DURATION_SECONDS",
    "MAX_VOICE_FILE_BYTES",
    "MODEL_FILE",
    "MODEL_REPO",
    "MODEL_REVISION",
    "NUM_THREADS",
    "SAMPLE_RATE",
    "SHERPA_ONNX_VERSION",
    "TOKENS_FILE",
    "FfmpegDecoder",
    "GigaAMRecognizer",
    "TelegramVoiceFetcher",
    "TranscriptRecognizer",
    "VOICE_EMPTY_REPLY",
    "VOICE_ERROR_REPLY",
    "VOICE_TOO_LARGE_REPLY",
    "VOICE_UNAVAILABLE_REPLY",
    "VoiceError",
    "VoiceFileFetcher",
    "VoicePipeline",
    "WaveformDecoder",
    "build_pipeline",
    "check_voice_bounds",
    "decode_ogg_bytes",
    "ensure_model_files",
    "model_file_urls",
    "voice_error_reply",
]
