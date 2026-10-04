"""Ephemeral acoustic voice-presentation routing (issue #78).

Fixed implementation contract (no classifier research):

- model: ``Alice-Sabrina-Ivy/voice-gender-classifier-onnx-q8-v2``;
- file: ``onnx/model_quantized.onnx``;
- required SHA-256:
  ``fdc2dbdcf99b9217977f7472f7d677dd48219c4759ca3f38d0626b600d86c252``;
- runtime: ``onnxruntime==1.30.0``, CPU only;
- input: raw mono float32 at 16 kHz from the already-decoded #76 audio;
- labels are interpreted only as acoustic ``male-presenting`` /
  ``female-presenting`` (never a claim about gender identity).

Deterministic decision rule (exactly as specified):

1. Remove leading/trailing near-silence with a simple RMS gate
   (in-memory only, never persisted).
2. Less than 2.25 s of usable audio remains -> ``unknown``.
3. Evaluate up to the first five non-overlapping 0.75 s voiced windows.
4. For each window obtain the model female probability ``p``.
5. Fewer than three valid windows -> ``unknown``.
6. ``female-presenting`` only when median(p) >= 0.90 and at least 80%
   of windows have p >= 0.80.
7. ``male-presenting`` only when median(p) <= 0.10 and at least 80%
   of windows have p <= 0.20.
8. Otherwise -> ``unknown``.

TTS mapping (owned by ``aa.telegram.tts.voice_for_presentation``):

- ``male-presenting`` -> Silero ``xenia``;
- ``female-presenting`` -> Silero ``eugene``;
- ``unknown`` or classifier error -> Silero ``xenia``.

Privacy: classification exists only for the current turn. This module
never logs the class, probabilities, embeddings/features, or raw
audio; only window counts, sizes, and failure categories are logged.
Nothing is stored in chat/session history or any user profile, and the
inferred class is never exposed to the user. Transcript text is never
used for this decision and no speaker recognition/identity tracking
exists.
"""

from __future__ import annotations

import hashlib
import logging
import math
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

logger = logging.getLogger("aa.telegram.voice_presentation")

PRESENTATION_MODEL_REPO = "Alice-Sabrina-Ivy/voice-gender-classifier-onnx-q8-v2"
PRESENTATION_MODEL_FILE = "onnx/model_quantized.onnx"
PRESENTATION_MODEL_SHA256 = "fdc2dbdcf99b9217977f7472f7d677dd48219c4759ca3f38d0626b600d86c252"
ONNXRUNTIME_VERSION = "1.30.0"
PRESENTATION_SAMPLE_RATE = 16000

WINDOW_SECONDS = 0.75
WINDOW_SAMPLES = int(PRESENTATION_SAMPLE_RATE * WINDOW_SECONDS)
MIN_USABLE_SECONDS = 2.25
MIN_USABLE_SAMPLES = int(PRESENTATION_SAMPLE_RATE * MIN_USABLE_SECONDS)
MAX_WINDOWS = 5
MIN_WINDOWS = 3

FEMALE_MEDIAN_THRESHOLD = 0.90
FEMALE_WINDOW_THRESHOLD = 0.80
MALE_MEDIAN_THRESHOLD = 0.10
MALE_WINDOW_THRESHOLD = 0.20
AGREEMENT_FRACTION = 0.80

RMS_FRAME_SAMPLES = 320
RMS_THRESHOLD = 0.02

Presentation = Literal["male-presenting", "female-presenting", "unknown"]

_HF_MODEL_URL = (
    f"https://huggingface.co/{PRESENTATION_MODEL_REPO}/resolve/main/{PRESENTATION_MODEL_FILE}"
)


class PresentationError(Exception):
    """Bounded presentation failure (defaults to ``unknown``/``xenia``)."""

    def __init__(self, category: str, detail: str = "") -> None:
        super().__init__(f"presentation failed [{category}]" + (f": {detail}" if detail else ""))
        self.category = category
        self.detail = detail


class PresentationClassifier(Protocol):
    """Abstract per-turn acoustic presentation classifier."""

    @property
    def available(self) -> bool:
        """Whether the classifier is loaded and ready."""
        raise NotImplementedError

    def classify(self, samples: Sequence[float]) -> str:
        """Classify raw 16 kHz mono samples (never raises to caller)."""
        raise NotImplementedError


def presentation_model_url() -> str:
    """Return the pinned Hugging Face URL for the ONNX model file."""
    return _HF_MODEL_URL


def verify_model_sha256(model_path: Path) -> None:
    """Verify the pinned SHA-256; raise :class:`PresentationError` on mismatch."""
    try:
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PresentationError("model-unavailable", "model file is unreadable") from exc
    if digest.lower() != PRESENTATION_MODEL_SHA256.lower():
        raise PresentationError("model-checksum-mismatch", "model SHA-256 mismatch")


def ensure_presentation_model_file(model_path: Path) -> Path:
    """Ensure the pinned model file exists locally and matches the SHA-256.

    Downloads exactly the pinned Hugging Face file over HTTPS with the
    standard library only. Any failure (including a checksum mismatch)
    raises :class:`PresentationError` so the caller defaults to
    ``unknown``/``xenia`` without failing the voice reply.
    """
    if model_path.is_file() and model_path.stat().st_size > 0:
        verify_model_sha256(model_path)
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_suffix(model_path.suffix + ".part")
    try:
        with urllib.request.urlopen(presentation_model_url(), timeout=300) as resp:  # noqa: S310
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
        raise PresentationError("model-unavailable", "could not provision model") from exc
    verify_model_sha256(model_path)
    if model_path.stat().st_size == 0:
        raise PresentationError("model-unavailable", "model file is empty")
    return model_path


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    total = 0.0
    for item in values:
        value = float(item)
        total += value * value
    return math.sqrt(total / float(len(values)))


def trim_silence(samples: Sequence[float]) -> list[float]:
    """Remove leading/trailing near-silence with a simple RMS gate.

    Operates in memory only; the result is never persisted. Frames are
    20 ms (320 samples at 16 kHz); a frame is voiced when its RMS is at
    least ``RMS_THRESHOLD``. Returns the slice from the first to the
    last voiced frame (empty when nothing is voiced).
    """
    values = [float(item) for item in samples]
    if not values:
        return []
    frame = RMS_FRAME_SAMPLES
    voiced: list[bool] = []
    index = 0
    while index < len(values):
        chunk = values[index : index + frame]
        voiced.append(_rms(chunk) >= RMS_THRESHOLD)
        index += frame
    if not any(voiced):
        return []
    first = voiced.index(True)
    last = len(voiced) - 1 - voiced[::-1].index(True)
    start = first * frame
    end = min(len(values), (last + 1) * frame)
    return values[start:end]


def extract_voiced_windows(trimmed: Sequence[float]) -> list[list[float]]:
    """Return up to the first five non-overlapping 0.75 s voiced windows.

    ``trimmed`` is the output of :func:`trim_silence`. Windows with RMS
    below the gate are skipped as non-voiced; callers treat skipped
    windows as invalid (not counted toward the three-window minimum).
    """
    values = [float(item) for item in trimmed]
    windows: list[list[float]] = []
    offset = 0
    while offset + WINDOW_SAMPLES <= len(values) and len(windows) < MAX_WINDOWS:
        window = values[offset : offset + WINDOW_SAMPLES]
        offset += WINDOW_SAMPLES
        if _rms(window) < RMS_THRESHOLD:
            continue
        windows.append(window)
    return windows


def median(values: Sequence[float]) -> float:
    """Return the statistical median of ``values`` (non-empty)."""
    ordered = sorted(float(item) for item in values)
    count = len(ordered)
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def decide_presentation(probabilities: Sequence[float]) -> Presentation:
    """Apply the fixed confidence rule to per-window female probabilities.

    ``probabilities`` holds one validated ``p`` per valid window.
    Fewer than three entries, or any disagreement that breaks the 80%
    agreement requirement, deterministically yields ``unknown``; no
    diarization is performed.
    """
    probs = [float(item) for item in probabilities]
    if len(probs) < MIN_WINDOWS:
        return "unknown"
    for item in probs:
        if not math.isfinite(item) or item < 0.0 or item > 1.0:
            return "unknown"
    med = median(probs)
    female_hits = sum(1 for item in probs if item >= FEMALE_WINDOW_THRESHOLD)
    male_hits = sum(1 for item in probs if item <= MALE_WINDOW_THRESHOLD)
    total = float(len(probs))
    if med >= FEMALE_MEDIAN_THRESHOLD and female_hits / total >= AGREEMENT_FRACTION:
        return "female-presenting"
    if med <= MALE_MEDIAN_THRESHOLD and male_hits / total >= AGREEMENT_FRACTION:
        return "male-presenting"
    return "unknown"


def classify_with_predictor(
    samples: Sequence[float],
    predict: Callable[[Sequence[float]], float],
) -> Presentation:
    """Classify ``samples`` with the fixed rule and a ``predict`` callback.

    ``predict`` maps one 0.75 s window to the model female probability
    ``p``. Any error, invalid probability, or short/non-voiced input
    yields ``unknown``; this function never raises for classification
    reasons and never uses transcript text or speaker identity.
    """
    try:
        usable = trim_silence(samples)
    except Exception:
        return "unknown"
    if len(usable) < MIN_USABLE_SAMPLES:
        return "unknown"
    candidate_windows = extract_voiced_windows(usable)
    if len(candidate_windows) < MIN_WINDOWS:
        return "unknown"
    # Evaluate at most the first five windows (already bounded).
    probs: list[float] = []
    for window in candidate_windows[:MAX_WINDOWS]:
        try:
            value = float(predict(window))
        except Exception:
            continue
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            continue
        probs.append(value)
        # Free per-window audio as soon as its score is known.
        del window
    del usable
    del candidate_windows
    if len(probs) < MIN_WINDOWS:
        return "unknown"
    return decide_presentation(probs)


def classify_samples(
    samples: Sequence[float],
    predict: Callable[[Sequence[float]], float],
) -> Presentation:
    """Public helper for the fixed rule with an injected predictor (tests)."""
    return classify_with_predictor(samples, predict)


def softmax_female_probability(logits: Sequence[float]) -> float:
    """Convert pre-softmax ``[male, female]`` logits to the female probability."""
    values = [float(item) for item in logits]
    if len(values) != 2:
        raise PresentationError("inference-failed", "unexpected model output shape")
    peak = max(values)
    exp_male = math.exp(values[0] - peak)
    exp_female = math.exp(values[1] - peak)
    total = exp_male + exp_female
    if total <= 0.0 or not math.isfinite(total):
        raise PresentationError("inference-failed", "invalid model logits")
    return exp_female / total


class VoicePresentationClassifier:
    """Pinned ONNX acoustic presentation classifier (CPU only).

    The model is loaded once and reused for all turns. ``onnxruntime``
    is imported lazily so text-only workers and unit tests never require
    it. Any other installed onnxruntime version fails initialization
    closed. Inference consumes raw mono float32 at 16 kHz; the single
    model output is pre-softmax ``[male, female]`` logits converted to
    the female probability via softmax.
    """

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._session: Any | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        """Whether the ONNX session loaded successfully."""
        return self._session is not None

    @property
    def load_error(self) -> str | None:
        """Initialization failure detail (category only, no paths)."""
        return self._load_error

    def ensure_loaded(self) -> None:
        """Load the pinned model once; raise :class:`PresentationError` if unusable."""
        if self._session is not None:
            return
        try:
            import onnxruntime  # type: ignore[import-untyped]  # noqa: PLC0415
        except ImportError as exc:
            self._load_error = "onnxruntime is not installed"
            raise PresentationError("model-unavailable", "onnxruntime is missing") from exc
        installed = str(getattr(onnxruntime, "__version__", ""))
        if installed != ONNXRUNTIME_VERSION:
            self._load_error = "unexpected onnxruntime version"
            raise PresentationError(
                "model-unavailable",
                f"onnxruntime must be {ONNXRUNTIME_VERSION}",
            )
        try:
            model_file = ensure_presentation_model_file(self._model_path)
        except PresentationError as exc:
            self._load_error = exc.category
            raise
        except OSError as exc:
            self._load_error = "model provisioning failed"
            raise PresentationError("model-unavailable", "model provisioning failed") from exc
        try:
            import numpy as _np  # noqa: PLC0415

            session = onnxruntime.InferenceSession(
                str(model_file),
                providers=["CPUExecutionProvider"],
            )
            self._input_name = str(session.get_inputs()[0].name)
            self._output_name = str(session.get_outputs()[0].name)
            _ = _np
            self._session = session
        except PresentationError:
            raise
        except Exception as exc:
            self._load_error = "session construction failed"
            raise PresentationError("model-unavailable", "session failed") from exc
        logger.info("presentation classifier loaded")

    def female_probability(self, window: Sequence[float]) -> float:
        """Return the model female probability for one 0.75 s window."""
        session = self._session
        if session is None or self._input_name is None or self._output_name is None:
            raise PresentationError("model-unavailable", "classifier is not loaded")
        if len(window) != WINDOW_SAMPLES:
            raise PresentationError("inference-failed", "window has unexpected length")
        try:
            import numpy as _np  # noqa: PLC0415

            audio = _np.asarray([list(float(item) for item in window)], dtype=_np.float32)
            raw = session.run([self._output_name], {self._input_name: audio})
        except PresentationError:
            raise
        except Exception as exc:
            raise PresentationError("inference-failed", "model inference failed") from exc
        try:
            import numpy as _np2  # noqa: PLC0415

            logits = _np2.asarray(raw[0]).reshape(-1).tolist()
            return softmax_female_probability([float(item) for item in logits])
        except PresentationError:
            raise
        except Exception as exc:
            raise PresentationError("inference-failed", "invalid model output") from exc

    def classify(self, samples: Sequence[float]) -> Presentation:
        """Classify one turn ephemerally; ``unknown`` on any error.

        Never raises for classification reasons, never logs the class,
        probabilities, features, or audio, and never persists anything.
        """
        try:
            return classify_with_predictor(samples, self.female_probability)
        except Exception:
            return "unknown"


__all__ = [
    "AGREEMENT_FRACTION",
    "FEMALE_MEDIAN_THRESHOLD",
    "FEMALE_WINDOW_THRESHOLD",
    "MALE_MEDIAN_THRESHOLD",
    "MALE_WINDOW_THRESHOLD",
    "MAX_WINDOWS",
    "MIN_USABLE_SAMPLES",
    "MIN_USABLE_SECONDS",
    "MIN_WINDOWS",
    "ONNXRUNTIME_VERSION",
    "PRESENTATION_MODEL_FILE",
    "PRESENTATION_MODEL_REPO",
    "PRESENTATION_MODEL_SHA256",
    "PRESENTATION_SAMPLE_RATE",
    "RMS_FRAME_SAMPLES",
    "RMS_THRESHOLD",
    "WINDOW_SAMPLES",
    "WINDOW_SECONDS",
    "Presentation",
    "PresentationClassifier",
    "PresentationError",
    "VoicePresentationClassifier",
    "classify_samples",
    "classify_with_predictor",
    "decide_presentation",
    "ensure_presentation_model_file",
    "extract_voiced_windows",
    "median",
    "presentation_model_url",
    "softmax_female_probability",
    "trim_silence",
    "verify_model_sha256",
]
