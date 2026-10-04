"""Optional public voice-model cache helpers (issue #79).

This layer is pure acceleration for ephemeral GitHub-hosted runners. It
caches only the fixed public voice assets introduced by #76-#78:

- GigaAM ``large/model.int8.onnx`` + ``large/tokens.txt`` at the pinned
  revision (issue #76);
- Silero TTS ``v5_5_ru.pt`` (issue #77);
- the pinned voice-presentation ONNX file with its required SHA-256
  (issue #78).

The runtime must work identically when the cache is missing, stale,
corrupt, or unavailable. No cache hit is trusted solely because GitHub
returned one: every reuse revalidates the pinned identity from
``corpus/voice.lock.json`` before use.

This module reuses the #59 cache semantics from
:mod:`aa.corpus.public_cache` instead of designing another cache
subsystem: the same enable flag, the same sensitive-path guard, the
same exact-key discipline (no broad fallback restore key for model
files whose exact identity is required), the same verify-before-use /
prune-then-repair flow, and the same non-fatal save behaviour (owned by
the workflow ``continue-on-error`` steps).

Never cached here: Telegram voice/PCM/OGG bytes, transcripts, generated
answers, speaker probabilities/classes/features, session/user data, or
secrets. Logs and markers contain only paths, sizes, digests, and the
hit/miss/rejected category.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from aa.corpus.public_cache import (
    PublicCacheError,
    assert_cache_paths_safe,
    is_cache_enabled,
)

__all__ = [
    "PYTHON_VERSION",
    "VOICE_CACHE_VERSION",
    "VOICE_LOCK_FORMAT",
    "VOICE_MARKER_NAME",
    "PublicCacheError",
    "assert_cache_paths_safe",
    "assert_voice_cache_paths_safe",
    "default_gigaam_dir",
    "default_presentation_model_path",
    "default_tts_model_path",
    "default_voice_lock_path",
    "invalidate_gigaam_model",
    "invalidate_presentation_model",
    "invalidate_tts_model",
    "is_cache_enabled",
    "load_voice_lock",
    "verify_gigaam_model",
    "verify_presentation_model",
    "verify_tts_model",
    "voice_cache_keys",
    "voice_gigaam_key",
    "voice_lock_digest",
    "voice_presentation_key",
    "voice_silero_key",
]

VOICE_LOCK_FORMAT = "aa-voice-lock/1"
VOICE_CACHE_VERSION = "v1"
VOICE_MARKER_NAME = "aa-voice-cache.json"
PYTHON_VERSION = "3.12"

# Fragments of user-derived voice data that must never enter the cache.
# Checked case-insensitively against the joined path string, in addition
# to the #59 sensitive-path guard.
VOICE_SENSITIVE_PATH_FRAGMENTS = (
    ".ogg",
    ".opus",
    ".pcm",
    ".wav",
    "transcript",
    "generated-answer",
    "speaker-prob",
    "voice-presentation-score",
    "sendvoice",
)


def repo_root() -> Path:
    """Return the repository root for the checked-in voice lock file."""
    return Path(__file__).resolve().parents[3]


def default_voice_lock_path() -> Path:
    """Return the default ``corpus/voice.lock.json`` path."""
    return repo_root() / "corpus" / "voice.lock.json"


def default_gigaam_dir(env: dict[str, str] | None = None) -> Path:
    """Return the default GigaAM model directory (env override aware)."""
    source = os.environ if env is None else env
    raw = str(source.get("AA_VOICE_MODEL_DIR", "")).strip()
    if raw:
        return Path(raw)
    return repo_root() / "models" / "gigaam"


def default_tts_model_path(env: dict[str, str] | None = None) -> Path:
    """Return the default Silero model file path (env override aware)."""
    source = os.environ if env is None else env
    raw = str(source.get("AA_TTS_MODEL_PATH", "")).strip()
    if raw:
        return Path(raw)
    return repo_root() / "models" / "tts" / "v5_5_ru.pt"


def default_presentation_model_path(env: dict[str, str] | None = None) -> Path:
    """Return the default presentation model file path (env override aware)."""
    source = os.environ if env is None else env
    raw = str(source.get("AA_VOICE_PRESENTATION_MODEL_PATH", "")).strip()
    if raw:
        return Path(raw)
    return repo_root() / "models" / "voice-presentation" / "model_quantized.onnx"


def load_voice_lock(path: str | Path) -> dict[str, Any]:
    """Load and validate the pinned voice-model lock file."""
    lock_path = Path(path)
    try:
        raw = lock_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PublicCacheError(f"voice lock is missing: {lock_path}: {exc}") from exc
    try:
        lock = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PublicCacheError(f"voice lock is not valid JSON: {exc}") from exc
    if not isinstance(lock, dict):
        raise PublicCacheError("voice lock must be a JSON object")
    if lock.get("format") != VOICE_LOCK_FORMAT:
        raise PublicCacheError(f"unsupported voice lock format: {lock.get('format')!r}")

    gigaam = lock.get("gigaam")
    if not isinstance(gigaam, dict):
        raise PublicCacheError("voice lock gigaam must be an object")
    if gigaam.get("model_id") != "fussraider/GigaAM-Multilingual-sherpa-onnx-ctc":
        raise PublicCacheError(f"unexpected GigaAM model id: {gigaam.get('model_id')!r}")
    if not _is_hex_revision(gigaam.get("revision")):
        raise PublicCacheError(
            f"GigaAM revision must be a 40-char hex digest: {gigaam.get('revision')!r}"
        )
    if gigaam.get("model_file") != "large/model.int8.onnx":
        raise PublicCacheError(f"unexpected GigaAM model file: {gigaam.get('model_file')!r}")
    if gigaam.get("tokens_file") != "large/tokens.txt":
        raise PublicCacheError(f"unexpected GigaAM tokens file: {gigaam.get('tokens_file')!r}")

    silero = lock.get("silero")
    if not isinstance(silero, dict):
        raise PublicCacheError("voice lock silero must be an object")
    if silero.get("model_id") != "v5_5_ru":
        raise PublicCacheError(f"unexpected Silero model id: {silero.get('model_id')!r}")
    if silero.get("url") != "https://models.silero.ai/models/tts/ru/v5_5_ru.pt":
        raise PublicCacheError(f"unexpected Silero model URL: {silero.get('url')!r}")

    presentation = lock.get("presentation")
    if not isinstance(presentation, dict):
        raise PublicCacheError("voice lock presentation must be an object")
    if presentation.get("model_id") != "Alice-Sabrina-Ivy/voice-gender-classifier-onnx-q8-v2":
        raise PublicCacheError(
            f"unexpected presentation model id: {presentation.get('model_id')!r}"
        )
    if presentation.get("model_file") != "onnx/model_quantized.onnx":
        raise PublicCacheError(
            f"unexpected presentation model file: {presentation.get('model_file')!r}"
        )
    sha = presentation.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64 or not _is_hex(sha):
        raise PublicCacheError(f"presentation sha256 must be a 64-char hex digest: {sha!r}")
    return lock


def _is_hex_revision(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)
    )


def _is_hex(value: str) -> bool:
    return bool(value) and all(c in "0123456789abcdefABCDEF" for c in value)


def voice_lock_digest(lock: dict[str, Any]) -> str:
    """Return the SHA-256 digest of the canonical voice lock JSON."""
    payload = (json.dumps(lock, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_key_input(name: str, value: str) -> str:
    if not value:
        raise PublicCacheError(f"voice cache key input {name!r} must be non-empty")
    return value


def voice_gigaam_key(
    *,
    os_name: str,
    arch: str,
    python_version: str = PYTHON_VERSION,
    revision: str,
    deps_hash: str,
) -> str:
    """Build the exact compatibility/content key for the GigaAM cache."""
    _require_key_input("os_name", os_name)
    _require_key_input("arch", arch)
    _require_key_input("python_version", python_version)
    _require_key_input("deps_hash", deps_hash)
    if not _is_hex_revision(revision):
        raise PublicCacheError(f"GigaAM revision must be a 40-char hex digest: {revision!r}")
    return (
        f"aa-voice-gigaam-{VOICE_CACHE_VERSION}-{os_name}-{arch}"
        f"-py{python_version}-{revision}-{deps_hash}"
    )


def voice_silero_key(
    *,
    os_name: str,
    arch: str,
    python_version: str = PYTHON_VERSION,
    model_id: str,
    deps_hash: str,
) -> str:
    """Build the exact compatibility/content key for the Silero cache."""
    _require_key_input("os_name", os_name)
    _require_key_input("arch", arch)
    _require_key_input("python_version", python_version)
    _require_key_input("deps_hash", deps_hash)
    if model_id != "v5_5_ru":
        raise PublicCacheError(f"unexpected Silero model id: {model_id!r}")
    return (
        f"aa-voice-silero-{VOICE_CACHE_VERSION}-{os_name}-{arch}"
        f"-py{python_version}-{model_id}-{deps_hash}"
    )


def voice_presentation_key(
    *,
    os_name: str,
    arch: str,
    python_version: str = PYTHON_VERSION,
    sha256: str,
    deps_hash: str,
) -> str:
    """Build the exact compatibility/content key for the presentation cache."""
    _require_key_input("os_name", os_name)
    _require_key_input("arch", arch)
    _require_key_input("python_version", python_version)
    _require_key_input("deps_hash", deps_hash)
    if not isinstance(sha256, str) or len(sha256) != 64 or not _is_hex(sha256):
        raise PublicCacheError(f"presentation sha256 must be a 64-char hex digest: {sha256!r}")
    return (
        f"aa-voice-presentation-{VOICE_CACHE_VERSION}-{os_name}-{arch}"
        f"-py{python_version}-{sha256.lower()}-{deps_hash}"
    )


def voice_cache_keys(
    *,
    os_name: str,
    arch: str,
    python_version: str = PYTHON_VERSION,
    lock: dict[str, Any],
    deps_hash: str,
) -> dict[str, str]:
    """Build the exact cache key for each of the three voice model families."""
    gigaam = lock["gigaam"]
    silero = lock["silero"]
    presentation = lock["presentation"]
    assert isinstance(gigaam, dict)
    assert isinstance(silero, dict)
    assert isinstance(presentation, dict)
    return {
        "gigaam": voice_gigaam_key(
            os_name=os_name,
            arch=arch,
            python_version=python_version,
            revision=str(gigaam["revision"]),
            deps_hash=deps_hash,
        ),
        "silero": voice_silero_key(
            os_name=os_name,
            arch=arch,
            python_version=python_version,
            model_id=str(silero["model_id"]),
            deps_hash=deps_hash,
        ),
        "presentation": voice_presentation_key(
            os_name=os_name,
            arch=arch,
            python_version=python_version,
            sha256=str(presentation["sha256"]),
            deps_hash=deps_hash,
        ),
    }


def assert_voice_cache_paths_safe(paths: list[str] | tuple[str, ...]) -> None:
    """Fail closed when a voice-cache path could carry user-derived data.

    Reuses the #59 sensitive-path guard and additionally rejects audio,
    transcript, and speaker-signal artefacts that must never enter the
    model cache.
    """
    assert_cache_paths_safe(list(paths))
    for path in paths:
        lowered = str(path).replace("\\", "/").lower()
        for fragment in VOICE_SENSITIVE_PATH_FRAGMENTS:
            if fragment in lowered:
                raise PublicCacheError(f"refusing user-derived voice cache path: {path!r}")


def _marker_path(model_dir: Path) -> Path:
    return model_dir / VOICE_MARKER_NAME


def _read_marker(model_dir: Path) -> dict[str, Any] | None:
    try:
        raw = _marker_path(model_dir).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        marker = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return marker if isinstance(marker, dict) else None


def _write_marker(model_dir: Path, payload: dict[str, Any]) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    marker = _marker_path(model_dir)
    body = dict(payload)
    body["format"] = "aa-voice-cache-marker/1"
    marker.write_text(json.dumps(body, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return marker


def _non_empty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def verify_gigaam_model(model_dir: str | Path, lock: dict[str, Any]) -> bool:
    """Return True only when cached GigaAM files match the pinned lock."""
    gigaam = lock.get("gigaam")
    if not isinstance(gigaam, dict):
        return False
    revision = str(gigaam.get("revision"))
    if not _is_hex_revision(revision):
        return False
    root = Path(model_dir)
    if not _non_empty_file(root / "model.int8.onnx"):
        return False
    if not _non_empty_file(root / "tokens.txt"):
        return False
    marker = _read_marker(root)
    if marker is None:
        return False
    if marker.get("family") != "gigaam":
        return False
    if marker.get("revision") != revision:
        return False
    if marker.get("model_id") != gigaam.get("model_id"):
        return False
    return True


def verify_tts_model(model_path: str | Path, lock: dict[str, Any]) -> bool:
    """Return True only when the cached Silero file matches the pinned lock."""
    silero = lock.get("silero")
    if not isinstance(silero, dict):
        return False
    path = Path(model_path)
    if not _non_empty_file(path):
        return False
    marker = _read_marker(path.parent)
    if marker is None:
        return False
    if marker.get("family") != "silero":
        return False
    if marker.get("model_id") != silero.get("model_id"):
        return False
    if marker.get("url") != silero.get("url"):
        return False
    return True


def verify_presentation_model(model_path: str | Path, lock: dict[str, Any]) -> bool:
    """Return True only when the cached classifier matches the pinned SHA-256."""
    presentation = lock.get("presentation")
    if not isinstance(presentation, dict):
        return False
    expected = str(presentation.get("sha256"))
    if len(expected) != 64 or not _is_hex(expected):
        return False
    path = Path(model_path)
    if not _non_empty_file(path):
        return False
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return False
    if digest.lower() != expected.lower():
        return False
    marker = _read_marker(path.parent)
    if marker is None:
        return False
    if marker.get("family") != "presentation":
        return False
    if str(marker.get("sha256")).lower() != expected.lower():
        return False
    return True


def write_gigaam_marker(model_dir: str | Path, lock: dict[str, Any]) -> Path:
    """Record the verified GigaAM revision next to the cached model files."""
    gigaam = lock["gigaam"]
    assert isinstance(gigaam, dict)
    return _write_marker(
        Path(model_dir),
        {
            "family": "gigaam",
            "model_id": str(gigaam["model_id"]),
            "revision": str(gigaam["revision"]),
            "lock_sha256": voice_lock_digest(lock),
        },
    )


def write_tts_marker(model_path: str | Path, lock: dict[str, Any]) -> Path:
    """Record the verified Silero identity next to the cached model file."""
    silero = lock["silero"]
    assert isinstance(silero, dict)
    return _write_marker(
        Path(model_path).parent,
        {
            "family": "silero",
            "model_id": str(silero["model_id"]),
            "url": str(silero["url"]),
            "lock_sha256": voice_lock_digest(lock),
        },
    )


def write_presentation_marker(model_path: str | Path, lock: dict[str, Any]) -> Path:
    """Record the verified classifier SHA-256 next to the cached model file."""
    presentation = lock["presentation"]
    assert isinstance(presentation, dict)
    return _write_marker(
        Path(model_path).parent,
        {
            "family": "presentation",
            "model_id": str(presentation["model_id"]),
            "model_file": str(presentation["model_file"]),
            "sha256": str(presentation["sha256"]).lower(),
            "lock_sha256": voice_lock_digest(lock),
        },
    )


def invalidate_gigaam_model(model_dir: str | Path) -> None:
    """Remove a stale/corrupt GigaAM cache entry so download can repair it."""
    root = Path(model_dir)
    try:
        if _marker_path(root).is_file():
            _marker_path(root).unlink()
    except OSError:
        pass
    for name in ("model.int8.onnx", "tokens.txt"):
        try:
            candidate = root / name
            if candidate.is_file():
                candidate.unlink()
        except OSError:
            continue


def invalidate_tts_model(model_path: str | Path) -> None:
    """Remove a stale/corrupt Silero cache entry so download can repair it."""
    path = Path(model_path)
    try:
        if _marker_path(path.parent).is_file():
            _marker_path(path.parent).unlink()
    except OSError:
        pass
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


def invalidate_presentation_model(model_path: str | Path) -> None:
    """Remove a stale/corrupt classifier cache entry so download can repair it."""
    path = Path(model_path)
    try:
        if _marker_path(path.parent).is_file():
            _marker_path(path.parent).unlink()
    except OSError:
        pass
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass
