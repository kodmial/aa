"""Pinned local BGE cross-encoder reranker for the v2 pipeline (issue #116).

Production reranker contract (fixed):

- model: ``BAAI/bge-reranker-v2-m3`` at the immutable revision pinned in
  ``corpus/reranker.lock.json``;
- runtime API: the maintained ``FlagEmbedding`` reranker interface
  (``FlagEmbedding.FlagReranker``), local in-process CPU execution with
  batched scoring;
- no network calls on the per-turn hot path: bootstrap may populate and
  validate the model cache before runtime readiness, but an ordinary turn
  reuses the already-loaded model with ``HF_HUB_OFFLINE=1`` semantics.

Cache lifecycle reuses the repository model-cache pattern from
:mod:`aa.corpus.public_cache`: the cache key binds runner OS/arch,
Python/runtime versions, model revision/checksums and the reranker
library version. Every cache hit is validated before use; corrupt or
stale cache is discarded and rebuilt. One long-lived reranker instance
per worker scores candidates in batches.

Offline/test backend: explicit test injection only. Construct
``CrossEncoderReranker(lock, backend=OFFLINE_BACKEND_NAME)`` directly in
tests or pass ``allow_offline=True`` to :func:`get_reranker` for hermetic
CI without weights. Production (``allow_offline=False``) fails closed
with :class:`RerankerError` when the pinned snapshot or the
``FlagEmbedding`` stack is unavailable instead of silently serving
heuristic ordering.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aa.retrieval.normalize import ru_stem, ru_tokens

logger = logging.getLogger("aa.retrieval.reranker")

RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
RERANKER_LOCK_FORMAT = "aa-reranker-lock/1"
RERANKER_MODEL_DIR_NAME = "models--BAAI--bge-reranker-v2-m3"
RERANKER_MARKER_NAME = "aa-reranker-cache.json"

OFFLINE_BACKEND_NAME = "offline-stem-overlap/1"
FLAG_BACKEND_NAME = "bge-reranker-v2-m3-flag/1"

_HEX40_RE = frozenset("0123456789abcdef")


class RerankerError(ValueError):
    """Raised when the reranker lock, cache or scoring cannot be served."""


def _is_hex_revision(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(char in _HEX40_RE for char in value)


def _is_hex_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _HEX40_RE for char in value.casefold())
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lock_file_hashes(lock: dict[str, Any]) -> dict[str, str] | None:
    """Return optional pinned per-file SHA-256 digests from the lock."""
    raw = lock.get("files")
    if raw is None:
        raw = lock.get("file_hashes")
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        raise RerankerError("reranker lock file hashes must be a non-empty object")
    cleaned: dict[str, str] = {}
    for name, digest in raw.items():
        if not isinstance(name, str) or not name or ".." in name or name.startswith("/"):
            raise RerankerError(f"reranker lock file hash has bad name: {name!r}")
        if isinstance(digest, dict):
            digest = digest.get("sha256")
        if not _is_hex_sha256(digest):
            raise RerankerError(f"reranker lock file hash must be sha256 hex: {name!r}")
        assert isinstance(digest, str)
        cleaned[name] = digest.casefold()
    return cleaned


def repo_root() -> Path:
    """Return the repository root holding ``corpus/reranker.lock.json``."""
    return Path(__file__).resolve().parents[3]


def default_reranker_lock_path() -> Path:
    """Return the default committed reranker lock path."""
    return repo_root() / "corpus" / "reranker.lock.json"


def load_reranker_lock(path: str | Path) -> dict[str, Any]:
    """Load and validate the pinned reranker lock file (fails closed)."""
    lock_path = Path(path)
    try:
        raw = lock_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RerankerError(f"reranker lock is missing: {lock_path}: {exc}") from exc
    try:
        lock = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RerankerError(f"reranker lock is not valid JSON: {exc}") from exc
    if not isinstance(lock, dict):
        raise RerankerError("reranker lock must be a JSON object")
    if lock.get("format") != RERANKER_LOCK_FORMAT:
        raise RerankerError(f"unsupported reranker lock format: {lock.get('format')!r}")
    if lock.get("model_id") != RERANKER_MODEL_ID:
        raise RerankerError(f"unexpected reranker model id: {lock.get('model_id')!r}")
    revision = lock.get("revision")
    if not _is_hex_revision(revision):
        raise RerankerError(f"reranker revision must be a 40-char hex digest: {revision!r}")
    if lock.get("sha") is not None and lock.get("sha") != revision:
        raise RerankerError("reranker lock sha must equal the pinned revision")
    required = lock.get("required_files")
    if (
        not isinstance(required, list)
        or not required
        or not all(
            isinstance(name, str) and name and ".." not in name and not name.startswith("/")
            for name in required
        )
    ):
        raise RerankerError("reranker lock required_files must be a non-empty path list")
    if lock.get("files") is not None or lock.get("file_hashes") is not None:
        pinned = _lock_file_hashes(lock)
        assert pinned is not None
        missing = [name for name in required if name not in pinned]
        if missing:
            raise RerankerError(f"reranker lock file hashes miss required files: {missing!r}")
    runtime = lock.get("runtime")
    if not isinstance(runtime, dict):
        raise RerankerError("reranker lock must carry a runtime section")
    if runtime.get("interface") != "FlagEmbedding.FlagReranker":
        raise RerankerError("reranker runtime interface must be FlagEmbedding.FlagReranker")
    if runtime.get("device") != "cpu":
        raise RerankerError("reranker runtime device must stay pinned to cpu")
    return dict(lock)


def lock_digest(lock: dict[str, Any]) -> str:
    """Return the SHA-256 digest of the canonical lock JSON (key binding)."""
    payload = (json.dumps(lock, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_hf_cache_dir(env: dict[str, str] | None = None) -> Path:
    """Resolve the Hugging Face hub cache directory (public files only)."""
    source = os.environ if env is None else env
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        raw = str(source.get(key, "")).strip()
        if raw:
            base = Path(raw)
            if key == "HF_HOME":
                return base / "hub"
            if base.name == "hub":
                return base
            return base
    home = Path(str(source.get("HOME", Path.home().as_posix()))).expanduser()
    return home / ".cache" / "huggingface" / "hub"


def resolve_reranker_root(hf_cache_dir: str | Path) -> Path:
    """Return the hub directory holding the pinned reranker snapshot."""
    return Path(hf_cache_dir) / RERANKER_MODEL_DIR_NAME


def snapshot_dir(model_root: str | Path, revision: str) -> Path:
    """Return the hub snapshot directory for ``revision``."""
    return Path(model_root) / "snapshots" / revision


def normalize_os_name(value: str | None = None) -> str:
    """Normalize an OS name to the runner-style token."""
    import platform as _platform

    raw = (value or _platform.system()).strip().lower()
    if raw == "linux":
        return "Linux"
    if raw in {"darwin", "macos", "mac os x"}:
        return "macOS"
    if raw in {"windows", "windows_nt"}:
        return "Windows"
    return (value or _platform.system()).strip() or "Unknown-OS"


def normalize_arch(value: str | None = None) -> str:
    """Normalize a machine arch to the runner-style token."""
    import platform as _platform

    raw = (value or _platform.machine()).strip().lower()
    if raw in {"x86_64", "x64", "amd64"}:
        return "X64"
    if raw in {"aarch64", "arm64"}:
        return "ARM64"
    return (value or _platform.machine()).strip() or "Unknown-Arch"


def flag_embedding_version() -> str:
    """Return the installed FlagEmbedding version, or ``absent``."""
    try:
        import importlib.metadata as _metadata

        return str(_metadata.version("FlagEmbedding"))
    except Exception:
        return "absent"


def runtime_versions() -> dict[str, str]:
    """Return the runtime versions bound into the reranker cache key."""
    versions: dict[str, str] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_full": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "platform": platform.platform(),
        "flag_embedding": flag_embedding_version(),
    }
    try:
        import importlib.metadata as _metadata

        for dist in ("torch", "transformers"):
            try:
                versions[dist] = str(_metadata.version(dist))
            except Exception:
                versions[dist] = "absent"
    except Exception:
        versions.setdefault("torch", "absent")
        versions.setdefault("transformers", "absent")
    return versions


def reranker_cache_key(
    lock: dict[str, Any],
    *,
    os_name: str | None = None,
    arch: str | None = None,
) -> str:
    """Build the exact cache key for the pinned reranker snapshot.

    The key binds runner OS/arch, Python/runtime versions, model
    revision/checksums and the reranker library version. No fallback
    restore keys are used: a wrong revision must miss safely instead of
    reusing an incompatible snapshot.
    """
    runtime_versions_bound = runtime_versions()
    canonical = {
        "model_id": str(lock.get("model_id")),
        "revision": str(lock.get("revision")),
        "lock_sha256": lock_digest(lock),
        "os": normalize_os_name(os_name),
        "arch": normalize_arch(arch),
        "python": runtime_versions_bound["python"],
        "flag_embedding": runtime_versions_bound["flag_embedding"],
        "torch": runtime_versions_bound.get("torch", "absent"),
        "transformers": runtime_versions_bound.get("transformers", "absent"),
    }
    digest = hashlib.sha256(
        (json.dumps(canonical, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    ).hexdigest()
    slug = RERANKER_MODEL_ID.replace("/", "-").replace("_", "-").lower()
    return (
        f"aa-reranker-v1-{canonical['os']}-{canonical['arch']}"
        f"-{slug}-{canonical['revision']}-{digest[:16]}"
    )


def verify_cached_reranker(model_root: str | Path, lock: dict[str, Any]) -> bool:
    """Return True only when the cached reranker matches the pinned lock.

    Every cache hit is validated: the marker revision must equal the
    pinned revision, the marker must bind the exact lock digest, and
    every ``required_files`` entry must exist under
    ``snapshots/<revision>/`` with matching size and SHA-256 digest
    recorded in the marker (plus the lock-pinned digest when the lock
    carries per-file hashes). Truncated or bit-corrupted files with
    size > 0 fail closed instead of loading silently.
    """
    revision = str(lock.get("revision"))
    required = lock.get("required_files")
    if not _is_hex_revision(revision) or not isinstance(required, list):
        return False
    try:
        pinned = _lock_file_hashes(lock)
    except RerankerError:
        return False
    marker_path = Path(model_root) / RERANKER_MARKER_NAME
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(marker, dict):
        return False
    if marker.get("model_id") != lock.get("model_id"):
        return False
    if marker.get("revision") != revision:
        return False
    if marker.get("lock_sha256") != lock_digest(lock):
        return False
    recorded = marker.get("files")
    if not isinstance(recorded, dict):
        return False
    snapshot = snapshot_dir(model_root, revision)
    for name in required:
        if not isinstance(name, str) or not name:
            return False
        try:
            candidate = snapshot / name
            if not candidate.is_file():
                return False
            size = candidate.stat().st_size
            if size == 0:
                return False
            entry = recorded.get(name)
            if not isinstance(entry, dict):
                return False
            expected_sha = entry.get("sha256")
            expected_size = entry.get("size")
            if not _is_hex_sha256(expected_sha):
                return False
            if not isinstance(expected_size, int) or expected_size <= 0:
                return False
            if size != expected_size:
                return False
            if pinned is not None:
                want = pinned.get(name)
                if want is None or str(expected_sha).casefold() != want:
                    return False
            try:
                actual_sha = _sha256_file(candidate)
            except OSError:
                return False
            if actual_sha.casefold() != str(expected_sha).casefold():
                return False
        except OSError:
            return False
    return True


def write_reranker_marker(model_root: str | Path, lock: dict[str, Any]) -> Path:
    """Record the verified pinned reranker revision next to the cache."""
    root = Path(model_root)
    root.mkdir(parents=True, exist_ok=True)
    revision = str(lock.get("revision"))
    required = lock.get("required_files")
    files: dict[str, dict[str, Any]] = {}
    if isinstance(required, list):
        snapshot = snapshot_dir(root, revision)
        for name in required:
            if not isinstance(name, str) or not name:
                continue
            candidate = snapshot / name
            try:
                if not candidate.is_file():
                    continue
                size = candidate.stat().st_size
                if size <= 0:
                    continue
                files[name] = {"sha256": _sha256_file(candidate), "size": size}
            except OSError:
                continue
    marker = root / RERANKER_MARKER_NAME
    payload = {
        "format": "aa-reranker-model-marker/1",
        "model_id": str(lock.get("model_id")),
        "revision": revision,
        "lock_sha256": lock_digest(lock),
        "files": files,
    }
    marker.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return marker


def invalidate_cached_reranker(model_root: str | Path) -> None:
    """Discard a corrupt or stale cached reranker so bootstrap can repair it."""
    root = Path(model_root)
    marker = root / RERANKER_MARKER_NAME
    try:
        if marker.is_file():
            marker.unlink()
    except OSError:
        pass
    snapshots = root / "snapshots"
    if snapshots.is_dir():
        for child in sorted(snapshots.iterdir()):
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    for sub in sorted(child.rglob("*"), reverse=True):
                        try:
                            if sub.is_file() or sub.is_symlink():
                                sub.unlink()
                            elif sub.is_dir():
                                sub.rmdir()
                        except OSError:
                            pass
                    child.rmdir()
            except OSError:
                continue


def _offline_scores(query: str, texts: list[str]) -> list[float]:
    """Deterministic offline relevance scorer (no weights, no network).

    Stemmed token overlap between the canonical query and each exact
    child text, lightly smoothed with character-trigram overlap for
    typo tolerance. Scores only reorder candidates; text is untouched.
    """
    query_stems = {ru_stem(token) for token in ru_tokens(query)}
    query_collapsed = "".join(sorted(query_stems))
    query_tris = (
        {query_collapsed[pos : pos + 3] for pos in range(len(query_collapsed) - 2)}
        if len(query_collapsed) >= 3
        else set()
    )
    scores: list[float] = []
    for text in texts:
        doc_stems = [ru_stem(token) for token in ru_tokens(text)]
        if not doc_stems or not query_stems:
            scores.append(0.0)
            continue
        doc_set = set(doc_stems)
        overlap = len(query_stems & doc_set)
        # Recall-oriented: fraction of query stems present, with a small
        # precision term so longer exact matches win ties.
        recall = overlap / len(query_stems)
        precision = overlap / len(doc_set) if doc_set else 0.0
        doc_collapsed = "".join(sorted(doc_set))
        doc_tris = (
            {doc_collapsed[pos : pos + 3] for pos in range(len(doc_collapsed) - 2)}
            if len(doc_collapsed) >= 3
            else set()
        )
        tri = (
            len(query_tris & doc_tris) / len(query_tris | doc_tris)
            if (query_tris or doc_tris)
            else 0.0
        )
        scores.append(0.7 * recall + 0.2 * precision + 0.1 * tri)
    return scores


@dataclass
class CrossEncoderReranker:
    """One long-lived reranker instance per worker (reused across turns).

    The production backend loads ``BAAI/bge-reranker-v2-m3`` once through
    the maintained ``FlagEmbedding.FlagReranker`` interface on CPU and
    scores candidates in batches. Ordinary turns reuse this instance;
    no model download or network access occurs on the hot path.
    """

    lock: dict[str, Any]
    backend: str = OFFLINE_BACKEND_NAME
    _flag_reranker: Any = field(default=None, repr=False)

    @property
    def model_id(self) -> str:
        """Return the pinned reranker model id."""
        return str(self.lock.get("model_id"))

    @property
    def revision(self) -> str:
        """Return the pinned immutable model revision."""
        return str(self.lock.get("revision"))

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score ``(query, text)`` pairs in one batch (order preserved).

        ``query`` must be the context-resolved canonical query
        (``queries[0]``); every ``text`` must be exact Russian child
        chunk text. Returns one float per text, best-first independent.
        """
        if not isinstance(query, str) or not query.strip():
            raise RerankerError("reranker query must be a non-empty string")
        if not texts:
            return []
        for text in texts:
            if not isinstance(text, str) or not text:
                raise RerankerError("reranker candidates must be non-empty strings")
        if self._flag_reranker is not None:
            return self._score_flag(query, texts)
        return _offline_scores(query, texts)

    def _score_flag(self, query: str, texts: list[str]) -> list[float]:
        pairs = [[query, text] for text in texts]
        try:
            with _RERANKER_SCORE_LOCK:
                raw = self._flag_reranker.compute_score(pairs, normalize=True)
        except Exception as exc:
            raise RerankerError(f"FlagEmbedding reranker scoring failed: {exc}") from exc
        if isinstance(raw, float):
            return [float(raw)]
        try:
            return [float(value) for value in list(raw)]
        except TypeError as exc:
            raise RerankerError(f"FlagEmbedding reranker returned no scores: {exc}") from exc


_RERANKER_SINGLETONS: dict[str, CrossEncoderReranker] = {}

# Zero hot-path disk-read contract: the validated lock is parsed once and
# cached in process memory; warm turns reuse the singleton without
# touching the filesystem.
_RERANKER_LOCK_CACHE: dict[str, dict[str, Any]] = {}
_RERANKER_INIT_LOCK = threading.Lock()
_RERANKER_SCORE_LOCK = threading.Lock()


def _load_cached_lock(resolved: Path) -> dict[str, Any]:
    """Return the validated lock, reading disk only on the first miss."""
    key = str(resolved)
    cached = _RERANKER_LOCK_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    lock = load_reranker_lock(resolved)
    _RERANKER_LOCK_CACHE[key] = dict(lock)
    return dict(lock)


def _load_flag_reranker(lock: dict[str, Any]) -> Any:
    """Load the FlagEmbedding reranker from the validated local cache.

    Fails closed with :class:`RerankerError` when the stack or the pinned
    snapshot is unavailable; the caller must not silently serve heuristic
    ordering in production. Never downloads on the hot path.
    """
    try:
        import importlib
    except Exception as exc:
        raise RerankerError(f"FlagEmbedding stack is unavailable: {exc}") from exc
    try:
        flag_module = importlib.import_module("FlagEmbedding")
    except Exception as exc:
        raise RerankerError(f"FlagEmbedding stack is unavailable: {exc}") from exc
    flag_cls = getattr(flag_module, "FlagReranker", None)
    if flag_cls is None:
        raise RerankerError("FlagEmbedding.FlagReranker interface is unavailable")
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        model_root = resolve_reranker_root(resolve_hf_cache_dir())
        if not verify_cached_reranker(model_root, lock):
            raise RerankerError("pinned BGE snapshot is not cached locally; refusing network fetch")
        snapshot = snapshot_dir(model_root, str(lock.get("revision")))
        try:
            # Maintained FlagEmbedding reranker interface, CPU, local only.
            return flag_cls(str(snapshot), use_fp16=False, device="cpu")
        except Exception as exc:
            raise RerankerError(f"FlagEmbedding reranker load failed: {exc}") from exc
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


def _try_load_flag_reranker(lock: dict[str, Any]) -> Any | None:
    """Best-effort flag load returning None (explicit offline paths only)."""
    try:
        return _load_flag_reranker(lock)
    except RerankerError:
        return None


def get_reranker(
    lock_path: str | Path | None = None, *, allow_offline: bool = False
) -> CrossEncoderReranker:
    """Return the long-lived worker reranker for the pinned lock.

    Loads once per lock revision; every later turn reuses the instance
    and performs zero model downloads, zero network access and zero
    lock-file disk reads on the warm path (the validated lock is cached
    in process memory and the singleton is checked under a lock).

    Production (``allow_offline=False``) is BGE-only and fails closed
    with :class:`RerankerError` when the pinned snapshot or the
    ``FlagEmbedding`` stack is unavailable. Pass
    ``allow_offline=True`` only for explicit hermetic test injection
    where the deterministic stem-overlap backend is wanted.
    """
    resolved = Path(lock_path) if lock_path is not None else default_reranker_lock_path()
    cache_key = str(resolved)
    with _RERANKER_INIT_LOCK:
        cached_lock = _RERANKER_LOCK_CACHE.get(cache_key)
        if cached_lock is not None:
            key = f"{cached_lock.get('model_id')}@{cached_lock.get('revision')}"
            cached = _RERANKER_SINGLETONS.get(key)
            if cached is not None:
                return cached
            lock = dict(cached_lock)
        else:
            lock = load_reranker_lock(resolved)
            _RERANKER_LOCK_CACHE[cache_key] = dict(lock)
            key = f"{lock.get('model_id')}@{lock.get('revision')}"
            cached = _RERANKER_SINGLETONS.get(key)
            if cached is not None:
                return cached
        try:
            flag_reranker = _load_flag_reranker(lock)
        except RerankerError:
            if not allow_offline:
                raise
            flag_reranker = None
        if flag_reranker is not None:
            instance = CrossEncoderReranker(
                lock=lock, backend=FLAG_BACKEND_NAME, _flag_reranker=flag_reranker
            )
        else:
            instance = CrossEncoderReranker(lock=lock, backend=OFFLINE_BACKEND_NAME)
        _RERANKER_SINGLETONS[key] = instance
    logger.info("reranker ready backend=%s model=%s", instance.backend, key)
    return instance


def reset_reranker_cache() -> None:
    """Drop cached reranker singletons and the cached lock (tests only)."""
    _RERANKER_SINGLETONS.clear()
    _RERANKER_LOCK_CACHE.clear()


__all__ = [
    "FLAG_BACKEND_NAME",
    "OFFLINE_BACKEND_NAME",
    "RERANKER_LOCK_FORMAT",
    "RERANKER_MODEL_DIR_NAME",
    "RERANKER_MODEL_ID",
    "CrossEncoderReranker",
    "RerankerError",
    "default_reranker_lock_path",
    "flag_embedding_version",
    "get_reranker",
    "load_reranker_lock",
    "lock_digest",
    "reranker_cache_key",
    "reset_reranker_cache",
    "resolve_hf_cache_dir",
    "resolve_reranker_root",
    "runtime_versions",
    "snapshot_dir",
    "verify_cached_reranker",
]
