"""Optional public dependency/model cache helpers (issue #59).

This layer is pure acceleration for ephemeral GitHub-hosted runners. It
caches only reproducible public, non-sensitive assets:

- Python package download/wheel cache (``~/.cache/pip``);
- pinned ``intfloat/multilingual-e5-base`` model files under the
  Hugging Face hub cache.

The runtime must work identically when the cache is missing, corrupt,
incompatible, or unavailable. No cache hit is trusted solely because
GitHub returned one: every reuse revalidates the pinned identity from
``corpus/embedding.lock.json`` before use.

Never cached here: corpus text, retrieval indexes, secrets,
Telegram/user state, or any decrypted artifact. The text-bearing
retrieval cache belongs to #58 and is encrypted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

LOCK_FORMAT = "aa-public-embedding-lock/1"
PUBLIC_CACHE_VERSION = "v1"
MARKER_NAME = "aa-public-cache.json"
MODEL_ID = "intfloat/multilingual-e5-base"
MODEL_DIR_NAME = "models--intfloat--multilingual-e5-base"
PYTHON_VERSION = "3.12"

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

# Fragments that must never appear in a public-cache path. Checked
# case-insensitively against the joined path string.
SENSITIVE_PATH_FRAGMENTS = (
    "corpus/generated",
    "corpus\\generated",
    "corpus/source/raw",
    "corpus\\source\\raw",
    "corpus/source/encrypted",
    "corpus\\source\\encrypted",
    "corpus/source/fetch-state",
    "canonical.json",
    ".tar.zst",
    ".age",
    "age-secret-key",
    "telegram",
    "bot_token",
    "opencode.db",
    "sessions",
    "/storage/",
    "\\storage\\",
)


class PublicCacheError(ValueError):
    """Raised when the public-cache lock or paths are invalid."""


def is_hex_revision(value: object) -> bool:
    """Return True when ``value`` is a 40-char lowercase hex revision."""
    return isinstance(value, str) and _REVISION_RE.match(value) is not None


def repo_root() -> Path:
    """Return the repository root for the checked-in lock file."""
    return Path(__file__).resolve().parents[3]


def default_lock_path() -> Path:
    """Return the default ``corpus/embedding.lock.json`` path."""
    return repo_root() / "corpus" / "embedding.lock.json"


def load_embedding_lock(path: str | Path) -> dict[str, object]:
    """Load and validate the pinned embedding-model lock file."""
    lock_path = Path(path)
    try:
        raw = lock_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PublicCacheError(f"embedding lock is missing: {lock_path}: {exc}") from exc
    try:
        lock = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PublicCacheError(f"embedding lock is not valid JSON: {exc}") from exc
    if not isinstance(lock, dict):
        raise PublicCacheError("embedding lock must be a JSON object")
    if lock.get("format") != LOCK_FORMAT:
        raise PublicCacheError(f"unsupported embedding lock format: {lock.get('format')!r}")
    if lock.get("model_id") != MODEL_ID:
        raise PublicCacheError(f"unexpected embedding model id: {lock.get('model_id')!r}")
    revision = lock.get("revision")
    if not is_hex_revision(revision):
        raise PublicCacheError(
            f"embedding lock revision must be a 40-char hex digest: {revision!r}"
        )
    sha = lock.get("sha")
    if sha is not None and sha != revision:
        raise PublicCacheError("embedding lock sha must equal the pinned revision")
    required = lock.get("required_files")
    if (
        not isinstance(required, list)
        or not required
        or not all(
            isinstance(name, str) and name and ".." not in name and not name.startswith("/")
            for name in required
        )
    ):
        raise PublicCacheError("embedding lock required_files must be a non-empty path list")
    return lock


def lock_digest(lock: dict[str, object]) -> str:
    """Return the SHA-256 digest of the canonical lock JSON (for key binding)."""
    payload = (json.dumps(lock, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_slug(model_id: str) -> str:
    """Return the filesystem slug for a Hugging Face model id."""
    return model_id.replace("/", "--").replace("_", "--")


def public_pip_key(
    *,
    os_name: str,
    arch: str,
    python_version: str = PYTHON_VERSION,
    deps_hash: str,
) -> str:
    """Build the compatibility/content key for the public pip cache."""
    if not os_name or not arch or not python_version or not deps_hash:
        raise PublicCacheError("pip cache key inputs must all be non-empty")
    return f"aa-public-pip-{PUBLIC_CACHE_VERSION}-{os_name}-{arch}-py{python_version}-{deps_hash}"


def public_pip_restore_prefix(
    *, os_name: str, arch: str, python_version: str = PYTHON_VERSION
) -> str:
    """Build the fallback restore prefix for the public pip cache only."""
    if not os_name or not arch or not python_version:
        raise PublicCacheError("pip restore prefix inputs must all be non-empty")
    return f"aa-public-pip-{PUBLIC_CACHE_VERSION}-{os_name}-{arch}-py{python_version}-"


def public_model_key(
    *,
    os_name: str,
    arch: str,
    model_id: str = MODEL_ID,
    revision: str,
    lock_hash: str,
) -> str:
    """Build the exact compatibility/content key for the embedding model cache.

    No fallback restore-keys are used for the model cache: a wrong revision
    must miss safely instead of reusing an incompatible snapshot.
    """
    if not os_name or not arch or not lock_hash:
        raise PublicCacheError("model cache key inputs must all be non-empty")
    if model_id != MODEL_ID:
        raise PublicCacheError(f"unexpected embedding model id: {model_id!r}")
    if not is_hex_revision(revision):
        raise PublicCacheError(f"model revision must be a 40-char hex digest: {revision!r}")
    slug = model_slug(model_id).replace("--", "-")
    return f"aa-public-model-{PUBLIC_CACHE_VERSION}-{os_name}-{arch}-{slug}-{revision}-{lock_hash}"


def assert_cache_paths_safe(paths: list[str] | tuple[str, ...]) -> None:
    """Fail closed when a public-cache path could carry sensitive data."""
    for path in paths:
        lowered = str(path).replace("\\", "/").lower()
        for fragment in SENSITIVE_PATH_FRAGMENTS:
            if fragment.lower() in lowered:
                raise PublicCacheError(f"refusing sensitive public-cache path: {path!r}")


def is_cache_enabled(env: dict[str, str] | None = None) -> bool:
    """Return False only when explicitly disabled via ``AA_PUBLIC_CACHE_ENABLED``."""
    source = os.environ if env is None else env
    raw = str(source.get("AA_PUBLIC_CACHE_ENABLED", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off", "disabled"}


def resolve_hf_cache_dir(env: dict[str, str] | None = None) -> Path:
    """Resolve the Hugging Face hub cache directory (public model files only)."""
    source = os.environ if env is None else env
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        raw = str(source.get(key, "")).strip()
        if raw:
            base = Path(raw)
            # HF_HOME nests the hub cache under ``hub``.
            if key == "HF_HOME":
                return base / "hub"
            if base.name == "hub":
                return base
            return base
    home = Path(str(source.get("HOME", Path.home().as_posix()))).expanduser()
    return home / ".cache" / "huggingface" / "hub"


def resolve_model_root(hf_cache_dir: str | Path) -> Path:
    """Return the hub directory holding the pinned model snapshot."""
    return Path(hf_cache_dir) / MODEL_DIR_NAME


def marker_path(model_root: str | Path) -> Path:
    """Return the marker file recording the verified cached revision."""
    return Path(model_root) / MARKER_NAME


def snapshot_dir(model_root: str | Path, revision: str) -> Path:
    """Return the hub snapshot directory for ``revision``."""
    return Path(model_root) / "snapshots" / revision


def write_marker(model_root: str | Path, lock: dict[str, object]) -> Path:
    """Record the verified pinned revision next to the cached model."""
    root = Path(model_root)
    root.mkdir(parents=True, exist_ok=True)
    marker = marker_path(root)
    payload = {
        "format": "aa-public-model-marker/1",
        "model_id": str(lock.get("model_id")),
        "revision": str(lock.get("revision")),
        "lock_sha256": lock_digest(lock),
    }
    marker.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return marker


def read_marker(model_root: str | Path) -> dict[str, object] | None:
    """Read the cache marker, returning None when missing or malformed."""
    try:
        raw = marker_path(model_root).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        marker = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return marker if isinstance(marker, dict) else None


def verify_cached_model(model_root: str | Path, lock: dict[str, object]) -> bool:
    """Return True only when the cached model matches the pinned lock.

    Verification never trusts a cache hit alone: the marker revision must
    equal the pinned revision and every ``required_files`` entry must exist
    as a non-empty file under ``snapshots/<revision>/``.
    """
    revision = str(lock.get("revision"))
    required = lock.get("required_files")
    if not is_hex_revision(revision) or not isinstance(required, list):
        return False
    marker = read_marker(model_root)
    if marker is None:
        return False
    if marker.get("model_id") != lock.get("model_id"):
        return False
    if marker.get("revision") != revision:
        return False
    snapshot = snapshot_dir(model_root, revision)
    for name in required:
        if not isinstance(name, str) or not name:
            return False
        try:
            candidate = snapshot / name
            if not candidate.is_file() or candidate.stat().st_size == 0:
                return False
        except OSError:
            return False
    return True


def invalidate_cached_model(model_root: str | Path) -> None:
    """Remove a corrupt or incompatible cached model so download can repair it."""
    root = Path(model_root)
    marker = root / MARKER_NAME
    try:
        if marker.is_file():
            marker.unlink()
    except OSError:
        pass
    # Remove revision snapshots; keep the parent so the cache dir itself survives.
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
