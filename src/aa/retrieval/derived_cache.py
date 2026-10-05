"""Encrypted disposable cache for the derived AA retrieval index (issue #58).

This cache is acceleration only. Runtime correctness never depends on it:
a missing, stale, corrupt, or unavailable cache behaves exactly like a
first clean installation (deterministic rebuild from the canonical corpus).

Cached bundle (ready-to-open derived retrieval state):

- ``index.json`` — version metadata, RU chunk records, EN control metadata;
- ``lexical.db`` — SQLite FTS5/BM25 table over RU chunks;
- ``dense.json`` — normalized dense vectors parallel to chunk ids;
- one internal cache manifest (version bindings + file digests).

The bundle is text-bearing, so it never enters GitHub Actions cache as
plaintext: it is packaged deterministically (``tar`` + ``zstd``), encrypted
with the existing age recipient / ``AA_BOOK_AGE_IDENTITY`` contract, and
only the encrypted package plus non-secret cache metadata is cached.

Restored cache is treated as untrusted input until checksum, decryption,
version bindings, and a lightweight self-test all pass. Any failure
deletes the restored data and transparently rebuilds from canonical.

Logs carry only hit/miss/rejected status, rejection reason categories,
durations, key fingerprints, and rebuilt index versions — never corpus
text, user text, secrets, or decrypted cache contents.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import sqlite3
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zstandard as zstd

from aa.corpus.age_v1 import AgeError, decrypt_bytes, encrypt_bytes

DERIVED_CACHE_SCHEMA_VERSION = 1
DERIVED_CACHE_FORMAT = "aa-derived-retrieval-cache/1"
DERIVED_CACHE_KEY_PREFIX = "aa-derived-retrieval-v1"
MANIFEST_NAME = "derived-cache-manifest.json"
BUNDLE_FILES = ("index.json", "lexical.db", "dense.json")
ENCRYPTED_NAME = "derived-retrieval.tar.zst.age"
META_NAME = "derived-retrieval.meta.json"
MAX_DECOMPRESSED_BYTES = 512 * 1024 * 1024

IDENTITY_ENV = "AA_BOOK_AGE_IDENTITY"
ENABLE_ENV = "AA_DERIVED_CACHE_ENABLED"

# Fragments that must never appear in a derived-cache Actions path. The
# staging directory holds only the encrypted package plus non-secret
# metadata; plaintext corpus/index, secrets, and session state are refused.
SENSITIVE_PATH_FRAGMENTS = (
    "corpus/generated/retrieval",
    "corpus\\generated\\retrieval",
    "corpus/generated/canonical",
    "corpus/source/raw",
    "corpus\\source\\raw",
    "corpus/source/encrypted/plaintext",
    "corpus/source/fetch-state",
    "canonical.json",
    "canonical.ru.json",
    "corpus_structure.json",
    "fetch-state",
    "lexical.db",
    "dense.json",
    "index.json",
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


class DerivedCacheError(ValueError):
    """Raised when the derived retrieval cache cannot be used (fails to rebuild)."""


@dataclass(frozen=True)
class CacheKey:
    """Exact content-addressed cache identity plus a log-safe fingerprint."""

    key: str
    fingerprint: str
    digest: str


def is_cache_enabled(env: dict[str, str] | None = None) -> bool:
    """Return False only when explicitly disabled via ``AA_DERIVED_CACHE_ENABLED``."""
    source = os.environ if env is None else env
    raw = str(source.get(ENABLE_ENV, "1")).strip().lower()
    return raw not in {"0", "false", "no", "off", "disabled"}


def repo_root() -> Path:
    """Return the repository root for the checked-in binding files."""
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DerivedCacheError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DerivedCacheError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DerivedCacheError(f"{label} must be a JSON object: {path}")
    return payload


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's bytes."""
    return sha256_bytes(Path(path).read_bytes())


def sha256_canonical_json(payload: dict[str, Any]) -> str:
    """Return the SHA-256 of canonical (sorted-keys) JSON encoding."""
    raw = (json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    return sha256_bytes(raw)


def normalize_os_name(value: str | None = None) -> str:
    """Normalize an OS name to the ``runner.os``-style token."""
    raw = (value or platform.system()).strip().lower()
    if raw in {"linux"}:
        return "Linux"
    if raw in {"darwin", "macos", "mac os x"}:
        return "macOS"
    if raw in {"windows", "windows_nt"}:
        return "Windows"
    return (value or platform.system()).strip() or "Unknown-OS"


def normalize_arch(value: str | None = None) -> str:
    """Normalize a machine arch to the ``runner.arch``-style token."""
    raw = (value or platform.machine()).strip().lower()
    if raw in {"x86_64", "x64", "amd64"}:
        return "X64"
    if raw in {"aarch64", "arm64"}:
        return "ARM64"
    return (value or platform.machine()).strip() or "Unknown-Arch"


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    versions["python"] = f"{sys.version_info.major}.{sys.version_info.minor}"
    versions["python_full"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    try:
        versions["sqlite"] = str(sqlite3.sqlite_version)
    except Exception:  # noqa: BLE001 - version probing never fails the key
        versions["sqlite"] = "unknown"
    try:
        import importlib.metadata as metadata

        try:
            versions["zstandard"] = str(metadata.version("zstandard"))
        except Exception:  # noqa: BLE001 - optional version detail
            versions["zstandard"] = "unknown"
    except Exception:  # noqa: BLE001 - version probing never fails the key
        versions["zstandard"] = "unknown"
    try:
        import faiss  # type: ignore[import-untyped]

        versions["faiss"] = str(getattr(faiss, "__version__", "present"))
    except Exception:  # noqa: BLE001 - faiss is optional
        versions["faiss"] = "absent"
    return versions


def retrieval_abi() -> dict[str, Any]:
    """Return the index/tokenizer/library ABI versions bound into the key."""
    from aa.retrieval.dense import (
        DENSE_TOP_K,
        E5_BACKEND_NAME,
        HASHING_BACKEND_NAME,
        HASHING_DIM,
    )
    from aa.retrieval.fusion import (
        MAX_CANDIDATES_PER_ASPECT,
        MAX_PER_SECTION,
        RRF_K,
    )
    from aa.retrieval.index import INDEX_BUILDER_VERSION, INDEX_FORMAT
    from aa.retrieval.lexical import FTS_TABLE, LEXICAL_TOP_K
    from aa.retrieval.planner import SCHEMA_VERSION as PLANNER_SCHEMA

    return {
        "python": _library_versions()["python"],
        "sqlite": _library_versions()["sqlite"],
        "zstandard": _library_versions()["zstandard"],
        "faiss": _library_versions()["faiss"],
        "index_format": INDEX_FORMAT,
        "index_builder_version": INDEX_BUILDER_VERSION,
        "fts_table": FTS_TABLE,
        "lexical_top_k": LEXICAL_TOP_K,
        "dense_top_k": DENSE_TOP_K,
        "rrf_k": RRF_K,
        "max_per_aspect": MAX_CANDIDATES_PER_ASPECT,
        "max_per_section": MAX_PER_SECTION,
        "hashing_dim": HASHING_DIM,
        "hashing_backend": HASHING_BACKEND_NAME,
        "e5_backend": E5_BACKEND_NAME,
        "planner_schema": PLANNER_SCHEMA,
    }


def default_binding_paths(root: Path | None = None) -> dict[str, Path]:
    """Return the default authoritative binding file paths."""
    base = root if root is not None else repo_root()
    return {
        "ru_manifest": base / "corpus" / "canonical.ru.manifest.json",
        "en_manifest": base / "corpus" / "canonical.manifest.json",
        "structure": base / "corpus" / "structure.json",
        "decision": base / "qualification" / "ru_first_retrieval.v1.decision.json",
        "embedding_lock": base / "corpus" / "embedding.lock.json",
        "recipient": base / "corpus" / "source" / "encrypted" / "recipient.txt",
    }


def read_bindings(
    root: Path | None = None,
    *,
    os_name: str | None = None,
    arch: str | None = None,
) -> dict[str, Any]:
    """Read the live authoritative version bindings for the cache key.

    Bindings include at least: cache schema version, RU/EN canonical
    SHA-256, aligned structure checksum/version, final #47 retrieval
    configuration checksum/version, embedding model id + pinned
    revision/digest, index/tokenizer/library ABI versions, and OS/arch
    where binary compatibility matters. No branch name, run id,
    timestamp, hostname, or other ephemeral identity is bound.
    """
    paths = default_binding_paths(root)
    ru_manifest = _load_json(paths["ru_manifest"], "RU canonical manifest")
    en_manifest = _load_json(paths["en_manifest"], "EN canonical manifest")
    structure_raw = Path(paths["structure"]).read_bytes()
    try:
        structure = json.loads(structure_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DerivedCacheError(f"aligned structure is not valid JSON: {exc}") from exc
    if not isinstance(structure, dict):
        raise DerivedCacheError("aligned structure must be a JSON object")
    decision_raw = Path(paths["decision"]).read_bytes()
    try:
        decision = json.loads(decision_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DerivedCacheError(f"#47 decision artifact is not valid JSON: {exc}") from exc
    if not isinstance(decision, dict):
        raise DerivedCacheError("#47 decision artifact must be a JSON object")
    lock = _load_json(paths["embedding_lock"], "embedding lock")

    ru_sha = str(ru_manifest.get("artifact_sha256", ""))
    en_sha = str(en_manifest.get("artifact_sha256", ""))
    if not ru_sha or not en_sha:
        raise DerivedCacheError("canonical manifests must carry artifact_sha256")
    model_id = str(lock.get("model_id", ""))
    revision = str(lock.get("revision", ""))
    if not model_id or not revision:
        raise DerivedCacheError("embedding lock must carry model_id + revision")

    production = decision.get("production")
    if not isinstance(production, dict):
        raise DerivedCacheError("#47 decision artifact must carry a production configuration")
    index_config = decision.get("index_config")
    if not isinstance(index_config, dict):
        raise DerivedCacheError("#47 decision artifact must carry index_config")

    normalized_os = normalize_os_name(os_name)
    normalized_arch = normalize_arch(arch)
    return {
        "cache_schema_version": DERIVED_CACHE_SCHEMA_VERSION,
        "cache_format": DERIVED_CACHE_FORMAT,
        "ru_artifact_sha256": ru_sha,
        "ru_manifest_format": str(ru_manifest.get("format", "")),
        "en_artifact_sha256": en_sha,
        "en_manifest_format": str(en_manifest.get("format", "")),
        "structure_sha256": sha256_bytes(structure_raw),
        "structure_format": str(structure.get("format", "")),
        "structure_builder_version": structure.get("builder_version"),
        "retrieval_config_sha256": sha256_bytes(decision_raw),
        "retrieval_benchmark_version": str(decision.get("benchmark_version", "")),
        "retrieval_decision_format": str(decision.get("format", "")),
        "retrieval_production_config_id": str(production.get("config_id", "")),
        "retrieval_production_config_version": str(production.get("config_version", "")),
        "retrieval_planner_schema": str(decision.get("planner_schema", "")),
        "retrieval_gold_sha256": str(decision.get("gold_sha256", "")),
        "retrieval_index_config_sha256": sha256_canonical_json(
            {str(key): index_config[key] for key in sorted(index_config)}
        ),
        "embedding_model_id": model_id,
        "embedding_revision": revision,
        "embedding_lock_sha256": sha256_canonical_json(
            {str(key): lock[key] for key in sorted(lock)}
        ),
        "abi": retrieval_abi(),
        "os": normalized_os,
        "arch": normalized_arch,
    }


def derived_cache_key(
    bindings: dict[str, Any],
    *,
    os_name: str | None = None,
    arch: str | None = None,
) -> CacheKey:
    """Build the exact content-addressed Actions cache key from bindings."""
    canonical = {str(key): bindings[key] for key in sorted(bindings)}
    # Explicit overrides stay content-addressed (OS/arch), never ephemeral.
    if os_name is not None:
        canonical["os"] = normalize_os_name(os_name)
    if arch is not None:
        canonical["arch"] = normalize_arch(arch)
    for required in (
        "cache_schema_version",
        "ru_artifact_sha256",
        "en_artifact_sha256",
        "structure_sha256",
        "retrieval_config_sha256",
        "embedding_model_id",
        "embedding_revision",
        "os",
        "arch",
    ):
        value = canonical.get(required)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise DerivedCacheError(f"cache bindings are missing {required!r}")
    digest = sha256_canonical_json(canonical)
    key = f"{DERIVED_CACHE_KEY_PREFIX}-{canonical['os']}-{canonical['arch']}-{digest}"
    return CacheKey(key=key, fingerprint=digest[:16], digest=digest)


def key_fingerprint(key: str) -> str:
    """Return the log-safe fingerprint suffix of a cache key (digest tail)."""
    tail = key.rsplit("-", 1)[-1] if "-" in key else key
    return tail[:16]


def assert_cache_paths_safe(paths: list[str] | tuple[str, ...]) -> None:
    """Fail closed when an Actions cache path could carry sensitive data."""
    for path in paths:
        lowered = str(path).replace("\\", "/").lower()
        for fragment in SENSITIVE_PATH_FRAGMENTS:
            if fragment.lower() in lowered:
                raise DerivedCacheError(f"refusing sensitive derived-cache path: {path!r}")


def bundle_file_digests(retrieval_dir: str | Path) -> dict[str, str]:
    """Return SHA-256 digests for every expected bundle file."""
    directory = Path(retrieval_dir)
    digests: dict[str, str] = {}
    for name in BUNDLE_FILES:
        candidate = directory / name
        if not candidate.is_file():
            raise DerivedCacheError(f"retrieval bundle is missing {name}: {candidate}")
        digests[name] = sha256_file(candidate)
    return digests


def build_cache_manifest(
    *,
    cache_key: CacheKey,
    bindings: dict[str, Any],
    file_digests: dict[str, str],
    chunk_count: int,
    index_version: int,
) -> dict[str, Any]:
    """Build the deterministic internal cache manifest (no timestamps)."""
    if chunk_count <= 0:
        raise DerivedCacheError("refusing to cache an empty retrieval index")
    for name in BUNDLE_FILES:
        if name not in file_digests:
            raise DerivedCacheError(f"cache manifest is missing digest for {name!r}")
    return {
        "format": DERIVED_CACHE_FORMAT,
        "cache_schema_version": DERIVED_CACHE_SCHEMA_VERSION,
        "cache_key": cache_key.key,
        "cache_digest": cache_key.digest,
        "bindings": {str(key): bindings[key] for key in sorted(bindings)},
        "files": {str(key): file_digests[str(key)] for key in sorted(file_digests)},
        "chunk_count": chunk_count,
        "index_version": index_version,
    }


def create_package(retrieval_dir: str | Path, manifest: dict[str, Any]) -> bytes:
    """Package the retrieval bundle + manifest deterministically (tar.zst)."""
    directory = Path(retrieval_dir)
    manifest_bytes = (json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    members: list[tuple[str, bytes]] = []
    for name in sorted([*BUNDLE_FILES, MANIFEST_NAME]):
        if name == MANIFEST_NAME:
            members.append((name, manifest_bytes))
            continue
        members.append((name, (directory / name).read_bytes()))
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, payload in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            info.pax_headers = {}
            tar.addfile(info, io.BytesIO(payload))
    compressor = zstd.ZstdCompressor(level=3, threads=1)
    return compressor.compress(tar_buffer.getvalue())


def extract_package(data: bytes) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Unpack tar.zst bytes into ``({name: bytes}, manifest)``."""
    try:
        decompressed = zstd.ZstdDecompressor().decompress(
            data, max_output_size=MAX_DECOMPRESSED_BYTES
        )
    except zstd.ZstdError as exc:
        raise DerivedCacheError(f"derived cache decompression failed: {exc}") from exc
    buffer = io.BytesIO(decompressed)
    try:
        with tarfile.open(fileobj=buffer, mode="r") as tar:
            raw_members = tar.getmembers()
            names = [member.name for member in raw_members]
            expected = sorted([*BUNDLE_FILES, MANIFEST_NAME])
            if sorted(names) != expected:
                raise DerivedCacheError(f"unexpected derived cache members: {sorted(names)}")
            members = {member.name: member for member in raw_members}
            if any(not member.isreg() for member in members.values()):
                raise DerivedCacheError("derived cache members must be regular files")
            files: dict[str, bytes] = {}
            for name in BUNDLE_FILES:
                handle = tar.extractfile(members[name])
                if handle is None:
                    raise DerivedCacheError(f"derived cache is missing {name!r}")
                files[name] = handle.read()
            manifest_handle = tar.extractfile(members[MANIFEST_NAME])
            if manifest_handle is None:
                raise DerivedCacheError("derived cache is missing its manifest")
            manifest = json.loads(manifest_handle.read().decode("utf-8"))
    except (tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DerivedCacheError(f"derived cache archive is malformed: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DerivedCacheError("derived cache manifest is malformed")
    return files, manifest


def encrypt_package(package: bytes, recipient: str) -> bytes:
    """Encrypt a packaged bundle to the age recipient (fails closed)."""
    try:
        return encrypt_bytes(package, [recipient.strip()])
    except AgeError as exc:
        raise DerivedCacheError(f"derived cache encryption failed: {exc}") from exc


def decrypt_package(encrypted: bytes, identity: str) -> bytes:
    """Decrypt an encrypted package with ``AA_BOOK_AGE_IDENTITY`` (fails closed)."""
    try:
        return decrypt_bytes(encrypted, [identity.strip()])
    except AgeError as exc:
        raise DerivedCacheError(f"derived cache decryption failed: {exc}") from exc


def verify_manifest(
    manifest: dict[str, Any], *, live_bindings: dict[str, Any], expected_key: str
) -> None:
    """Verify the internal manifest against the live bindings (fails closed)."""
    if manifest.get("format") != DERIVED_CACHE_FORMAT:
        raise DerivedCacheError("derived cache format is unsupported")
    if manifest.get("cache_schema_version") != DERIVED_CACHE_SCHEMA_VERSION:
        raise DerivedCacheError("derived cache schema version is stale")
    if manifest.get("cache_key") != expected_key:
        raise DerivedCacheError("derived cache key does not match the live key")
    stored = manifest.get("bindings")
    if not isinstance(stored, dict):
        raise DerivedCacheError("derived cache manifest has no bindings")
    for key in sorted(live_bindings):
        if stored.get(key) != live_bindings.get(key):
            raise DerivedCacheError(f"derived cache binding {key!r} is stale")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise DerivedCacheError("derived cache manifest has no file digests")
    for name in BUNDLE_FILES:
        digest = files.get(name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise DerivedCacheError(f"derived cache file digest for {name!r} is malformed")


def verify_extracted_files(files: dict[str, bytes], manifest: dict[str, Any]) -> None:
    """Verify extracted file bytes against the internal manifest digests."""
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise DerivedCacheError("derived cache manifest has no file digests")
    for name in BUNDLE_FILES:
        payload = files.get(name)
        if payload is None:
            raise DerivedCacheError(f"derived cache is missing {name!r}")
        if sha256_bytes(payload) != expected.get(name):
            raise DerivedCacheError(f"derived cache file {name!r} failed checksum")


def write_extracted_bundle(
    files: dict[str, bytes], retrieval_dir: str | Path, manifest: dict[str, Any]
) -> None:
    """Write verified bundle bytes into the ignored ephemeral workspace."""
    directory = Path(retrieval_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE_FILES:
        target = directory / name
        tmp_path = target.with_name(target.name + ".tmp")
        tmp_path.write_bytes(files[name])
        if tmp_path.read_bytes() != files[name]:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise DerivedCacheError(f"verified bundle write failed for {name!r}")
        os.replace(tmp_path, target)
    # The internal manifest is ephemeral verification state, not part of the
    # ready-to-open index layout; keep it out of the retrieval directory so
    # ``open_hybrid_index`` sees exactly the built layout.
    _ = manifest


def self_test_index(retrieval_dir: str | Path) -> dict[str, Any]:
    """Open RAM-resident state and run a lightweight self-test (fails closed)."""
    from aa.retrieval.index import INDEX_BUILDER_VERSION, close_hybrid_index, open_hybrid_index
    from aa.retrieval.lexical import lexical_search_conn

    directory = Path(retrieval_dir)
    index = open_hybrid_index(directory)
    if index.chunk_count <= 0:
        raise DerivedCacheError("derived cache self-test found no chunks")
    # Exact-IP round-trip over a stored vector (Faiss or pure-Python path).
    stored_id = index.dense.ids[0]
    stored_vector = index.dense.vectors[0]
    top = index.dense.search(stored_vector, top_k=1)
    if not top or top[0][0] != stored_id:
        raise DerivedCacheError("derived cache dense self-test failed")
    # Lexical round-trip via the RAM-resident FTS5 connection (no disk I/O).
    lexical_ok = False
    if index.lexical_conn is None:
        raise DerivedCacheError("derived cache index is not RAM-resident")
    try:
        for record in list(index.chunks.values())[:5]:
            for token in record.text.split():
                cleaned = "".join(ch for ch in token if ch.isalnum()).strip()
                if len(cleaned) < 4:
                    continue
                try:
                    hits = lexical_search_conn(index.lexical_conn, cleaned, top_k=3)
                except ValueError as exc:
                    raise DerivedCacheError(
                        f"derived cache lexical self-test failed: {exc}"
                    ) from exc
                if hits:
                    lexical_ok = True
                    break
            if lexical_ok:
                break
        if not lexical_ok:
            raise DerivedCacheError("derived cache lexical self-test found no indexed token")
        return {
            "chunk_count": index.chunk_count,
            "index_version": int(index.metadata.get("builder_version", INDEX_BUILDER_VERSION)),
        }
    finally:
        close_hybrid_index(index)


def clear_directory(directory: str | Path) -> None:
    """Delete restored data after any verification failure (rebuild follows)."""
    target = Path(directory)
    if not target.is_dir():
        return
    for child in sorted(target.iterdir()):
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


def build_meta_payload(
    *,
    cache_key: CacheKey,
    encrypted_sha256: str,
    encrypted_bytes: int,
    bindings: dict[str, Any],
) -> dict[str, Any]:
    """Build the non-secret Actions cache metadata (no text, no secrets)."""
    return {
        "format": "aa-derived-retrieval-meta/1",
        "cache_key": cache_key.key,
        "cache_fingerprint": cache_key.fingerprint,
        "cache_digest": cache_key.digest,
        "encrypted_sha256": encrypted_sha256,
        "encrypted_bytes": encrypted_bytes,
        "encrypted_file": ENCRYPTED_NAME,
        "bindings_digest": sha256_canonical_json(
            {str(key): bindings[key] for key in sorted(bindings)}
        ),
        "retrieval_config_sha256": str(bindings.get("retrieval_config_sha256", "")),
        "structure_sha256": str(bindings.get("structure_sha256", "")),
        "ru_artifact_sha256": str(bindings.get("ru_artifact_sha256", "")),
        "en_artifact_sha256": str(bindings.get("en_artifact_sha256", "")),
        "embedding_model_id": str(bindings.get("embedding_model_id", "")),
        "embedding_revision": str(bindings.get("embedding_revision", "")),
    }
