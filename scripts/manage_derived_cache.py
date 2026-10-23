#!/usr/bin/env python3
"""Manage the encrypted disposable derived retrieval cache (issue #58).

Acceleration only: runtime correctness never depends on this cache. Every
mode exits 0 when a deterministic rebuild is possible, so a missing,
stale, corrupt, or unavailable cache behaves exactly like a first clean
installation.

Modes:

- ``--print-key`` — compute the exact content-addressed cache key from the
  authoritative manifests/config and print it (no secrets, no text);
- ``--print-l2-plan`` — print the key, the L1 Actions key (with the
  rotation suffix so an unrecoverable corrupt key can be rotated), and the
  planned immutable L2 OCI tag/pin for issue #429;
- ``--restore`` — exact cache lookup only: verify the encrypted-package
  checksum, decrypt into the ignored ephemeral workspace, verify the
  internal manifest and every version binding, open SQLite/Faiss and run a
  lightweight self-test. On any failure the restored data is deleted and
  the status reports ``miss``/``rejected`` with a reason category;
- ``--restore-or-rebuild`` — try ``--restore``; on miss/rejected, restore
  and validate the authoritative RU/EN canonical artifacts, build the
  final qualified retrieval state, and run integrity/self-tests;
- ``--restore-l2`` — pull the digest-pinned encrypted bundle from the
  persistent GHCR OCI artifact (L2, issue #429) into an L2-owned staging
  directory and validate it exactly like L1; a missing/placeholder/invalid
  pin, registry failure, or digest mismatch is reported fail-closed and
  never trusted. Issue #461: download, decryption, verification and any
  staging cleanup touch only the L2-owned staging directory — the
  original L1 cache directory is never mutated, overwritten or deleted
  by an L2 failure;
- ``--publish-l2`` — package/encrypt the final production E5 index and
  publish it as one immutable GHCR OCI artifact. Publishing is
  idempotent: identical bindings converge to the same content-addressed
  tag and manifest digest, so parallel publishers share one digest and no
  run id/date/source SHA enters the identity;
- ``--save`` — package the ready-built index deterministically, encrypt
  with the existing age recipient, and write only the encrypted package
  plus non-secret cache metadata into ``--cache-dir``. Save failure is
  non-fatal (runtime still starts).

Only the cache mode logs are privacy-safe: hit/miss/rejected status,
rejection reason category, restore/decrypt/build durations, cache key
fingerprint, and rebuilt index version. Never corpus text, user text,
secrets, or decrypted contents.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.retrieval.derived_cache import (  # noqa: E402
    DERIVED_CACHE_FORMAT,
    ENCRYPTED_NAME,
    META_NAME,
    DerivedCacheError,
    assert_cache_paths_safe,
    assert_required_backend,
    build_cache_manifest,
    build_meta_payload,
    bundle_file_digests,
    clear_directory,
    create_package,
    decrypt_package,
    derived_cache_key,
    encrypt_package,
    extract_package,
    is_cache_enabled,
    normalize_arch,
    normalize_os_name,
    read_bindings,
    self_test_index,
    sha256_bytes,
    verify_extracted_files,
    verify_manifest,
)
from aa.retrieval.oci_artifact import (  # noqa: E402
    ArtifactError,
    OCIRegistry,
    expected_l2_tag,
    open_session,
    parse_pinned_digest,
    pull_artifact,
    push_artifact,
    read_pinned_reference,
)

DEFAULT_RETRIEVAL_DIR = ROOT / "corpus" / "generated" / "retrieval"
DEFAULT_CACHE_DIR = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "aa-derived-cache"
DEFAULT_DIGEST_FILE = ROOT / "docker" / "aa-index.digest"
DEFAULT_L2_REGISTRY = "ghcr.io"
DEFAULT_L2_REPOSITORY = "kodmial/aa-index"
IDENTITY_ENV = "AA_BOOK_AGE_IDENTITY"
TOKEN_ENV = "AA_INDEX_PACKAGES_TOKEN"
ROTATION_ENV = "AA_DERIVED_CACHE_ROTATION"


def _fail(message: str) -> int:
    print(f"derived cache failed: {message}", file=sys.stderr)
    return 1


def _status(payload: dict[str, Any], args: argparse.Namespace | None = None) -> int:
    text = json.dumps(payload, sort_keys=True)
    if args is not None and getattr(args, "status_file", None):
        try:
            Path(args.status_file).write_text(text + "\n", encoding="utf-8")
        except OSError:
            # Metrics/status recording is never allowed to break the
            # bootstrap itself; the printed JSON stays authoritative.
            pass
    print(text)
    return 0


def _l1_actions_key(key: Any, rotation: str) -> str:
    """Return the L1 Actions cache key with the rotation suffix.

    The encrypted bundle validates itself against the immutable base key,
    so the suffix only namespaces evictable L1 storage and can be bumped
    through ``AA_DERIVED_CACHE_ROTATION`` when a cache key cannot be
    repaired in place.
    """
    suffix = str(rotation).strip() or "0"
    if not suffix.isdigit():
        suffix = "0"
    return f"{key.key}-rot-{suffix}"


def _live_key(
    *,
    root: Path,
    os_name: str | None,
    arch: str | None,
) -> tuple[dict[str, Any], Any]:
    bindings = read_bindings(
        root,
        os_name=normalize_os_name(os_name) if os_name else None,
        arch=normalize_arch(arch) if arch else None,
    )
    key = derived_cache_key(bindings)
    return bindings, key


def _read_recipient(recipient_file: Path | None) -> str | None:
    candidates: list[Path] = []
    if recipient_file is not None:
        candidates.append(recipient_file)
    candidates.append(ROOT / "corpus" / "source" / "encrypted" / "recipient.txt")
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if text:
            return text
    return None


def _read_identity(identity_file: Path | None) -> str | None:
    if identity_file is not None:
        try:
            text = identity_file.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return None
        return text or None
    raw = os.environ.get(IDENTITY_ENV, "").strip()
    return raw or None


def _cmd_print_key(args: argparse.Namespace) -> int:
    try:
        bindings, key = _live_key(root=ROOT, os_name=args.os, arch=args.arch)
    except DerivedCacheError as exc:
        return _fail(str(exc))
    rotation = _l1_rotation(args)
    return _status(
        {
            "cache_key": key.key,
            "cache_fingerprint": key.fingerprint,
            "cache_digest": key.digest,
            "actions_key": _l1_actions_key(key, rotation),
            "l1_rotation": rotation,
            "os": str(bindings.get("os", "")),
            "arch": str(bindings.get("arch", "")),
        },
        args,
    )


def _l1_rotation(_args: argparse.Namespace | None = None) -> str:
    """Read the L1 namespace rotation counter from the environment."""
    raw = os.environ.get(ROTATION_ENV, "").strip()
    return raw if raw and raw.isdigit() else "0"


def _cmd_print_l2_plan(args: argparse.Namespace) -> int:
    try:
        bindings, key = _live_key(root=ROOT, os_name=args.os, arch=args.arch)
    except DerivedCacheError as exc:
        return _fail(str(exc))
    rotation = _l1_rotation(args)
    digest_file = args.digest_file
    pin_reference = _read_digest_reference_text(digest_file)
    pinned: str | None = None
    pin_problem = ""
    try:
        pinned = _l2_pinned_digest(digest_file)
    except ArtifactError as exc:
        # An unpublished/corrupt pin is a normal transition state reported
        # honestly, never a permission to relax the strict backend.
        pin_problem = exc.category
        pinned = None
    return _status(
        {
            "cache_key": key.key,
            "cache_fingerprint": key.fingerprint,
            "cache_digest": key.digest,
            "actions_key": _l1_actions_key(key, rotation),
            "l1_rotation": rotation,
            "l2_registry": args.oci_registry,
            "l2_repository": args.oci_repository,
            "l2_tag": expected_l2_tag(key.digest),
            "l2_pinned_digest": pinned,
            "l2_pin_reference": pin_reference,
            "l2_pin_problem": pin_problem,
            "digest_file": str(digest_file),
        },
        args,
    )


def _read_digest_reference_text(digest_file: str | Path) -> str:
    return read_pinned_reference(digest_file)


def _l2_pinned_digest(digest_file: str | Path) -> str | None:
    """Return the pinned immutable digest, or None while unpublished."""
    return parse_pinned_digest(str(digest_file))


def _split_digest_reference(reference: str) -> tuple[str, str, str]:
    """Split ``registry/repository@sha256:...`` into its parts (fail closed)."""
    registry_repo, sep, digest = reference.partition("@")
    if not sep or "/" not in registry_repo:
        raise ArtifactError("l2-pin-mismatch", "digest pin is not a full image reference")
    registry, _, repository = registry_repo.partition("/")
    from aa.retrieval.oci_artifact import DIGEST_RE, ZERO_DIGEST

    if DIGEST_RE.fullmatch(digest) is None or digest == ZERO_DIGEST:
        raise ArtifactError("l2-pin-mismatch", "digest pin carries no valid sha256 digest")
    return registry, repository, digest


def _attempt_restore(
    *,
    cache_dir: Path,
    retrieval_dir: Path,
    bindings: dict[str, Any],
    key: Any,
    identity: str | None,
    require_backend: str = "none",
) -> dict[str, Any]:
    started = time.perf_counter()
    encrypted_path = cache_dir / ENCRYPTED_NAME
    meta_path = cache_dir / META_NAME
    if not encrypted_path.is_file() or not meta_path.is_file():
        return {
            "status": "miss",
            "reason": "cache-miss",
            "cache_fingerprint": key.fingerprint,
            "restore_ms": (time.perf_counter() - started) * 1000.0,
        }
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            "status": "rejected",
            "reason": "corrupt-metadata",
            "cache_fingerprint": key.fingerprint,
            "restore_ms": (time.perf_counter() - started) * 1000.0,
        }
    if not isinstance(meta, dict) or meta.get("cache_key") != key.key:
        return {
            "status": "rejected",
            "reason": "stale-version",
            "cache_fingerprint": key.fingerprint,
            "restore_ms": (time.perf_counter() - started) * 1000.0,
        }
    try:
        encrypted = encrypted_path.read_bytes()
    except OSError:
        return {
            "status": "miss",
            "reason": "backend-unavailable",
            "cache_fingerprint": key.fingerprint,
            "restore_ms": (time.perf_counter() - started) * 1000.0,
        }
    expected_sha = meta.get("encrypted_sha256")
    if not isinstance(expected_sha, str) or sha256_bytes(encrypted) != expected_sha:
        return {
            "status": "rejected",
            "reason": "corrupt-package",
            "cache_fingerprint": key.fingerprint,
            "restore_ms": (time.perf_counter() - started) * 1000.0,
        }
    if not identity:
        return {
            "status": "miss",
            "reason": "backend-unavailable",
            "cache_fingerprint": key.fingerprint,
            "restore_ms": (time.perf_counter() - started) * 1000.0,
        }
    decrypt_started = time.perf_counter()
    try:
        package = decrypt_package(encrypted, identity)
    except DerivedCacheError:
        return {
            "status": "rejected",
            "reason": "decrypt-failed",
            "cache_fingerprint": key.fingerprint,
            "restore_ms": (time.perf_counter() - started) * 1000.0,
            "decrypt_ms": (time.perf_counter() - decrypt_started) * 1000.0,
        }
    decrypt_ms = (time.perf_counter() - decrypt_started) * 1000.0
    try:
        files, manifest = extract_package(package)
        verify_manifest(manifest, live_bindings=bindings, expected_key=key.key)
        verify_extracted_files(files, manifest)
        # Issue #429: a restored bundle must record the strictly required
        # production dense backend (e5 + E5 dimensions) inside its own
        # index metadata; a hashing bundle is never a production hit.
        assert_required_backend(files, require_backend=require_backend)
    except DerivedCacheError as exc:
        message = str(exc)
        if "wrong-backend" in message or "backend mismatch" in message:
            reason = "wrong-backend"
        elif "stale" in message:
            reason = "stale-version"
        else:
            reason = "corrupt-package"
        return {
            "status": "rejected",
            "reason": reason,
            "cache_fingerprint": key.fingerprint,
            "restore_ms": (time.perf_counter() - started) * 1000.0,
            "decrypt_ms": decrypt_ms,
        }
    # Write verified bytes into the ignored ephemeral workspace, then prove
    # the ready-to-open state before reporting a hit.
    from aa.retrieval.derived_cache import write_extracted_bundle

    try:
        write_extracted_bundle(files, retrieval_dir, manifest)
        checked = self_test_index(retrieval_dir)
    except DerivedCacheError as exc:
        clear_directory(retrieval_dir)
        message = str(exc)
        if "stale" in message:
            reason = "stale-version"
        elif "self-test" in message:
            reason = "self-test-failed"
        else:
            reason = "corrupt-package"
        return {
            "status": "rejected",
            "reason": reason,
            "cache_fingerprint": key.fingerprint,
            "restore_ms": (time.perf_counter() - started) * 1000.0,
            "decrypt_ms": decrypt_ms,
        }
    total_ms = (time.perf_counter() - started) * 1000.0
    return {
        "status": "hit",
        "cache_fingerprint": key.fingerprint,
        "restore_ms": total_ms,
        "decrypt_ms": decrypt_ms,
        "chunk_count": checked["chunk_count"],
        "index_version": checked["index_version"],
    }


def _rebuild_index(*, retrieval_dir: Path, backend: str) -> dict[str, Any]:
    started = time.perf_counter()
    from aa.retrieval.index import HybridIndexError, build_hybrid_index

    full_path = ROOT / "corpus" / "generated" / "corpus_structure.json"
    ru_manifest_path = ROOT / "corpus" / "canonical.ru.manifest.json"
    en_manifest_path = ROOT / "corpus" / "canonical.manifest.json"
    lock_path = ROOT / "corpus" / "embedding.lock.json"
    try:
        full = json.loads(full_path.read_text(encoding="utf-8"))
        ru_manifest = json.loads(ru_manifest_path.read_text(encoding="utf-8"))
        en_manifest = json.loads(en_manifest_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DerivedCacheError(f"cannot load qualified retrieval inputs: {exc}") from exc
    if not isinstance(full, dict) or not isinstance(ru_manifest, dict):
        raise DerivedCacheError("qualified retrieval inputs are malformed")
    clear_directory(retrieval_dir)
    try:
        index = build_hybrid_index(
            full,
            ru_manifest=ru_manifest,
            en_manifest=en_manifest,
            embedding_lock=lock,
            out_dir=retrieval_dir,
            backend=backend,
        )
    except HybridIndexError as exc:
        clear_directory(retrieval_dir)
        raise DerivedCacheError(f"production index rebuild failed: {exc}") from exc
    checked = self_test_index(retrieval_dir)
    return {
        "chunk_count": index.chunk_count,
        "index_version": checked["index_version"],
        "build_ms": (time.perf_counter() - started) * 1000.0,
    }


def _cmd_restore(args: argparse.Namespace) -> int:
    if not is_cache_enabled():
        return _status({"status": "disabled"}, args)
    try:
        bindings, key = _live_key(root=ROOT, os_name=args.os, arch=args.arch)
    except DerivedCacheError as exc:
        return _fail(str(exc))
    identity = _read_identity(args.identity_file)
    result = _attempt_restore(
        cache_dir=args.cache_dir,
        retrieval_dir=args.retrieval_dir,
        bindings=bindings,
        key=key,
        identity=identity,
        require_backend=args.require_backend,
    )
    return _status(result, args)


def _cmd_restore_or_rebuild(args: argparse.Namespace) -> int:
    if not is_cache_enabled():
        try:
            rebuilt = _rebuild_index(retrieval_dir=args.retrieval_dir, backend=args.backend)
        except DerivedCacheError as exc:
            return _fail(str(exc))
        return _status(
            {
                "status": "rebuilt",
                "reason": "cache-disabled",
                "backend": args.backend,
                "build_ms": rebuilt["build_ms"],
                "chunk_count": rebuilt["chunk_count"],
                "index_version": rebuilt["index_version"],
            },
            args,
        )
    try:
        bindings, key = _live_key(root=ROOT, os_name=args.os, arch=args.arch)
    except DerivedCacheError as exc:
        return _fail(str(exc))
    identity = _read_identity(args.identity_file)
    result = _attempt_restore(
        cache_dir=args.cache_dir,
        retrieval_dir=args.retrieval_dir,
        bindings=bindings,
        key=key,
        identity=identity,
        require_backend=args.require_backend,
    )
    if result.get("status") == "hit":
        return _status(dict(result, source="restore"), args)
    # Miss, stale, corrupt, wrong-backend, or unavailable: transparent
    # deterministic bounded rebuild from the canonical corpus. The
    # rejection reason is preserved. With ``--require-backend e5`` the
    # rebuild itself never selects the hashing backend.
    try:
        rebuilt = _rebuild_index(retrieval_dir=args.retrieval_dir, backend=args.backend)
    except DerivedCacheError as exc:
        return _fail(str(exc))
    return _status(
        {
            "status": "rebuilt",
            "source": "rebuild",
            "backend": args.backend,
            "reason": str(result.get("reason", "cache-miss")),
            "cache_fingerprint": key.fingerprint,
            "restore_ms": result.get("restore_ms", 0.0),
            "build_ms": rebuilt["build_ms"],
            "chunk_count": rebuilt["chunk_count"],
            "index_version": rebuilt["index_version"],
        },
        args,
    )


def _l2_session(args: argparse.Namespace) -> Any:
    token = ""
    token_env = args.token_env or TOKEN_ENV
    raw = os.environ.get(token_env, "")
    if raw:
        token = raw.strip()
    username = "x-access-token" if token else None
    password = token or None
    registry = OCIRegistry(
        host=args.oci_registry,
        repository=args.oci_repository,
        scheme=getattr(args, "oci_scheme", "https"),
    )
    return registry, open_session(registry, username=username, password=password)


def _l2_staging_dir(args: argparse.Namespace) -> Path:
    """Return the L2-owned staging directory for the pinned L2 pull.

    Issue #461: the staging directory is never the L1 cache directory
    itself. Download, decryption, verification, and any staging cleanup
    touch only this L2-owned path, so an L2 401/403, transport failure,
    partial blob, corrupt artifact, bad manifest/backend, decryption or
    self-test failure can never mutate, overwrite, or delete the restored
    L1 directory that consistently shares its path in production.
    """
    staging = getattr(args, "staging_dir", None)
    if staging is not None:
        return staging
    return args.cache_dir.parent / f"{args.cache_dir.name}.l2-staging"


def _cmd_restore_l2(args: argparse.Namespace) -> int:
    if not is_cache_enabled():
        return _status({"status": "disabled"}, args)
    try:
        bindings, key = _live_key(root=ROOT, os_name=args.os, arch=args.arch)
    except DerivedCacheError as exc:
        return _fail(str(exc))
    pin_reference = _read_digest_reference_text(args.digest_file)
    try:
        # The committed pin must already be an immutable sha256 digest;
        # placeholders cannot even authorize the pull attempt.
        pinned_digest = _l2_pinned_digest(args.digest_file)
    except ArtifactError as exc:
        return _status(
            {
                "status": "rejected",
                "reason": exc.category,
                "source": "l2",
                "pin_reference": pin_reference,
                "cache_fingerprint": key.fingerprint,
            },
            args,
        )
    if not pin_reference:
        return _status(
            {
                "status": "rejected",
                "reason": "l2-unpublished",
                "source": "l2",
                "cache_fingerprint": key.fingerprint,
            },
            args,
        )
    try:
        pin_registry, pin_repository, pinned_digest = _split_digest_reference(pin_reference)
    except ArtifactError:
        return _status(
            {
                "status": "rejected",
                "reason": "l2-pin-mismatch",
                "source": "l2",
                "cache_fingerprint": key.fingerprint,
            },
            args,
        )
    if pin_registry != args.oci_registry or pin_repository != args.oci_repository:
        return _status(
            {
                "status": "rejected",
                "reason": "l2-pin-mismatch",
                "source": "l2",
                "l2_pinned_registry": pin_registry,
                "l2_pinned_repository": pin_repository,
                "cache_fingerprint": key.fingerprint,
            },
            args,
        )
    staging_dir = _l2_staging_dir(args)
    registry, session = _l2_session(args)
    # Issue #461 staging isolation: only the L2-owned staging directory is
    # ever written or cleared by this command. The preexisting L1 cache
    # directory stays byte-for-byte identical across every L2 failure mode
    # (transport, 401/partial blob, digest mismatch, decrypt/validation).
    clear_directory(staging_dir)
    try:
        pull_artifact(
            registry,
            reference=pinned_digest,
            session=session,
            dest_dir=staging_dir,
            expect_files=(ENCRYPTED_NAME, META_NAME),
        )
    except ArtifactError as exc:
        # Fail closed on the staging directory only: partial/untrusted pull
        # content is removed and never used; the caller decides on the
        # L1/validation and bounded emergency rebuild policy.
        clear_directory(staging_dir)
        return _status(
            {
                "status": "rejected",
                "reason": exc.category,
                "source": "l2",
                "l2_digest": pinned_digest,
                "cache_fingerprint": key.fingerprint,
            },
            args,
        )
    identity = _read_identity(args.identity_file)
    result = _attempt_restore(
        cache_dir=staging_dir,
        retrieval_dir=args.retrieval_dir,
        bindings=bindings,
        key=key,
        identity=identity,
        require_backend=args.require_backend,
    )
    payload: dict[str, Any] = dict(result)
    payload["source"] = "l2"
    payload["l2_digest"] = pinned_digest
    # Verified-hit staging is cleaned too: the bootstrap caller saves the
    # validated state into the L1 cache directory through the existing
    # save path, never by mutating it from this L2 command.
    clear_directory(staging_dir)
    return _status(payload, args)


def _l2_annotations(*, key: Any, chunk_count: int, index_version: int) -> dict[str, str]:
    # Identity comes only from genuinely index-influencing inputs; no run
    # id, timestamp, date, branch, or ordinary source SHA may enter here.
    return {
        "aa.derived-index.cache-key": key.key,
        "aa.derived-index.cache-fingerprint": key.fingerprint,
        "aa.derived-index.cache-digest": key.digest,
        "aa.derived-index.bundle-format": DERIVED_CACHE_FORMAT,
        "aa.derived-index.chunk-count": str(int(chunk_count)),
        "aa.derived-index.index-version": str(int(index_version)),
    }


def _publish_artifact(
    *,
    cache_dir: Path,
    key: Any,
    chunk_count: int,
    index_version: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    registry, session = _l2_session(args)
    encrypted_bytes = (cache_dir / ENCRYPTED_NAME).read_bytes()
    meta_bytes = (cache_dir / META_NAME).read_bytes()
    files = {ENCRYPTED_NAME: encrypted_bytes, META_NAME: meta_bytes}
    tag = expected_l2_tag(key.digest)
    last_error: ArtifactError | None = None
    for attempt in range(1, 4):
        try:
            digest = push_artifact(
                registry,
                files=files,
                tag=tag,
                session=session,
                annotations=_l2_annotations(
                    key=key, chunk_count=chunk_count, index_version=index_version
                ),
            )
            del last_error
            return {
                "status": "published",
                "l2_digest": digest,
                "l2_tag": tag,
                "l2_registry": registry.host,
                "l2_repository": registry.repository,
                "published": True,
            }
        except ArtifactError as exc:
            last_error = exc
            if exc.category not in ("l2-registry-unavailable",):
                break
            time.sleep(attempt * 2.0)
    detail = last_error.category if last_error is not None else "unknown"
    raise ArtifactError("l2-push-failed", f"publish exhausted retries: {detail}")


def _cmd_publish_l2(args: argparse.Namespace) -> int:
    """Publish the final production E5 index as the persistent L2 artifact.

    Trusted-publisher entrypoint (issue #429): the retrieval directory must
    already hold (or be deterministically rebuildable into) the strict
    production E5 backend; publishing a hashing index is refused outright.
    Failure is fatal in the trusted publisher job.
    """
    if args.require_backend != "e5":
        return _fail("publish-l2 publishes only the strict production E5 backend")
    try:
        bindings, key = _live_key(root=ROOT, os_name=args.os, arch=args.arch)
    except DerivedCacheError as exc:
        return _fail(str(exc))
    identity = _read_identity(args.identity_file)
    result = _attempt_restore(
        cache_dir=args.cache_dir,
        retrieval_dir=args.retrieval_dir,
        bindings=bindings,
        key=key,
        identity=identity,
        require_backend=args.require_backend,
    )
    source = "restore"
    if result.get("status") != "hit":
        try:
            rebuilt = _rebuild_index(retrieval_dir=args.retrieval_dir, backend="e5")
        except DerivedCacheError as exc:
            return _fail(str(exc))
        source = "rebuild"
    else:
        rebuilt = {
            "chunk_count": int(result.get("chunk_count", 0)),
            "index_version": int(result.get("index_version", 0)),
        }
    try:
        assert_cache_paths_safe([str(args.cache_dir)])
    except DerivedCacheError as exc:
        return _fail(str(exc))
    saved = _save_bundle(args, bindings, key)
    if saved.get("status") != "saved":
        return _fail(f"L2 publish staging failed: {saved.get('reason', 'unknown')}")

    def _saved_int(field: str, fallback: int) -> int:
        raw = int(saved.get(field, fallback))
        return raw if raw else fallback

    try:
        publish = _publish_artifact(
            cache_dir=args.cache_dir,
            key=key,
            chunk_count=_saved_int("chunk_count", int(rebuilt["chunk_count"])),
            index_version=_saved_int("index_version", int(rebuilt["index_version"])),
            args=args,
        )
    except ArtifactError as exc:
        return _fail(str(exc))
    payload = {
        "status": "published",
        "source": source,
        "cache_key": key.key,
        "cache_fingerprint": key.fingerprint,
        "cache_digest": key.digest,
        **publish,
        "encrypted_sha256": saved.get("encrypted_sha256", ""),
        "encrypted_bytes": saved.get("encrypted_bytes", 0),
        "chunk_count": int(saved.get("chunk_count", 0)),
        "index_version": int(saved.get("index_version", 0)),
    }
    return _status(payload, args)


def _save_bundle(args: argparse.Namespace, bindings: dict[str, Any], key: Any) -> dict[str, Any]:
    """Encrypt/package the validated retrieval state into the cache dir.

    Returns the save payload (``saved`` or ``save-skipped`` with a reason).
    Never logs or stores secrets or plaintext index content.
    """
    # Save path order: the retrieval state must already be the final
    # qualified state; prove it with integrity/self-tests before packaging.
    try:
        checked = self_test_index(args.retrieval_dir)
        digests = bundle_file_digests(args.retrieval_dir)
    except (DerivedCacheError, ValueError, OSError):
        # A missing/invalid index is not a fatal runtime error: report a
        # non-fatal save skip so startup still succeeds.
        return {
            "status": "save-skipped",
            "reason": "self-test-failed",
            "cache_fingerprint": key.fingerprint,
        }
    manifest = build_cache_manifest(
        cache_key=key,
        bindings=bindings,
        file_digests=digests,
        chunk_count=int(checked["chunk_count"]),
        index_version=int(checked["index_version"]),
    )
    started = time.perf_counter()
    try:
        package = create_package(args.retrieval_dir, manifest)
    except (OSError, DerivedCacheError):
        return {
            "status": "save-skipped",
            "reason": "package-failed",
            "cache_fingerprint": key.fingerprint,
        }
    recipient = _read_recipient(args.recipient_file)
    if not recipient:
        return {
            "status": "save-skipped",
            "reason": "backend-unavailable",
            "cache_fingerprint": key.fingerprint,
        }
    try:
        from aa.corpus.age_v1 import parse_recipient

        parse_recipient(recipient)
        encrypted = encrypt_package(package, recipient)
        # Plaintext must never enter the cache: the encrypted bytes must not
        # contain an indexed chunk probe. Check one short probe from the
        # manifest-independent index payload without logging any text.
        probe: bytes | None = None
        try:
            index_payload = json.loads(
                (args.retrieval_dir / "index.json").read_text(encoding="utf-8")
            )
            chunks = index_payload.get("chunks")
            if isinstance(chunks, list) and chunks and isinstance(chunks[0], dict):
                text = str(chunks[0].get("text", ""))
                if len(text) >= 32:
                    probe = text[:32].encode("utf-8")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            probe = None
        if probe is not None and probe in encrypted:
            return {
                "status": "save-skipped",
                "reason": "plaintext-leak",
                "cache_fingerprint": key.fingerprint,
            }
    except DerivedCacheError:
        return {
            "status": "save-skipped",
            "reason": "encrypt-failed",
            "cache_fingerprint": key.fingerprint,
        }
    encrypted_sha = sha256_bytes(encrypted)
    meta = build_meta_payload(
        cache_key=key,
        encrypted_sha256=encrypted_sha,
        encrypted_bytes=len(encrypted),
        bindings=bindings,
    )
    try:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        encrypted_path = args.cache_dir / ENCRYPTED_NAME
        tmp_encrypted = encrypted_path.with_name(encrypted_path.name + ".tmp")
        tmp_encrypted.write_bytes(encrypted)
        if tmp_encrypted.read_bytes() != encrypted:
            raise OSError("verified encrypted write failed")
        os.replace(tmp_encrypted, encrypted_path)
        meta_path = args.cache_dir / META_NAME
        meta_text = json.dumps(meta, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
        tmp_meta.write_text(meta_text, encoding="utf-8")
        if tmp_meta.read_text(encoding="utf-8") != meta_text:
            raise OSError("verified metadata write failed")
        os.replace(tmp_meta, meta_path)
    except OSError:
        return {
            "status": "save-skipped",
            "reason": "backend-unavailable",
            "cache_fingerprint": key.fingerprint,
        }
    return {
        "status": "saved",
        "cache_fingerprint": key.fingerprint,
        "encrypted_sha256": encrypted_sha,
        "encrypted_bytes": len(encrypted),
        "package_ms": (time.perf_counter() - started) * 1000.0,
        "chunk_count": int(checked["chunk_count"]),
        "index_version": int(checked["index_version"]),
    }


def _cmd_save(args: argparse.Namespace) -> int:
    if not is_cache_enabled():
        return _status({"status": "disabled"}, args)
    try:
        assert_cache_paths_safe([str(args.cache_dir)])
    except DerivedCacheError as exc:
        return _fail(str(exc))
    try:
        bindings, key = _live_key(root=ROOT, os_name=args.os, arch=args.arch)
    except DerivedCacheError as exc:
        return _fail(str(exc))
    return _status(_save_bundle(args, bindings, key), args)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the derived retrieval cache manager."""
    parser = argparse.ArgumentParser(description="Manage the encrypted derived retrieval cache.")
    parser.add_argument(
        "--mode",
        choices=(
            "print-key",
            "print-l2-plan",
            "restore",
            "restore-or-rebuild",
            "restore-l2",
            "publish-l2",
            "save",
        ),
        required=True,
        help="Cache operation to perform.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="L2-owned staging directory for ``--mode restore-l2`` (issue "
        "#461). Default: a ``.l2-staging`` sibling of ``--cache-dir``. The "
        "L1 cache directory is never written or cleared through staging.",
    )
    parser.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL_DIR)
    parser.add_argument("--recipient-file", type=Path, default=None)
    parser.add_argument("--identity-file", type=Path, default=None)
    parser.add_argument("--os", type=str, default=None)
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument(
        "--backend",
        choices=("auto", "hashing", "e5"),
        default="e5",
        help="Dense backend used when a deterministic rebuild is required. "
        "Production runtime/qualification always builds and restores E5; "
        "hashing is the offline hermetic fallback for tests only.",
    )
    parser.add_argument(
        "--require-backend",
        choices=("none", "hashing", "e5"),
        default="none",
        help="Strict production backend enforcement restored bundles must "
        "prove in their index metadata (use ``e5`` in production workflows).",
    )
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument(
        "--digest-file",
        type=Path,
        default=DEFAULT_DIGEST_FILE,
        help="Committed immutable L2 pin file (docker/aa-index.digest).",
    )
    parser.add_argument("--oci-registry", type=str, default=DEFAULT_L2_REGISTRY)
    parser.add_argument("--oci-repository", type=str, default=DEFAULT_L2_REPOSITORY)
    parser.add_argument(
        "--oci-scheme",
        choices=("https", "http"),
        default="https",
        help="Transport scheme for the OCI registry (http: offline tests).",
    )
    parser.add_argument(
        "--token-env",
        type=str,
        default="AA_INDEX_PACKAGES_TOKEN",
        help="Environment variable that carries the GHCR packages token.",
    )
    args = parser.parse_args(argv)
    if args.mode == "print-key":
        return _cmd_print_key(args)
    if args.mode == "print-l2-plan":
        return _cmd_print_l2_plan(args)
    if args.mode == "restore":
        return _cmd_restore(args)
    if args.mode == "restore-or-rebuild":
        return _cmd_restore_or_rebuild(args)
    if args.mode == "restore-l2":
        return _cmd_restore_l2(args)
    if args.mode == "publish-l2":
        return _cmd_publish_l2(args)
    return _cmd_save(args)


if __name__ == "__main__":
    raise SystemExit(main())
