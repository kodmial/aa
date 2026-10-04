#!/usr/bin/env python3
"""Manage the encrypted disposable derived retrieval cache (issue #58).

Acceleration only: runtime correctness never depends on this cache. Every
mode exits 0 when a deterministic rebuild is possible, so a missing,
stale, corrupt, or unavailable cache behaves exactly like a first clean
installation.

Modes:

- ``--print-key`` — compute the exact content-addressed cache key from the
  authoritative manifests/config and print it (no secrets, no text);
- ``--restore`` — exact cache lookup only: verify the encrypted-package
  checksum, decrypt into the ignored ephemeral workspace, verify the
  internal manifest and every version binding, open SQLite/Faiss and run a
  lightweight self-test. On any failure the restored data is deleted and
  the status reports ``miss``/``rejected`` with a reason category;
- ``--restore-or-rebuild`` — try ``--restore``; on miss/rejected, restore
  and validate the authoritative RU/EN canonical artifacts, build the
  final qualified retrieval state, and run integrity/self-tests;
- ``--save`` — package the ready-built index deterministically, encrypt
  with the existing age recipient, and write only the encrypted package
  plus non-secret cache metadata into ``--cache-dir``. Save failure is
  non-fatal (runtime still starts).

Only ``--print-key``/``--restore``/``--restore-or-rebuild``/``--save`` logs
are privacy-safe: hit/miss/rejected status, rejection reason category,
restore/decrypt/build durations, cache key fingerprint, and rebuilt index
version. Never corpus text, user text, secrets, or decrypted contents.
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
    ENCRYPTED_NAME,
    META_NAME,
    DerivedCacheError,
    assert_cache_paths_safe,
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
    key_fingerprint,
    normalize_arch,
    normalize_os_name,
    read_bindings,
    self_test_index,
    sha256_bytes,
    verify_extracted_files,
    verify_manifest,
)

DEFAULT_RETRIEVAL_DIR = ROOT / "corpus" / "generated" / "retrieval"
DEFAULT_CACHE_DIR = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "aa-derived-cache"
IDENTITY_ENV = "AA_BOOK_AGE_IDENTITY"


def _fail(message: str) -> int:
    print(f"derived cache failed: {message}", file=sys.stderr)
    return 1


def _status(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, sort_keys=True))
    return 0


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
    return _status(
        {
            "cache_key": key.key,
            "cache_fingerprint": key.fingerprint,
            "cache_digest": key.digest,
            "os": str(bindings.get("os", "")),
            "arch": str(bindings.get("arch", "")),
        }
    )


def _attempt_restore(
    *,
    cache_dir: Path,
    retrieval_dir: Path,
    bindings: dict[str, Any],
    key: Any,
    identity: str | None,
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
    except DerivedCacheError as exc:
        message = str(exc)
        reason = "stale-version" if "stale" in message else "corrupt-package"
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
    from aa.retrieval.index import build_hybrid_index

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
    index = build_hybrid_index(
        full,
        ru_manifest=ru_manifest,
        en_manifest=en_manifest,
        embedding_lock=lock,
        out_dir=retrieval_dir,
        backend=backend,
    )
    checked = self_test_index(retrieval_dir)
    return {
        "chunk_count": index.chunk_count,
        "index_version": checked["index_version"],
        "build_ms": (time.perf_counter() - started) * 1000.0,
    }


def _cmd_restore(args: argparse.Namespace) -> int:
    if not is_cache_enabled():
        return _status({"status": "disabled"})
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
    )
    return _status(result)


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
                "build_ms": rebuilt["build_ms"],
                "chunk_count": rebuilt["chunk_count"],
                "index_version": rebuilt["index_version"],
            }
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
    )
    if result.get("status") == "hit":
        return _status(result)
    # Miss, stale, corrupt, or unavailable: transparent deterministic rebuild
    # from the canonical corpus. The rejection reason is preserved.
    try:
        rebuilt = _rebuild_index(retrieval_dir=args.retrieval_dir, backend=args.backend)
    except DerivedCacheError as exc:
        return _fail(str(exc))
    return _status(
        {
            "status": "rebuilt",
            "reason": str(result.get("reason", "cache-miss")),
            "cache_fingerprint": key.fingerprint,
            "restore_ms": result.get("restore_ms", 0.0),
            "build_ms": rebuilt["build_ms"],
            "chunk_count": rebuilt["chunk_count"],
            "index_version": rebuilt["index_version"],
        }
    )


def _cmd_save(args: argparse.Namespace) -> int:
    if not is_cache_enabled():
        return _status({"status": "disabled"})
    try:
        assert_cache_paths_safe([str(args.cache_dir)])
    except DerivedCacheError as exc:
        return _fail(str(exc))
    try:
        bindings, key = _live_key(root=ROOT, os_name=args.os, arch=args.arch)
    except DerivedCacheError as exc:
        return _fail(str(exc))
    # Save path order: the retrieval state must already be the final
    # qualified state; prove it with integrity/self-tests before packaging.
    try:
        checked = self_test_index(args.retrieval_dir)
        digests = bundle_file_digests(args.retrieval_dir)
    except (DerivedCacheError, ValueError, OSError):
        # A missing/invalid index is not a fatal runtime error: report a
        # non-fatal save skip so startup still succeeds.
        return _status(
            {
                "status": "save-skipped",
                "reason": "self-test-failed",
                "cache_fingerprint": key.fingerprint,
            }
        )
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
        return _status(
            {
                "status": "save-skipped",
                "reason": "package-failed",
                "cache_fingerprint": key.fingerprint,
            }
        )
    recipient = _read_recipient(args.recipient_file)
    if not recipient:
        return _status(
            {
                "status": "save-skipped",
                "reason": "backend-unavailable",
                "cache_fingerprint": key.fingerprint,
            }
        )
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
            return _status(
                {
                    "status": "save-skipped",
                    "reason": "plaintext-leak",
                    "cache_fingerprint": key.fingerprint,
                }
            )
    except DerivedCacheError:
        return _status(
            {
                "status": "save-skipped",
                "reason": "encrypt-failed",
                "cache_fingerprint": key.fingerprint,
            }
        )
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
        return _status(
            {
                "status": "save-skipped",
                "reason": "backend-unavailable",
                "cache_fingerprint": key.fingerprint,
            }
        )
    _ = key_fingerprint(key.key)
    return _status(
        {
            "status": "saved",
            "cache_fingerprint": key.fingerprint,
            "encrypted_sha256": encrypted_sha,
            "encrypted_bytes": len(encrypted),
            "package_ms": (time.perf_counter() - started) * 1000.0,
            "chunk_count": int(checked["chunk_count"]),
            "index_version": int(checked["index_version"]),
        }
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the derived retrieval cache manager."""
    parser = argparse.ArgumentParser(description="Manage the encrypted derived retrieval cache.")
    parser.add_argument(
        "--mode",
        choices=("print-key", "restore", "restore-or-rebuild", "save"),
        required=True,
        help="Cache operation to perform.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL_DIR)
    parser.add_argument("--recipient-file", type=Path, default=None)
    parser.add_argument("--identity-file", type=Path, default=None)
    parser.add_argument("--os", type=str, default=None)
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument(
        "--backend",
        choices=("auto", "hashing", "e5"),
        default="hashing",
        help="Dense backend used when a deterministic rebuild is required.",
    )
    args = parser.parse_args(argv)
    if args.mode == "print-key":
        return _cmd_print_key(args)
    if args.mode == "restore":
        return _cmd_restore(args)
    if args.mode == "restore-or-rebuild":
        return _cmd_restore_or_rebuild(args)
    return _cmd_save(args)


if __name__ == "__main__":
    raise SystemExit(main())
