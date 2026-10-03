#!/usr/bin/env python3
"""Refresh the encrypted canonical-book snapshot (issue #24).

Reads the validated canonical artifact from ``scripts/build_canonical.py``,
packs the exact artifact bytes plus minimum provenance metadata into a
deterministic ``tar.zst`` archive, encrypts it with the committed public age
recipient, and writes only the encrypted archive plus non-secret
``metadata.json`` into ``corpus/source/encrypted/``.

When ``AA_BOOK_AGE_IDENTITY`` (or ``--identity-file``) is available, the
script additionally decrypts the freshly written archive and verifies that it
reproduces the expected canonical SHA-256. Without an identity the script
still produces the snapshot but reports that decrypt-verification was
skipped; the refresh workflow always provides the secret for verification.

Logs contain only paths, sizes, and SHA-256 digests — never the plaintext
corpus, the recipient secret, or the private identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.corpus.age_v1 import (  # noqa: E402
    AgeError,
    decrypt_bytes,
    encrypt_bytes,
    parse_recipient,
)
from aa.corpus.encrypted_snapshot import (  # noqa: E402
    ARCHIVE_NAME,
    METADATA_NAME,
    RECIPIENT_NAME,
    build_metadata,
    create_tar_zst,
    extract_tar_zst,
    sha256_bytes,
)

DEFAULT_MANIFEST = ROOT / "corpus" / "canonical.manifest.json"
DEFAULT_SOURCE_LOCK = ROOT / "corpus" / "source.lock.json"
DEFAULT_CANONICAL = ROOT / "corpus" / "generated" / "canonical.json"
DEFAULT_ENCRYPTED_DIR = ROOT / "corpus" / "source" / "encrypted"


def _fail(message: str) -> int:
    print(f"snapshot refresh failed: {message}", file=sys.stderr)
    return 1


def _read_recipient(*, recipient_value: str | None, recipient_file: Path | None) -> str | None:
    if recipient_value:
        return recipient_value.strip()
    path = recipient_file or (DEFAULT_ENCRYPTED_DIR / RECIPIENT_NAME)
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def _read_identity(*, identity_file: Path | None) -> str | None:
    if identity_file is not None:
        try:
            return identity_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
    raw = os.environ.get("AA_BOOK_AGE_IDENTITY", "").strip()
    return raw or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh the encrypted canonical snapshot.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--encrypted-dir", type=Path, default=DEFAULT_ENCRYPTED_DIR)
    parser.add_argument("--recipient", type=str, default=None)
    parser.add_argument("--recipient-file", type=Path, default=None)
    parser.add_argument("--identity-file", type=Path, default=None)
    parser.add_argument("--skip-decrypt-verify", action="store_true")
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _fail(f"manifest is missing: {args.manifest}")
    except json.JSONDecodeError as exc:
        return _fail(f"manifest is not valid JSON: {exc}")
    try:
        source_lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    except FileNotFoundError:
        source_lock = None
    except json.JSONDecodeError as exc:
        return _fail(f"source lock is not valid JSON: {exc}")

    try:
        canonical_bytes = args.canonical.read_bytes()
    except FileNotFoundError:
        return _fail(f"canonical artifact is missing; build it first: {args.canonical}")

    expected_sha = str(manifest.get("artifact_sha256", ""))
    actual_sha = sha256_bytes(canonical_bytes)
    if not expected_sha or actual_sha != expected_sha:
        return _fail(f"canonical SHA mismatch: expected={expected_sha!r} actual={actual_sha!r}")

    recipient = _read_recipient(recipient_value=args.recipient, recipient_file=args.recipient_file)
    if not recipient:
        return _fail("public age recipient is missing (recipient file or --recipient)")
    try:
        parse_recipient(recipient)
    except AgeError as exc:
        return _fail(f"invalid age recipient: {exc}")

    tar_zst = create_tar_zst(canonical_bytes, manifest=manifest, source_lock=source_lock)
    try:
        encrypted = encrypt_bytes(tar_zst, [recipient])
    except AgeError as exc:
        return _fail(f"encryption failed: {exc}")

    args.encrypted_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.encrypted_dir / ARCHIVE_NAME
    archive_path.write_bytes(encrypted)
    if archive_path.read_bytes() != encrypted:
        return _fail("written archive differs from the encrypted payload")

    encrypted_sha = hashlib.sha256(encrypted).hexdigest()
    metadata = build_metadata(
        canonical_sha256=actual_sha,
        encrypted_sha256=encrypted_sha,
        manifest=manifest,
        source_lock=source_lock,
        recipient=recipient,
    )
    metadata_path = args.encrypted_dir / METADATA_NAME
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "canonical_sha256": actual_sha,
                "canonical_bytes": len(canonical_bytes),
                "encrypted_sha256": encrypted_sha,
                "encrypted_bytes": len(encrypted),
            }
        )
    )

    if args.skip_decrypt_verify:
        print("decrypt verification skipped (no identity provided)")
        return 0

    identity = _read_identity(identity_file=args.identity_file)
    if not identity:
        print("decrypt verification skipped (AA_BOOK_AGE_IDENTITY is not set)")
        return 0
    try:
        recovered_zst = decrypt_bytes(encrypted, [identity])
        recovered_canonical, _ = extract_tar_zst(recovered_zst)
    except (AgeError, ValueError) as exc:
        return _fail(f"decrypt verification failed: {exc}")
    if sha256_bytes(recovered_canonical) != actual_sha:
        return _fail("decrypt verification failed: canonical SHA mismatch after round-trip")
    print("decrypt verification ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
