#!/usr/bin/env python3
"""Refresh the encrypted canonical-book snapshot (issues #24, #50).

Reads the validated canonical artifact from ``scripts/build_canonical.py``
(English) or ``scripts/build_canonical_ru.py`` (Russian via ``--lang ru`` or
RU ``--manifest`` / ``--canonical`` paths; ``--archive-name`` /
``--metadata-name`` / ``--canonical-name`` default from the resolved language
and may still be overridden explicitly),
packs the exact artifact bytes plus minimum provenance metadata into a
deterministic ``tar.zst`` archive, encrypts it with the committed public age
recipient, and writes only the encrypted archive plus non-secret
``metadata.json`` into ``corpus/source/encrypted/``.

When ``AA_BOOK_AGE_IDENTITY`` (or ``--identity-file``) is available, the
script decrypts the freshly produced archive in memory and verifies that it
reproduces the expected canonical SHA-256 before anything is written.
Without an identity the script fails closed and writes nothing, unless
``--skip-decrypt-verify`` is explicitly passed for local development; the
refresh workflow always provides the secret for verification.

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
    CANONICAL_NAME,
    METADATA_NAME,
    RECIPIENT_NAME,
    RU_ARCHIVE_NAME,
    RU_CANONICAL_NAME,
    RU_METADATA_NAME,
    build_metadata,
    create_tar_zst,
    extract_tar_zst,
    sha256_bytes,
)

DEFAULT_MANIFEST = ROOT / "corpus" / "canonical.manifest.json"
DEFAULT_SOURCE_LOCK = ROOT / "corpus" / "source.lock.json"
DEFAULT_CANONICAL = ROOT / "corpus" / "generated" / "canonical.json"
DEFAULT_RU_MANIFEST = ROOT / "corpus" / "canonical.ru.manifest.json"
DEFAULT_RU_SOURCE_LOCK = ROOT / "corpus" / "source.ru.lock.json"
DEFAULT_RU_CANONICAL = ROOT / "corpus" / "generated" / "canonical.ru.json"
DEFAULT_ENCRYPTED_DIR = ROOT / "corpus" / "source" / "encrypted"


def _is_default(path: Path, default: Path) -> bool:
    try:
        return Path(path).resolve() == default.resolve()
    except OSError:
        return False


def _resolve_lang(*, manifest: Path, canonical: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    try:
        if _is_default(manifest, DEFAULT_RU_MANIFEST) or _is_default(
            canonical, DEFAULT_RU_CANONICAL
        ):
            return "ru"
    except OSError:
        pass
    return "en"


def _fail(message: str) -> int:
    print(f"snapshot refresh failed: {message}", file=sys.stderr)
    return 1


def _read_recipient(*, recipient_value: str | None, recipient_file: Path | None) -> str | None:
    if recipient_value:
        return recipient_value.strip()
    path = recipient_file or (DEFAULT_ENCRYPTED_DIR / RECIPIENT_NAME)
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _read_identity(*, identity_file: Path | None) -> str | None:
    if identity_file is not None:
        try:
            return identity_file.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
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
    parser.add_argument(
        "--archive-name",
        type=str,
        default=None,
        help="Encrypted archive file name inside --encrypted-dir "
        "(default follows --lang/manifest/canonical; RU default: canonical.ru.tar.zst.age).",
    )
    parser.add_argument(
        "--metadata-name",
        type=str,
        default=None,
        help="Metadata file name inside --encrypted-dir "
        "(default follows --lang/manifest/canonical; RU default: metadata.ru.json).",
    )
    parser.add_argument(
        "--canonical-name",
        type=str,
        default=None,
        help="Canonical member name inside the tar.zst archive "
        "(default follows --lang/manifest/canonical; RU default: canonical.ru.json).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=("en", "ru"),
        default=None,
        help="Corpus language. Defaults to Russian when --manifest/--canonical "
        "are the RU defaults, English otherwise. --lang ru alone also "
        "selects the RU manifest/source-lock/canonical defaults.",
    )
    args = parser.parse_args(argv)

    lang = _resolve_lang(manifest=args.manifest, canonical=args.canonical, explicit=args.lang)
    if lang == "ru":
        if _is_default(args.manifest, DEFAULT_MANIFEST):
            args.manifest = DEFAULT_RU_MANIFEST
        if _is_default(args.source_lock, DEFAULT_SOURCE_LOCK):
            args.source_lock = DEFAULT_RU_SOURCE_LOCK
        if _is_default(args.canonical, DEFAULT_CANONICAL):
            args.canonical = DEFAULT_RU_CANONICAL
    archive_name = args.archive_name or (RU_ARCHIVE_NAME if lang == "ru" else ARCHIVE_NAME)
    metadata_name = args.metadata_name or (RU_METADATA_NAME if lang == "ru" else METADATA_NAME)
    canonical_name = args.canonical_name or (RU_CANONICAL_NAME if lang == "ru" else CANONICAL_NAME)

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

    tar_zst = create_tar_zst(
        canonical_bytes,
        manifest=manifest,
        source_lock=source_lock,
        canonical_name=canonical_name,
    )
    try:
        encrypted = encrypt_bytes(tar_zst, [recipient])
    except AgeError as exc:
        return _fail(f"encryption failed: {exc}")

    skip_verify = bool(args.skip_decrypt_verify)
    if not skip_verify:
        try:
            identity = _read_identity(identity_file=args.identity_file)
        except OSError as exc:
            return _fail(f"cannot read identity file: {exc}")
        if not identity:
            if args.identity_file is not None:
                return _fail(f"identity file is missing or empty: {args.identity_file}")
            return _fail(
                "decrypt verification requires an identity "
                "(AA_BOOK_AGE_IDENTITY or --identity-file); refusing to write "
                "an unverified snapshot (use --skip-decrypt-verify to override)"
            )
        try:
            recovered_zst = decrypt_bytes(encrypted, [identity])
            recovered_canonical, _ = extract_tar_zst(
                recovered_zst, expected_canonical_name=canonical_name
            )
        except (AgeError, ValueError) as exc:
            return _fail(f"decrypt verification failed: {exc}")
        if sha256_bytes(recovered_canonical) != actual_sha:
            return _fail("decrypt verification failed: canonical SHA mismatch after round-trip")

    args.encrypted_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.encrypted_dir / archive_name
    tmp_path = archive_path.with_name(archive_path.name + ".tmp")
    tmp_path.write_bytes(encrypted)
    if tmp_path.read_bytes() != encrypted:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return _fail("written archive differs from the encrypted payload")
    os.replace(tmp_path, archive_path)

    encrypted_sha = hashlib.sha256(encrypted).hexdigest()
    metadata = build_metadata(
        canonical_sha256=actual_sha,
        encrypted_sha256=encrypted_sha,
        manifest=manifest,
        source_lock=source_lock,
        recipient=recipient,
        encrypted_file=f"corpus/source/encrypted/{archive_name}",
        canonical_name=canonical_name,
    )
    metadata_path = args.encrypted_dir / metadata_name
    tmp_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
    metadata_text = json.dumps(metadata, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    tmp_metadata.write_text(metadata_text, encoding="utf-8")
    if tmp_metadata.read_text(encoding="utf-8") != metadata_text:
        try:
            tmp_metadata.unlink()
        except OSError:
            pass
        return _fail("written metadata differs from the expected payload")
    os.replace(tmp_metadata, metadata_path)

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

    if skip_verify:
        print("decrypt verification skipped (--skip-decrypt-verify)")
        return 0
    print("decrypt verification ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
