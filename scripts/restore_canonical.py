#!/usr/bin/env python3
"""Deterministic canonical bootstrap/restore entry point (issues #24, #50).

Runtime order (``--lang en`` default; ``--lang ru`` restores the Russian
artifact from ``corpus/canonical.ru.manifest.json`` into
``corpus/generated/canonical.ru.json`` via the ``canonical.ru.tar.zst.age``
snapshot encrypted to the same age recipient):

1. Reuse the decrypted ``corpus/generated/canonical.json`` when it already
   exists in the current runner workspace and verifies against the committed
   ``corpus/canonical.manifest.json`` (SHA-256 plus section validation).
2. Otherwise, when the encrypted repository snapshot exists and
   ``AA_BOOK_AGE_IDENTITY`` is available, decrypt it into
   ``corpus/generated/canonical.json`` and verify SHA/version.
3. Otherwise, fall back to the deterministic network fetch/build from #3
   (``scripts/fetch_aa_source.py`` + ``scripts/build_canonical.py``) only
   when explicitly allowed via ``--allow-network-fallback`` or
   ``AA_ALLOW_NETWORK_FETCH=1``.
4. Fail closed when no verified canonical artifact can be obtained.

Later tasks must call this entry point instead of inventing their own
source-loading path. Inside one running job, call it once at startup and
reuse the restored copy for all requests; never download or decrypt the book
per message. Plaintext book text and plaintext retrieval indexes are never
placed in GitHub Actions cache or in Git by this script.

Logs contain only paths, sizes, and SHA-256 digests — never the plaintext
corpus or the private identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.corpus.age_v1 import AgeError, decrypt_bytes, recipient_from_identity  # noqa: E402
from aa.corpus.canonical import (  # noqa: E402
    CanonicalCorpusError,
    load_canonical,
    load_canonical_ru,
)
from aa.corpus.encrypted_snapshot import (  # noqa: E402
    ARCHIVE_NAME,
    METADATA_NAME,
    RU_ARCHIVE_NAME,
    RU_CANONICAL_NAME,
    RU_METADATA_NAME,
    extract_tar_zst,
    sha256_bytes,
)

DEFAULT_MANIFEST = ROOT / "corpus" / "canonical.manifest.json"
DEFAULT_OUTPUT = ROOT / "corpus" / "generated" / "canonical.json"
DEFAULT_RU_MANIFEST = ROOT / "corpus" / "canonical.ru.manifest.json"
DEFAULT_RU_OUTPUT = ROOT / "corpus" / "generated" / "canonical.ru.json"
DEFAULT_ENCRYPTED_DIR = ROOT / "corpus" / "source" / "encrypted"

IDENTITY_ENV = "AA_BOOK_AGE_IDENTITY"
NETWORK_ENV = "AA_ALLOW_NETWORK_FETCH"


def _fail(message: str) -> int:
    print(f"canonical restore failed: {message}", file=sys.stderr)
    return 1


def _manifest_artifact_sha(manifest_path: Path) -> str:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"manifest is missing: {manifest_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is not valid JSON: {exc}") from exc
    sha = manifest.get("artifact_sha256")
    if not isinstance(sha, str) or not sha:
        raise ValueError("manifest has no artifact_sha256")
    return sha


def _valid_existing(path: Path, *, expected_sha: str, lang: str = "en") -> bool:
    try:
        payload = path.read_bytes()
    except OSError:
        return False
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        return False
    try:
        if lang == "ru":
            load_canonical_ru(path, expected_sha256=expected_sha)
        else:
            load_canonical(path, expected_sha256=expected_sha)
    except CanonicalCorpusError:
        return False
    return True


def _write_verified(path: Path, payload: bytes, *, expected_sha: str, lang: str = "en") -> bool:
    if sha256_bytes(payload) != expected_sha:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_bytes(payload)
    if tmp_path.read_bytes() != payload:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return False
    os.replace(tmp_path, path)
    try:
        if lang == "ru":
            load_canonical_ru(path, expected_sha256=expected_sha)
        else:
            load_canonical(path, expected_sha256=expected_sha)
    except CanonicalCorpusError:
        try:
            path.unlink()
        except OSError:
            pass
        return False
    return True


def _read_identity(*, identity_file: Path | None) -> str | None:
    if identity_file is not None:
        try:
            text = identity_file.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"cannot read identity file {identity_file}: {exc}") from exc
        if not text:
            raise ValueError(f"identity file is empty: {identity_file}")
        return text
    raw = os.environ.get(IDENTITY_ENV, "").strip()
    return raw or None


def _network_allowed(*, flag: bool) -> bool:
    if flag:
        return True
    return os.environ.get(NETWORK_ENV, "").strip().lower() in {"1", "true", "yes"}


def _run_network_fallback(*, output: Path, manifest: Path, lang: str = "en") -> int:
    if lang == "ru":
        allowed_output, allowed_manifest = DEFAULT_RU_OUTPUT, DEFAULT_RU_MANIFEST
        fetch = ROOT / "scripts" / "fetch_ru_source.py"
        build = ROOT / "scripts" / "build_canonical_ru.py"
    else:
        allowed_output, allowed_manifest = DEFAULT_OUTPUT, DEFAULT_MANIFEST
        fetch = ROOT / "scripts" / "fetch_aa_source.py"
        build = ROOT / "scripts" / "build_canonical.py"
    if (
        Path(output).resolve() != allowed_output.resolve()
        or Path(manifest).resolve() != allowed_manifest.resolve()
    ):
        print(
            "canonical restore failed: network fallback supports only "
            f"default --output/--manifest (got {output} / {manifest})",
            file=sys.stderr,
        )
        return 1
    steps: list[list[str]] = []
    if lang == "ru":
        # Preserve-bytes/reuse semantics: never re-download over a trusted
        # preserved TXT. Reuse it without network; bootstrap from the
        # provider only to provision a missing TXT.
        raw_txt = ROOT / "corpus" / "source" / "raw-ru" / "aa-big-book.txt"
        fetch_step = [sys.executable, str(fetch)]
        if not raw_txt.exists():
            fetch_step = [sys.executable, str(fetch), "--bootstrap-from-provider"]
        steps = [fetch_step, [sys.executable, str(build)]]
    else:
        steps = [[sys.executable, str(fetch)], [sys.executable, str(build)]]
    for step in steps:
        proc = subprocess.run(step, capture_output=True, text=True, cwd=ROOT)  # noqa: S603
        if proc.returncode != 0:
            step_name = Path(step[1]).name
            print(
                f"network fallback step failed: {step_name}: exit={proc.returncode}",
                file=sys.stderr,
            )
            return proc.returncode or 1
    try:
        expected = _manifest_artifact_sha(manifest)
    except ValueError as exc:
        return _fail(str(exc))
    if not _valid_existing(output, expected_sha=expected, lang=lang):
        return _fail("network fallback produced an unverified artifact")
    print(
        json.dumps(
            {
                "restored": "network-fallback",
                "language": lang,
                "artifact_sha256": expected,
                "artifact_bytes": output.stat().st_size,
            }
        )
    )
    return 0


def _resolve_lang(*, manifest: Path, output: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    try:
        if Path(manifest).resolve() == DEFAULT_RU_MANIFEST.resolve():
            return "ru"
    except OSError:
        pass
    try:
        if Path(output).resolve() == DEFAULT_RU_OUTPUT.resolve():
            return "ru"
    except OSError:
        pass
    return "en"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Restore the verified canonical artifact.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--encrypted-dir", type=Path, default=DEFAULT_ENCRYPTED_DIR)
    parser.add_argument("--identity-file", type=Path, default=None)
    parser.add_argument(
        "--archive-name",
        type=str,
        default=None,
        help="Encrypted archive file name (default follows --lang; "
        "RU default: canonical.ru.tar.zst.age).",
    )
    parser.add_argument(
        "--metadata-name",
        type=str,
        default=None,
        help="Snapshot metadata file name (default follows --lang; RU default: metadata.ru.json).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=("en", "ru"),
        default=None,
        help="Corpus language. Defaults to Russian when --manifest/--output "
        "are the RU defaults, English otherwise. --lang ru alone also "
        "selects the RU manifest/output defaults.",
    )
    parser.add_argument(
        "--allow-network-fallback",
        action="store_true",
        help="Permit the deterministic network fetch/build fallback.",
    )
    parser.add_argument(
        "--no-network-fallback",
        action="store_true",
        help="Forbid the network fallback even when the env var allows it.",
    )
    args = parser.parse_args(argv)

    lang = _resolve_lang(manifest=args.manifest, output=args.output, explicit=args.lang)
    if lang == "ru":
        if Path(args.manifest).resolve() == DEFAULT_MANIFEST.resolve():
            args.manifest = DEFAULT_RU_MANIFEST
        if Path(args.output).resolve() == DEFAULT_OUTPUT.resolve():
            args.output = DEFAULT_RU_OUTPUT
    archive_name = args.archive_name or (RU_ARCHIVE_NAME if lang == "ru" else ARCHIVE_NAME)
    metadata_name = args.metadata_name or (RU_METADATA_NAME if lang == "ru" else METADATA_NAME)
    canonical_name = RU_CANONICAL_NAME if lang == "ru" else "canonical.json"

    try:
        expected_sha = _manifest_artifact_sha(args.manifest)
    except ValueError as exc:
        return _fail(str(exc))

    # 1. Reuse a valid decrypted artifact already present in this workspace.
    if _valid_existing(args.output, expected_sha=expected_sha, lang=lang):
        print(
            json.dumps(
                {
                    "restored": "reused",
                    "language": lang,
                    "artifact_sha256": expected_sha,
                    "artifact_bytes": args.output.stat().st_size,
                }
            )
        )
        return 0

    # 2. Decrypt the committed encrypted snapshot when possible.
    archive = args.encrypted_dir / archive_name
    metadata_path = args.encrypted_dir / metadata_name
    identity: str | None
    try:
        identity = _read_identity(identity_file=args.identity_file)
    except ValueError as exc:
        return _fail(str(exc))
    snapshot_error: str | None = None
    if archive.exists() and identity:
        # Production activation (#28): when the committed public recipient is
        # present, the configured identity must match it. Same existing
        # keypair protects EN + RU; a mismatched identity fails closed here
        # before decryption is attempted. Never log the identity itself.
        try:
            recipient_path = args.encrypted_dir / "recipient.txt"
            if recipient_path.is_file():
                committed = recipient_path.read_text(encoding="utf-8").strip()
                if committed:
                    try:
                        derived = recipient_from_identity(identity)
                    except AgeError as exc:
                        snapshot_error = f"configured identity is invalid: {exc}"
                        derived = ""
                    if snapshot_error is None and derived.strip() != committed:
                        snapshot_error = (
                            "configured identity does not match the committed "
                            "public recipient; refusing decrypt with a different keypair"
                        )
        except (OSError, UnicodeDecodeError) as exc:
            snapshot_error = f"committed recipient is unreadable: {exc}"
        if snapshot_error is None:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                metadata = None  # type: ignore[assignment]
                snapshot_error = f"snapshot metadata is missing or invalid: {exc}"
        if snapshot_error is None:
            if not isinstance(metadata, dict):
                snapshot_error = "snapshot metadata is malformed"
            elif metadata.get("canonical_sha256") != expected_sha:
                snapshot_error = "snapshot metadata canonical SHA does not match the manifest"
        if snapshot_error is None:
            if not isinstance(metadata, dict):
                snapshot_error = "snapshot metadata is malformed"
            else:
                version = metadata.get("metadata_version")
                if version is not None and version not in (1, 2):
                    snapshot_error = f"unsupported snapshot metadata version: {version!r}"
        if snapshot_error is None:
            if not isinstance(metadata, dict):
                snapshot_error = "snapshot metadata is malformed"
            else:
                member = metadata.get("canonical_member")
                if member is not None and member != canonical_name:
                    snapshot_error = "snapshot metadata canonical member does not match language"
        if snapshot_error is None:
            if not isinstance(metadata, dict):
                snapshot_error = "snapshot metadata is malformed"
            else:
                encrypted_file = metadata.get("encrypted_file")
                if (
                    isinstance(encrypted_file, str)
                    and encrypted_file
                    and not encrypted_file.endswith(f"/{archive_name}")
                    and encrypted_file != archive_name
                ):
                    snapshot_error = "snapshot metadata encrypted file does not match archive"
        encrypted: bytes | None = None
        if snapshot_error is None:
            try:
                encrypted = archive.read_bytes()
            except OSError as exc:
                snapshot_error = f"cannot read encrypted snapshot: {exc}"
        if snapshot_error is None:
            if encrypted is None:
                snapshot_error = "snapshot encrypted payload is missing"
            elif not isinstance(metadata, dict):
                snapshot_error = "snapshot metadata is malformed"
            else:
                expected_encrypted_sha = metadata.get("encrypted_sha256")
                if expected_encrypted_sha != hashlib.sha256(encrypted).hexdigest():
                    snapshot_error = "snapshot encrypted SHA does not match metadata"
        if snapshot_error is None:
            if encrypted is None:
                snapshot_error = "snapshot encrypted payload is missing"
            else:
                try:
                    tar_zst = decrypt_bytes(encrypted, [identity])
                    canonical_bytes, _ = extract_tar_zst(
                        tar_zst, expected_canonical_name=canonical_name
                    )
                except (AgeError, ValueError) as exc:
                    snapshot_error = f"snapshot decrypt failed: {exc}"
        if snapshot_error is None:
            if _write_verified(args.output, canonical_bytes, expected_sha=expected_sha, lang=lang):
                print(
                    json.dumps(
                        {
                            "restored": "decrypted",
                            "language": lang,
                            "artifact_sha256": expected_sha,
                            "artifact_bytes": len(canonical_bytes),
                        }
                    )
                )
                return 0
            snapshot_error = "decrypted snapshot failed SHA/section verification"
        if snapshot_error is not None:
            print(f"snapshot restore failed: {snapshot_error}", file=sys.stderr)

    # 3. Deterministic network fetch/build fallback when explicitly allowed.
    allow_network = _network_allowed(flag=args.allow_network_fallback)
    if args.no_network_fallback:
        allow_network = False
    if allow_network:
        return _run_network_fallback(output=args.output, manifest=args.manifest, lang=lang)

    # 4. Fail closed.
    reasons = []
    if not archive.exists():
        reasons.append("encrypted snapshot is not present")
    elif not identity:
        reasons.append(f"{IDENTITY_ENV} is not set")
    else:
        reasons.append(snapshot_error or "encrypted snapshot could not be verified")
    reasons.append("network fallback was not explicitly allowed")
    return _fail("; ".join(reasons))


if __name__ == "__main__":
    raise SystemExit(main())
