#!/usr/bin/env python3
"""Deterministic canonical bootstrap/restore entry point (issue #24).

Runtime order:

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

from aa.corpus.age_v1 import AgeError, decrypt_bytes  # noqa: E402
from aa.corpus.canonical import CanonicalCorpusError, load_canonical  # noqa: E402
from aa.corpus.encrypted_snapshot import (  # noqa: E402
    ARCHIVE_NAME,
    METADATA_NAME,
    extract_tar_zst,
    sha256_bytes,
)

DEFAULT_MANIFEST = ROOT / "corpus" / "canonical.manifest.json"
DEFAULT_OUTPUT = ROOT / "corpus" / "generated" / "canonical.json"
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


def _valid_existing(path: Path, *, expected_sha: str) -> bool:
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        return False
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        return False
    try:
        load_canonical(path, expected_sha256=expected_sha)
    except CanonicalCorpusError:
        return False
    return True


def _write_verified(path: Path, payload: bytes, *, expected_sha: str) -> bool:
    if sha256_bytes(payload) != expected_sha:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    try:
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
        except FileNotFoundError:
            return None
        return text or None
    raw = os.environ.get(IDENTITY_ENV, "").strip()
    return raw or None


def _network_allowed(*, flag: bool) -> bool:
    if flag:
        return True
    return os.environ.get(NETWORK_ENV, "").strip().lower() in {"1", "true", "yes"}


def _run_network_fallback(*, output: Path, manifest: Path) -> int:
    if output != DEFAULT_OUTPUT or manifest != DEFAULT_MANIFEST:
        print(
            "canonical restore failed: network fallback supports only "
            f"default --output/--manifest (got {output} / {manifest})",
            file=sys.stderr,
        )
        return 1
    fetch = ROOT / "scripts" / "fetch_aa_source.py"
    build = ROOT / "scripts" / "build_canonical.py"
    for step in ([sys.executable, str(fetch)], [sys.executable, str(build)]):
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
    if not _valid_existing(output, expected_sha=expected):
        return _fail("network fallback produced an unverified artifact")
    print(
        json.dumps(
            {
                "restored": "network-fallback",
                "artifact_sha256": expected,
                "artifact_bytes": output.stat().st_size,
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Restore the verified canonical artifact.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--encrypted-dir", type=Path, default=DEFAULT_ENCRYPTED_DIR)
    parser.add_argument("--identity-file", type=Path, default=None)
    parser.add_argument(
        "--allow-network-fallback",
        action="store_true",
        help="Permit the deterministic #3 network fetch/build fallback.",
    )
    parser.add_argument(
        "--no-network-fallback",
        action="store_true",
        help="Forbid the network fallback even when the env var allows it.",
    )
    args = parser.parse_args(argv)

    try:
        expected_sha = _manifest_artifact_sha(args.manifest)
    except ValueError as exc:
        return _fail(str(exc))

    # 1. Reuse a valid decrypted artifact already present in this workspace.
    if _valid_existing(args.output, expected_sha=expected_sha):
        print(
            json.dumps(
                {
                    "restored": "reused",
                    "artifact_sha256": expected_sha,
                    "artifact_bytes": args.output.stat().st_size,
                }
            )
        )
        return 0

    # 2. Decrypt the committed encrypted snapshot when possible.
    archive = args.encrypted_dir / ARCHIVE_NAME
    metadata_path = args.encrypted_dir / METADATA_NAME
    identity = _read_identity(identity_file=args.identity_file)
    snapshot_error: str | None = None
    if archive.exists() and identity:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            metadata = None  # type: ignore[assignment]
            snapshot_error = f"snapshot metadata is missing or invalid: {exc}"
        if snapshot_error is None:
            if not isinstance(metadata, dict) or metadata.get("canonical_sha256") != expected_sha:
                snapshot_error = "snapshot metadata canonical SHA does not match the manifest"
        encrypted: bytes | None = None
        if snapshot_error is None:
            try:
                encrypted = archive.read_bytes()
            except OSError as exc:
                snapshot_error = f"cannot read encrypted snapshot: {exc}"
        if snapshot_error is None:
            assert encrypted is not None
            expected_encrypted_sha = metadata.get("encrypted_sha256")
            if expected_encrypted_sha != hashlib.sha256(encrypted).hexdigest():
                snapshot_error = "snapshot encrypted SHA does not match metadata"
        if snapshot_error is None:
            assert encrypted is not None
            try:
                tar_zst = decrypt_bytes(encrypted, [identity])
                canonical_bytes, _ = extract_tar_zst(tar_zst)
            except (AgeError, ValueError) as exc:
                snapshot_error = f"snapshot decrypt failed: {exc}"
        if snapshot_error is None:
            if _write_verified(args.output, canonical_bytes, expected_sha=expected_sha):
                print(
                    json.dumps(
                        {
                            "restored": "decrypted",
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
        return _run_network_fallback(output=args.output, manifest=args.manifest)

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
