"""Encrypted canonical-book snapshot helpers (issue #24).

The durable cross-run cache for this small corpus is the committed encrypted
snapshot ``corpus/source/encrypted/canonical.tar.zst.age``. Plaintext book
text and plaintext retrieval indexes are never stored in Git or in GitHub
Actions cache; inside one running job the decrypted
``corpus/generated/canonical.json`` is reused for all requests.

Archive layout (deterministic ``tar`` + ``zstd`` then ``age``):

- ``canonical.json`` — exact bytes of the validated canonical artifact;
- ``provenance.json`` — minimum non-literary provenance metadata.

``metadata.json`` (committed, non-secret) records the canonical SHA-256, the
encrypted archive SHA-256, manifest/source versions, the encryption
format/version, and creation/update instructions.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import zstandard as zstd

ENCRYPTION_FORMAT = "age-encryption.org/v1/X25519"
ENCRYPTION_IMPL = "aa-age-v1-py/1"
ARCHIVE_FORMAT = "canonical-tar-zst/1"
ARCHIVE_NAME = "canonical.tar.zst.age"
METADATA_NAME = "metadata.json"
RECIPIENT_NAME = "recipient.txt"
METADATA_VERSION = 1
MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's bytes."""
    return sha256_bytes(Path(path).read_bytes())


def _provenance_payload(
    *,
    canonical_sha256: str,
    manifest: dict[str, Any],
    source_lock: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "archive_format": ARCHIVE_FORMAT,
        "canonical_sha256": canonical_sha256,
        "manifest_format": manifest.get("format"),
        "manifest_builder_version": manifest.get("builder_version"),
        "manifest_edition": manifest.get("edition"),
        "manifest_artifact_sha256": manifest.get("artifact_sha256"),
        "source_lock_version": (source_lock or {}).get("version"),
        "source_lock_edition": (source_lock or {}).get("edition"),
        "encryption_format": ENCRYPTION_FORMAT,
    }


def create_tar_zst(
    canonical_bytes: bytes,
    *,
    manifest: dict[str, Any],
    source_lock: dict[str, Any] | None = None,
) -> bytes:
    """Pack the canonical artifact plus provenance into deterministic tar.zst."""
    canonical_sha = sha256_bytes(canonical_bytes)
    provenance = _provenance_payload(
        canonical_sha256=canonical_sha, manifest=manifest, source_lock=source_lock
    )
    provenance_bytes = (
        json.dumps(provenance, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")

    members: list[tuple[str, bytes]] = [
        ("canonical.json", canonical_bytes),
        ("provenance.json", provenance_bytes),
    ]
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, payload in sorted(members):
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


def extract_tar_zst(data: bytes) -> tuple[bytes, dict[str, Any]]:
    """Unpack tar.zst bytes into ``(canonical_bytes, provenance)``."""
    try:
        decompressed = zstd.ZstdDecompressor().decompress(
            data, max_output_size=MAX_DECOMPRESSED_BYTES
        )
    except zstd.ZstdError as exc:
        raise ValueError(f"snapshot decompression failed: {exc}") from exc
    buffer = io.BytesIO(decompressed)
    try:
        with tarfile.open(fileobj=buffer, mode="r") as tar:
            raw_members = tar.getmembers()
            names = [m.name for m in raw_members]
            if len(raw_members) != 2 or len(set(names)) != 2:
                raise ValueError(f"unexpected snapshot members: {sorted(names)}")
            members = {m.name: m for m in raw_members}
            if set(members) != {"canonical.json", "provenance.json"}:
                raise ValueError(f"unexpected snapshot members: {sorted(members)}")
            if any(not m.isreg() for m in members.values()):
                raise ValueError("snapshot members must be regular files")
            canonical_member = tar.extractfile(members["canonical.json"])
            provenance_member = tar.extractfile(members["provenance.json"])
            if canonical_member is None or provenance_member is None:
                raise ValueError("snapshot is missing required members")
            canonical_bytes = canonical_member.read()
            provenance = json.loads(provenance_member.read().decode("utf-8"))
    except (tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"snapshot archive is malformed: {exc}") from exc
    if not isinstance(provenance, dict):
        raise ValueError("snapshot provenance is malformed")
    if provenance.get("canonical_sha256") != sha256_bytes(canonical_bytes):
        raise ValueError("snapshot provenance canonical SHA does not match payload")
    return canonical_bytes, provenance


def build_metadata(
    *,
    canonical_sha256: str,
    encrypted_sha256: str,
    manifest: dict[str, Any],
    source_lock: dict[str, Any] | None,
    recipient: str,
) -> dict[str, Any]:
    """Build the committed non-secret ``metadata.json`` payload."""
    return {
        "metadata_version": METADATA_VERSION,
        "canonical_sha256": canonical_sha256,
        "encrypted_sha256": encrypted_sha256,
        "encrypted_file": f"corpus/source/encrypted/{ARCHIVE_NAME}",
        "manifest_format": manifest.get("format"),
        "manifest_builder_version": manifest.get("builder_version"),
        "manifest_artifact_sha256": manifest.get("artifact_sha256"),
        "source_lock_version": (source_lock or {}).get("version"),
        "source_lock_edition": (source_lock or {}).get("edition"),
        "encryption_format": ENCRYPTION_FORMAT,
        "encryption_impl": ENCRYPTION_IMPL,
        "archive_format": ARCHIVE_FORMAT,
        "recipient_hint": recipient[-16:] if len(recipient) >= 16 else "unknown",
        "creation_update": (
            "Regenerate via the manually dispatched "
            ".github/workflows/encrypted-corpus-refresh.yml workflow, or locally: "
            "python3 scripts/fetch_aa_source.py && "
            "python3 scripts/build_canonical.py && "
            "python3 scripts/refresh_encrypted_snapshot.py --recipient-file "
            "corpus/source/encrypted/recipient.txt "
            "(decrypt verification needs AA_BOOK_AGE_IDENTITY). "
            "Production activation (key/secret/snapshot) is tracked in #28."
        ),
    }
