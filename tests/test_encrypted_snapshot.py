"""Encrypted snapshot + deterministic restore tests (issue #24).

All keys are ephemeral test-only ``age`` pairs created inside each test and
destroyed afterward (local variables plus ``tmp_path`` files cleaned by
pytest). No test key, private key, or plaintext book is committed or logged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
from typing import Any

import pytest

from aa.corpus import age_v1
from aa.corpus import encrypted_snapshot as snap
from aa.corpus.age_v1 import AgeError


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _load_builder() -> Any:
    path = _repo_root() / "scripts" / "build_canonical.py"
    spec = importlib.util.spec_from_file_location("build_canonical_fixture", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_canonical_payload() -> tuple[bytes, dict[str, Any]]:
    builder = _load_builder()
    preamble = (
        "Fixture book.\r\n\r\nProvided as fixture software\r\n\r\n"
        "by The Anonymous Press\r\n\r\n~~~~\r\n"
    )
    titles = (
        "BILL'S STORY",
        "THERE IS A SOLUTION",
        "MORE ABOUT ALCOHOLISM",
        "WE AGNOSTICS",
        "HOW IT WORKS",
        "INTO ACTION",
        "WORKING WITH OTHERS",
        "TO WIVES",
        "THE FAMILY AFTERWARD",
        "TO EMPLOYERS",
        "A VISION FOR YOU",
    )
    parts = [preamble]
    for number, title in enumerate(titles, start=1):
        body = (
            f"Chapter {number}\r\n    \r\n    {title}\r\n    \r\n"
            f"Snapshot fixture body of chapter {number}. "
        )
        if number == 11:
            body += "The road ends here, until then.\n"
        parts.append(body)
    aa = "".join(parts).encode("utf-8")
    paragraphs = "".join(f'<p id="p{i}">Snapshot fixture paragraph {i}.</p>' for i in range(1, 42))
    opinion = (
        f"<html><body><article><h1>Opinion</h1>{paragraphs}</article></body></html>"
    ).encode()
    chapters = builder.slice_chapters(aa)
    opinion_text, _ = builder.extract_doctors_opinion(opinion)
    sections: list[dict[str, Any]] = [
        {
            "id": "doctors-opinion",
            "title": "The Doctor's Opinion",
            "source_id": "doctors-opinion",
            "paragraphs": [f"p{i}" for i in range(1, 42)],
            "text_sha256": builder.sha256(opinion_text.encode("utf-8")),
        }
    ]
    for chapter in chapters:
        text = str(chapter["text"])
        sections.append(
            {
                "id": f"chapter-{int(chapter['number'])}",
                "title": str(chapter["title"]),
                "source_id": "core-pages-1-164",
                "byte_start": int(chapter["byte_start"]),
                "byte_end": int(chapter["byte_end"]),
                "text_sha256": builder.sha256(text.encode("utf-8")),
            }
        )
    manifest = {
        "builder_version": 1,
        "sources": [
            {
                "id": "core-pages-1-164",
                "url": "https://example.invalid/AA.txt",
                "raw_path": "corpus/source/raw/AA.txt",
                "sha256": builder.sha256(aa),
                "bytes": len(aa),
            },
            {
                "id": "doctors-opinion",
                "url": "https://example.invalid/doctors-opinion",
                "raw_path": "corpus/source/raw/doctors-opinion.html",
                "sha256": builder.sha256(opinion),
                "bytes": len(opinion),
            },
        ],
        "sections": sections,
    }
    fetch_state = {
        "version": 1,
        "sources": [
            {
                "id": "core-pages-1-164",
                "url": "https://example.invalid/AA.txt",
                "path": "corpus/source/raw/AA.txt",
                "bytes": len(aa),
                "sha256": builder.sha256(aa),
            },
            {
                "id": "doctors-opinion",
                "url": "https://example.invalid/doctors-opinion",
                "path": "corpus/source/raw/doctors-opinion.html",
                "bytes": len(opinion),
                "sha256": builder.sha256(opinion),
            },
        ],
    }
    artifact = builder.build_canonical(
        manifest=manifest, fetch_state=fetch_state, aa_bytes=aa, opinion_html=opinion
    )
    payload = builder.serialize_artifact(artifact)
    return payload, manifest


def _run_script(path: pathlib.Path, args: list[str], *, env_extra: dict[str, str]) -> Any:
    env = dict(__import__("os").environ)
    env.update(env_extra)
    # Ensure the ephemeral identity never leaks via a stale secret.
    return subprocess.run(
        [sys.executable, str(path), *args],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
        env=env,
        check=False,
    )


def test_age_round_trip_with_ephemeral_keypair() -> None:
    identity, recipient = age_v1.generate_identity()
    try:
        assert identity.startswith("AGE-SECRET-KEY-")
        assert recipient.startswith("age1")
        plaintext = b"snapshot fixture plaintext " * 5000
        encrypted = age_v1.encrypt_bytes(plaintext, [recipient])
        assert encrypted.startswith(b"age-encryption.org/v1\n")
        assert plaintext not in encrypted
        assert identity.encode() not in encrypted
        assert age_v1.decrypt_bytes(encrypted, [identity]) == plaintext
        assert age_v1.decrypt_bytes(age_v1.encrypt_bytes(b"", [recipient]), [identity]) == b""
        big = b"Q" * (3 * 64 * 1024 + 17)
        assert age_v1.decrypt_bytes(age_v1.encrypt_bytes(big, [recipient]), [identity]) == big
    finally:
        identity = "destroyed"  # noqa: F841
        recipient = "destroyed"  # noqa: F841


def test_age_wrong_identity_and_tampering_fail_closed() -> None:
    identity, recipient = age_v1.generate_identity()
    other_identity, _ = age_v1.generate_identity()
    try:
        encrypted = age_v1.encrypt_bytes(b"closed fixture", [recipient])
        with pytest.raises(AgeError):
            age_v1.decrypt_bytes(encrypted, [other_identity])
        tampered = bytearray(encrypted)
        tampered[-10] ^= 0xFF
        with pytest.raises(AgeError):
            age_v1.decrypt_bytes(bytes(tampered), [identity])
        header_end = encrypted.find(b"\n--- ")
        assert header_end > 0
        mac_tampered = bytearray(encrypted)
        mac_tampered[header_end + 5] ^= 0x01
        with pytest.raises(AgeError):
            age_v1.decrypt_bytes(bytes(mac_tampered), [identity])
    finally:
        identity = "destroyed"  # noqa: F841
        other_identity = "destroyed"  # noqa: F841


def test_tar_zst_is_deterministic_and_validated() -> None:
    payload, manifest = _fixture_canonical_payload()
    first = snap.create_tar_zst(payload, manifest=manifest, source_lock={"version": 2})
    second = snap.create_tar_zst(payload, manifest=manifest, source_lock={"version": 2})
    assert first == second
    canonical_bytes, provenance = snap.extract_tar_zst(first)
    assert canonical_bytes == payload
    assert provenance["canonical_sha256"] == hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError):
        snap.extract_tar_zst(b"not a zst archive")


def test_refresh_and_restore_round_trip_with_ephemeral_keypair(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, recipient = age_v1.generate_identity()
    try:
        payload, fixture_manifest = _fixture_canonical_payload()
        canonical_path = tmp_path / "canonical.json"
        canonical_path.write_bytes(payload)
        manifest = {
            "format": "aa-canonical-manifest/1",
            "builder_version": 1,
            "edition": "fixture",
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        lock_path = tmp_path / "lock.json"
        lock_path.write_text(json.dumps({"version": 2, "edition": "fixture"}), encoding="utf-8")
        encrypted_dir = tmp_path / "encrypted"
        recipient_path = tmp_path / "recipient.txt"
        recipient_path.write_text(recipient + "\n", encoding="utf-8")
        identity_path = tmp_path / "identity.txt"
        identity_path.write_text(identity + "\n", encoding="utf-8")
        monkeypatch.delenv("AA_BOOK_AGE_IDENTITY", raising=False)

        refresh = _repo_root() / "scripts" / "refresh_encrypted_snapshot.py"
        proc = _run_script(
            refresh,
            [
                "--manifest",
                str(manifest_path),
                "--source-lock",
                str(lock_path),
                "--canonical",
                str(canonical_path),
                "--encrypted-dir",
                str(encrypted_dir),
                "--recipient-file",
                str(recipient_path),
                "--identity-file",
                str(identity_path),
            ],
            env_extra={},
        )
        assert proc.returncode == 0, proc.stderr
        assert "decrypt verification ok" in proc.stdout
        # Logs carry digests only: no plaintext body, no private identity.
        assert identity not in proc.stdout
        assert "Snapshot fixture body" not in proc.stdout
        archive = encrypted_dir / "canonical.tar.zst.age"
        metadata_path = encrypted_dir / "metadata.json"
        assert archive.exists()
        assert payload not in archive.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["canonical_sha256"] == hashlib.sha256(payload).hexdigest()
        assert metadata["encrypted_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
        assert metadata["encryption_format"] == snap.ENCRYPTION_FORMAT

        restore = _repo_root() / "scripts" / "restore_canonical.py"
        output = tmp_path / "generated" / "canonical.json"
        proc = _run_script(
            restore,
            [
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
                "--encrypted-dir",
                str(encrypted_dir),
                "--identity-file",
                str(identity_path),
                "--no-network-fallback",
            ],
            env_extra={},
        )
        assert proc.returncode == 0, proc.stderr
        assert output.read_bytes() == payload
        assert identity not in proc.stdout
        assert "Snapshot fixture body" not in proc.stdout

        # Reuse: a second restore without identity/network must reuse the copy.
        output_stat = output.stat().st_mtime_ns
        proc = _run_script(
            restore,
            [
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
                "--encrypted-dir",
                str(encrypted_dir),
                "--no-network-fallback",
            ],
            env_extra={},
        )
        assert proc.returncode == 0, proc.stderr
        assert "reused" in proc.stdout
        assert output.stat().st_mtime_ns == output_stat
    finally:
        identity = "destroyed"  # noqa: F841


def test_restore_fails_closed_without_snapshot_or_network(tmp_path: pathlib.Path) -> None:
    payload, _ = _fixture_canonical_payload()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "aa-canonical-manifest/1",
                "builder_version": 1,
                "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    restore = _repo_root() / "scripts" / "restore_canonical.py"
    proc = _run_script(
        restore,
        [
            "--manifest",
            str(manifest_path),
            "--output",
            str(tmp_path / "out.json"),
            "--encrypted-dir",
            str(tmp_path / "missing-encrypted"),
            "--no-network-fallback",
        ],
        env_extra={"AA_BOOK_AGE_IDENTITY": "", "AA_ALLOW_NETWORK_FETCH": ""},
    )
    assert proc.returncode != 0


def test_provision_generates_valid_pair_without_storing_identity(
    tmp_path: pathlib.Path,
) -> None:
    provision = _repo_root() / "scripts" / "age_provision.py"
    recipient_path = tmp_path / "recipient.txt"
    # age_provision must refuse CI execution; simulate a local run.
    proc = _run_script(
        provision,
        ["--write-recipient", str(recipient_path)],
        env_extra={"CI": "", "GITHUB_ACTIONS": ""},
    )
    assert proc.returncode == 0, proc.stderr
    recipient = recipient_path.read_text(encoding="utf-8").strip()
    assert recipient.startswith("age1")
    # The identity is shown once for the operator but never written to disk.
    assert "AGE-SECRET-KEY-" in proc.stdout
    assert list(tmp_path.glob("*.key")) == []
    assert list(tmp_path.glob("*identity*")) == []


def test_no_private_key_or_plaintext_committed() -> None:
    root = _repo_root()
    tracked = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        cwd=root,
        check=True,
    ).stdout.splitlines()
    assert not [p for p in tracked if "AGE-SECRET-KEY" in p or p.endswith(".key")]
    assert not [
        p
        for p in tracked
        if p.startswith("corpus/source/encrypted/")
        and p.endswith((".tar.zst", ".json"))
        and p
        not in (
            "corpus/source/encrypted/README.md",
            "corpus/source/encrypted/metadata.json",
            "corpus/source/encrypted/metadata.ru.json",
        )
        and not p.endswith(".age")
    ]
    assert not [
        p
        for p in tracked
        if p.startswith("corpus/source/encrypted/")
        and p.endswith(".txt")
        and p != "corpus/source/encrypted/recipient.txt"
    ]
    for path in tracked:
        if (
            path.startswith("corpus/source/encrypted/")
            and path.endswith(".age")
            and path
            not in (
                "corpus/source/encrypted/canonical.tar.zst.age",
                "corpus/source/encrypted/canonical.ru.tar.zst.age",
            )
        ):
            raise AssertionError(f"unexpected committed snapshot fixture: {path}")
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "/corpus/generated/" in gitignore
    assert "/corpus/source/raw/" in gitignore
    assert "/corpus/source/raw-ru/" in gitignore
    assert "AGE-SECRET-KEY" in gitignore


def test_en_ru_share_single_recipient() -> None:
    """EN (#24) and RU (#50) share one committed recipient; no second key.

    The public recipient is provisioned once by the repository owner in #28
    and is absent until then, so this test skips pre-activation while still
    forbidding any second recipient file.
    """
    root = _repo_root()
    recipient_path = root / "corpus" / "source" / "encrypted" / "recipient.txt"
    tracked = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        cwd=root,
        check=True,
    ).stdout.splitlines()
    recipient_files = [p for p in tracked if "recipient" in p.lower()]
    assert recipient_files in (
        [],
        ["corpus/source/encrypted/recipient.txt"],
    ), f"EN and RU must share one recipient file, found: {recipient_files}"
    if not recipient_path.is_file():
        pytest.skip("no provisioned recipient yet (production activation is tracked in #28)")
    text = recipient_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    lines = text.splitlines()
    assert len(lines) == 1
    recipient = lines[0].strip()
    assert recipient.startswith("age1")
    age_v1.parse_recipient(recipient)
    # One recipient filename constant is shared by both snapshots.
    assert snap.RECIPIENT_NAME == "recipient.txt"
    assert snap.ARCHIVE_NAME == "canonical.tar.zst.age"
    assert snap.RU_ARCHIVE_NAME == "canonical.ru.tar.zst.age"
    assert snap.RU_ARCHIVE_NAME != snap.ARCHIVE_NAME
    for workflow in (
        "encrypted-corpus-refresh.yml",
        "encrypted-corpus-refresh-ru.yml",
    ):
        workflow_text = (root / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert "corpus/source/encrypted/recipient.txt" in workflow_text
        lowered = workflow_text.lower()
        assert "recipient-ru" not in lowered
        assert "recipient_ru" not in lowered
    russian_doc = (root / "docs" / "russian-corpus.md").read_text(encoding="utf-8")
    assert "no second" in russian_doc.lower()


def test_refresh_workflow_is_manual_trusted_and_cache_safe() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "encrypted-corpus-refresh.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_dispatch" in workflow
    assert "pull_request" not in workflow
    assert "AA_BOOK_AGE_IDENTITY" in workflow
    assert "actions/cache" not in workflow
    # The default token must not be given secret administration.
    assert "administration" not in workflow.lower()
