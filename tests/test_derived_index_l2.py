"""L2 OCI artifact store tests: fake local registry, digest-pin enforcement.

Covers issue #429: the persistent derived index layer must roundtrip the
encrypted bundle through a registry-v2 artifact with fail-closed digest
checks, rejected placeholder pins, and idempotent publication.
"""

from __future__ import annotations

import http.server
import json
import pathlib
import socket
import subprocess
import sys
import threading
import typing
import uuid

import pytest

from aa.corpus import age_v1
from aa.retrieval import derived_cache as dc
from aa.retrieval.oci_artifact import (
    ArtifactError,
    OCIRegistry,
    digest_bytes,
    expected_l2_tag,
    open_session,
    parse_pinned_digest,
    pull_artifact,
    push_artifact,
)

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "tests"))

from test_derived_cache import _build_fixture_index  # type: ignore[import-not-found]  # noqa: E402


def _server_port(server: typing.Any) -> int:
    return int(server.server_port)


def _server_state(server: typing.Any) -> dict[str, typing.Any]:
    return typing.cast("dict[str, typing.Any]", server.state)


class _FakeRegistryHandler(http.server.BaseHTTPRequestHandler):
    """Minimal registry-v2 subset with bearer auth and digest checks."""

    def log_message(self, *args: object) -> None:  # noqa: D102
        pass

    def _auth_ok(self) -> bool:
        if _server_state(self.server).get("auth_reject"):
            # Simulated authentication failure: API requests still carry the
            # standard bearer challenge, but the token endpoint itself
            # refuses, so the session can never resolve a usable token.
            self.send_response(401)
            host = f"http://127.0.0.1:{_server_port(self.server)}"
            self.send_header(
                "WWW-Authenticate",
                f'Bearer realm="{host}/token",service="fake",scope="repository:atest:pull"',
            )
            self.end_headers()
            return False
        header = self.headers.get("Authorization", "")
        if header.startswith("Basic ") or header.startswith("Bearer "):
            return True
        self.send_response(401)
        host = f"http://127.0.0.1:{_server_port(self.server)}"
        self.send_header(
            "WWW-Authenticate",
            f'Bearer realm="{host}/token",service="fake",scope="repository:atest:push,pull"',
        )
        self.end_headers()
        return False

    def _token(self) -> None:
        payload = json.dumps({"token": "fake-token", "expires_in": 300}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path.startswith("/token"):
            # The token endpoint rejects in the simulated auth-failure mode
            # (plain 401 without a follow-up challenge).
            if _server_state(self.server).get("auth_reject"):
                self.send_response(401)
                self.end_headers()
                return
            self._token()
            return
        if not self._auth_ok():
            return
        state = _server_state(self.server)
        if "/blobs/" in self.path:
            digest = self.path.rsplit("/", 1)[-1]
            blob = state["blobs"].get(digest)
            if blob is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)
            return
        if "/manifests/" in self.path:
            reference = self.path.rsplit("/", 1)[-1]
            entry = state["manifests_by_tag"].get(reference) or state["manifests_by_digest"].get(
                reference
            )
            if entry is None:
                self.send_response(404)
                self.end_headers()
                return
            manifest_bytes, manifest_digest = entry
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.oci.image.manifest.v1+json")
            self.send_header("Docker-Content-Digest", manifest_digest)
            self.send_header("Content-Length", str(len(manifest_bytes)))
            self.end_headers()
            self.wfile.write(manifest_bytes)
            return
        self.send_response(404)
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        if not self._auth_ok():
            return
        if not self.path.rstrip("/").endswith("/blobs/uploads"):
            self.send_response(404)
            self.end_headers()
            return
        host = f"http://127.0.0.1:{_server_port(self.server)}"
        upload_id = uuid.uuid4().hex
        self.send_response(202)
        location = f"/v2/atest/blobs/uploads/{upload_id}"
        if not getattr(self.server, "relative_upload_location", False):
            location = host + location
        self.send_header("Location", location)
        self.end_headers()

    def do_PUT(self) -> None:
        if not self._auth_ok():
            return
        state = _server_state(self.server)
        path = self.path
        if "/blobs/uploads/" in path:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            digest = digest_bytes(body)
            state["blobs"][digest] = body
            self.send_response(201)
            self.send_header("Docker-Content-Digest", digest)
            self.end_headers()
            return
        if "/manifests/" in path:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            digest = digest_bytes(body)
            reference = path.rsplit("/", 1)[-1]
            state["manifests_by_tag"][reference] = (body, digest)
            state["manifests_by_digest"][digest] = (body, digest)
            self.send_response(201)
            self.send_header("Docker-Content-Digest", digest)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()


class _RegistryFixture:
    def __init__(self) -> None:
        self.state: dict[str, typing.Any] = {
            "blobs": {},
            "manifests_by_tag": {},
            "manifests_by_digest": {},
        }
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeRegistryHandler)
        self.server.state = self.state  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = int(_server_port(self.server))
        self.registry = OCIRegistry(
            host=f"127.0.0.1:{self.port}", repository="kodmial/aa-index", scheme="http"
        )
        self.session = open_session(self.registry, username="user", password="testcred")

    def shutdown(self) -> None:
        self.server.shutdown()


@pytest.fixture()
def fake_registry() -> typing.Iterator[_RegistryFixture]:
    fixture = _RegistryFixture()
    try:
        yield fixture
    finally:
        fixture.shutdown()


_MANAGER_SCRIPT = _REPO / "scripts" / "manage_derived_cache.py"


def _run_restore_l2(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_MANAGER_SCRIPT), "--mode", "restore-l2", *args],
        capture_output=True,
        text=True,
        cwd=_REPO,
        check=False,
    )


def _seed_preexisting_l1(cache_dir: pathlib.Path, payload: bytes) -> dict[str, str]:
    """Create a preexisting L1 cache directory with known bytes (issue #461)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    encrypted = cache_dir / dc.ENCRYPTED_NAME
    meta = cache_dir / dc.META_NAME
    marker = cache_dir / "l1-marker.txt"
    encrypted.write_bytes(payload)
    meta.write_bytes(b'{"cache_key": "preexisting-marker"}\n')
    marker.write_bytes(b"l1-must-survive\n")
    return {
        dc.ENCRYPTED_NAME: dc.sha256_bytes(payload),
        dc.META_NAME: dc.sha256_bytes(meta.read_bytes()),
        "l1-marker.txt": dc.sha256_bytes(marker.read_bytes()),
    }


def _l1_dir_digested(cache_dir: pathlib.Path) -> dict[str, str]:
    return {
        name: dc.sha256_bytes(path.read_bytes())
        for name in sorted(path.name for path in cache_dir.iterdir())
        for path in (cache_dir / name,)
        if path.is_file()
    }


def _staging_clean_or_absent(staging_dir: pathlib.Path) -> bool:
    """Issue #461: staging cleanup leaves the directory empty (or created-back-to-absent)."""
    return (not staging_dir.exists()) or list(staging_dir.iterdir()) == []


def _encrypted_fixture(tmp_path: pathlib.Path) -> tuple[str, bytes, bytes, str]:
    """Build a valid hashing-backend bundle (plus age identity) for tests."""
    identity, recipient = age_v1.generate_identity()
    src = tmp_path / "src"
    _build_fixture_index(src)
    bindings = dc.read_bindings(_REPO, os_name="Linux", arch="X64")
    key = dc.derived_cache_key(bindings)
    digests = dc.bundle_file_digests(src)
    checked = dc.self_test_index(src)
    manifest = dc.build_cache_manifest(
        cache_key=key,
        bindings=bindings,
        file_digests=digests,
        chunk_count=int(checked["chunk_count"]),
        index_version=int(checked["index_version"]),
    )
    encrypted = dc.encrypt_package(dc.create_package(src, manifest), recipient)
    meta_payload = dc.build_meta_payload(
        cache_key=key,
        encrypted_sha256=dc.sha256_bytes(encrypted),
        encrypted_bytes=len(encrypted),
        bindings=bindings,
    )
    return (
        identity,
        encrypted,
        json.dumps(meta_payload).encode("utf-8"),
        key.digest,
    )


@pytest.mark.parametrize("relative_upload_location", [False, True])
def test_push_pull_roundtrip(
    fake_registry: _RegistryFixture, tmp_path: pathlib.Path, relative_upload_location: bool
) -> None:
    fake_registry.server.relative_upload_location = relative_upload_location  # type: ignore[attr-defined]
    identity, encrypted, meta, cache_digest = _encrypted_fixture(tmp_path)
    files = {dc.ENCRYPTED_NAME: encrypted, dc.META_NAME: meta}
    tag = expected_l2_tag(cache_digest)
    first_digest = push_artifact(
        fake_registry.registry, files=files, tag=tag, session=fake_registry.session
    )
    second_digest = push_artifact(
        fake_registry.registry, files=files, tag=tag, session=fake_registry.session
    )
    assert first_digest == second_digest
    dest = tmp_path / "restored"
    pulled = pull_artifact(
        fake_registry.registry,
        reference=first_digest,
        session=fake_registry.session,
        dest_dir=dest,
        expect_files=(dc.ENCRYPTED_NAME, dc.META_NAME),
    )
    assert sorted(p.name for p in pulled.values()) == [dc.META_NAME, dc.ENCRYPTED_NAME]


def test_push_refuses_bad_file_names(
    fake_registry: _RegistryFixture, tmp_path: pathlib.Path
) -> None:
    identity, encrypted, meta, cache_digest = _encrypted_fixture(tmp_path)
    files = {"bad name.txt": encrypted, dc.META_NAME: meta}
    with pytest.raises(ArtifactError) as excinfo:
        push_artifact(
            fake_registry.registry,
            files=files,
            tag=expected_l2_tag(cache_digest),
            session=fake_registry.session,
        )
    assert excinfo.value.category == "l2-push-failed"


def test_pull_rejects_tampered_blob(
    fake_registry: _RegistryFixture, tmp_path: pathlib.Path
) -> None:
    identity, encrypted, meta, cache_digest = _encrypted_fixture(tmp_path)
    files = {dc.META_NAME: meta, dc.ENCRYPTED_NAME: encrypted}
    digest = push_artifact(
        fake_registry.registry,
        files=files,
        tag=expected_l2_tag(cache_digest),
        session=fake_registry.session,
    )
    manifest_bytes, manifest_digest = _server_state(fake_registry.server)["manifests_by_digest"][
        digest
    ]
    manifest = json.loads(manifest_bytes)
    target_digest = manifest["layers"][0]["digest"]
    _server_state(fake_registry.server)["blobs"][target_digest] = b"tampered"
    dest = tmp_path / "restored"
    with pytest.raises(ArtifactError) as excinfo:
        pull_artifact(
            fake_registry.registry,
            reference=digest,
            session=fake_registry.session,
            dest_dir=dest,
        )
    assert excinfo.value.category in ("l2-blob-missing", "l2-blob-digest-mismatch")
    assert list(dest.iterdir() if dest.exists() else []) == []
    assert manifest_digest == digest  # manifest bytes stay verified


def test_pull_requires_digest_not_tag(
    fake_registry: _RegistryFixture, tmp_path: pathlib.Path
) -> None:
    identity, encrypted, meta, cache_digest = _encrypted_fixture(tmp_path)
    files = {dc.ENCRYPTED_NAME: encrypted, dc.META_NAME: meta}
    tag = expected_l2_tag(cache_digest)
    push_artifact(fake_registry.registry, files=files, tag=tag, session=fake_registry.session)
    with pytest.raises(ArtifactError) as excinfo:
        pull_artifact(
            fake_registry.registry,
            reference=tag,
            session=fake_registry.session,
            dest_dir=tmp_path / "restored",
        )
    assert excinfo.value.category == "l2-digest-invalid"


def test_parse_pinned_digest_fail_closed(tmp_path: pathlib.Path) -> None:
    placeholder = tmp_path / "placeholder.digest"
    placeholder.write_text("ghcr.io/kodmial/aa-index@sha256:" + "0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ArtifactError) as excinfo:
        parse_pinned_digest(placeholder)
    assert excinfo.value.category == "l2-unpublished"
    with pytest.raises(ArtifactError):
        parse_pinned_digest(tmp_path / "missing.digest")
    digest = "sha256:" + "a" * 64
    valid = tmp_path / "valid.digest"
    valid.write_text("ghcr.io/kodmial/aa-index@" + digest + "\n", encoding="utf-8")
    assert parse_pinned_digest(valid) == digest


def test_restore_l2_placeholder_pin_is_rejected(tmp_path: pathlib.Path) -> None:
    placeholder_pin = tmp_path / "placeholder.digest"
    placeholder_pin.write_text(
        "ghcr.io/kodmial/aa-index@sha256:" + "0" * 64 + "\n", encoding="utf-8"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "manage_derived_cache.py"),
            "--mode",
            "restore-l2",
            "--cache-dir",
            "unused-cache-dir",
            "--digest-file",
            str(placeholder_pin),
            "--require-backend",
            "e5",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO,
        check=False,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["status"] == "rejected"
    assert payload["reason"] == "l2-unpublished"


def test_restore_l2_roundtrip_via_registry(
    fake_registry: _RegistryFixture, tmp_path: pathlib.Path
) -> None:
    identity, encrypted, meta, cache_digest = _encrypted_fixture(tmp_path)
    files = {dc.ENCRYPTED_NAME: encrypted, dc.META_NAME: meta}
    tag = expected_l2_tag(cache_digest)
    digest = push_artifact(
        fake_registry.registry, files=files, tag=tag, session=fake_registry.session
    )
    identity_file = tmp_path / "identity.txt"
    identity_file.write_text(identity, encoding="utf-8")
    pin_file = tmp_path / "pin.digest"
    pin_file.write_text(
        f"127.0.0.1:{fake_registry.port}/kodmial/aa-index@{digest}\n", encoding="utf-8"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "manage_derived_cache.py"),
            "--mode",
            "restore-l2",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--identity-file",
            str(identity_file),
            "--digest-file",
            str(pin_file),
            "--oci-registry",
            f"127.0.0.1:{fake_registry.port}",
            "--oci-repository",
            "kodmial/aa-index",
            "--oci-scheme",
            "http",
            "--require-backend",
            "hashing",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "hit"
    assert payload["source"] == "l2"
    assert payload["l2_digest"] == digest
    # Restored production paths fail closed: the hashing fixture is not a
    # valid strict production (E5) hit.
    strict = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "manage_derived_cache.py"),
            "--mode",
            "restore-l2",
            "--cache-dir",
            str(tmp_path / "cache-strict"),
            "--identity-file",
            str(identity_file),
            "--digest-file",
            str(pin_file),
            "--oci-registry",
            f"127.0.0.1:{fake_registry.port}",
            "--oci-repository",
            "kodmial/aa-index",
            "--oci-scheme",
            "http",
            "--require-backend",
            "e5",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO,
        check=False,
    )
    assert strict.returncode == 0
    strict_payload = json.loads(strict.stdout)
    assert strict_payload["status"] == "rejected"
    assert strict_payload["reason"] == "wrong-backend"


# --- Issue #461: L2 staging isolation + L1 -> L2 -> rebuild ordering -------


def test_restore_l2_auth_and_transport_failures_never_touch_preexisting_l1(
    fake_registry: _RegistryFixture, tmp_path: pathlib.Path
) -> None:
    """Simulated L2 failures can never mutate, overwrite or delete L1."""
    identity, encrypted, meta, cache_digest = _encrypted_fixture(tmp_path)
    files = {dc.ENCRYPTED_NAME: encrypted, dc.META_NAME: meta}
    digest = push_artifact(
        fake_registry.registry,
        files=files,
        tag=expected_l2_tag(cache_digest),
        session=fake_registry.session,
    )
    identity_file = tmp_path / "identity.txt"
    identity_file.write_text(identity, encoding="utf-8")
    pin_file = tmp_path / "pin.digest"
    pin_file.write_text(
        f"127.0.0.1:{fake_registry.port}/kodmial/aa-index@{digest}\n", encoding="utf-8"
    )
    cache_dir = tmp_path / "l1-cache"
    before = _seed_preexisting_l1(cache_dir, payload=b"preexisting-l1-encrypted-bytes")
    # Seed stale bytes in the default staging sibling to prove it is the
    # exactly targeted (and cleaned) L2-owned path.
    default_staging = pathlib.Path(str(cache_dir) + ".l2-staging")
    default_staging.mkdir(parents=True, exist_ok=True)
    (default_staging / "stale-staging.bin").write_bytes(b"stale-staging-junk")

    # Simulated authentication failure (401 on every request incl. token).
    _server_state(fake_registry.server)["auth_reject"] = True
    auth = _run_restore_l2(
        "--cache-dir",
        str(cache_dir),
        "--identity-file",
        str(identity_file),
        "--digest-file",
        str(pin_file),
        "--oci-registry",
        f"127.0.0.1:{fake_registry.port}",
        "--oci-repository",
        "kodmial/aa-index",
        "--oci-scheme",
        "http",
        "--require-backend",
        "e5",
    )
    _server_state(fake_registry.server)["auth_reject"] = False

    assert auth.returncode == 0, auth.stderr
    auth_payload = json.loads(auth.stdout)
    assert auth_payload["status"] == "rejected"
    assert auth_payload["reason"] == "l2-authentication"
    assert auth_payload["source"] == "l2"
    assert _l1_dir_digested(cache_dir) == before
    # The default staging sibling is used and cleaned up; L1 stays intact.
    assert list(default_staging.iterdir()) == []

    # Simulated transport failure (registry port closed).
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    transport_pin = tmp_path / "transport-pin.digest"
    transport_pin.write_text(f"127.0.0.1:{dead_port}/kodmial/aa-index@{digest}\n", encoding="utf-8")
    transport = _run_restore_l2(
        "--cache-dir",
        str(cache_dir),
        "--identity-file",
        str(identity_file),
        "--digest-file",
        str(transport_pin),
        "--oci-registry",
        f"127.0.0.1:{dead_port}",
        "--oci-repository",
        "kodmial/aa-index",
        "--oci-scheme",
        "http",
        "--require-backend",
        "e5",
        "--staging-dir",
        str(tmp_path / "staging-explicit"),
    )
    assert transport.returncode == 0, transport.stderr
    transport_payload = json.loads(transport.stdout)
    assert transport_payload["status"] == "rejected"
    assert transport_payload["reason"] == "l2-registry-unavailable"
    assert _l1_dir_digested(cache_dir) == before
    staging_explicit = tmp_path / "staging-explicit"
    assert _staging_clean_or_absent(staging_explicit)


def test_restore_l2_bad_bundle_failures_never_touch_preexisting_l1(
    fake_registry: _RegistryFixture, tmp_path: pathlib.Path
) -> None:
    """Digest mismatch and decryption failure are isolated to L2 staging."""
    before_identity, encrypted, meta, cache_digest = _encrypted_fixture(tmp_path)
    files = {dc.ENCRYPTED_NAME: encrypted, dc.META_NAME: meta}
    digest = push_artifact(
        fake_registry.registry,
        files=files,
        tag=expected_l2_tag(cache_digest),
        session=fake_registry.session,
    )
    pin_file = tmp_path / "pin.digest"
    pin_file.write_text(
        f"127.0.0.1:{fake_registry.port}/kodmial/aa-index@{digest}\n", encoding="utf-8"
    )
    cache_dir = tmp_path / "l1-cache"
    before = _seed_preexisting_l1(cache_dir, payload=b"preexisting-l1-encrypted-bytes")

    # Digest-mismatch failure: tamper a blob server-side.
    manifest_bytes, _manifest_digest = _server_state(fake_registry.server)["manifests_by_digest"][
        digest
    ]
    manifest = json.loads(manifest_bytes)
    _server_state(fake_registry.server)["blobs"][manifest["layers"][0]["digest"]] = b"tampered"
    mismatch = _run_restore_l2(
        "--cache-dir",
        str(cache_dir),
        "--digest-file",
        str(pin_file),
        "--oci-registry",
        f"127.0.0.1:{fake_registry.port}",
        "--oci-repository",
        "kodmial/aa-index",
        "--oci-scheme",
        "http",
        "--require-backend",
        "e5",
        "--staging-dir",
        str(tmp_path / "staging-mismatch"),
    )
    assert mismatch.returncode == 0, mismatch.stderr
    mismatch_payload = json.loads(mismatch.stdout)
    assert mismatch_payload["status"] == "rejected"
    assert mismatch_payload["reason"] in ("l2-blob-missing", "l2-blob-digest-mismatch")
    assert _l1_dir_digested(cache_dir) == before
    assert _staging_clean_or_absent(tmp_path / "staging-mismatch")

    # Decryption failure: the artifact holds a bundle for another identity.
    package = dc.decrypt_package(encrypted, before_identity)
    other_identity, other_recipient = age_v1.generate_identity()
    try:
        other_encrypted = dc.encrypt_package(package, other_recipient)
        other_meta = json.dumps(
            dc.build_meta_payload(
                cache_key=dc.derived_cache_key(
                    dc.read_bindings(_REPO, os_name="Linux", arch="X64")
                ),
                encrypted_sha256=dc.sha256_bytes(other_encrypted),
                encrypted_bytes=len(other_encrypted),
                bindings=dc.read_bindings(_REPO, os_name="Linux", arch="X64"),
            )
        ).encode("utf-8")
    finally:
        other_identity = "destroyed"  # noqa: F841
        other_recipient = "destroyed"  # noqa: F841
    alt_digest = push_artifact(
        fake_registry.registry,
        files={dc.ENCRYPTED_NAME: other_encrypted, dc.META_NAME: other_meta},
        tag=expected_l2_tag(cache_digest),
        session=fake_registry.session,
    )
    identity_file = tmp_path / "fixture-identity.txt"
    identity_file.write_text(before_identity, encoding="utf-8")
    alt_pin = tmp_path / "alt-pin.digest"
    alt_pin.write_text(
        f"127.0.0.1:{fake_registry.port}/kodmial/aa-index@{alt_digest}\n", encoding="utf-8"
    )
    decrypt_fail = _run_restore_l2(
        "--cache-dir",
        str(cache_dir),
        "--identity-file",
        str(identity_file),
        "--digest-file",
        str(alt_pin),
        "--oci-registry",
        f"127.0.0.1:{fake_registry.port}",
        "--oci-repository",
        "kodmial/aa-index",
        "--oci-scheme",
        "http",
        "--require-backend",
        "e5",
        "--staging-dir",
        str(tmp_path / "staging-decrypt"),
    )
    assert decrypt_fail.returncode == 0, decrypt_fail.stderr
    decrypt_payload = json.loads(decrypt_fail.stdout)
    assert decrypt_payload["status"] == "rejected"
    assert decrypt_payload["reason"] == "decrypt-failed"
    assert _l1_dir_digested(cache_dir) == before
    assert _staging_clean_or_absent(tmp_path / "staging-decrypt")


def test_restore_l2_hit_keeps_preexisting_l1_and_cleans_staging(
    fake_registry: _RegistryFixture, tmp_path: pathlib.Path
) -> None:
    """A verified L2 hit writes the staging/retrieval layers only."""
    identity, encrypted, meta, cache_digest = _encrypted_fixture(tmp_path)
    digest = push_artifact(
        fake_registry.registry,
        files={dc.ENCRYPTED_NAME: encrypted, dc.META_NAME: meta},
        tag=expected_l2_tag(cache_digest),
        session=fake_registry.session,
    )
    identity_file = tmp_path / "identity.txt"
    identity_file.write_text(identity, encoding="utf-8")
    pin_file = tmp_path / "pin.digest"
    pin_file.write_text(
        f"127.0.0.1:{fake_registry.port}/kodmial/aa-index@{digest}\n", encoding="utf-8"
    )
    cache_dir = tmp_path / "l1-cache"
    before = _seed_preexisting_l1(cache_dir, payload=b"unrelated-preexisting-bytes")
    retrieval_dir = tmp_path / "retrieval"
    staging_dir = tmp_path / "explicit-staging"
    hit = _run_restore_l2(
        "--cache-dir",
        str(cache_dir),
        "--staging-dir",
        str(staging_dir),
        "--retrieval-dir",
        str(retrieval_dir),
        "--identity-file",
        str(identity_file),
        "--digest-file",
        str(pin_file),
        "--oci-registry",
        f"127.0.0.1:{fake_registry.port}",
        "--oci-repository",
        "kodmial/aa-index",
        "--oci-scheme",
        "http",
        "--require-backend",
        "hashing",
    )
    assert hit.returncode == 0, hit.stderr
    hit_payload = json.loads(hit.stdout)
    assert hit_payload["status"] == "hit"
    assert hit_payload["source"] == "l2"
    assert hit_payload["l2_digest"] == digest
    # Original L1 bytes are never mutated even by a verified L2 hit.
    assert _l1_dir_digested(cache_dir) == before
    # Staging is cleaned; the validated bundle lives in the retrieval dir.
    assert list(staging_dir.iterdir()) == []
    assert (retrieval_dir / "index.json").is_file()
    assert (retrieval_dir / "lexical.db").is_file()
    assert dc.self_test_index(retrieval_dir)["chunk_count"] > 0


def test_publish_l2_refuses_non_e5_backend(tmp_path: pathlib.Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "manage_derived_cache.py"),
            "--mode",
            "publish-l2",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--require-backend",
            "hashing",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO,
        check=False,
    )
    assert proc.returncode == 1
    assert "only the strict production E5 backend" in proc.stderr
