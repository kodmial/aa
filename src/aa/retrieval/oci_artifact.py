"""Minimal OCI distribution client for the persistent derived index (issue #429).

The encrypted derived retrieval bundle is published as an immutable
GHCR OCI artifact (L2) and pulled directly by digest, so a GitHub Actions
cache eviction (L1) can no longer cost a full E5 re-embedding. This module
implements the small registry-v2 subset AA needs with Python stdlib only:

- Bearer-token authentication driven by the standard ``WWW-Authenticate``
  challenge (realm host must belong to the registry domain, fail closed);
- manifest push with idempotent no-op when the exact same manifest digest
  already exists at the content-addressed tag, and blob-existence checks
  plus an in-process publish lock so parallel publishers do not duplicate
  identical content-addressed uploads;
- digest-pinned manifest pull where every blob and the manifest bytes
  themselves are verified against their sha256 digests before writing.

Security properties: only the encrypted package plus non-secret metadata
enters the artifact; the manifest reference must be an immutable sha256
digest (mutable tags never authorize a pull); credentials are read from the
environment and never logged; transport failures raise :class:`ArtifactError`
with a safe category, never a partial restore (files are written atomically
only after full verification).

Uses the same OCI media types as ``oras``; no external tooling is required.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"
CONFIG_MEDIA_TYPE = "application/vnd.oci.empty.v1+json"
BUNDLE_LAYER_MEDIA_TYPE = "application/vnd.aa.derived-index.bundle.v1"
ARTIFACT_TYPE = "application/vnd.aa.derived-index.artifact.v1"
FILE_NAME_ANNOTATION = "aa.derived-index.file-name.v1"
DEFAULT_DIGEST_FILE = "docker/aa-index.digest"

DIGEST_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
ZERO_DIGEST = "sha256:" + "0" * 64
LAYER_FILE_RE = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
MAX_LAYER_BYTES = 1024 * 1024 * 1024
_FLOW_ATTEMPTS = 3
# Serializes the manifest HEAD -> blob upload -> manifest PUT sequence in
# this process. Cross-process publishers remain idempotent because every
# object is content-addressed and blob existence is checked before upload.
_PUBLISH_LOCK = threading.Lock()
# Explicit, reviewed token services for registries that delegate auth to a
# host outside their own DNS subtree (GHCR currently advertises ghcr.io/token,
# but this keeps documented alternates working without accepting arbitrary
# challenge hosts).
_KNOWN_TOKEN_HOSTS: dict[str, frozenset[str]] = {
    "ghcr.io": frozenset({"ghcr.io", "token.actions.githubusercontent.com"}),
    "registry-1.docker.io": frozenset({"auth.docker.io"}),
}


class ArtifactError(ValueError):
    """Raised when the L2 OCI artifact cannot be published or restored.

    ``category`` is a stable, privacy-safe reason code for workflow logs;
    details never contain credentials, decrypted content, or book text.
    """

    def __init__(self, category: str, detail: str = "") -> None:
        message = f"L2 artifact failure [{category}]"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class OCIRegistry:
    """Resolved registry plus repository identity for GCR-style endpoints.

    ``host`` is ``ghcr.io`` in production; offline tests run an HTTP fixture.
    """

    host: str
    repository: str
    scheme: str = "https"

    def base_url(self) -> str:
        # Contract: no trailing slash, so callers can append a rooted path
        # (``/v2/...``) without producing a doubled separator.
        return f"{self.scheme}://{self.host.rstrip('/')}"

    def api_url(self, path: str) -> str:
        return f"{self.base_url()}/v2/{self.repository}{path}"


def is_pinned_digest(reference: str) -> bool:
    """Return True only for a real immutable ``sha256:<64-hex>`` reference."""
    match = DIGEST_RE.fullmatch(reference.strip())
    return bool(match) and reference.strip() != ZERO_DIGEST


def digest_bytes(data: bytes) -> str:
    """Return the ``sha256:...`` digest of bytes."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    return f"http {exc.code} {exc.reason}"


@dataclass
class _Challenge:
    realm: str
    service: str
    scope: str


def _parse_challenge(header: str, *, repository: str) -> _Challenge | None:
    """Parse ``WWW-Authenticate: Bearer realm=...,service=...,scope=...``.

    kodmial/aa#462: values are quoted, and a quoted value may itself carry
    commas (GHCR scopes such as ``"repository:org/pkg:push,pull"``), so
    the field separator is only a comma outside double quotes.
    """
    if not header.strip().lower().startswith("bearer"):
        return None
    body = header.strip()[len("bearer") :].strip()
    fields: dict[str, str] = {}
    for part in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', body):
        key_raw, sep, value_raw = part.partition("=")
        if not sep:
            continue
        key = key_raw.strip().lower()
        value = value_raw.strip().strip('"')
        fields[key] = value
    realm = fields.get("realm", "")
    if not realm:
        return None
    return _Challenge(
        realm=realm,
        service=fields.get("service", ""),
        scope=fields.get("scope", "").replace("$repository", repository),
    )


class _AuthSession:
    """Carries the bearer token after the first 401 challenge is resolved."""

    def __init__(
        self,
        registry: OCIRegistry,
        *,
        username: str | None,
        password: str | None,
    ) -> None:
        self._registry = registry
        self._username = username
        self._password = password
        self._token: str | None = None
        self._token_expiry: float = 0.0
        self._token_scope: str | None = None
        self._challenge: _Challenge | None = None
        self._token_lock = threading.Lock()

    def credentials_header(self) -> str | None:
        if self._username is None or self._password is None:
            return None
        raw = self._username + ":" + self._password
        encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
        return "Basic " + encoded

    def _realm_host_allowed(self, host: str) -> bool:
        """Accept the registry host and its subdomains (GHCR token service).

        GHCR advertises ``https://ghcr.io/token`` today, but registries may
        delegate token issuance to a sibling subdomain. Restricting the
        realm to the registry's own domain keeps the challenge fail-closed
        against SSRF-style redirects to unrelated hosts.
        """
        registry_host = self._registry.host.lower()
        candidate = host.lower()
        allowed = _KNOWN_TOKEN_HOSTS.get(registry_host, frozenset())
        return (
            candidate == registry_host
            or candidate in allowed
            or candidate.endswith("." + registry_host)
        )

    def _token_url(self, challenge: _Challenge, scope: str) -> str:
        parsed = urllib.parse.urlparse(challenge.realm)
        if parsed.scheme not in {"https", "http"}:
            raise ArtifactError("l2-authentication", "invalid token realm scheme")
        if not self._realm_host_allowed(parsed.netloc):
            raise ArtifactError("l2-authentication", "token realm host does not match the registry")
        # kodmial/aa#462: the WWW-Authenticate challenge scope is
        # authoritative. GHCR answers HTTP 403 when the token request uses
        # an abstract capability such as ``scope=pull``; only the fully
        # qualified documented scope (``repository:<name>:<actions>``
        # asserted by the challenge itself) is valid at its token realm.
        # The requested fallback scope is used only when a challenge
        # carries no scope at all.
        params: dict[str, str] = {"scope": challenge.scope or scope}
        if challenge.service:
            params["service"] = challenge.service
        query = urllib.parse.urlencode(params)
        separator = "&" if parsed.query else "?"
        return f"{challenge.realm}{separator}{query}"

    def _fetch_token(self, challenge: _Challenge, scope: str) -> str:
        url = self._token_url(challenge, scope)
        headers: dict[str, str] = {}
        basic = self.credentials_header()
        if basic is not None:
            headers["Authorization"] = basic
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactError(
                "l2-authentication", f"token fetch failed: {type(exc).__name__}"
            ) from exc
        raw = payload.get("token") if isinstance(payload, dict) else None
        if not isinstance(raw, str) or not raw:
            raise ArtifactError("l2-authentication", "token response carried no token")
        expires_in = payload.get("expires_in", 300) if isinstance(payload, dict) else 300
        try:
            lifetime = float(expires_in)
        except (TypeError, ValueError):
            # GHCR does not always advertise expiry; assume a conservative
            # default rather than caching the token indefinitely.
            lifetime = 300.0
        if lifetime <= 0:
            lifetime = 300.0
        margin = min(30.0, lifetime / 2.0)
        self._token_expiry = time.time() + lifetime - margin
        self._challenge = challenge
        self._token_scope = scope
        return raw

    def _ensure_valid_token(
        self,
        challenge: _Challenge,
        scope: str,
        *,
        stale_token: str | None = None,
    ) -> str:
        """Return a live token, refreshing once under the token lock.

        ``stale_token`` is the token the caller just saw rejected. If another
        thread already replaced it, reuse that fresh token instead of causing
        a thundering herd of token fetches.
        """
        with self._token_lock:
            if (
                self._token is not None
                and time.time() < self._token_expiry
                and self._token != stale_token
            ):
                return self._token
            token = self._fetch_token(challenge, scope)
            self._token = token
            return token

    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        auth_scope: str | None = None,
        max_redirects: int = 2,
        timeout_s: float = 300.0,
    ) -> tuple[int, dict[str, str], bytes]:
        """Issue one request, transparently resolving the 401 bearer challenge."""
        attempt_headers = dict(headers or {})
        for _hop in range(max_redirects + 2):
            request_headers = dict(attempt_headers)
            if (
                auth_scope is not None
                and self._challenge is not None
                and self._token_scope == auth_scope
                and self._token is not None
                and time.time() >= self._token_expiry
            ):
                # Refresh proactively so a long upload never starts with an
                # already-expired token.
                self._token = self._ensure_valid_token(self._challenge, auth_scope)
            if self._token is not None:
                request_headers.setdefault("Authorization", "Bearer " + self._token)
            sent_token = self._token
            request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=timeout_s) as response:
                    body = response.read() if method != "HEAD" else b""
                    return int(response.status), {k: v for k, v in response.headers.items()}, body
            except urllib.error.HTTPError as exc:
                if exc.code == 401 and auth_scope is not None:
                    challenge_raw = exc.headers.get("WWW-Authenticate", "")
                    challenge = _parse_challenge(
                        challenge_raw, repository=self._registry.repository
                    )
                    if challenge is not None:
                        self._token = self._ensure_valid_token(
                            challenge, auth_scope, stale_token=sent_token
                        )
                        continue
                return int(exc.code), {k: v for k, v in exc.headers.items()}, exc.read()
            except (urllib.error.URLError, OSError) as exc:
                raise ArtifactError("l2-registry-unavailable", str(exc.__class__.__name__)) from exc
            except (UnicodeDecodeError, ValueError) as exc:
                raise ArtifactError(
                    "l2-manifest-invalid", f"response unreadable: {type(exc).__name__}"
                ) from exc
        raise ArtifactError("l2-registry-unavailable", "too many redirects")


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def expected_l2_tag(cache_digest: str) -> str:
    """Return the content-addressed OCI tag for a derived-cache digest.

    The tag never carries a run id, date, or source SHA: identical index
    inputs converge to one immutable artifact that parallel publish runs
    idempotently share.
    """
    if not DIGEST_RE.fullmatch("sha256:" + cache_digest):
        raise ArtifactError("l2-digest-invalid", "cache digest is malformed")
    return f"idx-{cache_digest}"


def read_pinned_reference(digest_file: str | Path) -> str:
    """Return the last non-empty line of the committed L2 pin file or ''."""
    try:
        lines = Path(digest_file).read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return ""
    return lines[-1].strip() if lines else ""


def parse_pinned_digest(digest_file: str | Path) -> str:
    """Return the pinned digest or raise a folded category (fail closed)."""
    try:
        text = Path(digest_file).read_text(encoding="utf-8").strip().splitlines()
    except OSError as exc:
        raise ArtifactError(
            "l2-unpublished", f"digest file unreadable: {type(exc).__name__}"
        ) from exc
    reference = text[-1].strip() if text else ""
    registry_repo, sep, digest = reference.partition("@")
    if sep and "/" in registry_repo:
        reference = digest
    if not DIGEST_RE.fullmatch(reference):
        raise ArtifactError("l2-unpublished", "digest file is not an immutable digest pin")
    if reference == ZERO_DIGEST:
        raise ArtifactError("l2-unpublished", "L2 publication pin is still a placeholder")
    return reference


def push_artifact(
    registry: OCIRegistry,
    *,
    files: dict[str, bytes],
    tag: str,
    session: _AuthSession,
    annotations: dict[str, str] | None = None,
) -> str:
    """Publish files as one immutable OCI artifact; idempotent per exact content.

    Returns the immutable manifest digest. When the identical manifest
    already exists at ``tag`` the registry is not modified again.
    """
    for name, payload in files.items():
        if not LAYER_FILE_RE.fullmatch(name):
            raise ArtifactError("l2-push-failed", f"file name rejected: {name!r}")
        if len(payload) > MAX_LAYER_BYTES:
            raise ArtifactError("l2-push-failed", f"file too large: {name}")
    config_payload = _canonical_json(
        {
            "architecture": "unknown",
            "os": "unknown",
            "rootfs": {"type": "layers", "diff_ids": []},
        }
    )
    config_digest = digest_bytes(config_payload)
    layers: list[dict[str, Any]] = []
    for name in sorted(files):
        payload = files[name]
        layers.append(
            {
                "mediaType": BUNDLE_LAYER_MEDIA_TYPE,
                "size": len(payload),
                "digest": digest_bytes(payload),
                "annotations": {
                    FILE_NAME_ANNOTATION: name,
                    "aa.derived-index.file-sha256.v1": hashlib.sha256(payload).hexdigest(),
                },
            }
        )
    manifest_payload: dict[str, Any] = {
        "schemaVersion": 2,
        "mediaType": MANIFEST_MEDIA_TYPE,
        "artifactType": ARTIFACT_TYPE,
        "config": {
            "mediaType": CONFIG_MEDIA_TYPE,
            "size": len(config_payload),
            "digest": config_digest,
        },
        "layers": layers,
    }
    if annotations:
        manifest_payload["annotations"] = {
            str(key): str(value) for key, value in sorted(annotations.items())
        }
    manifest_bytes = _canonical_json(manifest_payload)
    manifest_digest = digest_bytes(manifest_bytes)

    with _PUBLISH_LOCK:
        head_status, head_headers, _ = session.request(
            registry.api_url(f"/manifests/{tag}"),
            method="HEAD",
            headers={"Accept": MANIFEST_MEDIA_TYPE},
            auth_scope="pull",
        )
        if head_status == 200 and head_headers.get("Docker-Content-Digest") == manifest_digest:
            return manifest_digest

        _upload_blob(registry, session, config_payload, config_digest)
        for layer in layers:
            name = layer["annotations"][FILE_NAME_ANNOTATION]
            payload = files[name]
            _upload_blob(registry, session, payload, layer["digest"])

        status, headers, _ = session.request(
            registry.api_url(f"/manifests/{tag}"),
            method="PUT",
            data=manifest_bytes,
            headers={"Content-Type": MANIFEST_MEDIA_TYPE},
            auth_scope="push,pull",
        )
        if status not in (201, 202):
            raise ArtifactError("l2-push-failed", f"manifest push failed with http {status}")
        location_digest = headers.get("Docker-Content-Digest", manifest_digest)
        if location_digest != manifest_digest:
            raise ArtifactError("l2-push-failed", "registry assigned an unexpected manifest digest")
        return manifest_digest


def _blob_exists(registry: OCIRegistry, session: _AuthSession, digest: str) -> bool:
    status, _, _ = session.request(
        registry.api_url(f"/blobs/{digest}"),
        method="HEAD",
        auth_scope="pull",
    )
    return status == 200


def _upload_blob(registry: OCIRegistry, session: _AuthSession, blob: bytes, digest: str) -> None:
    # Content-addressed blobs are shared by all publishers. Skipping an
    # already-present blob removes the duplicate bandwidth that two
    # concurrent publishers would otherwise spend on identical uploads.
    if _blob_exists(registry, session, digest):
        return
    status, headers, _ = session.request(
        registry.api_url("/blobs/uploads/"), method="POST", auth_scope="push,pull", data=b""
    )
    if status != 202:
        raise ArtifactError("l2-push-failed", f"blob upload rejected with http {status}")
    location: str = headers.get("Location", "")
    if not location:
        raise ArtifactError("l2-push-failed", "blob upload carried no upload location")
    for _hop in range(4):
        # GHCR may return a registry-relative Location (RFC 7231), not an
        # absolute URL. Resolve only a single-slash path against this registry;
        # scheme-relative locations must not redirect uploads to another host.
        if location.startswith("/") and not location.startswith("//"):
            location = registry.base_url() + location
        separator = "&" if "?" in location else "?"
        put_url = f"{location}{separator}digest={urllib.parse.quote(digest)}"
        status, headers, _ = session.request(
            put_url,
            method="PUT",
            data=blob,
            headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(blob))},
            auth_scope="push,pull",
        )
        if status in (201, 202, 200):
            return
        if status == 307 and headers.get("Location"):
            location = headers["Location"]
            continue
        break
    raise ArtifactError("l2-push-failed", f"blob upload failed with http {status}")


def pull_artifact(
    registry: OCIRegistry,
    *,
    reference: str,
    session: _AuthSession,
    dest_dir: str | Path,
    expect_files: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Pull a digest-pinned artifact into ``dest_dir`` after full verification.

    Fails closed: the reference must be an immutable digest, the manifest
    bytes must hash to it, every blob must hash to its layer digest, and
    only the exact expected file set is written (atomically, only after all
    verification passed).
    """
    if not is_pinned_digest(reference):
        raise ArtifactError("l2-digest-invalid", "pull reference is not an immutable digest")
    digest = reference
    status, headers, body = session.request(
        registry.api_url(f"/manifests/{digest}"),
        method="GET",
        headers={"Accept": MANIFEST_MEDIA_TYPE},
        auth_scope="pull",
    )
    if status == 404:
        raise ArtifactError("l2-manifest-missing", "pinned manifest does not exist")
    if status == 403:
        # kodmial/aa#462: an authenticated-but-denied pull is a package ACL
        # result, not a transport failure; report the exact safe category.
        raise ArtifactError("l2-access-denied", "registry denied the manifest pull: http 403")
    if status != 200:
        raise ArtifactError("l2-registry-unavailable", f"manifest get failed with http {status}")
    if digest_bytes(body) != digest:
        raise ArtifactError("l2-digest-mismatch", "manifest bytes do not match the pinned digest")
    try:
        manifest = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            "l2-manifest-invalid", f"manifest JSON invalid: {type(exc).__name__}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 2:
        raise ArtifactError("l2-manifest-invalid", "manifest is not an OCI image manifest")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ArtifactError("l2-manifest-invalid", "manifest carries no layers")
    files: dict[str, bytes] = {}
    for layer in layers:
        if not isinstance(layer, dict):
            raise ArtifactError("l2-manifest-invalid", "manifest layer is malformed")
        layer_digest = str(layer.get("digest", ""))
        if not is_pinned_digest(layer_digest):
            raise ArtifactError("l2-manifest-invalid", "layer digest is not an immutable digest")
        annotations = layer.get("annotations")
        if not isinstance(annotations, dict):
            raise ArtifactError("l2-manifest-invalid", "layer carries no file annotations")
        name = str(annotations.get(FILE_NAME_ANNOTATION, ""))
        if not LAYER_FILE_RE.fullmatch(name):
            raise ArtifactError("l2-manifest-invalid", f"layer file name rejected: {name!r}")
        blob_status, _, blob = session.request(
            registry.api_url(f"/blobs/{layer_digest}"),
            method="GET",
            auth_scope="pull",
        )
        if blob_status == 403:
            raise ArtifactError("l2-access-denied", "registry denied the blob pull: http 403")
        if blob_status != 200:
            raise ArtifactError(
                "l2-blob-missing", f"blob {layer_digest[:20]} failed with http {blob_status}"
            )
        if len(blob) > MAX_LAYER_BYTES or digest_bytes(blob) != layer_digest:
            raise ArtifactError(
                "l2-blob-digest-mismatch", f"blob {layer_digest[:20]} failed digest check"
            )
        if name in files:
            raise ArtifactError("l2-manifest-invalid", f"duplicate layer for {name!r}")
        files[name] = blob
    if expect_files is not None and sorted(files) != sorted(expect_files):
        raise ArtifactError(
            "l2-manifest-invalid",
            "pinned artifact file set does not match the expected bundle",
        )
    directory = Path(dest_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name in sorted(files):
        target = directory / name
        tmp_path = target.with_name(target.name + ".l2tmp")
        tmp_path.write_bytes(files[name])
        if tmp_path.read_bytes() != files[name]:
            tmp_path.unlink(missing_ok=True)
            raise ArtifactError("l2-restore-write", f"verified write failed for {name!r}")
        tmp_path.replace(target)
        written[name] = target
    return written


def open_session(
    registry: OCIRegistry,
    *,
    username: str | None,
    password: str | None,
) -> _AuthSession:
    """Create an authenticated session wrapper (credentials stay in memory)."""
    return _AuthSession(registry, username=username, password=password)
