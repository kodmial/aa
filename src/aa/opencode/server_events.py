"""Bounded passive server-side evidence for the local OpenCode server (kodmial/aa#453).

The client-side attempt ledger and the #448 ``StallSnapshotObserver`` only
prove local HTTP requests. This module adds the second vantage point the
outer watchdog needs before it kills the live lane: what the pinned
``opencode serve`` process itself reports about its sessions while a
request is in flight.

Probed against the pinned local ``opencode serve`` binary
(docker/opencode.version). Both the v1 surface retained for compatibility
and the native v2 surface are probed, v2 first:

- ``GET /openapi.json`` (v2) or ``GET /doc`` (v1) — documented OpenAPI
  surface (availability proof only);
- ``GET /api/session/active`` (v2: foreground drains owned by this OpenCode
  process) or ``GET /session/status`` (v1: closed vocabulary
  ``idle``/``busy``/``retry`` plus numeric ``attempt``);
- ``GET /api/event`` (v2) or ``GET /event`` / ``GET /global/event`` (v1) —
  bounded passive read of the SSE stream, extracting only the documented
  session lifecycle event types, fixed-vocabulary error names and numeric
  provider status codes. v2 ``Session.StructuredError`` is reduced to the
  same closed vocabulary; its message/body text is never retained.

Everything else is deliberately dropped: session ids stay in memory (or
never leave the parsed payload at all), event payload text, provider error
messages and response bodies are never retained. A session state, health
OK, or the process being alive is NEVER token/provider progress — every
snapshot carries ``progress_meaning: "none"``.

When the event surface is unavailable, broken, or of an unexpected shape,
the evidence explicitly reports ``unsupported``/``disconnected``/
``unobserved`` instead of ``server_idle`` or a fictitious provider stall.
The v2 server requires Basic authentication on these routes; when a
password is configured it is sent only in the request header and is never
stored in evidence.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import urllib.error
import urllib.request
from typing import Any

SCHEMA_SERVER_EVIDENCE = "aa-opencode-server-evidence/1"
SERVER_EVIDENCE_SOURCE = "opencode-server-events"

# Evidence of the availability of the documented local-only surface.
SERVER_STATES = ("supported", "unsupported", "disconnected", "unobserved", "unavailable")

# Closed vocabulary from the pinned server's own OpenAPI (SessionStatus and
# EventSessionError error union). Anything else is never reported as a
# state — schema drift therefore shows as unobserved, not as an idle lie.
SESSION_STATE_VOCAB = ("idle", "busy", "retry")
SESSION_ERROR_NAME_VOCAB = (
    "ProviderAuthError",
    "UnknownError",
    "MessageOutputLengthError",
    "MessageAbortedError",
    "StructuredOutputError",
    "ContextOverflowError",
    "ContentFilterError",
    "APIError",
)
SESSION_EVENT_VOCAB = ("session.status", "session.idle", "session.error")
# v2 reports terminal execution failures as their own durable events.
SESSION_V2_FAILURE_EVENT_VOCAB = (
    "session.execution.failed",
    "session.step.failed",
    "session.compaction.failed",
    "session.retry.scheduled",
)
SESSION_V2_EVENT_VOCAB = (
    *SESSION_EVENT_VOCAB,
    *SESSION_V2_FAILURE_EVENT_VOCAB,
    "session.execution.started",
    "session.execution.succeeded",
    "session.execution.interrupted",
)

# Native v2 routes are tried first, then the v1 routes retained for the
# compatibility client. A route that answers with the SPA HTML shell is not
# an event/status surface and never counts as support.
DOC_ENDPOINTS: tuple[str, ...] = ("/openapi.json", "/doc")
SESSION_STATUS_ENDPOINTS: tuple[str, ...] = ("/api/session/active", "/session/status")
EVENT_ENDPOINTS: tuple[str, ...] = ("/api/event", "/event", "/global/event")

# v2 ``/api/session/active`` only lists foreground Session drains currently
# owned by the process. The closed evidence vocabulary keeps the same
# counters while mapping the documented ``running`` shape to ``busy``.
SESSION_ACTIVE_STATE_MAP = {"running": "busy"}

_MAX_BYTES = 0x1000
_DEFAULT_TIMEOUT_S = 1.0


def unavailable_evidence(reason: str) -> dict[str, Any]:
    """Typed fail-closed snapshot when no server-side evidence exists."""
    return {
        "schema": SCHEMA_SERVER_EVIDENCE,
        "source": SERVER_EVIDENCE_SOURCE,
        "state": "unobserved",
        "reason": str(reason or "unknown")[:48],
        "progress_meaning": "none",
    }


def _classify_error(exc: BaseException) -> str:
    """Fixed-vocabulary classification of one probe failure."""
    if isinstance(exc, urllib.error.HTTPError):
        return "http-error"
    if isinstance(exc, urllib.error.URLError) or isinstance(exc, TimeoutError):
        return "disconnected"
    return "probe-error"


# Failure facts of one ``/session/status`` read mapped to typed verdicts.
_SESSION_STATUS_FAILURE_TOKENS = (
    ("session-status-http-error", "http-error"),
    ("session-status-body-timeout", "body-timeout"),
    ("session-status-body-error", "body-error"),
    ("session-status-shape", "shape"),
    ("session-status-error", "disconnected"),
)


def _session_status_failure(facts: dict[str, int]) -> str:
    """Typed failure token for one read ("" when this read succeeded)."""
    for key, token in _SESSION_STATUS_FAILURE_TOKENS:
        if int(facts.get(key, 0) or 0) > 0:
            return token
    return ""


class OpenCodeServerEventProbe:
    """Bounded passive reader of local server-side session evidence."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        password: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._password = str(password or "")
        self._doc_state = ""
        self._doc_endpoint = ""
        self._session_status_endpoint = ""
        self._event_endpoint = ""
        self._session_state_counts: dict[str, int] = {name: 0 for name in SESSION_STATE_VOCAB}
        self._last_session_state = ""
        self._retry_attempt = -1
        self._event_type_counts: dict[str, int] = {}
        self._server_error_counts: dict[str, int] = {}
        self._last_server_error_kind = ""
        self._last_server_error_status = -1
        self._last_server_error_retryable: bool | None = None
        self._last_observed_s = 0.0
        self._last_evidence: dict[str, Any] = unavailable_evidence("never-observed")

    # ------------------------------------------------------------------
    # Bounded network reads (block in threads via ``snapshot()``)
    # ------------------------------------------------------------------

    def _auth_headers(self, accept: str) -> dict[str, str]:
        """Request headers; Basic auth is sent but never retained."""
        headers = {"Accept": accept}
        if self._password:
            token = base64.b64encode(f"opencode:{self._password}".encode()).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        return headers

    def _read_doc(self) -> tuple[str, int]:
        """Probe the documented OpenAPI surface; closed-vocabulary result.

        v2 serves ``/openapi.json`` and the SPA shell on legacy ``/doc``; v1
        serves ``/doc``. A JSON document with a ``paths`` map is the only
        accepted proof, so the HTML shell can never be mistaken for support.
        """
        last_status = 0
        for endpoint in DOC_ENDPOINTS:
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(
                        f"{self._base_url}{endpoint}",
                        headers=self._auth_headers("application/json"),
                    ),
                    timeout=self._timeout_s,
                ) as response:
                    status = int(response.status)
                    read_payload = response.read(1024 * 1024)
            except urllib.error.HTTPError as exc:
                last_status = int(exc.code)
                continue
            except Exception:
                return "disconnected", 0
            if status != 200:
                last_status = status
                continue
            try:
                payload = json.loads(read_payload.decode("utf-8", errors="replace"))
            except ValueError:
                last_status = status
                continue
            if not isinstance(payload, dict) or not isinstance(payload.get("paths"), dict):
                last_status = status
                continue
            self._doc_endpoint = endpoint
            return "supported", status
        return "unsupported", last_status

    def _read_session_status_at(self, endpoint: str) -> dict[str, int]:
        """Read one session-activity route and reduce it to bounded counts."""
        counts = {"session-status": 0, "unknown-state": 0}
        response: Any = None
        try:
            response = urllib.request.urlopen(
                urllib.request.Request(
                    f"{self._base_url}{endpoint}",
                    headers=self._auth_headers("application/json"),
                ),
                timeout=self._timeout_s,
            )
        except urllib.error.HTTPError as exc:
            return {**counts, "session-status-http-error": int(exc.code)}
        except Exception:
            # Only a failure to establish the connection is "disconnected".
            # Once the server has answered, any later failure below is
            # schema drift / an unobserved shape, never a dead connection.
            return {**counts, "session-status-error": 1}
        try:
            status = int(response.status)
            body = response.read(_MAX_BYTES)
        except TimeoutError:
            return {**counts, "session-status-body-timeout": 1}
        except Exception:
            return {**counts, "session-status-body-error": 1}
        finally:
            try:
                response.close()
            except Exception:
                pass
        if status != 200:
            return {**counts, "session-status-http-error": int(status)}
        try:
            payload = json.loads(body.decode("utf-8", errors="replace"))
        except ValueError:
            return {**counts, "session-status-shape": 1}
        if not isinstance(payload, dict):
            return {**counts, "session-status-shape": 1}
        # v2 wraps the active-session map in ``data``; v1 is the bare map.
        mapping = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        if not isinstance(mapping, dict):
            return {**counts, "session-status-shape": 1}
        counts["session-status"] = len(mapping)
        for value in mapping.values():
            if not isinstance(value, dict):
                counts["unknown-state"] += 1
                continue
            kind = str(value.get("type", ""))[:24]
            kind = SESSION_ACTIVE_STATE_MAP.get(kind, kind)
            if kind in SESSION_STATE_VOCAB:
                self._session_state_counts[kind] = self._session_state_counts.get(kind, 0) + 1
                self._last_session_state = kind
                if kind == "retry":
                    attempt = value.get("attempt")
                    if isinstance(attempt, int):
                        self._retry_attempt = attempt
                counts[kind] = counts.get(kind, 0) + 1
            else:
                counts["unknown-state"] += 1
        return counts

    def _read_session_status(self) -> dict[str, int]:
        """Poll the session activity surface; counts only closed-vocabulary states.

        Session ids (the mapping keys) are never retained; per-session
        facts collapse into aggregate counters. The native v2
        ``/api/session/active`` route is tried first, then the v1
        ``/session/status`` route. An HTML shell or absent route falls
        through; an authentication or body failure stays fail-closed.
        """
        last: dict[str, int] = {"session-status": 0, "unknown-state": 0}
        self._session_status_endpoint = ""
        for endpoint in SESSION_STATUS_ENDPOINTS:
            facts = self._read_session_status_at(endpoint)
            last = facts
            http_error = int(facts.get("session-status-http-error", 0) or 0)
            if http_error in (404, 405):
                continue
            if "session-status-shape" in facts:
                # The SPA HTML shell (legacy routes on v2) parses as the
                # wrong shape; try the next documented route.
                continue
            if http_error or any(
                key in facts
                for key in (
                    "session-status-error",
                    "session-status-body-timeout",
                    "session-status-body-error",
                )
            ):
                return facts
            self._session_status_endpoint = endpoint
            return facts
        return last

    async def session_status_snapshot(self) -> dict[str, Any]:
        """Light poll of ``GET /session/status`` for the #448 observer probe.

        Runs one bounded loopback request in a thread, updates the same
        closed-vocabulary counters as :meth:`snapshot`, and returns the
        bounded verdict payload that the observer persists next to each
        stall snapshot. The observer only invokes this when writing a
        non-responding attempt snapshot, so the request rate stays tiny.
        No session id, status payload body or server log line is retained.
        """
        try:
            facts = await asyncio.to_thread(self._read_session_status)
        except Exception as exc:  # noqa: BLE001 - probe must never break the scan
            payload: dict[str, Any] = {
                "observed": False,
                "verdict": "server-events-unobserved",
                "error": type(exc).__name__[:64],
                "progress_meaning": "none",
            }
            return payload
        return self.session_status_probe_payload(facts)

    def _read_sse_window_at(self, endpoint: str) -> dict[str, int]:
        """Read one bounded SSE window from one candidate endpoint."""
        facts: dict[str, int] = {}
        response: Any = None
        read_state = ""
        try:
            response = urllib.request.urlopen(
                urllib.request.Request(
                    f"{self._base_url}{endpoint}",
                    headers=self._auth_headers("text/event-stream"),
                ),
                timeout=self._timeout_s,
            )
            facts["event-stream-status"] = int(response.status)
            content_type = str(response.headers.get("Content-Type", ""))[:64]
            facts["event-stream-streamed"] = int("event-stream" in content_type)
            if not facts["event-stream-streamed"]:
                # v2 serves the SPA shell on legacy v1 routes. A non-SSE
                # 200 is an absent surface, not a quiet stream.
                facts["event-stream-not-streamed"] = 1
                read_state = "not-streamed"
            else:
                read_state = self._consume_sse_frames(response)
        except urllib.error.HTTPError as exc:
            facts["event-stream-http-error"] = exc.code
        except TimeoutError:
            # A timeout after the stream opened is a quiet window; before
            # that it is a connection-level failure.
            if response is not None and "event-stream-status" in facts:
                read_state = "quiet"
            else:
                facts["event-stream-error"] = 1
        except Exception:
            # A failure before the stream was opened (or before its headers
            # arrived) is a connection-level failure.
            facts["event-stream-error"] = 1
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
        # A live-but-silent stream is a normal SSE condition, not a broken
        # connection: keep it distinct from ``event-stream-error``.
        if read_state == "quiet":
            facts["event-stream-quiet"] = 1
        elif read_state == "error":
            facts["event-stream-error"] = 1
        return facts

    def _read_sse_window(self) -> dict[str, int]:
        """Read a bounded window of the SSE event stream.

        Only closed-vocabulary facts are extracted: the documented event
        type tokens, the error union name (fixed vocabulary), the numeric
        ``statusCode`` and the boolean ``isRetryable``. Raw payload text,
        messages and ids are discarded. Retained facts survive only in
        this probe's bounded counters.

        The pinned server may expose either ``/event`` or ``/global/event``.
        A missing (404/405) or failed route falls through to the next
        candidate; a route that opened and answered is authoritative. When
        no candidate answers, the last typed failure is returned so the
        evidence stays fail-closed instead of becoming a fictitious idle.
        """
        last: dict[str, int] = {}
        self._event_endpoint = ""
        for endpoint in EVENT_ENDPOINTS:
            facts = self._read_sse_window_at(endpoint)
            last = facts
            status = int(facts.get("event-stream-status", 0) or 0)
            http_error = int(facts.get("event-stream-http-error", 0) or 0)
            if status == 200 and not facts.get("event-stream-not-streamed"):
                self._event_endpoint = endpoint
                return facts
            if facts.get("event-stream-not-streamed"):
                # SPA shell, not an SSE surface: try the next route.
                continue
            if http_error in (404, 405):
                continue
            if "event-stream-error" in facts:
                continue
            return facts
        return last

    def _read_sse_bytes(self, stream: Any) -> tuple[bytes, str]:
        """Read one bounded SSE window without waiting for a full buffer.

        ``HTTPResponse.read(n)`` blocks on a chunked SSE response until it
        collects ``n`` bytes or reaches EOF. A live pinned server sends one
        frame (for example ``server.connected``) and then keeps the stream
        open, so the old fixed-size read timed out and discarded the frame
        that had already arrived. Per-chunk reads preserve every observed
        byte; a timeout *after* a partial read is a successful bounded window,
        while a timeout with no bytes stays an honest quiet stream.
        """
        window = bytearray()
        timed_out = False
        failed = False
        while len(window) < _MAX_BYTES:
            reader = getattr(stream, "read1", None)
            try:
                if callable(reader):
                    chunk = reader(min(0x400, _MAX_BYTES - len(window)))
                else:
                    chunk = stream.read(min(0x400, _MAX_BYTES - len(window)))
            except TimeoutError:
                timed_out = True
                break
            except Exception:
                failed = True
                break
            if not chunk:
                break
            window.extend(chunk)
        if window:
            self._last_observed_s = time.time()
            return bytes(window), "ok"
        if failed:
            return b"", "error"
        return b"", "quiet" if timed_out else "ok"

    def _consume_sse_frames(self, stream: Any) -> str:
        """Parse one bounded read window of SSE ``data:`` frames.

        Returns ``"ok"`` when any bounded read completed (including EOF),
        ``"quiet"`` when a healthy open stream produced no bytes before the
        socket timeout, and ``"error"`` for a read failure before any byte.
        """
        window, state = self._read_sse_bytes(stream)
        if state != "ok":
            return state
        for raw_line in window.decode("utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            try:
                payload = json.loads(line[len("data:") :].strip())
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            event_type = str(payload.get("type", ""))[:48]
            if event_type in SESSION_V2_EVENT_VOCAB:
                self._event_type_counts[event_type] = self._event_type_counts.get(event_type, 0) + 1
            else:
                self._event_type_counts["unrecognized-event"] = (
                    self._event_type_counts.get("unrecognized-event", 0) + 1
                )
                continue
            # v1 carries ``properties``; v2 carries ``data``. Only the
            # closed-vocabulary subset below is ever read.
            properties: Any = payload.get("properties")
            if not isinstance(properties, dict):
                data = payload.get("data")
                properties = data if isinstance(data, dict) else {}
            if event_type == "session.status":
                status = properties.get("status")
                if isinstance(status, dict):
                    kind = str(status.get("type", ""))[:24]
                    if kind in SESSION_STATE_VOCAB:
                        self._session_state_counts[kind] = (
                            self._session_state_counts.get(kind, 0) + 1
                        )
                        self._last_session_state = kind
                        attempt = status.get("attempt")
                        if kind == "retry" and isinstance(attempt, int):
                            self._retry_attempt = attempt
                continue
            if event_type == "session.idle":
                self._session_state_counts["idle"] = self._session_state_counts.get("idle", 0) + 1
                self._last_session_state = "idle"
                continue
            if event_type in SESSION_V2_FAILURE_EVENT_VOCAB or event_type == "session.error":
                # Session ids are never retained; only the documented,
                # fixed-vocabulary error name and numeric status code.
                error = properties.get("error")
                kind = "unknown"
                if isinstance(error, dict):
                    name = str(error.get("name") or error.get("type") or "")[:64]
                    kind = name if name in SESSION_ERROR_NAME_VOCAB else "unknown"
                    data = error.get("data")
                    code: Any = None
                    retryable: Any = None
                    if isinstance(data, dict):
                        code = data.get("statusCode")
                        retryable = data.get("isRetryable")
                    else:
                        # v2 ``Session.StructuredError`` keeps ``status`` at
                        # the top level. ``retryAfterMs`` is not inferred as
                        # a retryability claim.
                        code = error.get("status")
                    if isinstance(code, int):
                        self._last_server_error_status = code
                    if isinstance(retryable, bool):
                        self._last_server_error_retryable = retryable
                self._last_server_error_kind = kind
                self._server_error_counts[kind] = self._server_error_counts.get(kind, 0) + 1
        return "ok"

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _probe_sync(self) -> tuple[str, dict[str, Any]]:
        doc_state, doc_status = self._read_doc()
        self._doc_state = doc_state
        session_facts = self._read_session_status()
        sse_facts = self._read_sse_window()
        # Fail-closed state resolution: supported only with the documented
        # OpenAPI surface; a connection-level failure (no HTTP answer) is
        # ``disconnected``; everything else is an unsupported shape, never
        # a fictitious idle observation.
        http_error_code = int(
            sse_facts.get("event-stream-http-error", 0)
            or session_facts.get("session-status-http-error", 0)
        )
        connection_failed = (
            "session-status-error" in session_facts or "event-stream-error" in sse_facts
        )
        if doc_state == "supported":
            state = "supported"
        elif http_error_code:
            state = "unsupported"
        elif connection_failed:
            state = "disconnected"
        else:
            state = "unsupported" if sse_facts else "unobserved"
        evidence: dict[str, Any] = {
            "schema": SCHEMA_SERVER_EVIDENCE,
            "source": SERVER_EVIDENCE_SOURCE,
            "state": state,
            "progress_meaning": "none",
            "doc_endpoint": self._doc_endpoint or "unobserved",
            "session_status_endpoint": self._session_status_endpoint or "unobserved",
            "event_endpoint": self._event_endpoint or "unobserved",
            "endpoints": {
                "/doc": {"http_status": int(doc_status), "state": doc_state},
                "/session/status": {str(k)[:48]: int(v) for k, v in session_facts.items()},
                self._event_endpoint or "/event": {
                    str(k)[:48]: int(v) for k, v in sse_facts.items()
                },
            },
            "session_state_counts": {
                **{
                    name: int(self._session_state_counts.get(name, 0))
                    for name in SESSION_STATE_VOCAB
                },
            },
            "event_type_counts": {
                str(k)[:48]: int(v) for k, v in sorted(self._event_type_counts.items()) if k
            }
            or {},
            "uncertainty": {
                "server_alive_or_busy_is_not_provider_progress": True,
            },
        }
        if self._last_session_state:
            evidence["last_session_state"] = self._last_session_state
        if self._last_session_state == "retry" and self._retry_attempt >= 0:
            evidence["last_retry_attempt"] = int(self._retry_attempt)
        if self._last_server_error_kind:
            evidence["server_error_kind"] = self._last_server_error_kind
            evidence["server_error_counts"] = {
                str(k)[:48]: int(v) for k, v in self._server_error_counts.items()
            }
            if self._last_server_error_status >= 0:
                # Only published when a real event actually carried it.
                evidence["server_error_status_code"] = int(self._last_server_error_status)
            if self._last_server_error_retryable is not None:
                evidence["server_error_is_retryable"] = self._last_server_error_retryable
        if self._last_observed_s:
            evidence["last_stream_observed_s"] = round(self._last_observed_s, 1)
        return state, evidence

    async def snapshot(self) -> dict[str, Any]:
        """One bounded privacy-safe server-side evidence snapshot."""
        try:
            _, evidence = await asyncio.to_thread(self._probe_sync)
            self._last_evidence = evidence
            return evidence
        except Exception as exc:  # noqa: BLE001 - evidence must never break the lane
            snapshot = unavailable_evidence(f"probe-error:{type(exc).__name__}")
            snapshot["state"] = "unavailable"
            return snapshot

    def cached_evidence(self) -> dict[str, Any]:
        """The last computed evidence snapshot (fail-closed when never probed).

        Used by synchronous lane callbacks (stage transitions) that must
        not perform blocking network work: they reuse the most recent
        probe result, or the explicit ``unobserved`` marker when no probe
        ever ran. Absence of a fresh network look stays honest here —
        nothing is reported as ``server_idle`` from a cached health state.
        """
        return self._last_evidence

    @property
    def last_session_state(self) -> str:
        """Latest closed-vocabulary session state seen ("" when none)."""
        return self._last_session_state

    def session_status_probe_payload(self, facts: dict[str, int] | None = None) -> dict[str, Any]:
        """Bounded verdict payload for the #448 observer's probe seam.

        When ``facts`` carries the outcome of the current read, a failed
        read is reported as unobserved with its typed failure token rather
        than reusing the last previously successful cached state.
        """
        if facts is not None:
            failure = _session_status_failure(facts)
            if failure:
                return {
                    "observed": False,
                    "verdict": f"server-events-session-{failure}"[:64],
                    "error": failure[:64],
                    "progress_meaning": "none",
                }
            has_state = any(int(facts.get(name, 0) or 0) > 0 for name in SESSION_STATE_VOCAB)
            state = self._last_session_state if has_state else ""
        else:
            state = self._last_session_state
        observed = state in SESSION_STATE_VOCAB
        return {
            "observed": observed,
            "verdict": (
                f"server-events-session-{state}" if observed else "server-events-session-unobserved"
            )[:64],
            "error": "",
            "progress_meaning": "none",
        }


def probe_for_runtime(runtime: object) -> OpenCodeServerEventProbe | None:
    """Return a cached probe for one runtime (None when no base URL exists)."""
    cached = getattr(runtime, "_server_event_probe", None)
    if isinstance(cached, OpenCodeServerEventProbe):
        return cached
    config = getattr(runtime, "config", None)
    base_url = str(getattr(config, "base_url", "") or "") if config is not None else ""
    if not base_url:
        return None
    # The native v2 evidence routes require the same Basic authentication as
    # every other v2 call. Prefer the runtime's provisioned password, then
    # the already-authenticated client. Neither value enters evidence.
    password = str(getattr(runtime, "_auth_password", "") or "")
    if not password:
        client = getattr(runtime, "client", None)
        password = str(getattr(client, "_password", "") or "")
    probe = OpenCodeServerEventProbe(base_url, password=password)
    try:
        runtime._server_event_probe = probe  # type: ignore[attr-defined]
    except Exception:
        return probe
    return probe


async def evidence_for_runtime(runtime: object) -> dict[str, Any]:
    """Bounded server-side evidence snapshot for one runtime (fail-closed)."""
    probe = probe_for_runtime(runtime)
    if probe is None:
        return unavailable_evidence("no-runtime-server-endpoint")
    return await probe.snapshot()


__all__ = [
    "DOC_ENDPOINTS",
    "EVENT_ENDPOINTS",
    "SCHEMA_SERVER_EVIDENCE",
    "SESSION_ACTIVE_STATE_MAP",
    "SESSION_STATUS_ENDPOINTS",
    "SERVER_EVIDENCE_SOURCE",
    "SERVER_STATES",
    "SESSION_ERROR_NAME_VOCAB",
    "SESSION_EVENT_VOCAB",
    "SESSION_STATE_VOCAB",
    "OpenCodeServerEventProbe",
    "evidence_for_runtime",
    "probe_for_runtime",
    "unavailable_evidence",
]
