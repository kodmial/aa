"""Bounded passive server-side evidence for the local OpenCode server (kodmial/aa#453).

The client-side attempt ledger and the #448 ``StallSnapshotObserver`` only
prove local HTTP requests. This module adds the second vantage point the
outer watchdog needs before it kills the live lane: what the pinned
``opencode serve`` process itself reports about its sessions while a
request is in flight.

Probed against the pinned version ``1.19.3`` (docker/opencode.version):

- ``GET /doc`` — documented OpenAPI surface (availability proof only);
- ``GET /session/status`` — poll-only map of session states with the
  closed vocabulary ``idle``/``busy``/``retry`` (+ numeric ``attempt``);
- ``GET /event`` — bounded passive read of the SSE stream, extracting only
  the ``session.status`` / ``session.error`` event types, the fixed error
  names of the documented error union and numeric ``statusCode`` /
  ``isRetryable`` fields.

Everything else is deliberately dropped: session ids stay in memory (or
never leave the parsed payload at all), event payload text, provider error
messages and response bodies are never retained. A session state, health
OK, or the process being alive is NEVER token/provider progress — every
snapshot carries ``progress_meaning: "none"``.

When the event surface is unavailable, broken, or of an unexpected shape,
the evidence explicitly reports ``unsupported``/``disconnected``/
``unobserved`` instead of ``server_idle`` or a fictitious provider stall.
"""

from __future__ import annotations

import asyncio
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


class OpenCodeServerEventProbe:
    """Bounded passive reader of local server-side session evidence."""

    def __init__(self, base_url: str, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._doc_state = ""
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

    def _read_doc(self) -> tuple[str, int]:
        """Probe ``GET /doc``; returns ``(state, http_status)`` closed-vocab."""
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    f"{self._base_url}/doc", headers={"Accept": "application/json"}
                ),
                timeout=self._timeout_s,
            ) as response:
                status = int(response.status)
                read_payload = response.read(1024 * 1024)
            if status != 200:
                return "unsupported", status
            try:
                payload = json.loads(read_payload.decode("utf-8", errors="replace"))
            except ValueError:
                return "unsupported", status
            if not isinstance(payload, dict) or "paths" not in payload:
                return "unsupported", status
            return "supported", status
        except urllib.error.HTTPError as exc:
            return "unsupported", int(exc.code)
        except Exception:
            return "disconnected", 0

    def _read_session_status(self) -> dict[str, int]:
        """Poll ``GET /session/status``; counts only closed-vocabulary states.

        Session ids (the mapping keys) are never retained; per-session
        facts collapse into aggregate counters.
        """
        counts = {"session-status": 0, "unknown-state": 0}
        response: Any = None
        try:
            response = urllib.request.urlopen(
                urllib.request.Request(
                    f"{self._base_url}/session/status",
                    headers={"Accept": "application/json"},
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
        counts["session-status"] = len(payload)
        for value in payload.values():
            if not isinstance(value, dict):
                counts["unknown-state"] += 1
                continue
            kind = str(value.get("type", ""))[:24]
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
            await asyncio.to_thread(self._read_session_status)
        except Exception as exc:  # noqa: BLE001 - probe must never break the scan
            payload: dict[str, Any] = {
                "observed": False,
                "verdict": "server-events-unobserved",
                "error": type(exc).__name__[:64],
                "progress_meaning": "none",
            }
            return payload
        return self.session_status_probe_payload()

    def _read_sse_window(self) -> dict[str, int]:
        """Read a bounded window of the SSE event stream.

        Only closed-vocabulary facts are extracted: the documented event
        type tokens, the error union name (fixed vocabulary), the numeric
        ``statusCode`` and the boolean ``isRetryable``. Raw payload text,
        messages and ids are discarded. Retained facts survive only in
        this probe's bounded counters.
        """
        facts: dict[str, int] = {}
        response: Any = None
        read_state = ""
        try:
            response = urllib.request.urlopen(
                urllib.request.Request(
                    f"{self._base_url}/event",
                    headers={"Accept": "text/event-stream"},
                ),
                timeout=self._timeout_s,
            )
            facts["event-stream-status"] = int(response.status)
            content_type = str(response.headers.get("Content-Type", ""))[:64]
            facts["event-stream-streamed"] = int("event-stream" in content_type)
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

    def _consume_sse_frames(self, stream: Any) -> str:
        """Parse one bounded read window of SSE ``data:`` frames.

        Returns ``"ok"`` when the bounded read completed (including EOF),
        ``"quiet"`` when a healthy open stream produced no bytes before the
        socket timeout, and ``"error"`` for any other read failure.
        """
        try:
            window = stream.read(_MAX_BYTES)
        except TimeoutError:
            return "quiet"
        except Exception:
            return "error"
        self._last_observed_s = time.time()
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
            self._event_type_counts[event_type] = self._event_type_counts.get(event_type, 0) + 1
            if event_type not in SESSION_EVENT_VOCAB:
                continue
            properties = payload.get("properties")
            if not isinstance(properties, dict):
                continue
            if event_type == "session.status":
                status = properties.get("status")
                if isinstance(status, dict):
                    kind = str(status.get("type", ""))[:24]
                    if kind in SESSION_STATE_VOCAB:
                        self._session_state_counts[kind] = (
                            self._session_state_counts.get(kind, 0) + 1
                        )
                        self._last_session_state = kind
                continue
            if event_type == "session.idle":
                self._session_state_counts["idle"] = self._session_state_counts.get("idle", 0) + 1
                self._last_session_state = "idle"
                continue
            if event_type == "session.error":
                # Session ids are never retained; only the documented,
                # fixed-vocabulary error name and numeric retryability.
                error = properties.get("error")
                kind = "unknown"
                if isinstance(error, dict):
                    name = str(error.get("name", ""))[:64]
                    kind = name if name in SESSION_ERROR_NAME_VOCAB else "unknown"
                    data = error.get("data")
                    if isinstance(data, dict):
                        code = data.get("statusCode")
                        if isinstance(code, int):
                            self._last_server_error_status = code
                        retryable = data.get("isRetryable")
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
            "endpoints": {
                "/doc": {"http_status": int(doc_status), "state": doc_state},
                "/session/status": {str(k)[:48]: int(v) for k, v in session_facts.items()},
                "/event": {str(k)[:48]: int(v) for k, v in sse_facts.items()},
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

    def session_status_probe_payload(self) -> dict[str, Any]:
        """Bounded verdict payload for the #448 observer's probe seam."""
        observed = self._last_session_state in SESSION_STATE_VOCAB
        return {
            "observed": observed,
            "verdict": (
                f"server-events-session-{self._last_session_state}"
                if observed
                else "server-events-session-unobserved"
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
    probe = OpenCodeServerEventProbe(base_url)
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
    "SCHEMA_SERVER_EVIDENCE",
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
