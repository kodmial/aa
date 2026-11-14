"""Short Gate C diagnostic canary (kodmial/aa#514).

This module implements the bounded preflight that must run *before* the
unchanged 18-scenario Gate C campaign. It is deliberately not a second
qualification controller and it never issues a product PASS/FAIL:

1. a deterministic, provider-free observer self-test drives the real
   :class:`~aa.opencode.client.HttpOpenCodeClient`,
   :class:`~aa.opencode.observer.StallSnapshotObserver`, watchdog writer and
   passive server-evidence probe against a controlled local HTTP server;
2. one or two existing core scenarios run through the real production raw
   Telegram seam, real planner/retrieval/answer/verifier and the pinned local
   OpenCode runtime;
3. only a genuinely bounded diagnostic budget is consumed. An externally
   unobserved model response is ``INCONCLUSIVE`` (never a fabricated
   ``provider-failed``);
4. the result is written as privacy-safe ``gate-c-canary-result.json``.

The full Gate C/E/F qualification remains the only authoritative verdict.
"""

from __future__ import annotations

import asyncio
import base64
import http.server
import json
import os
import shutil
import threading
import time
import urllib.parse
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aa.conversation.answer_causality import KNOWN_REASONS
from aa.conversation.answer_causality import UNKNOWN as UNKNOWN_FAILURE_CAUSE
from aa.opencode.client import HttpOpenCodeClient
from aa.opencode.observer import StallSnapshotObserver
from aa.opencode.server_events import OpenCodeServerEventProbe

SCHEMA_VERSION = "aa-gate-c-canary/1"

STATUS_READY = "READY"
STATUS_BLOCKED = "BLOCKED"
STATUS_FAIL = "FAIL"
STATUS_INCONCLUSIVE = "INCONCLUSIVE"

EXIT_BY_STATUS = {
    STATUS_READY: 0,
    STATUS_BLOCKED: 2,
    STATUS_INCONCLUSIVE: 3,
    STATUS_FAIL: 4,
}

DEFAULT_PREFLIGHT_BUDGET_S = 600.0
DEFAULT_LIVE_PROBE_BUDGET_S = 300.0
DEFAULT_SELF_TEST_TIMEOUT_S = 60.0
DEFAULT_SELF_TEST_DELAYS_S: tuple[float, ...] = (0.15, 0.30)
DEFAULT_SELF_TEST_INTERVAL_S = 0.02
DEFAULT_SCENARIO_LIMIT = 2
# Reserve inside the live probe budget for the required terminal trace/flush
# and Application shutdown. Scenario waits stop before this reserve so a
# budget-edge run still returns its in-flight evidence instead of being
# cancelled during cleanup.
DEFAULT_REAL_PATH_CLEANUP_RESERVE_S = 20.0

BUDGET_WITHIN = "within-budget"
BUDGET_EXPIRED = "external-progress-unobserved"

REASON_READY = "canary-ready"
REASON_SELF_TEST_FAILED = "diagnostic-self-test-failed"
REASON_MISSING_DIAGNOSTIC_CHAIN = "missing-diagnostic-chain"
REASON_UNSUPPORTED_SERVER = "unsupported-server-telemetry"
REASON_EXTERNAL_UNOBSERVED = "external-progress-unobserved"
REASON_FAILED_TYPED = "typed-product-path-failure"
REASON_ERROR = "surface-error"

FAIL_CLOSED_REASONS = frozenset(KNOWN_REASONS - {UNKNOWN_FAILURE_CAUSE})
"""Every typed product failure cause, except the literal ``unknown``.

The canary only stops the full campaign for a *concrete* typed cause. An
unattributed ``unknown`` stays a missing-diagnostic / Gate C concern and is
never promoted into a fabricated canary failure.
"""

SERVICE_ERROR_DELIVERED = "service-error-delivered"

# Privacy: no prompt, reply, passage, session id or credential may enter
# canary artifacts. The self-test embeds explicit sentinels in the fake
# server payloads and fails if any of them reaches a persisted file.
_FORBIDDEN_KEYS = frozenset(
    {
        "text",
        "exact_text",
        "utterance",
        "answer",
        "generated_answer",
        "evidence_text",
        "source_text",
        "passage_text",
        "book_text",
        "content",
        "synthetic_input",
        "transcript",
        "summary",
        "prompt",
        "message",
        "messages",
        "chain_of_thought",
        "secret",
        "identity",
        "private_key",
        "token",
        "session_id",
        "sessionID",
        "url",
    }
)

# Artificial markers. They exist only in the controlled test server and
# must never appear in an artifact or stdout line.
SELF_TEST_PROMPT_SENTINEL = "CANARY-PROMPT-SENTINEL-514"
SELF_TEST_SESSION_SENTINEL = "ses_canary_514_private"
SELF_TEST_SSE_SECRET_SENTINEL = "CANARY-SSE-PART-SECRET-514"
# Artificial Basic-auth password for the controlled v2 evidence surface. It
# proves the authenticated pinned-server path is exercised; it is not a
# production secret and never enters an artifact.
SELF_TEST_SERVER_PASSWORD = "canary-514-test-password"


class GateCCanaryError(ValueError):
    """Raised when canary evidence is malformed (fail-closed)."""


def assert_no_text_leak(payload: Any, owner: str = "gate-c-canary") -> None:
    """Reject text-carrying forbidden keys in canary payloads (fail-closed)."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = key.decode("utf-8", "replace") if isinstance(key, bytes) else str(key)
            if key_text.lower() in _FORBIDDEN_KEYS and isinstance(value, str) and value:
                raise ValueError(f"{owner}: forbidden key {key_text!r} carries text")
            assert_no_text_leak(value, f"{owner}.{key_text}")
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            assert_no_text_leak(item, f"{owner}[{index}]")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _bounded(value: object, limit: int = 128) -> str:
    return str(value or "")[:limit]


@dataclass(frozen=True)
class SelfTestEvidence:
    """Bounded deterministic readiness evidence (no provider involved)."""

    passed: bool
    missing_fields: tuple[str, ...] = ()
    snapshot_count: int = 0
    attempt_start_observed: bool = False
    trace_observed: bool = False
    flush_observed: bool = False
    process_alive_observed: bool = False
    scenario_correlation_observed: bool = False
    stage_attempt_stdout_observed: bool = False
    watchdog_marker_observed: bool = False
    server_evidence: dict[str, Any] = field(default_factory=dict)
    server_unobserved_verified: bool = False
    privacy_validated: bool = False
    orphan_absent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "missing_fields": list(self.missing_fields),
            "snapshot_count": self.snapshot_count,
            "attempt_start_observed": self.attempt_start_observed,
            "trace_observed": self.trace_observed,
            "flush_observed": self.flush_observed,
            "process_alive_observed": self.process_alive_observed,
            "scenario_correlation_observed": self.scenario_correlation_observed,
            "stage_attempt_stdout_observed": self.stage_attempt_stdout_observed,
            "watchdog_marker_observed": self.watchdog_marker_observed,
            "server_evidence": dict(self.server_evidence),
            "server_unobserved_verified": self.server_unobserved_verified,
            "privacy_validated": self.privacy_validated,
            "orphan_absent": self.orphan_absent,
        }


def self_test_contract_missing(evidence: SelfTestEvidence) -> tuple[str, ...]:
    """Fields a self-declared ``passed`` result still has to prove.

    ``observer_self_test`` derives its own missing list, but the orchestration
    also accepts injected runners (CI doubles and future adapters). A caller
    must never be able to set ``passed=True`` while omitting the persisted
    in-flight/terminal evidence or the server-side facts that authorize the
    18-case campaign. The checks are intentionally evidence-shaped, not a
    second copy of the observer implementation.
    """
    missing: list[str] = []
    if evidence.snapshot_count < 2:
        missing.append("observer.snapshot_count")
    if not evidence.attempt_start_observed:
        missing.append("observer.attempt_start_event")
    if not evidence.trace_observed:
        missing.append("observer.terminal_trace")
    if not evidence.flush_observed:
        missing.append("observer.flush_index")
    if not evidence.process_alive_observed:
        missing.append("observer.process_alive")
    if not evidence.scenario_correlation_observed:
        missing.append("observer.scenario_correlation")
    if not evidence.stage_attempt_stdout_observed:
        missing.append("stdout.stage_attempt")
    if not evidence.watchdog_marker_observed:
        missing.append("watchdog.marker")
    if str(evidence.server_evidence.get("state", "")) != "supported":
        missing.append("server_events.state")
    if not evidence.server_unobserved_verified:
        missing.append("server_events.unobserved")
    if not evidence.privacy_validated:
        missing.append("privacy.sentinel_leak")
    if not evidence.orphan_absent:
        missing.append("observer.orphan_attempt")
    return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class ScenarioEvidence:
    """One real-path canary scenario outcome, privacy-safe only."""

    family: str
    delivered: bool
    answer_outcome: str = ""
    product_outcome: str = ""
    failure_cause: str = ""
    attempt_observed: bool = False
    stage_observed: bool = False
    causality_observed: bool = False
    server_evidence_state: str = ""
    model: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": _bounded(self.family, 64),
            "delivered": bool(self.delivered),
            "answer_outcome": _bounded(self.answer_outcome, 64),
            "product_outcome": _bounded(self.product_outcome, 64),
            "failure_cause": _bounded(self.failure_cause, 64),
            "attempt_observed": bool(self.attempt_observed),
            "stage_observed": bool(self.stage_observed),
            "causality_observed": bool(self.causality_observed),
            "server_evidence_state": _bounded(self.server_evidence_state, 32),
            "model": _bounded(self.model, 128),
            "latency_ms": round(max(0.0, float(self.latency_ms)), 1),
        }


@dataclass(frozen=True)
class RealPathEvidence:
    """Bounded real-path canary evidence."""

    scenarios: tuple[ScenarioEvidence, ...] = ()
    server_evidence: dict[str, Any] = field(default_factory=dict)
    budget_expired: bool = False
    in_flight_preserved: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "server_evidence": dict(self.server_evidence),
            "budget_expired": bool(self.budget_expired),
            "in_flight_preserved": bool(self.in_flight_preserved),
            "error": _bounded(self.error, 64),
        }


@dataclass(frozen=True)
class CanaryResult:
    """The single privacy-safe ``gate-c-canary-result.json`` payload."""

    main_sha: str
    run_id: str
    status: str
    reason: str
    self_test: SelfTestEvidence
    real_path: RealPathEvidence
    preflight_budget_s: float
    live_probe_budget_s: float
    elapsed_s: float
    budget_category: str
    missing_fields: tuple[str, ...] = ()

    @property
    def exit_code(self) -> int:
        return EXIT_BY_STATUS.get(self.status, EXIT_BY_STATUS[STATUS_BLOCKED])

    def evidence_coverage(self) -> dict[str, str]:
        scenarios = self.real_path.scenarios
        return {
            "stage": (
                "observed"
                if scenarios and all(item.stage_observed for item in scenarios)
                else "missing"
            ),
            "attempt": (
                "observed"
                if scenarios and all(item.attempt_observed for item in scenarios)
                else "missing"
            ),
            "causality": (
                "observed"
                if scenarios and all(item.causality_observed for item in scenarios)
                else "missing"
            ),
            "server_events": (
                "observed"
                if str(self.real_path.server_evidence.get("state", "")) == "supported"
                else "unobserved"
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": SCHEMA_VERSION,
            "main_sha": self.main_sha,
            "run_id": self.run_id,
            "status": self.status,
            "reason": self.reason,
            "product_qualified": False,
            "gate_c_executed": False,
            "budget": {
                "preflight_budget_s": self.preflight_budget_s,
                "live_probe_budget_s": self.live_probe_budget_s,
                "elapsed_s": round(max(0.0, self.elapsed_s), 1),
                "category": self.budget_category,
            },
            "observer_self_test": self.self_test.to_dict(),
            "real_path": self.real_path.to_dict(),
            "evidence_coverage": self.evidence_coverage(),
            "real_turn_count": len(self.real_path.scenarios),
            "missing_fields": list(self.missing_fields),
            "privacy_validated": bool(self.self_test.privacy_validated),
            "artifacts": {
                "result": "gate-c-canary-result.json",
                "self_test_dir": "self-test",
                "attempt_observer_dir": "real-path/attempt-observer",
            },
        }
        assert_no_text_leak(payload)
        return payload


# ---------------------------------------------------------------------------
# Controlled local OpenCode server (provider-free)
# ---------------------------------------------------------------------------


class _CanaryServerState:
    def __init__(
        self,
        *,
        event_enabled: bool = True,
        event_route: str = "/api/event",
        session_status: bool = True,
        error_event: bool = True,
    ) -> None:
        self.event_enabled = event_enabled
        self.event_route = event_route
        self.session_status = session_status
        self.error_event = error_event
        self.clock = 1_700_000_000_000
        self.sessions: dict[str, dict[str, Any]] = {}
        self.release = threading.Event()
        self.prompt_seen = threading.Event()
        self.prompt_text = ""
        self.served_model = "mock/mock-model"

    def next_clock(self) -> int:
        self.clock += 1
        return self.clock

    def complete_turn(self, session: dict[str, Any]) -> None:
        messages = session["messages"]
        assert isinstance(messages, list)
        messages.append(
            {
                "id": f"msg_assistant_{len(messages) + 1:04d}",
                "time": {"created": self.next_clock()},
                "type": "assistant",
                "agent": "build",
                "model": {"providerID": "mock", "id": "mock-model"},
                "content": [{"type": "text", "text": "approved canary reply"}],
                "finish": "stop",
                "tokens": {
                    "input": 11,
                    "output": 7,
                    "reasoning": 0,
                    "cache": {"read": 3, "write": 0},
                },
            }
        )
        messages.append(
            {
                "id": f"msg_idle_{len(messages) + 1:04d}",
                "time": {"created": self.next_clock()},
                "type": "idle",
                "outcome": "succeeded",
            }
        )
        session["outcome"] = "succeeded"


class _CanaryHandler(http.server.BaseHTTPRequestHandler):
    server_version = "CanaryOpenCode/2.2.6"

    def log_message(self, *args: object) -> None:
        return

    @property
    def state(self) -> _CanaryServerState:
        server = self.server
        assert isinstance(server, _CanaryServer)
        return server.state

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = "Basic " + base64.b64encode(
            f"opencode:{SELF_TEST_SERVER_PASSWORD}".encode()
        ).decode("ascii")
        if self.headers.get("Authorization", "") == expected:
            return True
        self._send_json(401, {"_tag": "UnauthorizedError"})
        return False

    def _send_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        # Native v2 framing: one JSON object per ``data:`` line carrying
        # ``type`` and ``data`` (the pinned 2.2.6 server shape).
        frames: list[dict[str, Any]] = [
            {
                "type": "session.status",
                "data": {
                    "sessionID": SELF_TEST_SESSION_SENTINEL,
                    "status": {"type": "busy"},
                },
            }
        ]
        if self.state.error_event:
            frames.append(
                {
                    "type": "session.execution.failed",
                    "data": {
                        "sessionID": SELF_TEST_SESSION_SENTINEL,
                        "error": {
                            "type": "APIError",
                            "status": 503,
                            # Deliberate secrets: the parser must drop them.
                            "message": SELF_TEST_SSE_SECRET_SENTINEL,
                        },
                    },
                }
            )
        frames.append(
            {
                "type": "session.text.delta",
                "data": {
                    "sessionID": SELF_TEST_SESSION_SENTINEL,
                    "delta": SELF_TEST_SSE_SECRET_SENTINEL,
                },
            }
        )
        for frame in frames:
            self.wfile.write(f"data: {json.dumps(frame)}\n".encode())
            self.wfile.write(b"\n")
        self.wfile.flush()

    def _session(self, session_id: str) -> dict[str, Any] | None:
        session = self.state.sessions.get(session_id)
        if session is None:
            self._send_json(404, {"_tag": "SessionNotFoundError"})
        return session

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        if path in ("/openapi.json", "/doc"):
            if not self._authorized():
                return
            self._send_json(
                200,
                {
                    "paths": {
                        "/api/session/active": {},
                        "/api/event": {},
                        "/event": {},
                        "/global/event": {},
                    }
                },
            )
            return
        if path in ("/api/session/active", "/session/status"):
            if not self._authorized():
                return
            if not self.state.session_status:
                self._send_json(200, {"data": {}})
                return
            # Session ids stay on the server; the probe must never retain them.
            self._send_json(200, {"data": {SELF_TEST_SESSION_SENTINEL: {"type": "running"}}})
            return
        if path in ("/api/event", "/event", "/global/event"):
            if not self._authorized():
                return
            if not self.state.event_enabled or path != self.state.event_route:
                self._send_json(404, {"_tag": "UnknownError"})
                return
            self._send_sse()
            return
        if path == "/api/info":
            self._send_json(200, {"version": "2.2.6", "pid": 1234})
            return
        if path.startswith("/api/session/"):
            parts = path.split("/")
            if len(parts) < 4:
                self._send_json(404, {"_tag": "UnknownError"})
                return
            session_id = parts[3]
            session = self._session(session_id)
            if session is None:
                return
            if path.endswith("/message"):
                order = (query.get("order") or ["desc"])[0]
                messages = list(session["messages"])
                if order == "desc":
                    messages.reverse()
                self._send_json(200, {"data": messages, "cursor": {"next": None}})
                return
            self._send_json(
                200,
                {
                    "data": {
                        "id": session_id,
                        "title": session.get("title", ""),
                        "outcome": session.get("outcome"),
                    }
                },
            )
            return
        self._send_json(404, {"_tag": "UnknownError"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:
            body = {}
        if path == "/api/session":
            session_id = SELF_TEST_SESSION_SENTINEL
            self.state.sessions[session_id] = {
                "id": session_id,
                "title": "",
                "model": None,
                "messages": [],
                "outcome": None,
            }
            self._send_json(200, {"data": {"id": session_id, "title": ""}})
            return
        if path.startswith("/api/session/"):
            parts = path.split("/")
            if len(parts) < 4:
                self._send_json(404, {"_tag": "UnknownError"})
                return
            session_id = parts[3]
            session = self._session(session_id)
            if session is None:
                return
            if path.endswith("/model"):
                session["model"] = body.get("model") if isinstance(body, dict) else None
                self._send_json(204, None)
                return
            if path.endswith("/agent"):
                session["agent"] = body.get("agent") if isinstance(body, dict) else None
                self._send_json(204, None)
                return
            if path.endswith("/prompt"):
                prompt = str(body.get("text", "")) if isinstance(body, dict) else ""
                self.state.prompt_text = prompt
                self.state.prompt_seen.set()
                user = {
                    "id": f"msg_user_{len(session['messages']) + 1:04d}",
                    "time": {"created": self.state.next_clock()},
                    "type": "user",
                    "text": prompt,
                }
                session["messages"].append(user)
                self._send_json(
                    200,
                    {
                        "data": {
                            "id": user["id"],
                            "sessionID": session_id,
                            "time": user["time"],
                            "type": "user",
                        }
                    },
                )
                return
            if path.endswith("/interrupt"):
                # Let the wedged worker finish after the native interrupt so
                # the self-test proves a terminal attempt and no orphan.
                self.state.complete_turn(session)
                self.state.release.set()
                self._send_json(200, {"interrupted": True})
                return
        if path.startswith("/api/experimental/session/") and path.endswith("/wait"):
            # The real v2 wait route only reports agent-loop idle. Hold it
            # until the abort so the client attempt stays genuinely in flight.
            self.state.prompt_seen.wait(timeout=0.1)
            self.state.release.wait(timeout=10.0)
            self._send_json(204, None)
            return
        self._send_json(404, {"_tag": "UnknownError"})

    def do_DELETE(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        if path.startswith("/api/session/"):
            session_id = path.split("/")[3]
            self.state.sessions.pop(session_id, None)
            self._send_json(204, None)
            return
        self._send_json(404, {"_tag": "UnknownError"})


class _CanaryServer(http.server.ThreadingHTTPServer):
    state: _CanaryServerState

    def __init__(self, state: _CanaryServerState) -> None:
        super().__init__(("127.0.0.1", 0), _CanaryHandler)
        self.state = state
        self.daemon_threads = True


def _start_server(state: _CanaryServerState) -> tuple[_CanaryServer, threading.Thread, str]:
    server = _CanaryServer(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


def _stop_server(server: _CanaryServer, thread: threading.Thread) -> None:
    server.state.release.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=3.0)


async def _server_observation_checks(
    *,
    error_event: bool,
    event_enabled: bool = True,
    event_route: str = "/api/event",
    session_status: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Probe a controlled server and return bounded evidence plus unobserved flag."""
    state = _CanaryServerState(
        event_enabled=event_enabled,
        event_route=event_route,
        session_status=session_status,
        error_event=error_event,
    )
    server, thread, base_url = _start_server(state)
    try:
        probe = OpenCodeServerEventProbe(
            base_url, timeout_s=0.2, password=SELF_TEST_SERVER_PASSWORD
        )
        evidence = await probe.snapshot()
        endpoints = evidence.get("endpoints")
        status_facts = endpoints.get("/session/status") if isinstance(endpoints, dict) else None
        payload = probe.session_status_probe_payload(
            status_facts if isinstance(status_facts, dict) else {}
        )
        return evidence, not bool(payload.get("observed"))
    finally:
        _stop_server(server, thread)


def _scan_for_sentinels(paths: Sequence[Path]) -> bool:
    """Whether any persisted canary artifact contains a test sentinel."""
    markers = (
        SELF_TEST_PROMPT_SENTINEL.encode("utf-8"),
        SELF_TEST_SESSION_SENTINEL.encode("utf-8"),
        SELF_TEST_SSE_SECRET_SENTINEL.encode("utf-8"),
        SELF_TEST_SERVER_PASSWORD.encode("utf-8"),
    )
    for path in paths:
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        if any(marker in payload for marker in markers):
            return True
    return False


async def observer_self_test(
    out_dir: Path,
    *,
    delays: Sequence[float] = DEFAULT_SELF_TEST_DELAYS_S,
    interval_s: float = DEFAULT_SELF_TEST_INTERVAL_S,
    timeout_s: float = DEFAULT_SELF_TEST_TIMEOUT_S,
) -> SelfTestEvidence:
    """Provider-free deterministic proof that Gate C diagnostics are wired.

    Drives the real HTTP client, attempt ledger, watchdog writer, stall
    observer, terminal trace writer and passive server-evidence probe against
    a controlled loopback server. Any missing persisted field is returned as
    a typed missing field; the caller must not start the long Gate C lane.
    """
    from aa.conversation.turn_causality import clear_turn_causality, set_turn_causality
    from aa.qualification.live_watchdog import (
        ENV_PROGRESS_PATH,
        read_live_progress,
        record_live_progress,
    )
    from aa.qualification.product_contract_live import (
        _format_bounded_attempt_event,
        _format_bounded_live_fields,
        _live_attempt_snapshot_fields,
    )

    out_dir = Path(out_dir)
    observer_dir = out_dir / "attempt-observer"
    progress_path = out_dir / "canary-watchdog-progress.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    # This self-test may only claim evidence it produced in this run.
    if observer_dir.exists():
        shutil.rmtree(observer_dir)
    progress_path.unlink(missing_ok=True)
    missing: list[str] = []

    # (1) Independent server-side evidence: /doc, /session/status, SSE error.
    error_evidence, _ = await _server_observation_checks(error_event=True)
    if str(error_evidence.get("state", "")) != "supported":
        missing.append("server_events.state")
    if int(error_evidence.get("server_error_status_code", 0) or 0) != 503:
        missing.append("server_events.terminal_error_code")
    # The controlled surface deliberately mirrors the pinned native v2
    # routes; a parser that only understands the legacy v1 surface must
    # fail closed instead of claiming readiness against the wrong server.
    if error_evidence.get("doc_endpoint") != "/openapi.json":
        missing.append("server_events.v2_doc")
    if error_evidence.get("session_status_endpoint") != "/api/session/active":
        missing.append("server_events.v2_session_status")
    if error_evidence.get("event_endpoint") != "/api/event":
        missing.append("server_events.v2_event")
    # Independent server facts rather than one blended verdict: the controlled
    # server really reported an active drain, a session-status event and a
    # terminal failure event, and the numeric provider status survived.
    state_counts = error_evidence.get("session_state_counts")
    state_counts = state_counts if isinstance(state_counts, dict) else {}
    if int(state_counts.get("busy", 0) or 0) < 1:
        missing.append("server_events.session_active")
    event_counts = error_evidence.get("event_type_counts")
    event_counts = event_counts if isinstance(event_counts, dict) else {}
    if int(event_counts.get("session.status", 0) or 0) < 1:
        missing.append("server_events.session_status_event")
    if int(event_counts.get("session.execution.failed", 0) or 0) < 1:
        missing.append("server_events.terminal_error_event")
    if error_evidence.get("server_error_kind") != "APIError":
        missing.append("server_events.terminal_error_kind")

    # (2) A second controlled server proves the explicit unobserved shape when
    # no session event is present; no stale busy/idle state is fabricated.
    absent_evidence, unobserved_verified = await _server_observation_checks(
        error_event=False,
        event_enabled=False,
        session_status=False,
    )
    if str(absent_evidence.get("state", "")) != "supported":
        missing.append("server_events.doc_surface")
    if not unobserved_verified:
        missing.append("server_events.unobserved")
    if absent_evidence.get("event_type_counts"):
        missing.append("server_events.absent_event_shape")
    absent_state_counts = absent_evidence.get("session_state_counts")
    absent_state_counts = absent_state_counts if isinstance(absent_state_counts, dict) else {}
    if int(absent_state_counts.get("busy", 0) or 0) or str(
        absent_evidence.get("last_session_state", "")
    ):
        missing.append("server_events.absent_state_shape")

    state = _CanaryServerState()
    server, thread, base_url = _start_server(state)
    client = HttpOpenCodeClient(base_url, protocol="v2", request_timeout=5.0)
    events: list[dict[str, Any]] = []
    task: asyncio.Task[str] | None = None
    observer: StallSnapshotObserver | None = None
    previous_progress = os.environ.get(ENV_PROGRESS_PATH)
    try:
        session = await client.create_session()
        causality = "canary-self-test-causality"
        set_turn_causality(causality, stage="graph-turn")
        task = asyncio.create_task(
            client.send_message(
                session.id,
                SELF_TEST_PROMPT_SENTINEL,
                agent="planner",
                model="mock/mock-model",
            )
        )

        async def _wait_in_flight() -> dict[str, Any]:
            deadline = time.monotonic() + timeout_s
            last: dict[str, Any] = {}
            while time.monotonic() < deadline:
                for entry in reversed(client.attempt_audit):
                    last = dict(entry)
                    if entry.get("state") == "in-flight":
                        return dict(entry)
                await asyncio.sleep(0.01)
            return last

        attempt = await _wait_in_flight()
        attempt_id = str(attempt.get("attempt_id", ""))
        if not attempt_id or attempt.get("state") != "in-flight":
            missing.append("client_ledger.in_flight_attempt")
        if str(attempt.get("causality_id", "")) != causality:
            missing.append("client_ledger.scenario_correlation")

        def _sink(payload: Mapping[str, Any]) -> None:
            events.append(dict(payload))

        observer = StallSnapshotObserver(
            attempt_source=lambda: client.attempt_audit,
            out_dir=observer_dir,
            delays=tuple(float(delay) for delay in delays),
            interval_s=interval_s,
            process_state_source=lambda: {
                "owned_by_runtime": True,
                "alive": True,
                "spawned": True,
                "started": True,
            },
            event_sink=_sink,
            session_status_probe=OpenCodeServerEventProbe(
                base_url,
                timeout_s=0.2,
                password=SELF_TEST_SERVER_PASSWORD,
            ).session_status_snapshot,
        )
        await observer.start()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if observer.snapshot_file_count >= len(tuple(delays)):
                break
            await asyncio.sleep(0.01)

        if observer.snapshot_file_count < len(tuple(delays)):
            missing.append("observer.snapshot_count")
        if not any(event.get("event") == "attempt-start" for event in events):
            missing.append("observer.attempt_start_event")
        if not any(
            event.get("event") == "stall-snapshot"
            and str(event.get("attempt_id", "")) == attempt_id
            for event in events
        ):
            missing.append("observer.stall_snapshot_event")

        snapshot_paths = sorted(observer_dir.glob(f"attempt-{attempt_id}-snapshot-*.json"))
        process_alive = False
        correlation = False
        for snapshot_path in snapshot_paths:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            snapshot_attempt = payload.get("attempt")
            if isinstance(snapshot_attempt, dict):
                if str(snapshot_attempt.get("state", "")) == "in-flight":
                    if snapshot_attempt.get("causality_id") == causality:
                        correlation = True
            process = payload.get("process")
            if isinstance(process, dict) and process.get("alive") is True:
                process_alive = True
        if not process_alive:
            missing.append("observer.process_alive")
        if not correlation:
            missing.append("observer.scenario_correlation")

        # Bounded stdout stage/attempt fields, exactly as the live lane prints.
        fields = _live_attempt_snapshot_fields(
            type("_CanaryApp", (), {"opencode_runtime": type("_R", (), {"client": client})()})(),
            causality_id=causality,
            last_active_stage="graph-turn",
        )
        line = "Gate C scenario 1/1 phase=stage stage=graph-turn " + _format_bounded_live_fields(
            fields
        )
        event_line = "Gate C opencode-attempt " + _format_bounded_attempt_event(
            events[-1] if events else {"event": "attempt-start", "attempt_id": attempt_id}
        )
        if "attempt_state=in-flight" not in line or "stage=graph-turn" not in line:
            missing.append("stdout.stage_attempt")
        if "attempt_id=" not in event_line:
            missing.append("stdout.attempt_event")
        # Real live stdout causality evidence: both lines are rendered from
        # the allow-listed attempt projection, never from prompt text.
        print(line, flush=True)
        print(event_line, flush=True)

        # Watchdog marker/writer must persist the same bounded evidence.
        os.environ[ENV_PROGRESS_PATH] = str(progress_path)
        record_live_progress(
            "canary-self-test",
            real_progress=True,
            attempt_observer_status="active",
            attempt_count=observer.snapshot_attempt_count,
            snapshot_count=observer.snapshot_file_count,
        )
        persisted = read_live_progress(progress_path)
        watchdog_marker = (
            persisted.get("phase") == "canary-self-test"
            and persisted.get("attempt_observer_status") == "active"
            and int(persisted.get("snapshot_count", 0) or 0) >= len(tuple(delays))
        )
        if not watchdog_marker:
            missing.append("watchdog.marker")

        # Simulated abort: cancel the real HTTP attempt, release the fake
        # server, and require an explicit terminal trace + flush index.
        assert task is not None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        state.release.set()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if observer.traced_attempt_count >= 1 and not any(
                entry.get("state") == "in-flight" for entry in client.attempt_audit
            ):
                break
            await asyncio.sleep(0.01)
        await observer.stop(flush_reason="canary-self-test")
        trace_path = observer_dir / f"attempt-{attempt_id}-trace.json"
        flush_path = observer_dir / "attempt-observer-index.json"
        trace_observed = trace_path.exists()
        flush_observed = flush_path.exists()
        if not trace_observed:
            missing.append("observer.terminal_trace")
        if not flush_observed:
            missing.append("observer.flush_index")
        orphan_absent = not observer.is_running() and not any(
            entry.get("state") == "in-flight" for entry in client.attempt_audit
        )
        if not orphan_absent:
            missing.append("observer.orphan_attempt")

        privacy_validated = not _scan_for_sentinels(
            [
                *observer_dir.glob("*.json"),
                progress_path,
            ]
        )
        if not privacy_validated:
            missing.append("privacy.sentinel_leak")

        return SelfTestEvidence(
            passed=not missing,
            missing_fields=tuple(dict.fromkeys(missing)),
            snapshot_count=observer.snapshot_file_count,
            attempt_start_observed=any(event.get("event") == "attempt-start" for event in events),
            trace_observed=trace_observed,
            flush_observed=flush_observed,
            process_alive_observed=process_alive,
            scenario_correlation_observed=correlation,
            stage_attempt_stdout_observed=(
                "stdout.stage_attempt" not in missing and "stdout.attempt_event" not in missing
            ),
            watchdog_marker_observed=watchdog_marker,
            server_evidence=error_evidence,
            server_unobserved_verified=unobserved_verified,
            privacy_validated=privacy_validated,
            orphan_absent=orphan_absent,
        )
    except Exception as exc:  # noqa: BLE001 - self-test must fail closed
        missing.append(f"self-test.{type(exc).__name__}")
        return SelfTestEvidence(
            passed=False,
            missing_fields=tuple(dict.fromkeys(missing)),
            server_evidence=error_evidence,
            server_unobserved_verified=unobserved_verified,
        )
    finally:
        clear_turn_causality()
        if task is not None and not task.done():
            task.cancel()
        try:
            if observer is not None and observer.is_running():
                await observer.stop(flush_reason="canary-self-test-cleanup")
        finally:
            state.release.set()
            try:
                _stop_server(server, thread)
            finally:
                if previous_progress is None:
                    os.environ.pop(ENV_PROGRESS_PATH, None)
                else:
                    os.environ[ENV_PROGRESS_PATH] = previous_progress


# ---------------------------------------------------------------------------
# Real production path
# ---------------------------------------------------------------------------


async def run_real_path_scenarios(
    out_dir: Path,
    *,
    budget_s: float = DEFAULT_LIVE_PROBE_BUDGET_S,
    scenario_limit: int = DEFAULT_SCENARIO_LIMIT,
) -> RealPathEvidence:
    """Run one or two existing core scenarios through the production seam.

    The production ``Application``, raw Telegram update parser, dispatcher,
    LangGraph planner/retrieval/answer/verifier and pinned local OpenCode
    runtime are used unchanged. Only the Telegram network is a deterministic
    in-process seam, exactly as in the authoritative live lane.
    """
    from dataclasses import replace

    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.semantic_work_plan import set_stage_observer
    from aa.conversation.turn_causality import (
        clear_turn_causality,
        current_causality_id,
        set_turn_causality,
    )
    from aa.opencode.server_events import probe_for_runtime
    from aa.qualification.live_partial_evidence import stage_latency_coverage
    from aa.qualification.product_contract_live import (
        GATE_C_CANARY_SCENARIOS,
        QualificationTelegramApi,
        _format_bounded_attempt_event,
        _format_bounded_live_fields,
        _live_attempt_snapshot_fields,
        _opencode_attempt_entries,
        _opencode_process_state,
        _snapshot_product_failure_reason,
        delivered_product_accounting,
    )
    from aa.telegram.transport import PollingTelegramTransport

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(1.0, float(budget_s))
    scenarios_evidence: list[ScenarioEvidence] = []
    server_evidence: dict[str, Any] = {}
    in_flight_preserved = False
    budget_exhausted = False

    api = QualificationTelegramApi()
    settings = replace(
        Settings.from_env(None),
        bot_session_duration_seconds=0.0,
        typing_heartbeat_seconds=0.05,
    )
    transport = PollingTelegramTransport(
        token="canary-telegram-network-seam",
        api=api,
        poll_timeout_seconds=0,
        retry_base_delay_seconds=0.01,
        retry_max_delay_seconds=0.05,
    )
    app = Application(settings, transport=transport)
    observer: StallSnapshotObserver | None = None
    stage_seen: set[str] = set()
    causality_by_chat: dict[int, str] = {}
    causality_scenario_index: dict[str, int] = {}

    def _stage_observer(stage: str) -> None:
        stage_name = _bounded(stage, 64)
        stage_seen.add(stage_name)
        # Live bounded stdout evidence for each production stage. The helper
        # exports only allow-listed attempt/process fields, so no prompt,
        # reply, passage, session id or secret can reach the log.
        fields = _live_attempt_snapshot_fields(
            app,
            causality_id=current_causality_id(),
            last_active_stage=stage_name,
        )
        print(
            f"Gate C canary stage={stage_name} " + _format_bounded_live_fields(fields),
            flush=True,
        )

    def _attribute_turn(processor: Any) -> Any:
        async def attributed(incoming: Any) -> None:
            causality = causality_by_chat.get(int(incoming.chat_id), "")
            if causality:
                set_turn_causality(causality, stage="graph-turn")
            try:
                await processor(incoming)
            finally:
                clear_turn_causality()

        return attributed

    try:
        await asyncio.wait_for(app.start(), timeout=max(1.0, deadline - time.monotonic()))
        app.dispatcher.set_process_wrapper(_attribute_turn)
        set_stage_observer(_stage_observer)
        probe = probe_for_runtime(app.opencode_runtime)
        if probe is None:
            return RealPathEvidence(error="server-events-unavailable")
        server_evidence = await probe.snapshot()
        # The real pinned server must expose the observer's native status and
        # event surfaces before spending the live diagnostic budget on model
        # turns. A present-but-incomplete telemetry source is a typed block,
        # not a provider wait and not a reason to start the 18-case lane.
        if str(server_evidence.get("state", "")) != "supported" or _server_surface_missing(
            server_evidence
        ):
            return RealPathEvidence(
                server_evidence=server_evidence,
                error="unsupported-server-telemetry",
            )

        def _entries_source() -> tuple[dict[str, Any], ...]:
            return _opencode_attempt_entries(getattr(app, "opencode_runtime", None))

        def _event_sink(payload: Mapping[str, Any]) -> None:
            print(
                "Gate C opencode-attempt "
                + _format_bounded_attempt_event(payload, causality_scenario_index),
                flush=True,
            )

        observer = StallSnapshotObserver(
            attempt_source=_entries_source,
            out_dir=out_dir / "attempt-observer",
            process_state_source=lambda: _opencode_process_state(
                getattr(app, "opencode_runtime", None)
            ),
            session_status_probe=probe.session_status_snapshot,
            event_sink=_event_sink,
            register_atexit_flush=True,
        )
        await observer.start()

        scenarios = tuple(GATE_C_CANARY_SCENARIOS)[: max(1, int(scenario_limit))]
        for position, (family, chat_id, prompt) in enumerate(scenarios, start=1):
            if time.monotonic() >= deadline:
                budget_exhausted = True
                break
            stage_seen.clear()
            causality = uuid.uuid4().hex
            causality_by_chat[int(chat_id)] = causality
            causality_scenario_index[causality] = position
            scenario_started = time.perf_counter()
            before = len(api.sent_texts)
            raw = {
                "update_id": 940000 + position,
                "message": {
                    "message_id": position,
                    "date": 1,
                    "chat": {"id": chat_id, "type": "private"},
                    "text": prompt,
                },
            }
            api.pending_updates.append(raw)
            delivered = False
            while time.monotonic() < deadline:
                if len(api.sent_texts) > before:
                    delivered = True
                    break
                await asyncio.sleep(0.02)
            elapsed_ms = (time.perf_counter() - scenario_started) * 1000.0

            entries = _opencode_attempt_entries(app.opencode_runtime)
            matching = [
                entry for entry in entries if str(entry.get("causality_id", "")) == causality
            ]
            snapshot_attempts = {
                str(entry.get("attempt_id", "")) for entry in matching if entry.get("attempt_id")
            }
            attempt_observed = bool(matching) or bool(
                observer is not None
                and any(observer.has_snapshot_for(attempt_id) for attempt_id in snapshot_attempts)
            )
            if not delivered:
                budget_exhausted = True
                # Diagnostic budget exhausted: preserve the in-flight attempt
                # as INCONCLUSIVE evidence, never as a provider failure.
                in_flight_ids = {
                    str(entry.get("attempt_id", ""))
                    for entry in entries
                    if str(entry.get("state", "")) == "in-flight" and entry.get("attempt_id")
                }
                in_flight_preserved = bool(in_flight_ids) and bool(
                    observer is not None
                    and any(observer.has_snapshot_for(item) for item in in_flight_ids)
                )
                scenarios_evidence.append(
                    ScenarioEvidence(
                        family=family,
                        delivered=False,
                        attempt_observed=attempt_observed,
                        stage_observed=bool(stage_seen),
                        causality_observed=bool(matching),
                        server_evidence_state=str(server_evidence.get("state", "")),
                        latency_ms=elapsed_ms,
                    )
                )
                break

            reply = api.sent_texts[-1]
            snapshot: dict[str, Any] = {}
            try:
                graph = app.graph_runtime
                if graph is not None:
                    snapshot = graph.last_telemetry_for_thread(graph.thread_id(chat_id))
            except Exception:  # noqa: BLE001 - evidence must not break the canary
                snapshot = {}
            product_outcome, failure_reason = delivered_product_accounting(reply, snapshot)
            answer_outcome = _bounded(snapshot.get("answer_outcome", ""), 64)
            failure_cause = _bounded(
                failure_reason or _snapshot_product_failure_reason(snapshot), 64
            )
            served = ""
            served_audit = getattr(app.opencode_runtime.client, "served_model_audit", ())
            for item in reversed(tuple(served_audit)):
                candidate = str(item.get("served", ""))[:128]
                if candidate:
                    served = candidate
                    break
            if not served and matching:
                served = _bounded(matching[-1].get("model", ""), 128)
            coverage = stage_latency_coverage(snapshot)
            stage_observed = bool(stage_seen) or any(
                value == "measured" for value in coverage.values()
            )
            scenarios_evidence.append(
                ScenarioEvidence(
                    family=family,
                    delivered=True,
                    answer_outcome=answer_outcome,
                    product_outcome=product_outcome,
                    failure_cause=failure_cause,
                    attempt_observed=attempt_observed,
                    stage_observed=stage_observed,
                    causality_observed=bool(matching),
                    server_evidence_state=str(server_evidence.get("state", "")),
                    model=served,
                    latency_ms=elapsed_ms,
                )
            )
            if _scenario_failure(scenarios_evidence[-1]):
                # The first real-path turn already reproduced a known false
                # path. Do not replay the 18-case campaign.
                break

        server_evidence = await probe.snapshot()
        return RealPathEvidence(
            scenarios=tuple(scenarios_evidence),
            server_evidence=server_evidence,
            budget_expired=budget_exhausted,
            in_flight_preserved=in_flight_preserved,
        )
    except Exception as exc:  # noqa: BLE001 - the canary must fail closed
        return RealPathEvidence(
            scenarios=tuple(scenarios_evidence),
            server_evidence=server_evidence,
            budget_expired=time.monotonic() >= deadline,
            in_flight_preserved=in_flight_preserved,
            error=type(exc).__name__,
        )
    finally:
        set_stage_observer(None)
        clear_turn_causality()
        if observer is not None:
            try:
                await asyncio.wait_for(observer.stop(flush_reason="canary-real-path"), timeout=5.0)
            except (TimeoutError, asyncio.CancelledError):
                pass
        try:
            await asyncio.wait_for(app.stop(), timeout=10.0)
        except (TimeoutError, asyncio.CancelledError):
            pass


# ---------------------------------------------------------------------------
# Decision and orchestration
# ---------------------------------------------------------------------------


def _scenario_failure(scenario: ScenarioEvidence) -> str:
    """Concrete typed failure of one real-path turn ("" when none)."""
    if scenario.failure_cause in FAIL_CLOSED_REASONS:
        return scenario.failure_cause
    if not scenario.delivered:
        return ""
    if scenario.product_outcome == "service_error":
        return SERVICE_ERROR_DELIVERED
    if scenario.product_outcome == "rejected":
        return scenario.answer_outcome or "rejected-product-outcome"
    # ``unknown``/missing product accounting is not a canary failure: the
    # canary proves diagnostic wiring, not answer quality (Gate C owns that).
    return ""


def _server_surface_missing(evidence: Mapping[str, Any]) -> list[str]:
    """Missing native-surface fields of the *real* pinned server evidence.

    ``state: supported`` alone only proves the OpenAPI document exists. The
    canary additionally requires the session-activity and SSE routes to have
    answered on the same live server, because the observer's stall snapshots
    are meaningless without them.
    """
    missing: list[str] = []
    endpoints = evidence.get("endpoints")
    endpoints = endpoints if isinstance(endpoints, dict) else {}
    doc = endpoints.get("/doc")
    doc = doc if isinstance(doc, dict) else {}
    if doc.get("state") != "supported" or evidence.get("doc_endpoint") in ("", None):
        missing.append("server_events.doc_surface")
    status_facts = endpoints.get("/session/status")
    status_facts = status_facts if isinstance(status_facts, dict) else {}
    status_failure = any(
        int(status_facts.get(key, 0) or 0) > 0
        for key in (
            "session-status-http-error",
            "session-status-error",
            "session-status-body-timeout",
            "session-status-body-error",
            "session-status-shape",
        )
    )
    if (
        evidence.get("session_status_endpoint") in ("", None, "unobserved")
        or status_failure
        or "session-status" not in status_facts
    ):
        missing.append("server_events.session_status_surface")
    event_endpoint = str(evidence.get("event_endpoint", "") or "")
    event_facts = endpoints.get(event_endpoint)
    event_facts = event_facts if isinstance(event_facts, dict) else {}
    if (
        event_endpoint in ("", "unobserved")
        or int(event_facts.get("event-stream-status", 0) or 0) != 200
        or int(event_facts.get("event-stream-streamed", 0) or 0) != 1
        # An opened route whose bounded read failed is not a usable event
        # surface, even though its status headers look healthy.
        or int(event_facts.get("event-stream-error", 0) or 0)
    ):
        missing.append("server_events.event_stream_surface")
    return missing


def evaluate_canary(
    *,
    main_sha: str,
    run_id: str,
    self_test: SelfTestEvidence,
    real_path: RealPathEvidence,
    preflight_budget_s: float,
    live_probe_budget_s: float,
    elapsed_s: float,
    scenario_limit: int = DEFAULT_SCENARIO_LIMIT,
) -> CanaryResult:
    """Reduce bounded evidence to one honest canary status (never a product PASS)."""
    missing = list(self_test.missing_fields)
    # A self-declared ``passed`` is not enough: every required persisted and
    # server-side fact must actually be present. This also protects direct
    # callers that bypass ``run_canary``'s injected-runner validation.
    self_test_contract = self_test_contract_missing(self_test) if self_test.passed else ()
    if not self_test.passed and not missing:
        missing.append("self-test.failed")
    missing.extend(self_test_contract)
    server_state = str(real_path.server_evidence.get("state", ""))
    server_surfaces_missing = _server_surface_missing(real_path.server_evidence)
    if not server_state:
        missing.append("server_events.real_runtime_state")
    elif server_state != "supported":
        missing.append("server_events.real_runtime_state")
    else:
        missing.extend(server_surfaces_missing)
    status = STATUS_READY
    reason = REASON_READY

    if not self_test.passed or self_test_contract:
        status = STATUS_BLOCKED
        reason = REASON_SELF_TEST_FAILED
    elif not self_test.privacy_validated:
        status = STATUS_BLOCKED
        reason = REASON_SELF_TEST_FAILED
        missing.append("privacy.self_test")
    elif server_state and server_state != "supported":
        # A present but unsupported telemetry surface is a hard diagnostic
        # block, never an inconclusive provider wait.
        status = STATUS_BLOCKED
        reason = REASON_UNSUPPORTED_SERVER
    elif server_state == "supported" and server_surfaces_missing:
        # The OpenAPI document exists but the observer's status/SSE routes do
        # not answer on this pinned server: no readiness claim is possible.
        status = STATUS_BLOCKED
        reason = REASON_UNSUPPORTED_SERVER
    elif real_path.budget_expired or real_path.error == "diagnostic-budget-expired":
        status = STATUS_INCONCLUSIVE
        reason = REASON_EXTERNAL_UNOBSERVED
    elif real_path.error in ("server-events-unavailable", "unsupported-server-telemetry"):
        status = STATUS_BLOCKED
        reason = REASON_UNSUPPORTED_SERVER
    elif real_path.error:
        # A harness/runtime surface exception is infrastructure, not a
        # product failure and never a fabricated provider-death verdict.
        status = STATUS_BLOCKED
        reason = REASON_ERROR
    elif server_state != "supported":
        status = STATUS_BLOCKED
        reason = REASON_UNSUPPORTED_SERVER
    elif not real_path.scenarios:
        status = STATUS_BLOCKED
        reason = REASON_MISSING_DIAGNOSTIC_CHAIN
        missing.append("real_path.scenarios")
    else:
        for scenario in real_path.scenarios:
            if not (
                scenario.attempt_observed
                and scenario.stage_observed
                and scenario.causality_observed
            ):
                status = STATUS_BLOCKED
                reason = REASON_MISSING_DIAGNOSTIC_CHAIN
                if not scenario.attempt_observed:
                    missing.append("real_path.attempt_observed")
                if not scenario.stage_observed:
                    missing.append("real_path.stage_observed")
                if not scenario.causality_observed:
                    missing.append("real_path.causality_observed")
                break
            failure = _scenario_failure(scenario)
            if failure:
                status = STATUS_FAIL
                reason = failure
                break
        if status == STATUS_READY and len(real_path.scenarios) < max(
            1, min(int(scenario_limit), DEFAULT_SCENARIO_LIMIT)
        ):
            status = STATUS_BLOCKED
            reason = REASON_MISSING_DIAGNOSTIC_CHAIN
            missing.append("real_path.scenario_count")

    return CanaryResult(
        main_sha=main_sha,
        run_id=run_id,
        status=status,
        reason=reason,
        self_test=self_test,
        real_path=real_path,
        preflight_budget_s=preflight_budget_s,
        live_probe_budget_s=live_probe_budget_s,
        elapsed_s=elapsed_s,
        budget_category=(
            BUDGET_EXPIRED
            if real_path.budget_expired or elapsed_s >= preflight_budget_s
            else BUDGET_WITHIN
        ),
        missing_fields=tuple(dict.fromkeys(missing)),
    )


async def run_canary(
    *,
    main_sha: str,
    run_id: str,
    out_dir: Path,
    preflight_budget_s: float = DEFAULT_PREFLIGHT_BUDGET_S,
    live_probe_budget_s: float = DEFAULT_LIVE_PROBE_BUDGET_S,
    self_test_timeout_s: float = DEFAULT_SELF_TEST_TIMEOUT_S,
    scenario_limit: int = DEFAULT_SCENARIO_LIMIT,
    self_test_runner: (Callable[[Path, float], Awaitable[SelfTestEvidence]] | None) = None,
    real_path_runner: (Callable[[Path, float], Awaitable[RealPathEvidence]] | None) = None,
) -> CanaryResult:
    """Run the bounded preflight and return its privacy-safe result."""
    started = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Never let a previous run's READY artifact authorize this run's Gate C.
    (out_dir / "gate-c-canary-result.json").unlink(missing_ok=True)
    preflight_budget_s = max(1.0, float(preflight_budget_s))
    live_probe_budget_s = max(1.0, float(live_probe_budget_s))
    deadline = started + preflight_budget_s
    self_test = SelfTestEvidence(passed=False, missing_fields=("self-test.not-run",))
    real_path = RealPathEvidence(error="real-path-not-run")

    self_test_remaining = min(self_test_timeout_s, max(1.0, deadline - time.monotonic()))
    if self_test_runner is None:
        try:
            self_test = await asyncio.wait_for(
                observer_self_test(out_dir / "self-test", timeout_s=self_test_remaining),
                timeout=self_test_remaining,
            )
        except (TimeoutError, asyncio.CancelledError):
            self_test = SelfTestEvidence(passed=False, missing_fields=("self-test.timeout",))
        except Exception as exc:  # noqa: BLE001
            self_test = SelfTestEvidence(
                passed=False,
                missing_fields=(f"self-test.{type(exc).__name__}",),
            )
    else:
        try:
            self_test = await asyncio.wait_for(
                self_test_runner(out_dir / "self-test", self_test_remaining),
                timeout=self_test_remaining,
            )
        except (TimeoutError, asyncio.CancelledError):
            self_test = SelfTestEvidence(passed=False, missing_fields=("self-test.timeout",))
        except Exception as exc:  # noqa: BLE001 - injected runner must fail closed
            self_test = SelfTestEvidence(
                passed=False,
                missing_fields=(f"self-test.{type(exc).__name__}",),
            )
    # A runner (real or injected) still has to satisfy the same evidence
    # contract; it cannot self-declare success while omitting the persisted
    # in-flight/terminal or server-side facts.
    if self_test.passed:
        contract_missing = self_test_contract_missing(self_test)
        if contract_missing or self_test.missing_fields:
            self_test = SelfTestEvidence(
                passed=False,
                missing_fields=tuple(dict.fromkeys((*self_test.missing_fields, *contract_missing))),
            )

    if self_test.passed and time.monotonic() < deadline:
        live_budget = min(live_probe_budget_s, max(1.0, deadline - time.monotonic()))
        scenario_budget = max(
            1.0, live_budget - min(DEFAULT_REAL_PATH_CLEANUP_RESERVE_S, live_budget / 4.0)
        )
        try:
            if real_path_runner is None:
                real_path = await asyncio.wait_for(
                    run_real_path_scenarios(
                        out_dir / "real-path",
                        budget_s=scenario_budget,
                        scenario_limit=scenario_limit,
                    ),
                    timeout=live_budget,
                )
            else:
                real_path = await asyncio.wait_for(
                    real_path_runner(out_dir / "real-path", scenario_budget),
                    timeout=live_budget,
                )
        except (TimeoutError, asyncio.CancelledError):
            real_path = RealPathEvidence(
                budget_expired=True,
                server_evidence={},
                error="diagnostic-budget-expired",
            )
        except Exception as exc:  # noqa: BLE001
            real_path = RealPathEvidence(error=type(exc).__name__)
    elif not self_test.passed:
        real_path = RealPathEvidence(error="skipped-self-test-blocked")
    else:
        # The self-test consumed the whole preflight budget; no real turn was
        # allowed to start. This is external progress that was never observed,
        # never a provider failure.
        real_path = RealPathEvidence(
            budget_expired=True,
            error="diagnostic-budget-expired",
        )

    elapsed = time.monotonic() - started
    result = evaluate_canary(
        main_sha=main_sha,
        run_id=run_id,
        self_test=self_test,
        real_path=real_path,
        preflight_budget_s=preflight_budget_s,
        live_probe_budget_s=live_probe_budget_s,
        elapsed_s=elapsed,
        scenario_limit=scenario_limit,
    )
    write_canary_result(out_dir, result)
    return result


def write_canary_result(out_dir: Path, result: CanaryResult) -> Path:
    """Persist the one privacy-safe canary result artifact."""
    payload = result.to_dict()
    path = Path(out_dir) / "gate-c-canary-result.json"
    _write_json(path, payload)
    if _scan_for_sentinels([path]):
        # Defense in depth: the artifact itself is the boundary. Remove a
        # leaking file instead of leaving it for artifact upload.
        path.unlink(missing_ok=True)
        raise GateCCanaryError("canary artifact failed the privacy sentinel scan")
    return path


__all__ = [
    "BUDGET_EXPIRED",
    "BUDGET_WITHIN",
    "CanaryResult",
    "DEFAULT_LIVE_PROBE_BUDGET_S",
    "DEFAULT_PREFLIGHT_BUDGET_S",
    "DEFAULT_SCENARIO_LIMIT",
    "DEFAULT_REAL_PATH_CLEANUP_RESERVE_S",
    "DEFAULT_SELF_TEST_DELAYS_S",
    "EXIT_BY_STATUS",
    "FAIL_CLOSED_REASONS",
    "GateCCanaryError",
    "RealPathEvidence",
    "SCHEMA_VERSION",
    "SERVICE_ERROR_DELIVERED",
    "STATUS_BLOCKED",
    "STATUS_FAIL",
    "STATUS_INCONCLUSIVE",
    "STATUS_READY",
    "ScenarioEvidence",
    "SelfTestEvidence",
    "assert_no_text_leak",
    "evaluate_canary",
    "observer_self_test",
    "run_canary",
    "run_real_path_scenarios",
    "self_test_contract_missing",
    "write_canary_result",
]
