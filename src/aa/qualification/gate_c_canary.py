"""Bounded Gate C canary preflight (kodmial/aa#454).

The 18-case Gate C campaign is expensive and previously spent its full
allowance before proving that its own diagnostics were wired. This module
owns the cheap, fail-closed preflight that runs BEFORE the unchanged
campaign:

1. a deterministic observer/server self-test against a controlled local
   HTTP server using the real ``HttpOpenCodeClient``,
   ``StallSnapshotObserver`` and ``OpenCodeServerEventProbe``. It proves
   persisted in-flight snapshots, terminal traces, attempt/causality
   correlation, process state, watchdog progress writing, partial-evidence
   coverage, privacy and absence of orphans without any LLM call;
2. a bounded real-path canary of one or two existing core Gate C
   scenarios through the production Telegram seam, real planner,
   retrieval, answer and verifier. It is executed by the existing live
   lane with a small scenario subset, never by a second qualification
   controller.

Nothing here assigns a Product Contract PASS/FAIL. ``READY`` only means
the diagnostic chain is observable and the real path did not return a
known typed failure, so the unchanged 18-case Gate C may run exactly once.
Every budget expiry or missing observation stays typed and fail-closed.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from aa.opencode.server_events import OpenCodeServerEventProbe
from aa.qualification.live_partial_evidence import (
    COVERAGE_MISSING,
    COVERAGE_OBSERVED,
    SOURCES,
    build_partial_evidence_snapshot,
    coverage_from_partial,
    validated_partial_evidence,
)
from aa.qualification.live_watchdog import (
    ENV_PROGRESS_PATH,
    read_live_progress,
    record_live_progress,
)

SCHEMA_GATE_C_CANARY_RESULT = "aa-gate-c-canary-result/1"
SCHEMA_OBSERVER_SELF_TEST = "aa-gate-c-canary-observer-self-test/1"

STATUS_READY = "READY"
STATUS_BLOCKED = "BLOCKED"
STATUS_FAIL = "FAIL"
STATUS_INCONCLUSIVE = "INCONCLUSIVE"

EXIT_BY_STATUS = {
    STATUS_READY: 0,
    STATUS_FAIL: 1,
    STATUS_BLOCKED: 2,
    STATUS_INCONCLUSIVE: 3,
}

DEFAULT_PREFLIGHT_BUDGET_S = 600.0
DEFAULT_LIVE_BUDGET_S = 300.0
MIN_BUDGET_S = 10.0

# Nominal production observer schedule; the self-test runs the same schedule
# accelerated so the whole preflight stays in seconds.
NOMINAL_SNAPSHOT_DELAYS_S: tuple[float, ...] = (10.0, 25.0)
ACCELERATED_SNAPSHOT_DELAYS_S: tuple[float, ...] = (0.2, 0.5)

SELF_TEST_CAUSALITY = "canary-observer-selftest"
# A distinctive prompt crosses the real HTTP boundary once. It must never
# appear in any snapshot, trace, event, progress marker or result file.
SELF_TEST_PROMPT_SENTINEL = "GATECCANARYPROMPTSENTINEL454"
# A raw SSE text part crosses the same boundary. The bounded probe must
# extract only typed facts and never persist the raw event payload.
SELF_TEST_SSE_SENTINEL = "GATECCANARYSSESENTINEL454"
SELF_TEST_FORBIDDEN_MARKERS: tuple[str, ...] = (
    SELF_TEST_PROMPT_SENTINEL,
    SELF_TEST_SSE_SENTINEL,
    "Authorization",
    "Bearer ",
    "sk-",
)

# Known first-turn product failures that must stop the campaign early.
TYPED_CANARY_FAILURE_CAUSES: tuple[str, ...] = (
    "conversational-generation-failed",
    "clarification-unavailable",
    "guard-internal-terms",
    "generation-empty-candidate",
    "provider-timeout",
    "verifier-unavailable",
    "service_error",
)

# Closed vocabulary of canary decision categories.
CATEGORY_READY = "diagnostic-and-real-path-ready"
CATEGORY_SELF_TEST_FAILED = "observer-self-test-failed"
CATEGORY_UNSUPPORTED_SERVER = "unsupported-server-telemetry"
CATEGORY_REAL_PATH_NOT_EXECUTED = "real-path-not-executed"
CATEGORY_DIAGNOSTIC_MISSING = "diagnostic-evidence-missing"
CATEGORY_EXTERNAL_PROGRESS_UNOBSERVED = "external-progress-unobserved"
CATEGORY_REAL_PATH_TYPED_FAILURE = "real-path-typed-failure"
CATEGORY_REAL_PATH_INCOMPLETE = "real-path-incomplete"
CATEGORY_PRIVACY_VIOLATION = "privacy-violation"
CATEGORY_ORPHAN_WORK = "orphan-work"
CATEGORY_OPENCODE_UNAVAILABLE = "pinned-opencode-unavailable"
CATEGORY_PRECONDITION_FAILED = "precondition-failed"


@dataclass(frozen=True)
class CanaryDecision:
    """One typed canary decision (never a Product Contract verdict)."""

    status: str
    category: str
    missing_field: str = ""
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.status == STATUS_READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "category": self.category,
            "missing_field": self.missing_field,
            "detail": self.detail,
            "product_qualified": False,
        }


class _CanaryHttpState:
    """Mutable state for the controlled self-test HTTP server."""

    def __init__(self) -> None:
        self.abort_event = threading.Event()
        self.message_delay_s = 30.0
        self.message_requests = 0
        self.abort_requests = 0
        self.status_payload: dict[str, Any] = {"canary-session-1": {"type": "busy", "attempt": 1}}
        self.event_body = _default_event_body()


def _default_event_body() -> bytes:
    return (
        b'data: {"type":"session.status","properties":{"status":{"type":"busy"}}}\n\n'
        b'data: {"type":"message.part","properties":{"text":"GATECCANARYSSESENTINEL454"}}\n\n'
        b'data: {"type":"session.error","properties":{"error":{"name":"APIError",'
        b'"data":{"statusCode":503,"isRetryable":true}}}}\n\n'
    )


class _CanaryRequestHandler(BaseHTTPRequestHandler):
    """Minimal loopback OpenCode-compatible surface for the self-test."""

    protocol_version = "HTTP/1.1"
    state: _CanaryHttpState

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self._send_body(status, body, "application/json")

    def _send_body(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8", errors="replace"))
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urllib.parse.urlparse(self.path).path
        if path == "/doc":
            self._send_json(
                200,
                {
                    "openapi": "3.0.0",
                    "paths": {
                        "/session/status": {},
                        "/event": {},
                        "/session/{id}/message": {},
                    },
                },
            )
            return
        if path == "/global/health":
            self._send_json(200, {"healthy": True, "version": "canary-selftest"})
            return
        if path == "/session/status":
            self._send_json(200, dict(self.state.status_payload))
            return
        if path == "/event":
            self._send_body(200, bytes(self.state.event_body), "text/event-stream")
            return
        if path.startswith("/session/"):
            self._send_json(200, {"id": "canary-session-1"})
            return
        self._send_json(404, {"error": "not-found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urllib.parse.urlparse(self.path).path
        if path == "/session":
            self._read_json()
            self._send_json(200, {"id": "canary-session-1", "title": "canary"})
            return
        if path.endswith("/abort"):
            self._read_json()
            self.state.abort_requests += 1
            self.state.abort_event.set()
            self._send_json(200, True)
            return
        if path.endswith("/message"):
            self._read_json()
            self.state.message_requests += 1
            aborted = self.state.abort_event.wait(timeout=self.state.message_delay_s)
            if aborted:
                # A deterministic terminal failure, never a fabricated
                # provider response or progress signal.
                self._send_json(409, {"error": "aborted"})
                return
            self._send_json(
                200,
                {"info": {}, "parts": [{"type": "text", "text": "canary-selftest"}]},
            )
            return
        self._send_json(404, {"error": "not-found"})

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._send_json(200, True)


def _start_server(state: _CanaryHttpState) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    class Handler(_CanaryRequestHandler):
        pass

    Handler.state = state
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever, name="aa-gate-c-canary-http", daemon=True
    )
    thread.start()
    host, port = server.server_address[:2]
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    return server, thread, f"http://{host}:{port}"


async def _wait_for_attempt(client: Any, *, state: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.1, timeout_s)
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        for entry in reversed(client.attempt_audit):
            if str(entry.get("state", "")) == state:
                return dict(entry)
            last = dict(entry)
        await asyncio.sleep(0.01)
    return last


def _privacy_scan(paths: Sequence[Path], markers: Sequence[str]) -> tuple[bool, str]:
    """Scan only the artifacts owned by the self-test.

    The shared qualification output directory also contains downloaded A/B
    evidence in a full campaign. Scanning unrelated files there would make
    the canary's privacy verdict depend on evidence it did not produce.
    """
    for root in paths:
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = sorted(root.rglob("*"))
        else:
            continue
        for path in candidates:
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return False, path.name
            for marker in markers:
                if marker and marker in content:
                    return False, path.name
    return True, ""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


async def run_observer_self_test(
    out_dir: Path | str,
    *,
    nominal_delays: Sequence[float] = NOMINAL_SNAPSHOT_DELAYS_S,
    accelerated_delays: Sequence[float] = ACCELERATED_SNAPSHOT_DELAYS_S,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Run the deterministic, LLM-free observer/server self-test.

    The returned report is privacy-safe and contains only booleans,
    closed-vocabulary categories, counters, bounded paths and timings.
    """
    started = time.monotonic()
    out = Path(out_dir)
    observer_dir = out / "attempt-observer"
    checks: dict[str, bool] = {}
    failures: list[str] = []

    def check(name: str, ok: bool) -> None:
        checks[name] = bool(ok)
        if not ok:
            failures.append(name)

    state = _CanaryHttpState()
    server, thread, base_url = _start_server(state)
    events: list[dict[str, Any]] = []
    client: Any = None
    observer: Any = None
    task: asyncio.Task[Any] | None = None
    server_telemetry: dict[str, Any] = {}
    observer_stopped = False
    snapshot_payloads: list[dict[str, Any]] = []
    validated: dict[str, Any] | None = None
    coverage: dict[str, Any] = {}
    previous_progress_env = os.environ.get(ENV_PROGRESS_PATH)
    try:
        from aa.conversation.turn_causality import (
            clear_turn_causality,
            set_turn_causality,
        )
        from aa.opencode.client import HttpOpenCodeClient
        from aa.opencode.observer import StallSnapshotObserver

        probe = OpenCodeServerEventProbe(base_url, timeout_s=1.0)
        evidence = await probe.snapshot()
        server_telemetry = {
            "state": str(evidence.get("state", "unobserved")),
            "doc_state": str(
                (evidence.get("endpoints", {}).get("/doc", {}) or {}).get("state", "")
            ),
            "session_status_observed": bool(
                int(
                    (evidence.get("endpoints", {}).get("/session/status", {}) or {}).get(
                        "session-status", 0
                    )
                    or 0
                )
                > 0
            ),
            "sse_observed": bool(
                int(
                    (evidence.get("endpoints", {}).get("/event", {}) or {}).get(
                        "event-stream-status", 0
                    )
                    or 0
                )
                == 200
            ),
            "server_error_kind": str(evidence.get("server_error_kind", "")),
            "server_error_status_code": evidence.get("server_error_status_code"),
        }
        check("server-telemetry-supported", server_telemetry["state"] == "supported")
        check("server-doc-supported", server_telemetry["doc_state"] == "supported")
        check("server-session-status-observed", server_telemetry["session_status_observed"])
        check("server-sse-observed", server_telemetry["sse_observed"])
        check(
            "server-error-typed",
            server_telemetry["server_error_kind"] == "APIError"
            and server_telemetry["server_error_status_code"] == 503,
        )

        # Explicitly prove the absence path: no event/status fact must be
        # reported as unobserved, never as a fictitious idle/busy state.
        state.status_payload = {}
        state.event_body = b""
        absent_probe = OpenCodeServerEventProbe(base_url, timeout_s=1.0)
        absent = await absent_probe.snapshot()
        absent_counts = absent.get("session_state_counts", {})
        check(
            "server-absent-event-unobserved",
            "last_session_state" not in absent
            and all(int(value or 0) == 0 for value in absent_counts.values()),
        )
        check(
            "server-absent-probe-unobserved",
            absent_probe.session_status_probe_payload({}).get("observed") is False,
        )

        # Restore the observable surface for the observer snapshot.
        state.status_payload = {"canary-session-1": {"type": "busy", "attempt": 1}}
        state.event_body = _default_event_body()
        state.abort_event.clear()
        state.message_delay_s = 30.0

        client = HttpOpenCodeClient(base_url, request_timeout=timeout_s)
        health = await client.health()
        check("client-health", bool(health.healthy))
        session = await client.create_session("canary")
        check("client-session", bool(session.id))

        def _event_sink(payload: Mapping[str, Any]) -> None:
            events.append(dict(payload))

        set_turn_causality(SELF_TEST_CAUSALITY, stage="answer")
        observer = StallSnapshotObserver(
            attempt_source=lambda: client.attempt_audit,
            out_dir=observer_dir,
            delays=tuple(accelerated_delays),
            interval_s=0.02,
            process_state_source=lambda: {
                "owned_by_runtime": True,
                "alive": True,
                "started": True,
            },
            session_status_probe=probe.session_status_snapshot,
            event_sink=_event_sink,
        )
        await observer.start()
        task = asyncio.create_task(
            client.send_message(
                session.id,
                SELF_TEST_PROMPT_SENTINEL,
                timeout=timeout_s,
                agent="canary",
            )
        )
        in_flight = await _wait_for_attempt(client, state="in-flight", timeout_s=timeout_s)
        check("attempt-start-observed", bool(in_flight.get("attempt_id")))
        check(
            "attempt-causality-correlated",
            in_flight.get("causality_id") == SELF_TEST_CAUSALITY,
        )
        check("attempt-stage-correlated", in_flight.get("stage") == "answer")

        deadline = time.monotonic() + max(0.1, timeout_s)
        while time.monotonic() < deadline and observer.snapshot_file_count < len(
            tuple(accelerated_delays)
        ):
            await asyncio.sleep(0.02)
        check(
            "accelerated-snapshots-persisted",
            observer.snapshot_file_count >= len(tuple(accelerated_delays)),
        )
        for path in sorted(observer_dir.glob("attempt-*-snapshot-*.json")):
            try:
                snapshot_payloads.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        check(
            "snapshot-in-flight",
            any(
                str((payload.get("attempt") or {}).get("state", "")) == "in-flight"
                for payload in snapshot_payloads
            ),
        )
        check(
            "snapshot-process-alive",
            any(
                (payload.get("process") or {}).get("alive") is True for payload in snapshot_payloads
            ),
        )
        check(
            "snapshot-server-status-observed",
            any(
                (payload.get("session_status") or {}).get("session_status_observed") is True
                for payload in snapshot_payloads
            ),
        )

        aborted = await client.abort(session.id)
        check("simulated-abort-accepted", aborted is True)
        try:
            await task
        except Exception:  # noqa: BLE001 - the deterministic terminal failure
            pass
        task = None
        deadline = time.monotonic() + max(0.1, timeout_s)
        while time.monotonic() < deadline and observer.traced_attempt_count < 1:
            await asyncio.sleep(0.02)
        check("terminal-trace-persisted", observer.traced_attempt_count >= 1)
        await observer.stop(flush_reason="canary-selftest")
        observer_stopped = not observer.is_running()
        check("observer-stopped", observer_stopped)
        observer = None

        check(
            "attempt-start-event",
            any(event.get("event") == "attempt-start" for event in events),
        )
        check(
            "stall-snapshot-events",
            len([event for event in events if event.get("event") == "stall-snapshot"])
            >= len(tuple(accelerated_delays)),
        )
        check(
            "attempt-trace-event",
            any(event.get("event") == "attempt-trace" for event in events),
        )
        start_events = [event for event in events if event.get("event") == "attempt-start"]
        check(
            "attempt-event-stage-field",
            bool(start_events)
            and str(start_events[0].get("stage", "")) == "answer"
            and str(start_events[0].get("causality_id", "")) == SELF_TEST_CAUSALITY,
        )
        # The real live lane renders these allow-listed fields to stdout.
        # Prove the renderer carries stage + scenario identity while still
        # never emitting the prompt or a raw event line.
        from aa.qualification.product_contract_live import (
            _format_bounded_attempt_event,
        )

        formatted_event = (
            _format_bounded_attempt_event(start_events[0], {SELF_TEST_CAUSALITY: 1})
            if start_events
            else ""
        )
        if formatted_event:
            # Real stdout proof for the Actions log: the same allow-listed
            # renderer the live lane uses, with the preflight's scenario
            # correlation. It never carries the sentinel prompt or raw SSE.
            print(f"Gate C canary opencode-attempt {formatted_event}", flush=True)
        check(
            "stdout-attempt-stage-scenario",
            "stage=answer" in formatted_event
            and "scenario_index=1" in formatted_event
            and SELF_TEST_PROMPT_SENTINEL not in formatted_event
            and SELF_TEST_SSE_SENTINEL not in formatted_event,
        )
        check(
            "raw-sse-payload-not-retained",
            all(
                SELF_TEST_SSE_SENTINEL not in json.dumps(event, ensure_ascii=False)
                for event in events
            ),
        )

        progress_path = out / "self-test-progress.json"
        os.environ[ENV_PROGRESS_PATH] = str(progress_path)
        partial = build_partial_evidence_snapshot(
            phase="canary-selftest",
            causality_id=SELF_TEST_CAUSALITY,
            scenario_total=1,
            completed_scenario_count=1,
            scenarios_started=1,
            product_outcome_counts={"selftest": 1},
            stage_snapshots=[
                {
                    "planner_outcome": "planned",
                    "retrieval_outcome": "retrieved",
                    "answer_outcome": "answered",
                    "verifier_outcome": "verified",
                    "planner_latency_ms": 1.0,
                    "retrieval_latency_ms": 1.0,
                    "answer_latency_ms": 1.0,
                    "verifier_latency_ms": 1.0,
                }
            ],
            attempt_entries=client.attempt_audit,
            request_audit=client.request_latency_audit,
            process_state={"owned_by_runtime": True, "alive": True},
            server_evidence=evidence,
        )
        record_live_progress(
            "canary-selftest",
            attempt_observer_status="active",
            real_progress=False,
            partial_evidence=partial,
        )
        progress = read_live_progress(progress_path)
        validated = validated_partial_evidence(progress)
        check("watchdog-marker-written", bool(progress))
        check("partial-evidence-valid", validated is not None)
        coverage = coverage_from_partial(validated)
        check(
            "partial-attempt-coverage",
            coverage.get("client_attempt_ledger") == COVERAGE_OBSERVED,
        )
        check("partial-server-coverage", coverage.get("server_events") == COVERAGE_OBSERVED)
        check(
            "partial-process-coverage",
            coverage.get("runtime_process_state") == COVERAGE_OBSERVED,
        )
        check(
            "partial-stage-coverage",
            coverage.get("stage_telemetry") == COVERAGE_OBSERVED,
        )

        check("no-orphan-observer", observer_stopped and task is None)
        check("no-orphan-client-worker", bool(client.quiescence_proven(SELF_TEST_CAUSALITY)))
        privacy_ok, _privacy_path = _privacy_scan(
            [observer_dir, out / "self-test-progress.json"],
            SELF_TEST_FORBIDDEN_MARKERS,
        )
        check("privacy-safe-artifacts", privacy_ok)
    except Exception as exc:  # noqa: BLE001 - report a typed self-test failure
        check("self-test-completed", False)
        failures.append(f"self-test-error-{type(exc).__name__}"[:96])
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutdown only
                pass
        if observer is not None:
            try:
                await observer.stop(flush_reason="canary-selftest-shutdown")
            except Exception:  # noqa: BLE001 - shutdown only
                observer.flush("canary-selftest-shutdown", sync=True)
        try:
            from aa.conversation.turn_causality import clear_turn_causality

            clear_turn_causality()
        except Exception:  # noqa: BLE001 - shutdown only
            pass
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        if previous_progress_env is None:
            os.environ.pop(ENV_PROGRESS_PATH, None)
        else:
            os.environ[ENV_PROGRESS_PATH] = previous_progress_env

    if "no-orphan-http-server" not in checks:
        checks["no-orphan-http-server"] = not thread.is_alive()
        if not checks["no-orphan-http-server"]:
            failures.append("no-orphan-http-server")

    report = {
        "schema_version": SCHEMA_OBSERVER_SELF_TEST,
        "status": "PASS" if not failures else "FAIL",
        "checks": dict(sorted(checks.items())),
        "failed_checks": sorted(set(failures)),
        "nominal_snapshot_delays_s": [float(value) for value in nominal_delays],
        "accelerated_snapshot_delays_s": [float(value) for value in accelerated_delays],
        "persisted_in_flight_snapshots": sum(
            1
            for payload in snapshot_payloads
            if str((payload.get("attempt") or {}).get("state", "")) == "in-flight"
        ),
        "persisted_snapshot_files": len(snapshot_payloads),
        "persisted_traces": (
            int(observer.traced_attempt_count)
            if observer is not None
            else int(len(list(observer_dir.glob("attempt-*-trace.json"))))
        ),
        "attempt_start_events": len(
            [event for event in events if event.get("event") == "attempt-start"]
        ),
        "stall_snapshot_events": len(
            [event for event in events if event.get("event") == "stall-snapshot"]
        ),
        "server_telemetry": server_telemetry,
        "watchdog_marker_written": bool(checks.get("watchdog-marker-written")),
        "partial_evidence_coverage": {
            source: coverage_from_partial(validated).get(source, COVERAGE_MISSING)
            for source in SOURCES
        },
        "privacy_safe": bool(checks.get("privacy-safe-artifacts")),
        "no_orphan": bool(
            checks.get("no-orphan-observer")
            and checks.get("no-orphan-http-server")
            and checks.get("no-orphan-client-worker")
        ),
        "duration_s": round(max(0.0, time.monotonic() - started), 3),
        "llm_calls": 0,
    }
    _write_json(out / "observer-self-test.json", report)
    return report


async def run_bounded_real_path(lane_runner: Any, *, budget_s: float) -> Any:
    """Await one canary lane under a hard wall-clock bound.

    The production client uses executor threads, so a real HTTP hang still
    yields to the loop and ``asyncio.wait_for`` can cancel the lane. A
    timeout is external non-completion: it returns an ``INCOMPLETE`` lane
    with ``campaign-budget`` so :func:`evaluate_canary` stays typed as
    ``INCONCLUSIVE / external-progress-unobserved`` instead of blaming the
    provider or the product.
    """
    from dataclasses import replace

    from aa.qualification.product_contract_live import LaneResult

    bounded = max(1.0, float(budget_s))
    try:
        result = await asyncio.wait_for(lane_runner(), timeout=bounded)
    except TimeoutError:
        return LaneResult(
            lane="gate-c-canary-real-path",
            status="INCOMPLETE",
            incomplete=("canary-budget-expired",),
            metrics={
                "canary": True,
                "real_turns": 0,
                "scenario_diagnostics": [],
                "budget_expiration_category": "campaign-budget",
                "timeout_reason": "campaign-budget",
            },
        )
    if not isinstance(result, LaneResult):
        return result
    metrics = dict(getattr(result, "metrics", {}) or {})
    metrics.setdefault("canary", True)
    return replace(result, metrics=metrics)


def _bounded_str_list(values: Sequence[Any], *, limit: int = 8, width: int = 128) -> list[str]:
    """Deduplicate and bound privacy-safe identifier lists."""
    result: list[str] = []
    for value in values:
        text = str(value or "")[:width]
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _attempt_evidence_summary(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bounded attempt/state/model projection for the canary artifact."""
    states: dict[str, int] = {}
    operations: dict[str, int] = {}
    stages: dict[str, int] = {}
    models: list[str] = []
    failure_categories: list[str] = []
    for entry in entries:
        state = str(entry.get("state", "") or "")[:24]
        if state:
            states[state] = states.get(state, 0) + 1
        operation = str(entry.get("operation", "") or "")[:48]
        if operation:
            operations[operation] = operations.get(operation, 0) + 1
        stage = str(entry.get("stage", "") or "")[:48]
        if stage:
            stages[stage] = stages.get(stage, 0) + 1
        models.append(str(entry.get("model") or ""))
        failure = str(
            entry.get("failure_category", "") or entry.get("late_failure_category", "") or ""
        )[:64]
        if failure:
            failure_categories.append(failure)
    model_requests = sum(
        count
        for operation, count in operations.items()
        if operation in ("message-text", "message-structured")
    )
    return {
        "rows": len(entries),
        "state_counts": dict(sorted(states.items())),
        "operation_counts": dict(sorted(operations.items())),
        "stage_counts": dict(sorted(stages.items())),
        "model_requests": model_requests,
        "requested_models": _bounded_str_list(models),
        "failure_categories": _bounded_str_list(failure_categories, limit=8, width=64),
    }


def _server_state_observed(evidence: Mapping[str, Any] | None) -> bool:
    """Whether the pinned server itself reported at least one session fact.

    Endpoint availability alone (``/doc`` plus a well-shaped empty status
    map) is not server-side evidence of the real request. A READY canary
    requires an actual closed-vocabulary session state, an actual terminal
    error kind, or a parsed server event, all of which are extracted by
    the same pinned-server probe the live lane uses.
    """
    if not isinstance(evidence, Mapping):
        return False
    if str(evidence.get("last_session_state", "") or "") in (
        "idle",
        "busy",
        "retry",
    ):
        return True
    if str(evidence.get("server_error_kind", "") or ""):
        return True
    counts = evidence.get("session_state_counts")
    if isinstance(counts, Mapping):
        for state in ("idle", "busy", "retry"):
            try:
                if int(counts.get(state, 0) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                continue
    event_counts = evidence.get("event_type_counts")
    if isinstance(event_counts, Mapping):
        for event_type in ("session.status", "session.idle", "session.error"):
            try:
                if int(event_counts.get(event_type, 0) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _first_attempt_typed_cause(entries: Sequence[Mapping[str, Any]]) -> str:
    """First typed failure on a real model-message attempt, if any."""
    for entry in entries:
        operation = str(entry.get("operation", "") or "")
        if operation not in ("message-text", "message-structured"):
            continue
        cause = str(
            entry.get("failure_category", "") or entry.get("late_failure_category", "") or ""
        ).strip()
        if cause:
            return cause[:64]
    return ""


_SESSION_STATUS_FAILURE_KEYS: tuple[str, ...] = (
    "session-status-http-error",
    "session-status-body-timeout",
    "session-status-body-error",
    "session-status-shape",
    "session-status-error",
)


def server_endpoints_compatible(evidence: Mapping[str, Any] | None) -> bool:
    """Whether the pinned server exposes the exact canary telemetry surface.

    ``/doc`` alone is not enough: a documented OpenAPI page can coexist with
    broken ``/session/status`` or SSE routes. Canary readiness therefore
    requires an observed, parseable status map and a streamed ``/event``
    response with no failure token. An empty (but well-shaped) status map is
    still compatible; absence of a route or an unparsed shape is not.
    """
    if not isinstance(evidence, Mapping):
        return False
    endpoints = evidence.get("endpoints")
    if not isinstance(endpoints, Mapping):
        return False
    try:
        doc = endpoints.get("/doc")
        if not isinstance(doc, Mapping) or str(doc.get("state", "")) != "supported":
            return False
        session = endpoints.get("/session/status")
        if not isinstance(session, Mapping) or "session-status" not in session:
            return False
        if any(int(session.get(key, 0) or 0) > 0 for key in _SESSION_STATUS_FAILURE_KEYS):
            return False
        if int(session.get("unknown-state", 0) or 0) > 0:
            return False
        event = endpoints.get("/event")
        if not isinstance(event, Mapping):
            return False
        if int(event.get("event-stream-status", 0) or 0) != 200:
            return False
        if int(event.get("event-stream-streamed", 0) or 0) != 1:
            return False
        if int(event.get("event-stream-http-error", 0) or 0) > 0:
            return False
        return int(event.get("event-stream-error", 0) or 0) == 0
    except (TypeError, ValueError):
        return False


def build_real_path_report(lane_result: Any) -> dict[str, Any]:
    """Project a canary-mode ``LaneResult`` into a privacy-safe report."""
    metrics = getattr(lane_result, "metrics", {}) or {}
    diagnostics = metrics.get("scenario_diagnostics") or []
    diagnostics = [dict(item) for item in diagnostics if isinstance(item, Mapping)]
    partial = metrics.get("partial_evidence")
    coverage = coverage_from_partial(partial if isinstance(partial, Mapping) else None)
    server_evidence = metrics.get("server_evidence")
    if not isinstance(server_evidence, Mapping):
        server_evidence = {"state": "unobserved", "progress_meaning": "none"}
    attempt_audit = metrics.get("attempt_audit") or []
    attempt_audit = [dict(item) for item in attempt_audit if isinstance(item, Mapping)]
    in_flight = int(metrics.get("attempt_in_flight_count", 0) or 0)
    typed_cause = ""
    budget_category = str(metrics.get("budget_expiration_category", "none") or "none")
    for diagnostic in diagnostics:
        reason = str(
            diagnostic.get("failure_reason", "") or diagnostic.get("timeout_reason", "") or ""
        )
        if not reason:
            production = str(diagnostic.get("production_outcome", "") or "")
            if production in TYPED_CANARY_FAILURE_CAUSES:
                reason = production
        if reason and not typed_cause:
            typed_cause = reason[:64]
        if reason in ("campaign-budget", "no-real-progress"):
            budget_category = reason
    if not typed_cause:
        # The scenario diagnostic remains the primary attribution. When it
        # carries no cause, the client attempt ledger is the next honest
        # source: a failed real message attempt is a typed cause, never an
        # empty success. Timeout reasons stay in the budget category.
        attempt_cause = _first_attempt_typed_cause(attempt_audit)
        if attempt_cause and attempt_cause not in ("campaign-budget", "no-real-progress"):
            typed_cause = attempt_cause
    families: list[str] = []
    for diagnostic in diagnostics:
        family = str(diagnostic.get("family", "") or "")[:64]
        if family and family not in families:
            families.append(family)
    delivered_turns = sum(
        1
        for diagnostic in diagnostics
        if not bool(diagnostic.get("delivery_timeout"))
        and str(diagnostic.get("delivery_status", "")) == "delivered"
    )
    served_models: dict[str, list[str]] = {}
    if isinstance(partial, Mapping):
        raw_served = partial.get("served_models_by_agent")
        if isinstance(raw_served, Mapping):
            for agent, models in sorted(raw_served.items()):
                if isinstance(models, (list, tuple)):
                    served_models[str(agent)[:128]] = _bounded_str_list(models)
    stage_outcomes: dict[str, Any] = {}
    if isinstance(partial, Mapping):
        raw_outcomes = partial.get("stage_outcome_counts")
        if isinstance(raw_outcomes, Mapping):
            stage_outcomes = {
                str(stage)[:48]: dict(sorted(counts.items()))
                for stage, counts in sorted(raw_outcomes.items())
                if isinstance(counts, Mapping)
            }
    return {
        "status": str(getattr(lane_result, "status", "INCOMPLETE")),
        "lane": str(getattr(lane_result, "lane", "gate-c-canary-real-path")),
        "real_turns": len(diagnostics),
        "delivered_turns": delivered_turns,
        "families": families,
        "scenario_diagnostics": diagnostics,
        "typed_cause": typed_cause,
        "budget_expiration_category": budget_category,
        "attempt_coverage": coverage.get("client_attempt_ledger", COVERAGE_MISSING),
        "stage_coverage": coverage.get("stage_telemetry", COVERAGE_MISSING),
        "server_coverage": coverage.get("server_events", COVERAGE_MISSING),
        "served_model_coverage": coverage.get("served_model_audit", COVERAGE_MISSING),
        "runtime_process_coverage": coverage.get("runtime_process_state", COVERAGE_MISSING),
        "server_evidence_state": str(server_evidence.get("state", "unobserved")),
        "server_evidence": dict(server_evidence),
        "server_endpoints_compatible": server_endpoints_compatible(server_evidence),
        "server_state_observed": _server_state_observed(server_evidence),
        "attempt_in_flight_count": in_flight,
        "in_flight_attempt_preserved": bool(in_flight > 0),
        "attempt_evidence": _attempt_evidence_summary(attempt_audit),
        "served_models_by_agent": served_models,
        "stage_outcome_counts": stage_outcomes,
        "attempt_audit": attempt_audit[-48:],
        "observer": dict(metrics.get("observer") or {}),
        "product_outcome_counts": dict(metrics.get("product_outcome_counts") or {}),
        "delivered_service_error_count": int(metrics.get("delivered_service_error_count", 0) or 0),
    }


def evaluate_canary(
    *,
    self_test: Mapping[str, Any],
    real_path: Mapping[str, Any] | None,
    preflight_budget_s: float,
    live_budget_s: float,
    elapsed_s: float,
) -> CanaryDecision:
    """Pure fail-closed canary decision.

    ``READY`` is only possible with a passing deterministic self-test,
    supported pinned-server telemetry, at least one real turn, observed
    attempt+stage evidence and no typed real-path failure or budget expiry.
    """
    del preflight_budget_s, live_budget_s, elapsed_s
    checks = self_test.get("checks")
    if not isinstance(checks, Mapping) or not checks:
        return CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_SELF_TEST_FAILED,
            "observer_self_test.checks",
            "self-test produced no checks",
        )
    failed = [str(name) for name in self_test.get("failed_checks", []) if str(name)]
    if str(self_test.get("status", "")).upper() != "PASS" or failed:
        return CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_SELF_TEST_FAILED,
            failed[0] if failed else "observer_self_test.status",
            "observer/server self-test failed",
        )
    telemetry = self_test.get("server_telemetry")
    if not isinstance(telemetry, Mapping) or str(telemetry.get("state", "")) != "supported":
        return CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_UNSUPPORTED_SERVER,
            "observer_self_test.server_telemetry.state",
            "pinned server telemetry surface is not supported",
        )
    if not bool(self_test.get("privacy_safe")):
        return CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_PRIVACY_VIOLATION,
            "observer_self_test.privacy_safe",
            "self-test artifacts failed the privacy scan",
        )
    if not bool(self_test.get("no_orphan")):
        return CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_ORPHAN_WORK,
            "observer_self_test.no_orphan",
            "self-test left observer/server/client work alive",
        )
    if not isinstance(real_path, Mapping):
        return CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_REAL_PATH_NOT_EXECUTED,
            "real_path",
            "no real-path canary report",
        )
    # Budget expiry is checked before coverage: an unresponsive HTTP path is
    # an external non-completion, not a diagnostic/product failure. The
    # in-flight attempt and any partial evidence stay in the report.
    budget = str(real_path.get("budget_expiration_category", "none") or "none")
    if budget in ("campaign-budget", "no-real-progress"):
        return CanaryDecision(
            STATUS_INCONCLUSIVE,
            CATEGORY_EXTERNAL_PROGRESS_UNOBSERVED,
            "real_path.budget_expiration_category",
            f"bounded canary budget expired ({budget}); in-flight attempt preserved",
        )
    if int(real_path.get("real_turns", 0) or 0) < 1:
        return CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_REAL_PATH_NOT_EXECUTED,
            "real_path.real_turns",
            "no real production turn was executed",
        )
    if str(real_path.get("server_evidence_state", "")) != "supported":
        return CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_UNSUPPORTED_SERVER,
            "real_path.server_evidence_state",
            "the actual pinned server did not expose the documented telemetry",
        )
    if not bool(real_path.get("server_endpoints_compatible")):
        return CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_UNSUPPORTED_SERVER,
            "real_path.server_endpoints_compatible",
            "the actual pinned server did not expose /doc, /session/status and SSE /event together",
        )
    if not bool(real_path.get("server_state_observed")):
        return CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_UNSUPPORTED_SERVER,
            "real_path.server_state_observed",
            "the actual pinned server reported no session status/event fact for the real turn",
        )
    if str(real_path.get("attempt_coverage", "")) != COVERAGE_OBSERVED:
        return CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_DIAGNOSTIC_MISSING,
            "real_path.attempt_coverage",
            "no persisted client attempt ledger evidence",
        )
    if str(real_path.get("stage_coverage", "")) != COVERAGE_OBSERVED:
        return CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_DIAGNOSTIC_MISSING,
            "real_path.stage_coverage",
            "no persisted stage telemetry evidence",
        )
    cause = str(real_path.get("typed_cause", "") or "")
    if cause:
        return CanaryDecision(
            STATUS_FAIL,
            CATEGORY_REAL_PATH_TYPED_FAILURE,
            "real_path.typed_cause",
            f"first real-path turn returned typed cause {cause}",
        )
    attempt_evidence = real_path.get("attempt_evidence")
    model_requests = 0
    if isinstance(attempt_evidence, Mapping):
        try:
            model_requests = int(attempt_evidence.get("model_requests", 0) or 0)
        except (TypeError, ValueError):
            model_requests = 0
    if model_requests < 1:
        return CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_DIAGNOSTIC_MISSING,
            "real_path.attempt_evidence.model_requests",
            "no real OpenCode model-message attempt was observed",
        )
    lane_status = str(real_path.get("status", "")).upper()
    if lane_status == "FAIL":
        return CanaryDecision(
            STATUS_FAIL,
            CATEGORY_REAL_PATH_TYPED_FAILURE,
            "real_path.status",
            "real-path canary lane failed without a typed cause",
        )
    if lane_status != "PASS":
        return CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_REAL_PATH_INCOMPLETE,
            "real_path.status",
            f"real-path canary lane is {lane_status or 'unknown'}",
        )
    return CanaryDecision(STATUS_READY, CATEGORY_READY)


def build_canary_result(
    *,
    main_sha: str,
    run_id: str,
    self_test: Mapping[str, Any],
    real_path: Mapping[str, Any] | None,
    decision: CanaryDecision,
    preflight_budget_s: float,
    live_budget_s: float,
    elapsed_s: float,
    privacy_safe_paths: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the persisted ``gate-c-canary-result.json`` payload."""
    real_path_budget = str((real_path or {}).get("budget_expiration_category", "none"))
    budget_category = real_path_budget
    if decision.status == STATUS_INCONCLUSIVE and decision.missing_field.startswith("budget."):
        # The outer preflight allowance can expire before/around the live
        # lane, in which case the lane itself has no category to report.
        budget_category = "preflight-budget"
    budget_expired = bool(
        decision.status == STATUS_INCONCLUSIVE
        and budget_category in ("campaign-budget", "no-real-progress", "preflight-budget")
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_GATE_C_CANARY_RESULT,
        "status": decision.status,
        "ready": decision.ready,
        "category": decision.category,
        "missing_field": decision.missing_field,
        "detail": decision.detail,
        "main_sha": main_sha,
        "run_sha": main_sha,
        "run_id": str(run_id)[:64],
        "stage": "gate-c-canary",
        "stage_attempt_evidence_coverage": {
            "client_attempt_ledger": (real_path or {}).get("attempt_coverage", COVERAGE_MISSING),
            "stage_telemetry": (real_path or {}).get("stage_coverage", COVERAGE_MISSING),
            "server_events": (real_path or {}).get("server_coverage", COVERAGE_MISSING),
            "served_model_audit": (real_path or {}).get("served_model_coverage", COVERAGE_MISSING),
            "runtime_process_state": (
                (real_path or {}).get(
                    "runtime_process_coverage",
                    (self_test.get("partial_evidence_coverage") or {}).get(
                        "runtime_process_state", COVERAGE_MISSING
                    ),
                )
            ),
        },
        "observer_self_test": {
            "schema_version": str(self_test.get("schema_version", "")),
            "status": str(self_test.get("status", "")),
            "failed_checks": list(self_test.get("failed_checks", [])),
            "nominal_snapshot_delays_s": list(self_test.get("nominal_snapshot_delays_s", [])),
            "accelerated_snapshot_delays_s": list(
                self_test.get("accelerated_snapshot_delays_s", [])
            ),
            "persisted_in_flight_snapshots": int(
                self_test.get("persisted_in_flight_snapshots", 0) or 0
            ),
            "persisted_traces": int(self_test.get("persisted_traces", 0) or 0),
            "attempt_start_events": int(self_test.get("attempt_start_events", 0) or 0),
            "watchdog_marker_written": bool(self_test.get("watchdog_marker_written")),
            "server_telemetry": dict(self_test.get("server_telemetry") or {}),
            "privacy_safe": bool(self_test.get("privacy_safe")),
            "no_orphan": bool(self_test.get("no_orphan")),
            "duration_s": float(self_test.get("duration_s", 0.0) or 0.0),
            "llm_calls": int(self_test.get("llm_calls", 0) or 0),
        },
        "real_path": {
            "status": str((real_path or {}).get("status", "not-executed")),
            "real_turns": int((real_path or {}).get("real_turns", 0) or 0),
            "delivered_turns": int((real_path or {}).get("delivered_turns", 0) or 0),
            "families": list((real_path or {}).get("families", [])),
            "typed_cause": str((real_path or {}).get("typed_cause", "")),
            "budget_expiration_category": str(
                (real_path or {}).get("budget_expiration_category", "none")
            ),
            "attempt_coverage": str((real_path or {}).get("attempt_coverage", COVERAGE_MISSING)),
            "stage_coverage": str((real_path or {}).get("stage_coverage", COVERAGE_MISSING)),
            "server_evidence_state": str(
                (real_path or {}).get("server_evidence_state", "unobserved")
            ),
            "server_endpoints_compatible": bool(
                (real_path or {}).get("server_endpoints_compatible")
            ),
            "server_state_observed": bool((real_path or {}).get("server_state_observed")),
            # Persist the bounded server-side facts (including a typed
            # session.error name/status when one was really observed) so an
            # operator can see the causality -> attempt -> server chain
            # without any raw payload or session id.
            "server_evidence": dict((real_path or {}).get("server_evidence") or {}),
            "in_flight_attempt_preserved": bool(
                (real_path or {}).get("in_flight_attempt_preserved")
            ),
            "attempt_evidence": dict((real_path or {}).get("attempt_evidence") or {}),
            "served_models_by_agent": dict((real_path or {}).get("served_models_by_agent") or {}),
            "stage_outcome_counts": dict((real_path or {}).get("stage_outcome_counts") or {}),
            "observer": dict((real_path or {}).get("observer") or {}),
            "scenario_diagnostics": list((real_path or {}).get("scenario_diagnostics", [])),
        },
        "budget_expiration_category": budget_category,
        "real_turns": int((real_path or {}).get("real_turns", 0) or 0),
        "delivered_turns": int((real_path or {}).get("delivered_turns", 0) or 0),
        "typed_cause": str((real_path or {}).get("typed_cause", "")),
        "families": list((real_path or {}).get("families", [])),
        "budget": {
            "preflight_budget_s": float(preflight_budget_s),
            "live_budget_s": float(live_budget_s),
            "elapsed_s": round(float(elapsed_s), 3),
            "category": budget_category,
            "expired": budget_expired,
        },
        "privacy_safe_paths": [str(path)[:200] for path in privacy_safe_paths],
        "total_time_s": round(float(elapsed_s), 3),
        "product_qualified": False,
        "qualification_evidence": False,
        "gate_scenarios_executed": 0,
        "note": (
            "Bounded diagnostic canary only. READY never qualifies the "
            "Product Contract; the unchanged 18-scenario Gate C remains the "
            "only authoritative gate run."
        ),
    }
    if extra:
        result.update({str(key): value for key, value in extra.items()})
    return result


__all__ = [
    "ACCELERATED_SNAPSHOT_DELAYS_S",
    "CATEGORY_DIAGNOSTIC_MISSING",
    "CATEGORY_EXTERNAL_PROGRESS_UNOBSERVED",
    "CATEGORY_OPENCODE_UNAVAILABLE",
    "CATEGORY_PRECONDITION_FAILED",
    "CATEGORY_READY",
    "CATEGORY_REAL_PATH_TYPED_FAILURE",
    "CATEGORY_SELF_TEST_FAILED",
    "CATEGORY_UNSUPPORTED_SERVER",
    "CanaryDecision",
    "DEFAULT_LIVE_BUDGET_S",
    "DEFAULT_PREFLIGHT_BUDGET_S",
    "EXIT_BY_STATUS",
    "NOMINAL_SNAPSHOT_DELAYS_S",
    "SCHEMA_GATE_C_CANARY_RESULT",
    "SCHEMA_OBSERVER_SELF_TEST",
    "STATUS_BLOCKED",
    "STATUS_FAIL",
    "STATUS_INCONCLUSIVE",
    "STATUS_READY",
    "TYPED_CANARY_FAILURE_CAUSES",
    "build_canary_result",
    "build_real_path_report",
    "evaluate_canary",
    "run_bounded_real_path",
    "run_observer_self_test",
    "server_endpoints_compatible",
]
