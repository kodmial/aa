"""kodmial/aa#453: real HTTP-level fake OpenCode serve + forced watchdog kill.

The other #453 regressions drive the evidence contract with synthetic
audits; this test drives the REAL ``HttpOpenCodeClient`` against a local
fake ``opencode serve`` implementing the documented surface of the pinned
``1.19.3`` version:

- 2 completed conversations (one served turn + one provider service error
  confirmed BOTH by the ``info.error`` payload classification of the
  client and by a server-side ``session.error`` event carrying the
  documented numeric ``statusCode``),
- 1 hung in-flight request (the server accepts and never answers),
- then a wedged lane, preempted by the outer watchdog well inside one
  minute of wall clock.

The timeout FAIL lane must preserve the measured client-ledger facts
(attempt audit, served models, token counters, request latencies), the
provider-confirmed server-side error fact, honest latency aggregates over
the single real delivered sample, and distinct event identity — with the
prompt/reply sentinels and the fake session id never leaking into any
published artifact field.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
from collections import deque
from pathlib import Path

import pytest

from aa.conversation.turn_causality import set_turn_causality
from aa.opencode.client import HttpOpenCodeClient
from aa.opencode.errors import OpenCodeError
from aa.opencode.runtime import LocalOpenCodeRuntime, OpenCodeConfig
from aa.qualification.live_watchdog import (
    ENV_SCENARIO_INACTIVITY_S,
    ENV_TOTAL_DEADLINE_S,
    record_live_progress,
    supervise_live_gate_c,
)
from aa.qualification.product_contract_live import LaneResult

# Deliberate sentinels: if any of them reaches the published FAIL artifact,
# the privacy contract is broken.
PROMPT_SENTINEL = "PROMPTSENTINEL453"
PROMPT_SENTINEL_2 = "PROMPTSENTINEL453B"
REPLY_SENTINEL = "REPLYSENTINEL453"
SESSION_SENTINEL = "ses453opaque"

# The bounded causality id the child stamps on the hung turn.
CAUSALITY = "ca" * 16


class _FakeOpenCodeServe(http.server.BaseHTTPRequestHandler):
    """In-test double of the pinned ``opencode serve`` evidence surface."""

    protocol_version = "HTTP/1.0"
    server_version = "FakeOpenCodeServe/1.19.3"

    def log_message(self, *args: object) -> None:  # pragma: no cover - quiet
        return

    def _server(self) -> _FakeServe:
        server = self.server
        assert isinstance(server, _FakeServe)
        return server

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(str(self.headers.get("Content-Length", "0")) or "0")
        if length:
            return self.rfile.read(length)
        return b""

    def do_GET(self) -> None:
        if self.path == "/global/health":
            self._send_json(200, {"healthy": True, "version": "1.19.3"})
        elif self.path == "/doc":
            self._send_json(200, {"paths": {"/session": {}, "/event": {}}})
        elif self.path == "/session/status":
            # Session ids are the mapping keys ON THE SERVER (like the real
            # pinned surface); the probe never retains them.
            self._send_json(200, {SESSION_SENTINEL: {"type": "busy"}})
        elif self.path == "/event":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for frame in self._server().sse_frames:
                self.wfile.write((frame + "\n").encode("utf-8"))
        else:
            self._send_json(404, {"error": "unknown"})

    def do_POST(self) -> None:
        self._read_body()
        if self.path == "/session":
            self._send_json(200, {"id": SESSION_SENTINEL, "title": ""})
            return
        if "/message" in self.path:
            behavior = self._server().next_message_behavior()
            if behavior == "hang":
                # Park forever without closing the socket: the client stays
                # stuck inside its read exactly like a wedged opencode call.
                self._read_body()
                self._server().hang_pass.wait()
                return
            if behavior == "served":
                self._send_json(
                    200,
                    {
                        "info": {
                            "providerID": "prov-a",
                            "modelID": "model-a",
                            "tokens": {"input": 11, "output": 4},
                        },
                        "parts": [{"type": "text", "text": REPLY_SENTINEL}],
                    },
                )
                return
            if behavior == "service-error":
                self._send_json(
                    200,
                    {
                        "info": {
                            "providerID": "prov-a",
                            "modelID": "model-a",
                            "tokens": {"input": 7, "output": 0},
                            # The provider fault proofs only arrive in an
                            # actually received payload; the human-readable
                            # message is (deliberately) classified away.
                            "error": {
                                "name": "APIError",
                                "data": {"statusCode": 503, "isRetryable": True},
                            },
                        },
                        "parts": [{"type": "text", "text": REPLY_SENTINEL}],
                    },
                )
                return
            # The wedged turn: accepted, request fully consumed, then the
            # handler thread parks forever — exactly a hung server-side
            # session with no byte ever travelling back.
            return
        self._send_json(200, {"ok": True})


class _FakeServe(http.server.ThreadingHTTPServer):
    def __init__(self, message_behaviors: list[str], sse_frames: list[str]) -> None:
        self._behaviors = deque(message_behaviors)
        self.sse_frames = sse_frames
        self.hang_pass = threading.Event()  # never set: deliberate hung gate
        super().__init__(("127.0.0.1", 0), _FakeOpenCodeServe)
        self.daemon_threads = True

    def next_message_behavior(self) -> str:
        if not self._behaviors:
            return "hang"
        return self._behaviors.popleft()


async def run_live_child() -> None:
    """The child-side scenario: 2 done conversations + 1 hung + wedge."""
    import time

    from aa.opencode.server_events import probe_for_runtime
    from aa.qualification.live_partial_evidence import build_partial_evidence_snapshot
    from aa.qualification.live_turn_repair import request_state_from_audit
    from aa.qualification.product_contract_live import _opencode_attempt_entries

    server = _FakeServe(
        message_behaviors=["served", "service-error", "hang"],
        sse_frames=[
            "data: "
            + json.dumps({"type": "session.status", "properties": {"status": {"type": "busy"}}}),
            "data: "
            + json.dumps(
                {
                    "type": "session.error",
                    "properties": {
                        "error": {
                            "name": "APIError",
                            "data": {"statusCode": 503, "isRetryable": True},
                        }
                    },
                }
            ),
        ],
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    config = OpenCodeConfig(base_url=base_url, command="opencode", workdir=".")
    runtime = LocalOpenCodeRuntime(config, manage_process=False)
    client = runtime.client
    assert isinstance(client, HttpOpenCodeClient)
    assert probe_for_runtime(runtime) is not None

    session = (await client.create_session()).id
    set_turn_causality(CAUSALITY)

    # Turn 1: a real served conversation.
    started_perf = time.perf_counter()
    reply = await client.send_message(
        session, PROMPT_SENTINEL, agent="planner", model="prov-a/model-a"
    )
    served_ms = (time.perf_counter() - started_perf) * 1000.0
    assert REPLY_SENTINEL in reply

    # Turn 2: a completed conversation whose provider answer is a service
    # error (the client itself classifies it from the received payload).
    service_error_kind = ""
    try:
        await client.send_message(
            session, PROMPT_SENTINEL_2, agent="answer", model="prov-a/model-a"
        )
    except OpenCodeError as exc:
        service_error_kind = str(getattr(exc, "kind", ""))[:24]
    assert service_error_kind  # a real classified provider fault was observed

    # Turn 3: the hung in-flight conversation (wedged HTTP boundary).
    asyncio.create_task(
        client.send_message(session, PROMPT_SENTINEL, agent="answer", model="prov-a/model-a")
    )
    await asyncio.sleep(0.2)

    # Server-side vantage point over the same pinned documented surface.
    probe = probe_for_runtime(runtime)
    assert probe is not None
    server_evidence = await probe.snapshot()

    partial = build_partial_evidence_snapshot(
        phase="scenario-heartbeat",
        scenario_index=3,
        family="ordinary-text",
        causality_id=CAUSALITY,
        current_request_state=str(
            request_state_from_audit(client.request_latency_audit, CAUSALITY).get(
                "request_state", ""
            )
        ),
        scenario_total=3,
        completed_scenario_count=2,
        scenarios_started=3,
        scenarios_interrupted=0,
        product_outcome_counts={"served": 1, "service_error": 1},
        delivered_service_error_count=1,
        stage_snapshots=[
            {
                "scenario_index": 1,
                "planner_latency_ms": 12.5,
                "planner_outcome": "llm-primary",
                "retrieval_latency_ms": 3.5,
                "retrieval_outcome": "index-loaded",
                "answer_latency_ms": round(served_ms, 1),
                "answer_outcome": "served",
                "verifier_latency_ms": 0,
                "verifier_outcome": "",
            }
        ],
        attempt_entries=_opencode_attempt_entries(runtime),
        served_audit=client.served_model_audit,
        token_audit=client.token_usage_audit,
        request_audit=client.request_latency_audit,
        process_state=None,
        server_evidence=server_evidence,
    )
    record_live_progress(
        "scenario-stage",
        active_stage="planner",
        scenario_index=3,
        family="ordinary-text",
        scenario_total=3,
        completed_scenario_count=2,
        scenario_diagnostics=[
            {
                "scenario_index": 1,
                "family": "ordinary-text",
                "delivery_timeout": False,
                "production_outcome": "served",
            },
            {
                "scenario_index": 2,
                "family": "ordinary-text",
                "delivery_timeout": False,
                "production_outcome": "service_error",
            },
        ],
        latency_samples=[
            {
                "causality_id": "c1",
                "family": "ordinary-text",
                "outcome": "served",
                "latency_ms": round(served_ms, 1),
                "censored": False,
                "product_outcome": "served",
            }
        ],
        real_progress=True,
        partial_evidence=partial,
    )
    # ... then the lane wedges forever; the watchdog must preempt it.
    await asyncio.sleep(60.0)


def test_watchdog_kill_of_http_hung_lane_publishes_measured_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    progress_path = tmp_path / "p0453-http-progress.json"
    monkeypatch.setenv("AA_QUALIFICATION_LIVE_PROGRESS_PATH", str(progress_path))
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "20")
    monkeypatch.setenv(ENV_SCENARIO_INACTIVITY_S, "1.0")
    monkeypatch.delenv("AA_QUALIFICATION_WATCHDOG_RECOVERY_ALLOWANCE_S", raising=False)
    monkeypatch.delenv("AA_QUALIFICATION_JOB_STARTED_AT", raising=False)
    monkeypatch.delenv("AA_QUALIFICATION_JOB_STARTED_MONOTONIC", raising=False)

    async def lane(run_id: str) -> LaneResult:
        _ = run_id
        await run_live_child()
        # Unreachable on the wedged path; typed for the lane contract.
        return LaneResult(lane="live-telegram-evidence", status="FAIL")

    import time

    started = time.monotonic()
    result = supervise_live_gate_c(lane, run_id="p0453-http", main_sha="d" * 40)
    elapsed = time.monotonic() - started
    # DoD: fake hung OpenCode + simulated kill must complete inside one minute.
    assert elapsed < 60.0

    assert result.status == "FAIL"
    metrics = result.metrics
    assert metrics["watchdog_trigger"] == "no-real-progress"
    assert metrics["watchdog_diagnostics_status"] == "observed-partial"

    # --- Measured client-ledger facts survive the kill --------------------
    attempts = metrics["opencode_attempt_audit"]
    assert isinstance(attempts, list) and len(attempts) == 4
    operations = [str(entry.get("operation")) for entry in attempts]
    assert operations.count("session-create") == 1
    assert operations.count("message-text") == 3
    message_attempts = [entry for entry in attempts if entry.get("operation") == "message-text"]
    hung = message_attempts[-1]
    assert hung["state"] == "in-flight"
    # No measured latency exists for the hung attempt: the in-flight row's
    # placeholder is never published as a measured (0 ms) turn.
    assert hung.get("latency_ms") in (None, 0.0)
    assert metrics["opencode_attempt_in_flight_count"] == 1

    # --- Served model + token usage travel exactly as measured ------------
    assert metrics["served_models_by_agent"] == {"planner": ["prov-a/model-a"]}
    assert metrics["opencode_token_usage_by_agent"] == {"input": 11, "output": 4}

    # --- Request latencies: real measurements, hung row excluded ----------
    latencies = metrics["opencode_request_latency_ms"]
    assert latencies["session-create"]["count"] == 1
    message_latencies = latencies["message-text"]
    assert message_latencies["count"] == 2
    assert message_latencies["p50"] > 0
    assert message_latencies["p95"] >= message_latencies["p50"]

    # --- Honest scenario accounting + the one real delivered latency ------
    assert metrics["product_outcome_counts"] == {"served": 1, "service_error": 1}
    assert metrics["delivered_service_error_count"] == 1
    assert metrics["turn_latency_sample_count"] == 1
    assert metrics["turn_latency_p50_ms"] == metrics["turn_latency_p95_ms"]
    assert metrics["turn_latency_p50_ms"] > 0

    # --- Server-side provider-confirmed error fact ------------------------
    server_events = metrics["opencode_server_events"]
    assert server_events["state"] == "supported"  # real /doc + /session/status + /event
    assert server_events["progress_meaning"] == "none"  # never provider-token progress
    assert server_events["server_error_kind"] == "APIError"
    assert server_events["server_error_status_code"] == 503
    assert server_events["server_error_is_retryable"] is True
    event_counts = server_events["event_type_counts"]
    assert event_counts["session.error"] == 1
    assert event_counts["session.status"] == 1
    assert metrics["partial_evidence_coverage"]["server_events"] == "observed"

    # --- Event identity: exactly one watchdog termination row -------------
    watchdog_rows = [
        item
        for item in metrics["scenario_diagnostics"]
        if item.get("event_kind") == "watchdog-campaign-termination"
    ]
    assert len(watchdog_rows) == 1 and watchdog_rows[0]["event_id"]

    # --- Privacy: no prompt/reply/session/secret leakage in the artifact --
    encoded = json.dumps(metrics)
    for sentinel in (PROMPT_SENTINEL, PROMPT_SENTINEL_2, REPLY_SENTINEL, SESSION_SENTINEL):
        assert sentinel not in encoded
