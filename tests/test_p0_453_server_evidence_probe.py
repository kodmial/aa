"""kodmial/aa#453: server-side evidence probe and privacy of runtime capture.

The OpenCodeServerEventProbe must produce bounded, closed-vocabulary
evidence from a fake ``opencode serve`` HTTP surface — including the
documented numeric ``statusCode`` of a ``session.error`` event — and must
never retain session ids, provider messages or any raw event text (a
deliberate artificial secret marker is embedded in the fake server
payloads to prove exactly that). ``RuntimeEventCapture`` likewise never
retains any arbitrary substring of a matched server line.
"""

import http.server
import json
import subprocess
import threading
import time
from typing import cast

from aa.opencode.runtime import RuntimeEventCapture
from aa.opencode.server_events import OpenCodeServerEventProbe, probe_for_runtime

# Artificial secret marker: if any of it appears in a retained evidence
# payload, the privacy contract is broken.
SECRET = "artificial-secret-zz-marker-prompt-text"


class _FakeOpenCodeHandler(http.server.BaseHTTPRequestHandler):
    """In-test double of the pinned ``opencode serve`` evidence surface."""

    server_version = "FakeOpenCode/1.19.3"

    def log_message(self, *args: object) -> None:
        return

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @property
    def _state(self) -> dict[str, object]:
        server = self.server
        assert isinstance(server, _FakeServer)
        return server.state

    def do_GET(self) -> None:
        if self.path == "/doc":
            if bool(self._state.get("doc_supported", True)):
                self._send_json(200, self._state.get("doc_payload", {"paths": {"/session": {}}}))
            else:
                self._send_json(404, {"error": "not found"})
        elif self.path == "/session/status":
            payload = self._state.get("session_status")
            if payload is None:
                self._send_json(500, {"error": "unavailable"})
            else:
                self._send_json(200, payload)
        elif self.path == "/event":
            # A live but quiet stream sends headers, then nothing until the
            # probe's socket timeout fires.
            if bool(self._state.get("sse_quiet")):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.flush()
                quiet_s = self._state.get("sse_quiet_s", 0.4)
                time.sleep(float(quiet_s) if isinstance(quiet_s, (int, float)) else 0.4)
                return
            # Streaming surface: write the bounded window then close so the
            # probe's fixed-size read returns at EOF.
            payload = self._state.get("events")
            if payload is None:
                self._send_json(404, {"error": "not found"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for frame in str(payload).splitlines():
                if frame:
                    self.wfile.write((frame + "\n").encode("utf-8"))
        else:
            self._send_json(404, {"error": "unknown"})


class _FakeServer(http.server.ThreadingHTTPServer):
    state: dict[str, object]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _FakeOpenCodeHandler)
        self.state = {"doc_supported": True}
        self.daemon_threads = True


def _mk_server(state: dict[str, object] | None = None) -> _FakeServer:
    server = _FakeServer()
    if state is not None:
        server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


_URL = "http://127.0.0.1:0"


def _probe_for(server: _FakeServer, timeout_s: float = 1.0) -> OpenCodeServerEventProbe:
    return OpenCodeServerEventProbe(
        f"http://127.0.0.1:{server.server_address[1]}", timeout_s=timeout_s
    )


def test_cached_evidence_fails_closed_before_any_probe() -> None:
    """Cached evidence before a probe is explicitly unobserved, never idle."""
    probe = OpenCodeServerEventProbe(_URL)
    cached = probe.cached_evidence()
    assert cached["state"] == "unobserved"
    assert cached["progress_meaning"] == "none"
    payload = probe.session_status_probe_payload()
    assert payload == {
        "observed": False,
        "verdict": "server-events-session-unobserved",
        "error": "",
        "progress_meaning": "none",
    }


def test_snapshot_supported_and_closed_vocabulary() -> None:
    server = _mk_server({"doc_supported": True, "session_status": {"ses_1": {"type": "busy"}}})
    try:
        probe = _probe_for(server)
        evidence = _run(probe.snapshot())
        assert evidence["state"] == "supported"
        assert evidence["progress_meaning"] == "none"
        assert evidence["last_session_state"] == "busy"
        usage = cast(dict[str, int], evidence["session_state_counts"])
        assert usage.get("busy") == 1
        endpoints = cast(dict[str, object], evidence["endpoints"])
        doc_state = cast(dict[str, object], endpoints["/doc"])
        assert doc_state["state"] == "supported"
        # Cached evidence now mirrors the fresh snapshot.
        assert probe.cached_evidence()["last_session_state"] == "busy"
        payload = probe.session_status_probe_payload()
        assert payload == {
            "observed": True,
            "verdict": "server-events-session-busy",
            "error": "",
            "progress_meaning": "none",
        }
    finally:
        server.shutdown()
        server.server_close()


def _run(awaitable: object) -> dict[str, object]:
    import asyncio

    return asyncio.run(awaitable)  # type: ignore[arg-type]


def test_live_but_silent_sse_stream_is_not_disconnected() -> None:
    """A quiet open SSE stream is typed quiet, not a connection failure."""
    server = _mk_server(
        {
            "doc_payload": {"openapi": "3.0.0"},
            "session_status": {"ses_1": {"type": "busy"}},
            "sse_quiet": True,
            "sse_quiet_s": 0.4,
        }
    )
    try:
        probe = _probe_for(server, timeout_s=0.1)
        evidence = _run(probe.snapshot())
        # /doc lacks the documented shape, but the server answered and the
        # SSE stream opened: this is not a dead connection.
        assert evidence["state"] != "disconnected"
        assert evidence["state"] == "unsupported"
        endpoints = cast(dict[str, object], evidence["endpoints"])
        event_facts = cast(dict[str, int], endpoints["/event"])
        assert event_facts.get("event-stream-status") == 200
        assert event_facts.get("event-stream-streamed") == 1
        assert event_facts.get("event-stream-quiet") == 1
        assert "event-stream-error" not in event_facts
        assert evidence["last_session_state"] == "busy"
        assert evidence["progress_meaning"] == "none"
    finally:
        server.shutdown()
        server.server_close()


def test_sse_error_event_is_provider_confirmed_and_text_free() -> None:
    """The numeric statusCode travels; payload text and ids never do."""
    frames = "\n".join(
        [
            "data: "
            + json.dumps(
                {
                    "type": "session.status",
                    "properties": {
                        "status": {"type": "busy"},
                        "sessionID": SECRET,
                        SECRET: SECRET,
                    },
                }
            ),
            "data: "
            + json.dumps(
                {
                    "type": "session.error",
                    "properties": {
                        "error": {
                            "name": "APIError",
                            "data": {"statusCode": 429, "isRetryable": True},
                            "message": SECRET,
                        },
                        "sessionID": SECRET,
                    },
                }
            ),
        ]
    )
    server = _mk_server({"session_status": {}, "events": frames})
    try:
        probe = _probe_for(server)
        evidence = _run(probe.snapshot())
        assert evidence["state"] == "supported"
        assert evidence["server_error_kind"] == "APIError"
        assert evidence["server_error_status_code"] == 429
        assert evidence["server_error_is_retryable"] is True
        event_counts = cast(dict[str, int], evidence["event_type_counts"])
        assert event_counts.get("session.error") == 1
        encoded = json.dumps(evidence)
        assert SECRET not in encoded
        assert "sessionID" not in encoded or SECRET not in encoded
        # The provider-documented numeric fact upgrades the source type.
        from aa.qualification.live_partial_evidence import build_partial_evidence_snapshot

        partial = build_partial_evidence_snapshot(phase="heartbeat", server_evidence=evidence)
        assert partial["source_types"]["server_events"] == "provider-response-confirmed"
    finally:
        server.shutdown()
        server.server_close()


def test_sse_unknown_error_name_stays_unknown_and_text_free() -> None:
    """Schema drift is typed 'unknown', never aNamed fabricated state."""
    frames = "data: " + json.dumps(
        {
            "type": "session.error",
            "properties": {
                "error": {"name": "NewFancyServerError", "data": {"statusCode": 500}},
                "message": SECRET,
            },
        }
    )
    server = _mk_server({"session_status": {}, "events": frames})
    try:
        probe = _probe_for(server)
        evidence = _run(probe.snapshot())
        assert evidence["server_error_kind"] == "unknown"
        assert evidence["server_error_status_code"] == 500
        assert SECRET not in json.dumps(evidence)
    finally:
        server.shutdown()
        server.server_close()


def test_unsupported_and_disconnected_states_fail_closed() -> None:
    server = _mk_server({"doc_supported": False, "session_status": None, "events": None})
    try:
        probe = _probe_for(server)
        evidence = _run(probe.snapshot())
        # /doc unsupported, /session/status + /event HTTP errors.
        assert evidence["state"] == "unsupported"
        assert "last_session_state" not in evidence or evidence["last_session_state"] == ""
    finally:
        server.shutdown()
        server.server_close()

    # Dead endpoint: connection-level failure is a typed "disconnected".
    probe = OpenCodeServerEventProbe("http://127.0.0.1:9", timeout_s=0.2)
    evidence2 = _run(probe.snapshot())
    assert evidence2["state"] == "disconnected"
    assert evidence2["progress_meaning"] == "none"


def test_probe_for_runtime_caches_and_returns_none_without_endpoint() -> None:
    class _Config:
        base_url = ""

    class _NoUrlRuntime:
        config = None

    class _Runtime:
        class config:  # noqa: N801 - minimal fake
            base_url = "http://127.0.0.1:11111"

    assert probe_for_runtime(_NoUrlRuntime()) is None
    runtime = _Runtime()
    probe = probe_for_runtime(runtime)
    assert probe is not None
    assert probe_for_runtime(runtime) is probe  # cached on the runtime


def test_runtime_event_capture_never_retains_line_text() -> None:
    """Matched server lines are reduced to closed-vocabulary facts."""
    capture = RuntimeEventCapture()
    process = subprocess.Popen(
        [
            "/bin/sh",
            "-c",
            "echo fataal "
            + SECRET
            + "; echo listening on port 19191; echo 'done'; sleep 0.05; echo 'panic: boom "
            + SECRET
            + " 4422'",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    capture.attach(process)
    process.wait(timeout=15)
    for _ in range(50):
        time.sleep(0.05)
        events = capture.filtered_events()
        if events and len(events) >= 2:
            break
    capture.close()
    raw = json.dumps(list(capture.filtered_events()))
    assert SECRET not in raw
    codes = {event.get("event_code") for event in capture.filtered_events()}
    # Closed vocabulary only — never a free-form event payload.
    assert codes <= {"server-fatal-exit", "server-listening", "server-error"}
    assert all(event.get("bounded") is True for event in capture.filtered_events())
    numeric = [
        event["numeric_value"] for event in capture.filtered_events() if "numeric_value" in event
    ]
    assert numeric, "the bounded numeric fact (port/exit code) must survive"
    capture.record_process_exit(exit_code=0)
    exits = [event for event in capture.filtered_events() if event.get("event") == "process-exit"]
    assert exits and exits[-1].get("exit_code") == 0
