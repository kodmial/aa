"""kodmial/aa#555 regression tests: diagnostic health must speak OpenCode v2.

The A/B diagnostic used the retired v1 ``/global/health`` and ``/session``
routes. A healthy OpenCode 2 server therefore looked nonresponsive, the
watchdog set ``server_nonresponsive_sustained`` after four snapshots and the
Gate C observation window was aborted with a false ``server-error-proven``.

These tests run a stdlib fake OpenCode 2 server and prove the diagnostic
observers use the native, Basic-authenticated v2 surface instead:

- ``_health_probe`` reads ``GET /api/info`` and never calls ``/global/health``;
- the v1 health shape is rejected;
- ``_opencode_internal_observation`` reads ``/api/session`` and
  ``/api/session/{id}/message`` with the managed credential;
- the in-flight liveness observer passes the runtime password through;
- ``ChildWatchdog`` stays healthy against an authenticated v2 server even
  when the credential is bound after construction (the production ordering).
"""

from __future__ import annotations

import base64
import contextlib
import http.server
import importlib.util
import json
import pathlib
import sys
import threading
import time
import urllib.parse
from collections.abc import Iterator
from typing import Any

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "diagnose_gate_c_scenario2.py"
PASSWORD = "test-only-555-secret"


@pytest.fixture(scope="module")
def diag() -> Any:
    module_name = "diagnose_gate_c_scenario2_555"
    spec = importlib.util.spec_from_file_location(module_name, str(SCRIPT_PATH))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _V2State:
    """Mutable state for one isolated fake OpenCode 2 server."""

    def __init__(self) -> None:
        self.password = ""
        self.requests: list[dict[str, Any]] = []
        self.info_payload: dict[str, Any] = {
            "version": "2.2.6",
            "pid": 4321,
            "hostname": "fake",
            "urls": [],
            "paths": {"tmp": "/tmp"},
            "capabilities": {"persistentPty": True},
        }
        self.sessions: list[dict[str, Any]] = [
            {"id": "ses_555", "title": "", "time": {"created": 1, "updated": 1}}
        ]
        self.messages: list[dict[str, Any]] = [
            {"id": "msg_1", "parts": [{"type": "text"}, {"type": "tool"}]}
        ]


class _V2Handler(http.server.BaseHTTPRequestHandler):
    server_version = "FakeOpenCodeV2/2.2.6"

    def log_message(self, *args: object) -> None:
        return

    @property
    def state(self) -> _V2State:
        server = self.server
        assert isinstance(server, _V2FakeServer)
        return server.state

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if not self.state.password:
            return True
        token = base64.b64encode(f"opencode:{self.state.password}".encode()).decode("ascii")
        return self.headers.get("Authorization") == f"Basic {token}"

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        self._send_json(401, {"_tag": "UnauthorizedError", "message": "Authentication required"})
        return False

    def _record(self, method: str, path: str) -> None:
        self.state.requests.append({"method": method, "path": path})

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        self._record("GET", path)
        if path == "/api/info":
            if not self._require_auth():
                return
            self._send_json(200, self.state.info_payload)
            return
        if path == "/api/session":
            if not self._require_auth():
                return
            self._send_json(
                200,
                {
                    "data": self.state.sessions,
                    "cursor": {"previous": None, "next": None},
                },
            )
            return
        if path == "/api/session/active":
            if not self._require_auth():
                return
            self._send_json(200, {"ses_555": {"status": "busy"}})
            return
        if path.startswith("/api/session/") and path.endswith("/message"):
            if not self._require_auth():
                return
            self._send_json(
                200,
                {
                    "data": self.state.messages,
                    "cursor": {"previous": None, "next": None},
                },
            )
            return
        if path == "/global/health":
            # Retired v1 route: it still answers, but the diagnostic must not
            # depend on it. Requests are recorded so tests can assert that.
            self._send_json(200, {"healthy": True, "version": "v1"})
            return
        if path == "/session":
            self._send_json(200, self.state.sessions)
            return
        self._send_json(404, {"_tag": "UnknownError", "message": "unknown"})


class _V2FakeServer(http.server.ThreadingHTTPServer):
    state: _V2State

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _V2Handler)
        self.state = _V2State()
        self.daemon_threads = True


@contextlib.contextmanager
def _v2_server() -> Iterator[tuple[_V2State, str]]:
    server = _V2FakeServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.state, f"http://127.0.0.1:{int(server.server_address[1])}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _request_paths(state: _V2State) -> list[str]:
    return [str(item["path"]) for item in state.requests]


def _sink_events(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_health_probe_uses_authenticated_v2_info(diag: Any) -> None:
    with _v2_server() as (state, base_url):
        state.password = PASSWORD
        # No credential: fail closed with a typed credential error, never
        # fall back to the retired v1 route.
        assert diag._health_probe(base_url, 1.0) == (False, "auth-error")
        healthy, version = diag._health_probe(base_url, 1.0, PASSWORD)
        assert healthy is True
        assert version == "2.2.6"
        paths = _request_paths(state)
        assert paths.count("/api/info") == 2
        assert "/global/health" not in paths


def test_health_probe_rejects_retired_v1_health_shape(diag: Any) -> None:
    with _v2_server() as (state, base_url):
        state.info_payload = {"healthy": True, "version": "v1"}
        healthy, version = diag._health_probe(base_url, 1.0)
        assert healthy is False
        assert version == "v1"


def test_health_probe_requires_integer_pid(diag: Any) -> None:
    with _v2_server() as (state, base_url):
        state.info_payload = {"version": "2.2.6"}
        assert diag._health_probe(base_url, 1.0) == (False, "2.2.6")


def test_internal_observation_uses_authenticated_v2_routes(diag: Any) -> None:
    with _v2_server() as (state, base_url):
        state.password = PASSWORD
        observed = diag._opencode_internal_observation(base_url, "ses_555", password=PASSWORD)
        assert observed["responsive"] is True
        assert observed["sessions_endpoint_supported"] is True
        assert observed["sessions_seen"] == 1
        assert observed["messages_endpoint_supported"] is True
        assert observed["messages_seen"] == 1
        assert observed["part_kinds"] == ["text", "tool"]
        paths = _request_paths(state)
        assert "/api/session" in paths
        assert "/api/session/ses_555/message" in paths
        assert "/session" not in paths
        assert "/global/health" not in paths


def test_internal_observation_reads_native_v2_content_parts(diag: Any) -> None:
    """The real v2 message shape carries parts under ``content``."""
    with _v2_server() as (state, base_url):
        state.messages = [
            {
                "id": "msg_1",
                "content": [{"type": "text"}, {"type": "tool"}],
            }
        ]
        observed = diag._opencode_internal_observation(base_url, "ses_555")
        assert observed["responsive"] is True
        assert observed["messages_seen"] == 1
        assert observed["part_kinds"] == ["text", "tool"]


def test_internal_observation_is_fail_closed_without_credential(diag: Any) -> None:
    with _v2_server() as (state, base_url):
        state.password = PASSWORD
        observed = diag._opencode_internal_observation(base_url, "")
        assert observed["responsive"] is False
        paths = _request_paths(state)
        assert "/api/session" in paths
        assert "/session" not in paths


def test_runtime_password_prefers_runtime_then_client_then_environment(
    diag: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Runtime:
        _auth_password = "runtime-secret"

    class _RuntimeWithClient:
        _auth_password = ""
        client = type("_Client", (), {"_password": "client-secret"})()

    monkeypatch.setenv("OPENCODE_PASSWORD", "env-secret")
    assert diag._opencode_runtime_password(_Runtime()) == "runtime-secret"
    assert diag._opencode_runtime_password(_RuntimeWithClient()) == "client-secret"
    assert diag._opencode_runtime_password() == "env-secret"


def test_internal_observation_preserves_session_route_mismatch_category(
    diag: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_request(
        base_url: str,
        method: str,
        path: str,
        body: dict[str, object] | None,
        connect_timeout: float,
        op_timeout: float | None,
        password: str = "",
    ) -> tuple[bool, bool, dict[str, float], dict[str, Any]]:
        return False, False, {"status": 404}, {"fail_category": "http-error-404"}

    monkeypatch.setattr(diag, "_timed_request", fake_request)
    observed = diag._opencode_internal_observation("http://127.0.0.1:1", "ses_555")
    assert observed["sessions_endpoint_supported"] is False
    assert observed["fail_category"] == "http-error-404"


def test_internal_observation_preserves_message_route_mismatch_category(
    diag: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_request(
        base_url: str,
        method: str,
        path: str,
        body: dict[str, object] | None,
        connect_timeout: float,
        op_timeout: float | None,
        password: str = "",
    ) -> tuple[bool, bool, dict[str, float], dict[str, Any]]:
        if path == diag.OPENCODE_V2_SESSION_PATH:
            return True, False, {"status": 200}, {"_list": [{"id": "ses_555"}]}
        return False, False, {"status": 404}, {"fail_category": "http-error-404"}

    monkeypatch.setattr(diag, "_timed_request", fake_request)
    observed = diag._opencode_internal_observation("http://127.0.0.1:1", "ses_555")
    assert observed["messages_endpoint_supported"] is False
    assert observed["messages_fail_category"] == "http-error-404"


class _Empty200Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        return

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


def test_timed_request_accepts_empty_observation_response(diag: Any) -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Empty200Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{int(server.server_address[1])}"
        ok, provider_error, timings, _extra = diag._timed_request(
            base_url,
            "GET",
            diag.OPENCODE_V2_SESSION_ACTIVE_PATH,
            None,
            1.0,
            1.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert ok is True
    assert provider_error is False
    assert timings["status"] == 200


def test_watchdog_clears_auth_error_after_credential_recovery(
    diag: Any, tmp_path: pathlib.Path
) -> None:
    with _v2_server() as (state, base_url):
        state.password = PASSWORD
        counters: dict[str, Any] = {}
        sink = diag.EventSink(tmp_path / "watchdog-recovery-555.jsonl", echo=False)
        try:
            watchdog = diag.ChildWatchdog(
                probe=diag.PROBE_SCENARIO2,
                cadence_s=0.05,
                max_snapshots=1,
                anchor=threading.Event(),
                anchor_wait_timeout_s=1.0,
                sink=sink,
                checkpoint_dir=tmp_path,
                opencode_base_url=base_url,
                counters=counters,
                internal_probe=None,
                password_provider=lambda: "stale-password",
            )
            failed = watchdog._collect(at_s=0.0)
            assert failed["opencode_health"]["auth_error"] is True
            assert counters["auth_error"] is True
            watchdog.set_password_provider(lambda: PASSWORD)
            recovered = watchdog._collect(at_s=0.1)
            assert recovered["opencode_health"]["auth_error"] is False
            assert counters["auth_error"] is False
        finally:
            sink.close()


def test_watchdog_stays_healthy_against_authenticated_v2_server(
    diag: Any, tmp_path: pathlib.Path
) -> None:
    with _v2_server() as (state, base_url):
        state.password = PASSWORD
        counters: dict[str, Any] = {
            "last_stage": "opencode",
            "last_stage_at_s": 0.1,
            "last_stage_monotonic": time.monotonic(),
            "last_real_event": "stage-start:opencode",
            "last_real_monotonic": time.monotonic(),
            "typing_sends": 0,
            "delivered": False,
            "probe_start": time.monotonic(),
        }
        sink = diag.EventSink(tmp_path / "watchdog-555.jsonl", echo=False)
        anchor = threading.Event()
        watchdog = diag.ChildWatchdog(
            probe=diag.PROBE_SCENARIO2,
            cadence_s=0.05,
            max_snapshots=6,
            anchor=anchor,
            anchor_wait_timeout_s=15.0,
            sink=sink,
            checkpoint_dir=tmp_path,
            opencode_base_url=base_url,
            counters=counters,
            internal_probe=None,
        )
        # Production binds the generated credential after construction.
        watchdog.set_password_provider(lambda: PASSWORD)
        watchdog.start()
        anchor.set()
        watchdog.join(timeout=20.0)
        sink.close()

        assert not watchdog.is_alive()
        assert counters.get("server_nonresponsive_sustained") is None
        events = _sink_events(tmp_path / "watchdog-555.jsonl")
        snapshots = [event for event in events if event["event"] == "snapshot"]
        assert len(snapshots) == 6
        assert all(snapshot["opencode_health"]["ok"] is True for snapshot in snapshots)
        assert all(snapshot["consecutive_health_failures"] == 0 for snapshot in snapshots)
        assert not [event for event in events if event["event"] == "server-nonresponsive-sustained"]


def test_in_flight_liveness_passes_v2_credential(diag: Any, tmp_path: pathlib.Path) -> None:
    with _v2_server() as (state, base_url):
        state.password = PASSWORD
        sink = diag.EventSink(tmp_path / "liveness-555.jsonl", echo=False)
        stop_event = threading.Event()
        thread = threading.Thread(
            target=diag._observe_model_in_flight_liveness,
            kwargs={
                "sink": sink,
                "base_url": base_url,
                "session_id_provider": lambda: "ses_555",
                "started_monotonic": time.monotonic(),
                "stop_event": stop_event,
                "interval_s": 1.0,
                "password_provider": lambda: PASSWORD,
            },
            daemon=True,
        )
        thread.start()
        deadline = time.monotonic() + 10.0
        events: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            events = [
                event
                for event in _sink_events(tmp_path / "liveness-555.jsonl")
                if event["event"] == "model-in-flight-liveness"
            ]
            if events:
                break
            time.sleep(0.05)
        stop_event.set()
        thread.join(timeout=5.0)
        sink.close()

        assert not thread.is_alive()
        assert events
        assert events[0]["server_healthy"] is True
        assert events[0]["sessions_responsive"] is True
        assert events[0]["sessions_seen"] == 1
        assert events[0]["messages_seen"] == 1
        assert events[0]["part_kinds"] == ["text", "tool"]
        paths = _request_paths(state)
        assert "/api/info" in paths
        assert "/api/session" in paths
        assert "/global/health" not in paths
