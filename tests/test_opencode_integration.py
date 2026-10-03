"""OpenCode/Zen runtime integration tests (issue #2).

Covers the fake/local boundary, per-chat session create/continue/reset,
chat isolation, timeout/error classification, graceful shutdown, log
privacy and the application readiness gate. Live-server tests run against a
real local ``opencode serve`` when the binary is available and skip
otherwise; contract tests use a stdlib fake HTTP server so CI stays
hermetic.
"""

from __future__ import annotations

import asyncio
import http.server
import io
import json
import logging
import shutil
import socket
import subprocess
import threading
import time
from typing import Any

import pytest

from aa.app import Application
from aa.config import Settings
from aa.opencode.client import (
    FakeOpenCodeClient,
    HttpOpenCodeClient,
    OpenCodeClient,
    model_payload,
)
from aa.opencode.errors import (
    OpenCodeDeterministicError,
    OpenCodeSessionNotFoundError,
    OpenCodeStartupError,
    OpenCodeTimeoutError,
    OpenCodeTransientError,
    classify_http_status,
    classify_provider_error,
)
from aa.opencode.runtime import (
    LocalOpenCodeRuntime,
    OpenCodeConfig,
    StubOpenCodeRuntime,
    _probe_health,
    split_base_url,
)
from aa.sessions.coordinator import SessionCoordinator
from aa.telegram.transport import StubTelegramTransport, TelegramReply

HAS_OPENCODE = shutil.which("opencode") is not None
needs_opencode = pytest.mark.skipif(not HAS_OPENCODE, reason="opencode binary missing")


def _settings() -> Settings:
    return Settings.from_env({})


def _stub_runtime() -> StubOpenCodeRuntime:
    return StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------------------
# Session create / continue / reset over the fake boundary.
# ---------------------------------------------------------------------------


async def test_session_create_continue_reset() -> None:
    client = FakeOpenCodeClient()
    coordinator = SessionCoordinator()
    await coordinator.start()

    first = await coordinator.ensure_opencode_session(11, client)
    assert first.startswith("ses_")
    reply = await client.send_message(first, "hello")
    assert reply

    # Continuing the same chat reuses the OpenCode session.
    same = await coordinator.ensure_opencode_session(11, client)
    assert same == first
    await client.send_message(first, "second turn")
    history = await client.list_messages(first)
    assert [message.role for message in history] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]

    # Reset binds a fresh session and retires the old one.
    reset = await coordinator.reset_opencode_session(11, client)
    assert reset != first
    with pytest.raises(OpenCodeSessionNotFoundError):
        await client.get_session(first)
    session = coordinator.get_or_create(11)
    assert session.opencode_session_id == reset
    assert session.message_count == 0
    assert session.generation == 1
    await coordinator.stop()


async def test_two_chats_cannot_share_session() -> None:
    client = FakeOpenCodeClient()
    coordinator = SessionCoordinator()

    session_a = await coordinator.ensure_opencode_session(1, client)
    session_b = await coordinator.ensure_opencode_session(2, client)
    assert session_a != session_b

    await client.send_message(session_a, "message for chat one")
    assert await client.list_messages(session_b) == []
    assert coordinator.get_opencode_session_id(1) == session_a
    assert coordinator.get_opencode_session_id(2) == session_b


async def test_reset_missing_remote_session_still_succeeds() -> None:
    client = FakeOpenCodeClient()
    coordinator = SessionCoordinator()
    coordinator.set_opencode_session_id(9, "ses_does_not_exist")
    fresh = await coordinator.reset_opencode_session(9, client)
    assert fresh.startswith("ses_")
    assert coordinator.get_opencode_session_id(9) == fresh


async def test_worker_restart_recovers_via_opencode_source_of_truth() -> None:
    """Only the chat->session map is worker state; OpenCode owns history."""
    client = FakeOpenCodeClient()
    before = SessionCoordinator()
    session_id = await before.ensure_opencode_session(7, client)
    await client.send_message(session_id, "remember this")

    # Simulate a worker restart: the new coordinator re-attaches by id.
    after = SessionCoordinator()
    after.set_opencode_session_id(7, session_id)
    info = await client.get_session(session_id)
    assert info.id == session_id
    history = await client.list_messages(session_id)
    assert len(history) == 2

    # A dangling id (remote data gone) is detected and replaced.
    after.set_opencode_session_id(8, "ses_gone_after_restart")
    with pytest.raises(OpenCodeSessionNotFoundError):
        await client.get_session("ses_gone_after_restart")
    replacement = await after.reset_opencode_session(8, client)
    assert replacement != "ses_gone_after_restart"


# ---------------------------------------------------------------------------
# Error classification.
# ---------------------------------------------------------------------------


def test_classify_http_status() -> None:
    assert isinstance(classify_http_status(400), OpenCodeDeterministicError)
    assert isinstance(classify_http_status(404), OpenCodeSessionNotFoundError)
    assert isinstance(classify_http_status(429), OpenCodeTransientError)
    assert isinstance(classify_http_status(500), OpenCodeTransientError)
    assert isinstance(classify_http_status(503), OpenCodeTransientError)
    assert classify_http_status(500).transient
    assert not classify_http_status(400).transient


def test_classify_provider_error_retryable_flag() -> None:
    retryable = classify_provider_error(
        {"name": "APIError", "data": {"statusCode": 500, "isRetryable": True}}
    )
    assert isinstance(retryable, OpenCodeTransientError)
    fatal = classify_provider_error(
        {"name": "APIError", "data": {"statusCode": 400, "isRetryable": False}}
    )
    assert isinstance(fatal, OpenCodeDeterministicError)
    assert isinstance(classify_provider_error(None), OpenCodeDeterministicError)


def test_model_hint_parsing() -> None:
    assert model_payload("") is None
    assert model_payload("zen/spark") == {"providerID": "zen", "modelID": "spark"}
    with pytest.raises(OpenCodeDeterministicError):
        model_payload("bare-model-name")


async def test_send_empty_prompt_is_deterministic() -> None:
    client = FakeOpenCodeClient()
    info = await client.create_session("t")
    with pytest.raises(OpenCodeDeterministicError):
        await client.send_message(info.id, "   ")


async def test_fake_timeout_is_transient() -> None:
    client = FakeOpenCodeClient(latency=5.0)
    info = await client.create_session("t")
    with pytest.raises(OpenCodeTimeoutError):
        await client.send_message(info.id, "hi", timeout=0.01)


async def test_injected_transient_failure_surfaces() -> None:
    client = FakeOpenCodeClient(fail_next=OpenCodeTransientError("boom"))
    with pytest.raises(OpenCodeTransientError):
        await client.health()


def test_split_base_url_requires_explicit_port() -> None:
    host, port = split_base_url("http://127.0.0.1:4096")
    assert (host, port) == ("127.0.0.1", 4096)
    with pytest.raises(OpenCodeStartupError):
        split_base_url("http://127.0.0.1:0")
    with pytest.raises(OpenCodeStartupError):
        split_base_url("not-a-url")


# ---------------------------------------------------------------------------
# HTTP contract against a stdlib fake OpenCode server.
# ---------------------------------------------------------------------------


class _FakeServeHandler(http.server.BaseHTTPRequestHandler):
    """Minimal in-test double of the ``opencode serve`` REST surface."""

    server_version = "FakeOpenCode/1.0"

    def log_message(self, *args: object) -> None:
        return

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return None
        return json.loads(self.rfile.read(length).decode("utf-8"))

    @property
    def _state(self) -> dict[str, Any]:
        server = self.server
        assert isinstance(server, _FakeServeServer)
        return server.state

    def do_GET(self) -> None:
        if self.path == "/global/health":
            self._send_json(200, {"healthy": True, "version": "test-1.0"})
        elif self.path.startswith("/session/ses_missing"):
            self._send_json(404, {"error": "not found"})
        elif self.path.startswith("/session/") and self.path.endswith("/message"):
            self._send_json(200, [])
        elif self.path.startswith("/session/"):
            session_id = self.path.split("/")[2].split("?")[0]
            self._send_json(200, {"id": session_id, "title": "t"})
        else:
            self._send_json(404, {"error": "unknown"})

    def do_POST(self) -> None:
        body = self._read_json()
        self._state.setdefault("requests", []).append(
            {"method": "POST", "path": self.path, "body": body}
        )
        if self.path == "/session":
            self._send_json(200, {"id": "ses_contract01", "title": "t"})
        elif self.path.endswith("/abort"):
            self._send_json(200, True)
        elif self.path.endswith("/message"):
            session_id = self.path.split("/")[2]
            if session_id == "ses_provider_fail":
                self._send_json(
                    200,
                    {
                        "info": {
                            "error": {
                                "name": "APIError",
                                "data": {"statusCode": 400, "isRetryable": False},
                            }
                        },
                        "parts": [],
                    },
                )
            else:
                self._send_json(
                    200,
                    {
                        "info": {"role": "assistant"},
                        "parts": [{"type": "text", "text": "server-reply"}],
                    },
                )
        else:
            self._send_json(404, {"error": "unknown"})

    def do_DELETE(self) -> None:
        self._state.setdefault("requests", []).append(
            {"method": "DELETE", "path": self.path, "body": None}
        )
        if "/ses_missing" in self.path:
            self._send_json(404, {"error": "not found"})
        else:
            self._send_json(200, True)


class _FakeServeServer(http.server.ThreadingHTTPServer):
    state: dict[str, Any]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _FakeServeHandler)
        self.state = {}
        self.daemon_threads = True


def _run_fake_serve() -> tuple[_FakeServeServer, threading.Thread, str]:
    server = _FakeServeServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    return server, thread, f"http://127.0.0.1:{port}"


async def test_http_contract_as_expected_by_opencode() -> None:
    server, thread, base_url = _run_fake_serve()
    try:
        client: OpenCodeClient = HttpOpenCodeClient(base_url, request_timeout=5.0)
        health = await client.health()
        assert health.healthy
        assert health.version == "test-1.0"

        created = await client.create_session("t")
        assert created.id == "ses_contract01"
        assert (await client.get_session(created.id)).id == created.id

        reply = await client.send_message(created.id, "hello")
        assert reply == "server-reply"
        assert await client.abort(created.id) is True
        assert await client.delete_session(created.id) is True

        # The worker posts the narrow prompt shape and nothing else.
        posted = [
            entry
            for entry in server.state["requests"]
            if entry["path"] == f"/session/{created.id}/message"
        ]
        assert posted and posted[0]["body"] == {"parts": [{"type": "text", "text": "hello"}]}

        with pytest.raises(OpenCodeSessionNotFoundError):
            await client.get_session("ses_missing01")
        with pytest.raises(OpenCodeSessionNotFoundError):
            await client.delete_session("ses_missing01")
    finally:
        server.shutdown()
        thread.join(timeout=5.0)


async def test_http_provider_error_is_deterministic() -> None:
    server, thread, base_url = _run_fake_serve()
    try:
        client = HttpOpenCodeClient(base_url, request_timeout=5.0)
        with pytest.raises(OpenCodeDeterministicError):
            await client.send_message("ses_provider_fail", "hello")
    finally:
        server.shutdown()
        thread.join(timeout=5.0)


async def test_http_model_pointer_is_forwarded() -> None:
    server, thread, base_url = _run_fake_serve()
    try:
        client = HttpOpenCodeClient(base_url, request_timeout=5.0)
        created = await client.create_session("t")
        await client.send_message(created.id, "hi", model="zen/spark")
        posted = [entry for entry in server.state["requests"] if entry["path"].endswith("/message")]
        assert posted[-1]["body"]["model"] == {
            "providerID": "zen",
            "modelID": "spark",
        }
    finally:
        server.shutdown()
        thread.join(timeout=5.0)


async def test_http_timeout_is_transient() -> None:
    server, thread, base_url = _run_fake_serve()
    try:
        server.state["slow"] = True
        original = _FakeServeHandler.do_GET

        def slow_get(self: _FakeServeHandler) -> None:
            time.sleep(1.0)
            original(self)

        _FakeServeHandler.do_GET = slow_get  # type: ignore[method-assign]
        try:
            client = HttpOpenCodeClient(base_url, request_timeout=0.05)
            with pytest.raises(OpenCodeTimeoutError):
                await client.health()
        finally:
            _FakeServeHandler.do_GET = original  # type: ignore[method-assign]
    finally:
        server.shutdown()
        thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Startup classification without a server.
# ---------------------------------------------------------------------------


async def test_attach_without_server_is_startup_error() -> None:
    port = _free_port()
    runtime = LocalOpenCodeRuntime(
        OpenCodeConfig(base_url=f"http://127.0.0.1:{port}", command="opencode", workdir="."),
        manage_process=False,
        ready_timeout=1.0,
    )
    with pytest.raises(OpenCodeStartupError):
        await runtime.start()
    assert not runtime.running


async def test_missing_binary_is_startup_error() -> None:
    port = _free_port()
    runtime = LocalOpenCodeRuntime(
        OpenCodeConfig(
            base_url=f"http://127.0.0.1:{port}",
            command="opencode-binary-missing",
            workdir=".",
        ),
        ready_timeout=1.0,
    )
    with pytest.raises(OpenCodeStartupError):
        await runtime.start()


async def test_ensure_ready_before_start_is_not_ready() -> None:
    from aa.opencode.errors import OpenCodeNotReadyError

    runtime = LocalOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir="."),
        manage_process=False,
    )
    with pytest.raises(OpenCodeNotReadyError):
        await runtime.ensure_ready(timeout=0.5)


# ---------------------------------------------------------------------------
# Application readiness gate.
# ---------------------------------------------------------------------------


class _NeverReadyRuntime(StubOpenCodeRuntime):
    async def ensure_ready(self, timeout: float | None = None) -> None:
        raise OpenCodeStartupError("never ready")


async def test_app_refuses_traffic_until_opencode_ready() -> None:
    transport = StubTelegramTransport()
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_NeverReadyRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
    )
    with pytest.raises(OpenCodeStartupError):
        await app.start()
    assert not app.running
    assert not transport.running


async def test_full_turn_over_fake_boundary() -> None:
    client = FakeOpenCodeClient()
    runtime = StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir="."),
        client=client,
    )
    coordinator = SessionCoordinator()
    transport = StubTelegramTransport()
    app = Application(
        _settings(), transport=transport, opencode_runtime=runtime, sessions=coordinator
    )
    await app.start()
    try:
        assert app.opencode_runtime.ready
        session_id = await coordinator.ensure_opencode_session(42, app.opencode_runtime.client)
        reply = await app.opencode_runtime.client.send_message(session_id, "ping")
        assert reply
        await app.transport.send(TelegramReply(chat_id=42, text=reply))
        assert transport.sent[-1].chat_id == 42
    finally:
        await app.stop()


# ---------------------------------------------------------------------------
# Log privacy: prompts and credentials never reach the logs.
# ---------------------------------------------------------------------------


async def test_no_prompts_or_credentials_in_logs() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    target = logging.getLogger("aa.opencode")
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)
    secret_prompt = "transfer 100 USD, token 123456:ABCDEF-secret-token-value"
    try:
        client = FakeOpenCodeClient(fail_next=OpenCodeDeterministicError("injected failure"))
        coordinator = SessionCoordinator()
        session_id = await coordinator.ensure_opencode_session(5, FakeOpenCodeClient())
        with pytest.raises(OpenCodeDeterministicError):
            await client.send_message(session_id, secret_prompt)
        ok_client = FakeOpenCodeClient()
        sid = await coordinator.ensure_opencode_session(6, ok_client)
        await ok_client.send_message(sid, secret_prompt)
        await coordinator.reset_opencode_session(6, ok_client)
    finally:
        target.removeHandler(handler)
    output = stream.getvalue()
    assert secret_prompt not in output
    assert "ABCDEF" not in output
    assert "123456" not in output


# ---------------------------------------------------------------------------
# Live local server: graceful shutdown and attach semantics.
# ---------------------------------------------------------------------------


@needs_opencode
async def test_spawned_server_shuts_down_gracefully() -> None:
    port = _free_port()
    runtime = LocalOpenCodeRuntime(
        OpenCodeConfig(base_url=f"http://127.0.0.1:{port}", command="opencode", workdir="."),
        ready_timeout=30.0,
    )
    await runtime.start()
    assert runtime.ready
    assert runtime.owns_process
    process = runtime._process
    assert process is not None
    created = await runtime.client.create_session("shutdown-probe")
    assert (await runtime.client.get_session(created.id)).id == created.id
    await runtime.stop()
    assert not runtime.running
    assert not runtime.ready
    assert process.poll() is not None
    healthy, _ = await asyncio.to_thread(_probe_health, f"http://127.0.0.1:{port}", 1.0)
    assert not healthy
    # Stopping twice stays idempotent.
    await runtime.stop()


@needs_opencode
async def test_attach_mode_never_kills_foreign_server() -> None:
    port = _free_port()
    proc = subprocess.Popen(
        ["opencode", "serve", "--port", str(port), "--hostname", "127.0.0.1"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30.0
        healthy = False
        while time.monotonic() < deadline:
            healthy, _ = await asyncio.to_thread(_probe_health, f"http://127.0.0.1:{port}", 1.0)
            if healthy:
                break
            await asyncio.sleep(0.2)
        assert healthy, "could not start probe server"

        runtime = LocalOpenCodeRuntime(
            OpenCodeConfig(base_url=f"http://127.0.0.1:{port}", command="opencode", workdir="."),
            manage_process=False,
        )
        await runtime.start()
        assert runtime.ready
        assert not runtime.owns_process
        await runtime.stop()
        assert proc.poll() is None, "attach-mode stop must not kill the server"
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(proc.wait), 10.0)
            except TimeoutError:
                proc.kill()
