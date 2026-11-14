"""Real OpenCode 2.2.6 end-to-end runtime tests.

These tests start the actual pinned ``opencode`` binary from CI and drive the
production :class:`~aa.opencode.runtime.LocalOpenCodeRuntime` /
:class:`~aa.opencode.client.HttpOpenCodeClient` boundary. A local, offline
OpenAI-compatible mock provider makes the model turn deterministic without
any external token or Telegram access.

Covered here:

- the exact pinned binary exposes ``GET /api/info`` behind Basic auth;
- a real ``POST /api/session/{id}/prompt`` is only an enqueue: a turn that has
  not finished cannot be returned as a reply;
- the final reply is pulled after the real agent loop reaches its terminal
  ``idle`` message, with served-model identity, token usage and latency;
- strict structured JSON output round-trips;
- session CRUD and idle interrupt semantics.

The fake HTTP protocol contract lives in ``test_opencode_v2_protocol.py``.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from aa.opencode.client import HttpOpenCodeClient
from aa.opencode.errors import (
    OpenCodeSessionNotFoundError,
    OpenCodeTimeoutError,
)
from aa.opencode.runtime import LocalOpenCodeRuntime, OpenCodeConfig, _probe_health

ROOT = Path(__file__).resolve().parents[1]
PIN = (ROOT / "docker" / "opencode.version").read_text(encoding="utf-8").strip()

HAS_OPENCODE = shutil.which("opencode") is not None
needs_opencode = pytest.mark.skipif(not HAS_OPENCODE, reason="opencode binary missing")

MOCK_MODEL = "mock/mock-model"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _MockProviderServer(http.server.ThreadingHTTPServer):
    """Local OpenAI-compatible provider with an optional completion gate."""

    started: threading.Event
    release: threading.Event
    reply_text: str
    structured_reply: str
    fail_status: int
    requests: list[dict[str, Any]]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _MockProviderHandler)
        self.daemon_threads = True
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.reply_text = "Grounded answer [book:1]"
        self.structured_reply = '{"answer":"ok","citations":["book:1"]}'
        self.fail_status = 0
        self.requests = []


class _MockProviderHandler(http.server.BaseHTTPRequestHandler):
    """Streams one deterministic OpenAI chat-completion chunk sequence."""

    def log_message(self, *args: object) -> None:
        return

    @property
    def provider(self) -> _MockProviderServer:
        server = self.server
        assert isinstance(server, _MockProviderServer)
        return server

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "mock-model",
                            "object": "model",
                            "created": 0,
                            "owned_by": "mock",
                        }
                    ],
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw or b"{}")
        except ValueError:
            payload = {}
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send_json(404, {"error": "not found"})
            return
        provider = self.provider
        provider.requests.append(payload if isinstance(payload, dict) else {})
        provider.started.set()
        if not provider.release.wait(timeout=60.0):
            self._send_json(503, {"error": "mock completion gate timed out"})
            return
        if provider.fail_status:
            self._send_json(
                provider.fail_status,
                {
                    "error": {
                        "message": "injected provider failure",
                        "type": "rate_limit_error" if provider.fail_status == 429 else "api_error",
                        "code": provider.fail_status,
                    }
                },
            )
            return
        prompt = ""
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    prompt += content + "\n"
        if "JSON Schema:" in prompt:
            reply = provider.structured_reply
        else:
            reply = provider.reply_text
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        created = int(time.time())
        base: dict[str, Any] = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": created,
            "model": "mock-model",
        }

        def emit(delta: dict[str, Any], finish: str | None = None) -> None:
            chunk = dict(base)
            chunk["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        emit({"role": "assistant", "content": reply})
        emit({}, "stop")
        final = dict(base)
        final["choices"] = []
        final["usage"] = {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19,
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _start_provider() -> tuple[_MockProviderServer, threading.Thread]:
    server = _MockProviderServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_provider(server: _MockProviderServer, thread: threading.Thread) -> None:
    server.release.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5.0)


def _write_project_config(workdir: Path, provider_port: int) -> None:
    config = {
        "provider": {
            "mock": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Mock Provider",
                "options": {
                    "baseURL": f"http://127.0.0.1:{provider_port}/v1",
                    "apiKey": "test-only-key",
                },
                "models": {"mock-model": {"name": "Mock Model"}},
            }
        }
    }
    (workdir / "opencode.json").write_text(json.dumps(config), encoding="utf-8")


def _runtime(workdir: Path) -> LocalOpenCodeRuntime:
    port = _free_port()
    return LocalOpenCodeRuntime(
        OpenCodeConfig(
            base_url=f"http://127.0.0.1:{port}",
            command="opencode",
            workdir=str(workdir),
        ),
        ready_timeout=60.0,
        shutdown_timeout=10.0,
    )


def test_pinned_opencode_binary_is_22x() -> None:
    if not HAS_OPENCODE:
        pytest.skip("opencode binary missing")
    version = subprocess.run(
        ["opencode", "--version"], capture_output=True, text=True, check=True
    ).stdout
    assert PIN in version
    assert PIN.startswith("2.")


@needs_opencode
async def test_real_v2_info_requires_basic_auth(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    try:
        await runtime.start()
        healthy_without_password, _ = await asyncio.to_thread(
            _probe_health, runtime.config.base_url, 2.0
        )
        assert not healthy_without_password
        healthy, version = await asyncio.to_thread(
            _probe_health, runtime.config.base_url, 2.0, runtime._auth_password
        )
        assert healthy
        assert version == PIN
    finally:
        await runtime.stop()


@needs_opencode
async def test_real_v2_full_turn_completes_with_model_usage_and_latency(
    tmp_path: Path,
) -> None:
    provider, thread = _start_provider()
    _write_project_config(tmp_path, int(provider.server_address[1]))
    runtime = _runtime(tmp_path)
    try:
        await runtime.start()
        assert runtime.ready
        client = runtime.client
        assert isinstance(client, HttpOpenCodeClient)
        created = await client.create_session("real-v2")
        reply = await client.send_message(
            created.id,
            "grounded question",
            model=MOCK_MODEL,
            agent="build",
            timeout=60.0,
        )
        assert reply == provider.reply_text
        assert client.served_model_audit[-1] == {
            "agent": "build",
            "requested": MOCK_MODEL,
            "served": MOCK_MODEL,
        }
        usage = client.token_usage_audit[-1]
        assert usage["agent"] == "build"
        assert usage["input"] == 12
        assert usage["output"] == 7
        message_rows = [
            row for row in client.request_latency_audit if row["operation"] == "message-text"
        ]
        assert len(message_rows) == 1
        assert message_rows[0]["state"] == "completed"
        assert message_rows[0]["success"] is True
        latency = message_rows[0]["latency_ms"]
        assert isinstance(latency, (int, float))
        assert float(latency) >= 0.0
        history = await client.list_messages(created.id)
        assert [message.role for message in history] == ["user", "assistant"]
        assert history[-1].text == provider.reply_text

        password = runtime._auth_password
        assert password
        assert password not in repr(client.request_latency_audit)
        assert password not in repr(runtime.runtime_events)
        assert password not in repr(runtime.runtime_state)
    finally:
        await runtime.stop()
        _stop_provider(provider, thread)


@needs_opencode
async def test_real_v2_queued_prompt_is_not_a_completed_reply(tmp_path: Path) -> None:
    provider, thread = _start_provider()
    provider.release.clear()
    _write_project_config(tmp_path, int(provider.server_address[1]))
    runtime = _runtime(tmp_path)
    turn: asyncio.Task[str] | None = None
    try:
        await runtime.start()
        client = runtime.client
        created = await client.create_session("queued-v2")
        turn = asyncio.create_task(
            client.send_message(
                created.id,
                "slow grounded question",
                model=MOCK_MODEL,
                timeout=60.0,
            )
        )
        started = await asyncio.to_thread(provider.started.wait, 20.0)
        assert started, "the real provider request never started"
        assert not turn.done(), "enqueue must not be treated as a completed reply"

        in_flight = await client.list_messages(created.id)
        assert [message.role for message in in_flight] == ["user"]

        provider.release.set()
        reply = await asyncio.wait_for(turn, timeout=60.0)
        assert reply == provider.reply_text
        history = await client.list_messages(created.id)
        assert [message.role for message in history] == ["user", "assistant"]
        assert history[-1].text == provider.reply_text
    finally:
        provider.release.set()
        if turn is not None and not turn.done():
            turn.cancel()
        await runtime.stop()
        _stop_provider(provider, thread)


@needs_opencode
async def test_real_v2_queued_second_turn_never_reuses_previous_reply(tmp_path: Path) -> None:
    provider, thread = _start_provider()
    _write_project_config(tmp_path, int(provider.server_address[1]))
    runtime = _runtime(tmp_path)
    turn: asyncio.Task[str] | None = None
    try:
        await runtime.start()
        client = runtime.client
        created = await client.create_session("queue-second-v2")
        first = await client.send_message(
            created.id,
            "first grounded question",
            model=MOCK_MODEL,
            timeout=60.0,
        )
        assert first == provider.reply_text

        provider.release.clear()
        turn = asyncio.create_task(
            client.send_message(
                created.id,
                "second grounded question",
                model=MOCK_MODEL,
                timeout=2.0,
            )
        )
        deadline = time.monotonic() + 20.0
        while len(provider.requests) < 2 and not turn.done() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert len(provider.requests) >= 2, "the second real provider request never started"
        assert not turn.done(), "the queued second turn must not complete"

        with pytest.raises(OpenCodeTimeoutError):
            await turn
        history = await client.list_messages(created.id)
        # The previous turn's assistant reply remains in history but was not
        # returned as this turn's answer.
        assert [message.role for message in history] == ["user", "assistant", "user"]
        assert history[1].text == provider.reply_text
        assert history[0].text == "first grounded question"
        assert history[-1].text == "second grounded question"
    finally:
        provider.release.set()
        if turn is not None and not turn.done():
            turn.cancel()
        await runtime.stop()
        _stop_provider(provider, thread)


@needs_opencode
async def test_real_v2_bounded_wait_never_forges_a_reply(tmp_path: Path) -> None:
    provider, thread = _start_provider()
    provider.release.clear()
    _write_project_config(tmp_path, int(provider.server_address[1]))
    runtime = _runtime(tmp_path)
    try:
        await runtime.start()
        client = runtime.client
        created = await client.create_session("timeout-v2")
        with pytest.raises(OpenCodeTimeoutError):
            await client.send_message(
                created.id,
                "never finishes",
                model=MOCK_MODEL,
                timeout=0.5,
            )
        provider.release.set()
    finally:
        provider.release.set()
        await runtime.stop()
        _stop_provider(provider, thread)


@needs_opencode
async def test_real_v2_paged_history_follows_cursor(tmp_path: Path) -> None:
    provider, thread = _start_provider()
    _write_project_config(tmp_path, int(provider.server_address[1]))
    runtime = _runtime(tmp_path)
    try:
        await runtime.start()
        client = runtime.client
        created = await client.create_session("paged-v2")
        for prompt in ("first paged question", "second paged question"):
            reply = await client.send_message(created.id, prompt, model=MOCK_MODEL, timeout=60.0)
            assert reply == provider.reply_text
        history = await client.list_messages(created.id, limit=4)
        assert [message.role for message in history] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert history[0].text == "first paged question"
        assert history[2].text == "second paged question"
    finally:
        await runtime.stop()
        _stop_provider(provider, thread)


@needs_opencode
async def test_real_v2_session_survives_server_restart(tmp_path: Path) -> None:
    provider, thread = _start_provider()
    _write_project_config(tmp_path, int(provider.server_address[1]))
    first_runtime = _runtime(tmp_path)
    second_runtime: LocalOpenCodeRuntime | None = None
    try:
        await first_runtime.start()
        created = await first_runtime.client.create_session("restart-v2")
        reply = await first_runtime.client.send_message(
            created.id,
            "hello restart",
            model=MOCK_MODEL,
            timeout=60.0,
        )
        assert reply == provider.reply_text
        await first_runtime.stop()

        # A fresh server process on a different port, against the same state
        # directory, must still resolve the durable session id and history.
        second_runtime = _runtime(tmp_path)
        await second_runtime.start()
        reattached = await second_runtime.client.get_session(created.id)
        assert reattached.id == created.id
        history = await second_runtime.client.list_messages(created.id)
        assert [message.role for message in history] == ["user", "assistant"]
        assert history[-1].text == provider.reply_text
    finally:
        if second_runtime is not None:
            await second_runtime.stop()
        await first_runtime.stop()
        _stop_provider(provider, thread)


@needs_opencode
async def test_real_v2_structured_output_round_trip(tmp_path: Path) -> None:
    provider, thread = _start_provider()
    _write_project_config(tmp_path, int(provider.server_address[1]))
    runtime = _runtime(tmp_path)
    try:
        await runtime.start()
        client = runtime.client
        created = await client.create_session("structured-v2")
        result = await client.send_structured_message(
            created.id,
            "plan the retrieval",
            schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            model=MOCK_MODEL,
            timeout=60.0,
            retry_count=0,
        )
        assert result == {"answer": "ok", "citations": ["book:1"]}
        assert isinstance(client, HttpOpenCodeClient)
        assert client.served_model_audit[-1]["served"] == MOCK_MODEL
    finally:
        await runtime.stop()
        _stop_provider(provider, thread)


@needs_opencode
async def test_real_v2_session_crud_and_idle_interrupt(tmp_path: Path) -> None:
    provider, thread = _start_provider()
    _write_project_config(tmp_path, int(provider.server_address[1]))
    runtime = _runtime(tmp_path)
    try:
        await runtime.start()
        client = runtime.client
        created = await client.create_session("crud-v2")
        assert (await client.get_session(created.id)).id == created.id
        assert await client.abort(created.id) is False
        assert await client.delete_session(created.id) is True
        with pytest.raises(OpenCodeSessionNotFoundError):
            await client.get_session(created.id)
    finally:
        await runtime.stop()
        _stop_provider(provider, thread)
