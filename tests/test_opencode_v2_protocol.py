"""OpenCode v2 HTTP protocol contract tests.

These tests speak the exact OpenCode 2.2.6 HTTP surface against a stdlib fake
server: Basic-authenticated ``GET /api/info``, session CRUD, asynchronous
``POST /api/session/{id}/prompt`` admission, ``POST
/api/experimental/session/{id}/wait``, paged ``GET
/api/session/{id}/message`` and ``POST /api/session/{id}/interrupt``.

The central contract is that a successful prompt *enqueue* is never a reply:
the client must observe this turn's own terminal ``idle`` message (and its
assistant message) before returning text. Corresponding fail-closed cases
cover 401, malformed/non-JSON content, unsupported routes, 429/503 and
interrupted turns. Real-binary coverage lives in
``test_opencode_v2_real_runtime.py``.
"""

from __future__ import annotations

import asyncio
import base64
import http.server
import json
import threading
import time
import urllib.parse
from typing import Any

import pytest

from aa.opencode.client import (
    HttpOpenCodeClient,
    model_ref_payload,
)
from aa.opencode.errors import (
    OpenCodeDeterministicError,
    OpenCodeRateLimitError,
    OpenCodeSessionNotFoundError,
    OpenCodeTimeoutError,
    OpenCodeTransientError,
)
from aa.opencode.runtime import _probe_health

PASSWORD = "test-only-v2-secret"


class _V2FakeState:
    """Mutable server state for one isolated fake OpenCode v2 instance."""

    def __init__(self) -> None:
        self.password = ""
        self.clock = 1_700_000_000_000
        self.sessions: dict[str, dict[str, Any]] = {}
        self.requests: list[dict[str, Any]] = []
        self.mode = "complete"
        self.fail_status = 429
        self.page_size = 100
        self.structured_text = '{"answer":"ok","citations":["book:1"]}'
        self.served_model_override: dict[str, Any] | None = None
        self.invalid_json = False
        self.status_override: int | None = None
        self.info_payload: dict[str, Any] | None = None
        self.hide_admitted_user = False

    def next_clock(self) -> int:
        self.clock += 1
        return self.clock


class _V2FakeHandler(http.server.BaseHTTPRequestHandler):
    server_version = "FakeOpenCodeV2/2.2.6"

    def log_message(self, *args: object) -> None:
        return

    @property
    def state(self) -> _V2FakeState:
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

    def _send_html(self, status: int) -> None:
        body = b"<html><body>OpenCode SPA fallback</body></html>"
        self.send_response(status)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return None
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _authorized(self) -> bool:
        password = self.state.password
        if not password:
            return True
        token = base64.b64encode(f"opencode:{password}".encode()).decode("ascii")
        return self.headers.get("Authorization") == f"Basic {token}"

    def _record(self, method: str, path: str, body: Any = None) -> None:
        self.state.requests.append({"method": method, "path": path, "body": body})

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        self._send_json(401, {"_tag": "UnauthorizedError", "message": "Authentication required"})
        return False

    def _session_or_404(self, session_id: str) -> dict[str, Any] | None:
        session = self.state.sessions.get(session_id)
        if session is None:
            self._send_json(404, {"_tag": "SessionNotFoundError", "message": "missing"})
        return session

    def _append(self, session: dict[str, Any], message_type: str, **extra: Any) -> dict[str, Any]:
        messages = session["messages"]
        assert isinstance(messages, list)
        entry: dict[str, Any] = {
            "id": f"msg_{len(messages) + 1:04d}",
            "time": {"created": self.state.next_clock()},
            "type": message_type,
        }
        entry.update(extra)
        messages.append(entry)
        return entry

    def _assistant_extra(self, session: dict[str, Any], text: str) -> dict[str, Any]:
        model = (
            self.state.served_model_override
            or session.get("model")
            or {
                "providerID": "zen",
                "id": "fallback",
            }
        )
        return {
            "agent": session.get("agent") or "build",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "finish": "stop",
            "tokens": {
                "input": 11,
                "output": 7,
                "reasoning": 0,
                "cache": {"read": 3, "write": 0},
            },
        }

    def _complete_turn(self, session: dict[str, Any], prompt: str) -> None:
        mode = self.state.mode
        if mode == "queued":
            return
        if mode == "failed":
            error = {
                "type": "provider.rate-limit",
                "message": "provider body must never be propagated",
                "status": self.state.fail_status,
            }
            assistant = self._assistant_extra(session, "")
            assistant["finish"] = "error"
            assistant["error"] = error
            self._append(session, "assistant", **assistant)
            self._append(session, "idle", outcome="failed", error=error)
            session["outcome"] = "failed"
            return
        text = (
            self.state.structured_text if "JSON Schema:" in prompt else ("Grounded answer [book:2]")
        )
        self._append(session, "assistant", **self._assistant_extra(session, text))
        session["outcome"] = "succeeded"
        self._append(session, "idle", outcome="succeeded")

    def _message_page(self, session: dict[str, Any], query: dict[str, list[str]]) -> None:
        order = (query.get("order") or ["desc"])[0]
        cursor = (query.get("cursor") or [""])[0]
        limit = int((query.get("limit") or ["100"])[0])
        offset = 0
        page_size = min(limit, self.state.page_size)
        if cursor:
            cursor_order, _, raw_offset = cursor.partition(":")
            order = cursor_order or order
            offset = int(raw_offset or "0")
        messages = list(session["messages"])
        if order == "desc":
            messages.reverse()
        page = messages[offset : offset + page_size]
        next_offset = offset + page_size
        next_cursor = f"{order}:{next_offset}" if next_offset < len(messages) else None
        self._send_json(
            200,
            {"data": page, "cursor": {"previous": None, "next": next_cursor}},
        )

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        if path == "/api/info":
            self._record("GET", path)
            if not self._require_auth():
                return
            if self.state.invalid_json:
                self._send_html(200)
                return
            payload = self.state.info_payload or {
                "version": "2.2.6",
                "pid": 1234,
                "hostname": "test",
                "urls": [],
                "paths": {"tmp": "/tmp"},
                "capabilities": {"persistentPty": True},
            }
            self._send_json(200, payload)
            return
        if path.startswith("/api/session/"):
            self._record("GET", self.path)
            if not self._require_auth():
                return
            if self.state.status_override is not None:
                self._send_json(self.state.status_override, {"_tag": "Error"})
                return
            parts = path.split("/")
            session_id = parts[3]
            session = self._session_or_404(session_id)
            if session is None:
                return
            if path.endswith("/message"):
                self._message_page(session, query)
                return
            self._send_json(
                200,
                {
                    "data": {
                        "id": session["id"],
                        "projectID": "project-test",
                        "cost": 0,
                        "tokens": {"input": 0, "output": 0},
                        "time": {"created": session["created"], "updated": self.state.clock},
                        "title": session["title"],
                        "outcome": session.get("outcome"),
                    }
                },
            )
            return
        if path == "/global/health":
            # The retired v1 route remains available only to prove the v2
            # probe does not accept its shape.
            self._send_json(200, {"healthy": True, "version": "v1"})
            return
        self._send_json(404, {"_tag": "UnknownError", "message": "unknown"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        body = self._read_json()
        self._record("POST", path, body)
        if path == "/api/session":
            if not self._require_auth():
                return
            session_id = f"ses_v2_{len(self.state.sessions) + 1:03d}"
            session: dict[str, Any] = {
                "id": session_id,
                "title": body.get("title", "") if isinstance(body, dict) else "",
                "created": self.state.next_clock(),
                "model": None,
                "agent": None,
                "messages": [],
                "outcome": None,
            }
            self.state.sessions[session_id] = session
            self._send_json(
                200,
                {
                    "data": {
                        "id": session_id,
                        "projectID": "project-test",
                        "cost": 0,
                        "tokens": {"input": 0, "output": 0},
                        "time": {"created": session["created"], "updated": self.state.clock},
                        "title": session["title"],
                    }
                },
            )
            return
        if not path.startswith("/api/"):
            self._send_json(404, {"_tag": "UnknownError", "message": "unknown"})
            return
        if not self._require_auth():
            return
        parts = path.split("/")
        if len(parts) >= 4 and parts[2] == "session":
            session_id = parts[3]
            current = self._session_or_404(session_id)
            if current is None:
                return
            if path.endswith("/model"):
                model = body.get("model") if isinstance(body, dict) else None
                current["model"] = model
                self._append(current, "model-switched", model=model)
                self._send_json(204, None)
                return
            if path.endswith("/agent"):
                agent = body.get("agent") if isinstance(body, dict) else None
                current["agent"] = agent
                self._append(current, "agent-switched", agent=agent)
                self._send_json(204, None)
                return
            if path.endswith("/prompt"):
                prompt = body.get("text", "") if isinstance(body, dict) else ""
                if self.state.hide_admitted_user:
                    # Simulate an admitted prompt that history has not
                    # projected yet: the response carries an id that is not
                    # in the message timeline.
                    user = {
                        "id": f"msg_hidden_{len(current['messages']) + 1:04d}",
                        "time": {"created": self.state.next_clock()},
                        "type": "user",
                        "text": prompt,
                    }
                else:
                    user = self._append(current, "user", text=prompt)
                    self._complete_turn(current, prompt)
                idle_entries = [entry for entry in current["messages"] if entry["type"] == "idle"]
                if idle_entries:
                    current["outcome"] = idle_entries[-1]["outcome"]
                self._send_json(
                    200,
                    {
                        "data": {
                            "id": user["id"],
                            "sessionID": session_id,
                            "time": user["time"],
                            "type": "user",
                            "payload": {"text": prompt},
                            "delivery": "steer",
                        }
                    },
                )
                return
            if path.endswith("/interrupt"):
                active = self.state.mode == "queued"
                if active:
                    self._append(current, "idle", outcome="interrupted")
                    current["outcome"] = "interrupted"
                self._send_json(200, {"interrupted": active})
                return
        if path.startswith("/api/experimental/session/") and path.endswith("/wait"):
            session_id = path.split("/")[4]
            if self._session_or_404(session_id) is None:
                return
            self._send_json(204, None)
            return
        self._send_json(404, {"_tag": "UnknownError", "message": "unknown"})

    def do_DELETE(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        self._record("DELETE", path)
        if not path.startswith("/api/session/"):
            self._send_json(404, {"_tag": "UnknownError", "message": "unknown"})
            return
        if not self._require_auth():
            return
        session_id = path.split("/")[3]
        if self.state.sessions.pop(session_id, None) is None:
            self._send_json(404, {"_tag": "SessionNotFoundError", "message": "missing"})
            return
        self._send_json(204, None)


class _V2FakeServer(http.server.ThreadingHTTPServer):
    state: _V2FakeState

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _V2FakeHandler)
        self.state = _V2FakeState()
        self.daemon_threads = True


def _run_server() -> tuple[_V2FakeServer, threading.Thread, str]:
    server = _V2FakeServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    return server, thread, f"http://127.0.0.1:{port}"


def _client(base_url: str, password: str = PASSWORD) -> HttpOpenCodeClient:
    return HttpOpenCodeClient(base_url, protocol="v2", password=password, request_timeout=5.0)


def test_v2_info_health_shape_and_basic_auth() -> None:
    server, thread, base_url = _run_server()
    try:
        server.state.password = PASSWORD
        assert _probe_health(base_url, 1.0) == (False, "")
        healthy, version = _probe_health(base_url, 1.0, PASSWORD)
        assert healthy
        assert version == "2.2.6"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


async def test_v2_health_rejects_v1_shape() -> None:
    server, thread, base_url = _run_server()
    try:
        server.state.info_payload = {"healthy": True, "version": "v1"}
        client = _client(base_url)
        health = await client.health()
        assert not health.healthy
        assert health.version == "v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


async def test_v2_unauthorized_is_fail_closed_without_secret_leak() -> None:
    server, thread, base_url = _run_server()
    try:
        server.state.password = PASSWORD
        client = _client(base_url, password="wrong-secret-value")
        with pytest.raises(OpenCodeDeterministicError, match="http=401") as caught:
            await client.health()
        assert "wrong-secret-value" not in str(caught.value)
        assert PASSWORD not in str(caught.value)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


async def test_v2_non_json_content_is_fail_closed() -> None:
    server, thread, base_url = _run_server()
    try:
        server.state.invalid_json = True
        client = _client(base_url)
        with pytest.raises(OpenCodeDeterministicError, match="non-JSON"):
            await client.health()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_v2_unsupported_route_is_fail_closed() -> None:
    client = _client("http://127.0.0.1:1")
    with pytest.raises(OpenCodeDeterministicError, match="unsupported OpenCode v2"):
        client._sync_request("GET", "/healthz", None, 1.0)


async def test_v2_send_message_rejects_unsupported_native_format() -> None:
    client = _client("http://127.0.0.1:1")
    with pytest.raises(OpenCodeDeterministicError, match="send_structured_message"):
        await client.send_message(
            "ses_unknown",
            "structured please",
            format={"type": "json_schema", "schema": {"type": "object"}},
        )


async def test_v2_503_is_transient_and_429_is_rate_limit() -> None:
    server, thread, base_url = _run_server()
    try:
        client = _client(base_url)
        created = await client.create_session("errors")
        server.state.status_override = 503
        with pytest.raises(OpenCodeTransientError):
            await client.get_session(created.id)
        server.state.status_override = 429
        with pytest.raises(OpenCodeRateLimitError):
            await client.get_session(created.id)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


async def test_v2_queued_prompt_is_never_returned_as_a_reply() -> None:
    server, thread, base_url = _run_server()
    try:
        client = _client(base_url)
        created = await client.create_session("queue")
        server.state.mode = "queued"
        with pytest.raises(OpenCodeTimeoutError):
            await client.send_message(created.id, "queued question", timeout=0.4)

        # Only the admitted user message exists: no assistant reply, and the
        # failed message attempt never claims success.
        history = await client.list_messages(created.id)
        assert [(message.role, message.text) for message in history] == [
            ("user", "queued question")
        ]
        message_rows = [
            row for row in client.request_latency_audit if row["operation"] == "message-text"
        ]
        assert message_rows and message_rows[-1]["state"] == "failed"
        assert message_rows[-1]["success"] is False
        # The failed worker attempt is over: quiescence is proven, but no
        # completed reply was ever fabricated.
        assert client.quiescence_proven()
        # The wait route was consulted; admission alone was not completion.
        paths = [entry["path"] for entry in server.state.requests]
        assert f"/api/session/{created.id}/prompt" in paths
        assert "/api/experimental/session/" + f"{created.id}/wait" in paths
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


async def test_v2_queued_second_turn_never_reuses_previous_reply() -> None:
    server, thread, base_url = _run_server()
    try:
        client = _client(base_url)
        created = await client.create_session("queue-second")
        first = await client.send_message(created.id, "first question", timeout=5.0)
        assert first == "Grounded answer [book:2]"

        # The second prompt is admitted but never completes. Its history page
        # also contains the previous turn's assistant/idle entries, which must
        # not be mistaken for this turn's completion.
        server.state.mode = "queued"
        with pytest.raises(OpenCodeTimeoutError):
            await client.send_message(created.id, "second question", timeout=0.4)

        history = await client.list_messages(created.id)
        assert [(message.role, message.text) for message in history] == [
            ("user", "first question"),
            ("assistant", "Grounded answer [book:2]"),
            ("user", "second question"),
        ]
        message_rows = [
            row for row in client.request_latency_audit if row["operation"] == "message-text"
        ]
        assert [row["state"] for row in message_rows] == ["completed", "failed"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


async def test_v2_missing_admitted_turn_never_reuses_previous_reply() -> None:
    server, thread, base_url = _run_server()
    try:
        client = _client(base_url)
        created = await client.create_session("stale-history")
        first = await client.send_message(created.id, "first question", timeout=5.0)
        assert first == "Grounded answer [book:2]"

        # The prompt route admits a new user message, but history has not
        # projected it yet and still contains the previous turn's terminal
        # assistant/idle entries. Those entries are older than the admission
        # timestamp, so time alone cannot prove they are not this turn: the
        # client must require the admitted id and fail closed.
        server.state.mode = "queued"
        server.state.hide_admitted_user = True
        with pytest.raises(OpenCodeTimeoutError):
            await client.send_message(created.id, "second question", timeout=0.4)

        history = await client.list_messages(created.id)
        assert [(message.role, message.text) for message in history] == [
            ("user", "first question"),
            ("assistant", "Grounded answer [book:2]"),
        ]
        message_rows = [
            row for row in client.request_latency_audit if row["operation"] == "message-text"
        ]
        assert [row["state"] for row in message_rows] == ["completed", "failed"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


async def test_v2_cancelled_turn_issues_native_interrupt_and_proves_quiescence() -> None:
    server, thread, base_url = _run_server()
    turn: asyncio.Task[str] | None = None
    try:
        client = _client(base_url)
        created = await client.create_session("cancel")
        server.state.mode = "queued"
        turn = asyncio.create_task(client.send_message(created.id, "slow question", timeout=5.0))
        deadline = time.monotonic() + 2.0
        while not any(
            entry["path"] == f"/api/session/{created.id}/prompt" for entry in server.state.requests
        ):
            assert time.monotonic() < deadline, "prompt was never admitted"
            await asyncio.sleep(0.01)

        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

        # The cancellation path issues the v2 native interrupt and only
        # reports quiescence once the worker thread itself has terminated.
        deadline = time.monotonic() + 5.0
        while not client.quiescence_proven() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert client.quiescence_proven()
        paths = [entry["path"] for entry in server.state.requests]
        assert f"/api/session/{created.id}/interrupt" in paths
        message_rows = [
            row for row in client.request_latency_audit if row["operation"] == "message-text"
        ]
        assert message_rows[-1]["state"] == "cancelled"
        assert message_rows[-1]["success"] is False
    finally:
        if turn is not None and not turn.done():
            turn.cancel()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


async def test_v2_completed_turn_pulls_reply_after_idle_with_model_and_latency() -> None:
    server, thread, base_url = _run_server()
    try:
        client = _client(base_url)
        created = await client.create_session("complete")
        reply = await client.send_message(
            created.id,
            "grounded question",
            model="zen/spark",
            agent="aa-v2",
            timeout=5.0,
        )
        assert reply == "Grounded answer [book:2]"
        assert client.served_model_audit[-1] == {
            "agent": "aa-v2",
            "requested": "zen/spark",
            "served": "zen/spark",
        }
        assert client.token_usage_audit[-1] == {
            "agent": "aa-v2",
            "input": 11,
            "output": 7,
            "reasoning": 0,
            "cache_read": 3,
            "cache_write": 0,
        }
        rows = [row for row in client.request_latency_audit if row["operation"] == "message-text"]
        assert len(rows) == 1
        assert rows[0]["state"] == "completed"
        assert rows[0]["success"] is True
        assert isinstance(rows[0]["latency_ms"], float)
        assert rows[0]["latency_ms"] >= 0
        assert client.quiescence_proven()

        # The prompt was admitted before wait/history were consulted.
        paths = [entry["path"] for entry in server.state.requests]
        prompt_at = paths.index(f"/api/session/{created.id}/prompt")
        wait_at = paths.index("/api/experimental/session/" + f"{created.id}/wait")
        assert prompt_at < wait_at
        assert any("/message" in path for path in paths[wait_at:])
        # The model selector traveled as a v2 Model.Ref, not a v1 payload.
        switch = [
            entry
            for entry in server.state.requests
            if entry["path"] == f"/api/session/{created.id}/model"
        ]
        assert switch[-1]["body"] == {"model": model_ref_payload("zen/spark")}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


async def test_v2_failed_turn_is_classified_and_never_a_reply() -> None:
    server, thread, base_url = _run_server()
    try:
        client = _client(base_url)
        created = await client.create_session("failed")
        server.state.mode = "failed"
        server.state.fail_status = 429
        with pytest.raises(OpenCodeRateLimitError):
            await client.send_message(created.id, "quota question", timeout=5.0)
        history = await client.list_messages(created.id)
        # The failed assistant carries no visible text; the client returned no
        # reply for it, so a provider/quota failure cannot become a PASS.
        assert [(message.role, message.text) for message in history] == [
            ("user", "quota question"),
            ("assistant", ""),
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


async def test_v2_structured_output_is_strict_json() -> None:
    server, thread, base_url = _run_server()
    try:
        client = _client(base_url)
        created = await client.create_session("structured")
        result = await client.send_structured_message(
            created.id,
            "plan the retrieval",
            schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            model="zen/spark",
            agent="aa-planner-v2",
            retry_count=0,
            timeout=5.0,
        )
        assert result == {"answer": "ok", "citations": ["book:1"]}
        assert client.served_model_audit[-1]["served"] == "zen/spark"

        server.state.structured_text = "not a json object"
        with pytest.raises(OpenCodeDeterministicError, match="structured output missing"):
            await client.send_structured_message(
                created.id,
                "plan again",
                schema={"type": "object"},
                retry_count=0,
                timeout=5.0,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


async def test_v2_history_follows_paged_cursor() -> None:
    server, thread, base_url = _run_server()
    try:
        server.state.page_size = 2
        client = _client(base_url)
        created = await client.create_session("paged")
        await client.send_message(created.id, "one", timeout=5.0)
        await client.send_message(created.id, "two", timeout=5.0)
        history = await client.list_messages(created.id, limit=4)
        assert [(message.role, message.text) for message in history] == [
            ("user", "one"),
            ("assistant", "Grounded answer [book:2]"),
            ("user", "two"),
            ("assistant", "Grounded answer [book:2]"),
        ]
        message_requests = [
            entry
            for entry in server.state.requests
            if entry["path"].startswith(f"/api/session/{created.id}/message")
        ]
        cursor_requests = [entry for entry in message_requests if "cursor=" in entry["path"]]
        assert cursor_requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


async def test_v2_interrupt_reports_active_execution() -> None:
    server, thread, base_url = _run_server()
    try:
        client = _client(base_url)
        created = await client.create_session("interrupt")
        # An active (queued) turn reports that it was interrupted and
        # persists an interrupted idle; an idle turn is a false no-op.
        server.state.mode = "queued"
        assert await client.abort(created.id) is True
        server.state.mode = "complete"
        assert await client.abort(created.id) is False
        history = await client.list_messages(created.id)
        assert history == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


async def test_v2_session_crud_and_missing_session() -> None:
    server, thread, base_url = _run_server()
    try:
        client = _client(base_url)
        created = await client.create_session("crud")
        assert (await client.get_session(created.id)).id == created.id
        assert await client.delete_session(created.id) is True
        with pytest.raises(OpenCodeSessionNotFoundError):
            await client.get_session(created.id)
        with pytest.raises(OpenCodeSessionNotFoundError):
            await client.delete_session(created.id)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
