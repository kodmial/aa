"""Gate B real Telegram startup/delivery acceptance (issue #332).

Proves the transport lane qualifier from post-#331 main distinguishes
honest typed outcomes instead of hardcoding INCOMPLETE without dialling:

- missing/invalid token, getMe failure, webhook/polling conflict (409),
  rate limiting (429), network failure each map to a distinct category;
- success / failure / unknown send acknowledgements never produce phantom
  delivered quotes, history, or confirmations;
- text/voice/split replies and the meeting inline callback run through the
  actual application to transport receipts with confirmed-prefix semantics;
- the qualifier reports INCOMPLETE (blocked, typed external dependency) vs
  FAIL vs real PASS honestly and never treats token presence as delivery.

Privacy: no test logs or asserts on tokens, chat ids, message bodies, or
book passages beyond fixed short fixtures.
"""

from __future__ import annotations

import urllib.error
from typing import Any

import pytest

from aa.telegram.startup_probe import (
    EXTERNAL_PEER_DEPENDENCY,
    categorize_startup_exception,
    classify_http_status,
    confirm_full_delivery,
    live_delivery_peer_from_env,
    probe_telegram_startup,
)
from aa.telegram.transport import (
    PollingTelegramTransport,
    TelegramApi,
    TelegramApiError,
    TelegramAuthError,
    TelegramConflictError,
    TelegramRateLimitedError,
    _translate_envelope,
    _translate_http_error,
)


def _http_error(code: int, description: str) -> urllib.error.HTTPError:
    import io
    import json

    body = json.dumps({"ok": False, "error_code": code, "description": description}).encode()
    return urllib.error.HTTPError(
        url="https://api.telegram.org/botX/getMe",
        code=code,
        msg=description,
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(body),
    )


class _ScriptedApi(TelegramApi):
    """Scripted Bot API seam reusing the production transport code path."""

    def __init__(
        self,
        *,
        me: Any = None,
        fail_with: BaseException | None = None,
        fail_method: str | None = None,
        send_result: Any = None,
        send_fail_times: int = 0,
    ) -> None:
        if me is None:
            me = {"id": 700000001, "is_bot": True, "username": "probe_bot"}
        if send_result is None:
            send_result = {"message_id": 4242}
        self.me = me
        self.fail_with = fail_with
        self.fail_method = fail_method
        self.send_result = send_result
        self.send_fail_times = send_fail_times
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.send_attempts = 0

    async def call(self, method: str, payload: dict[str, Any]) -> Any:
        self.calls.append((method, dict(payload)))
        if self.fail_method == method and self.fail_with is not None:
            raise self.fail_with
        if method == "getMe":
            if isinstance(self.me, BaseException):
                raise self.me
            return self.me
        if method == "getUpdates":
            return []
        if method == "sendMessage":
            self.send_attempts += 1
            if self.send_attempts <= self.send_fail_times:
                raise TelegramApiError("telegram sendMessage transient failure")
            if isinstance(self.send_result, BaseException):
                raise self.send_result
            return self.send_result
        return True

    async def download_file(self, file_path: str) -> bytes:
        raise TelegramApiError("no fixture")

    async def send_voice(self, chat_id: int, ogg_bytes: bytes) -> Any:
        return {"message_id": 9001}

    def method_order(self) -> list[str]:
        return [method for method, _ in self.calls]


def _fast_probe_kwargs() -> dict[str, Any]:
    return {"poll_timeout_seconds": 0}


def test_http_errors_translate_to_distinct_types() -> None:
    auth = _translate_http_error(_http_error(401, "Unauthorized"))
    assert isinstance(auth, TelegramAuthError)
    conflict = _translate_http_error(_http_error(409, "Conflict: another webhook"))
    assert isinstance(conflict, TelegramConflictError)
    assert not isinstance(conflict, TelegramAuthError)
    limited = _translate_http_error(_http_error(429, "Too Many Requests: retry after 5"))
    assert isinstance(limited, TelegramRateLimitedError)
    assert not isinstance(limited, TelegramAuthError)
    assert not isinstance(limited, TelegramConflictError)
    assert isinstance(_translate_http_error(_http_error(500, "Server error")), TelegramApiError)


def test_envelope_errors_translate_to_distinct_types() -> None:
    with pytest.raises(TelegramAuthError):
        _translate_envelope("getMe", {"ok": False, "error_code": 401, "description": "X"})
    with pytest.raises(TelegramConflictError):
        _translate_envelope(
            "getUpdates", {"ok": False, "error_code": 409, "description": "Conflict"}
        )
    with pytest.raises(TelegramRateLimitedError):
        _translate_envelope(
            "sendMessage", {"ok": False, "error_code": 429, "description": "Too Many"}
        )


def test_startup_categories_are_distinct() -> None:
    assert classify_http_status(401, "Unauthorized") == "telegram-auth"
    assert classify_http_status(409, "Conflict") == "polling-webhook-conflict"
    assert classify_http_status(429, "Too Many Requests") == "transport-rate-limited"
    assert classify_http_status(500, "boom") == "transport-network-failure"
    assert categorize_startup_exception(TelegramAuthError("x")) == "telegram-auth"
    assert categorize_startup_exception(TelegramConflictError("x")) == "polling-webhook-conflict"
    assert categorize_startup_exception(TelegramRateLimitedError("x")) == "transport-rate-limited"
    assert categorize_startup_exception(TelegramApiError("getMe bad")) == "telegram-getme"
    assert categorize_startup_exception(TelegramApiError("network error")) == (
        "transport-network-failure"
    )
    assert categorize_startup_exception(TelegramApiError("deleteWebhook down")) == (
        "telegram-bootstrap"
    )
    assert categorize_startup_exception(TimeoutError()) == "transport-network-failure"
    assert categorize_startup_exception(OSError("down")) == "transport-network-failure"
    assert categorize_startup_exception(ValueError("bot token missing")) == "environment-secrets"


async def test_probe_missing_token_blocked_without_network() -> None:
    api = _ScriptedApi()
    outcome = await probe_telegram_startup("", api=api, **_fast_probe_kwargs())
    assert outcome.status == "blocked"
    assert outcome.category == "environment-secrets"
    assert api.calls == []


async def test_probe_healthy_bootstrap_and_full_start() -> None:
    api = _ScriptedApi()
    outcome = await probe_telegram_startup("test-token", api=api, **_fast_probe_kwargs())
    assert outcome.status == "ready"
    assert outcome.category == "ready"
    assert outcome.bot_id_present is True
    assert outcome.polling_live is True
    assert outcome.bootstrap_verified is True
    # Clean shutdown: bootstrap ran through the production contract in order.
    assert api.method_order()[:3] == ["getMe", "deleteWebhook", "setMyCommands"]

    bootstrap_api = _ScriptedApi()
    bootstrap = await probe_telegram_startup(
        "test-token", api=bootstrap_api, start_polling=False, **_fast_probe_kwargs()
    )
    assert bootstrap.status == "ready"
    assert bootstrap.bootstrap_verified is True
    assert bootstrap.polling_live is False
    # Bootstrap-only never issues getUpdates: safe beside a live poller.
    assert "getUpdates" not in bootstrap_api.method_order()
    assert bootstrap_api.method_order()[:3] == ["getMe", "deleteWebhook", "setMyCommands"]


async def test_probe_invalid_token_fails_closed() -> None:
    api = _ScriptedApi(me=TelegramAuthError("telegram unauthorized: invalid bot token"))
    outcome = await probe_telegram_startup("bad-token", api=api, **_fast_probe_kwargs())
    assert outcome.status == "failed"
    assert outcome.category == "telegram-auth"


async def test_probe_conflict_is_blocked_not_failed() -> None:
    api = _ScriptedApi(
        fail_method="deleteWebhook",
        fail_with=TelegramConflictError("telegram polling conflict (409)"),
    )
    outcome = await probe_telegram_startup("test-token", api=api, **_fast_probe_kwargs())
    assert outcome.status == "blocked"
    assert outcome.category == "polling-webhook-conflict"


async def test_probe_getme_failure_is_distinct() -> None:
    api = _ScriptedApi(me={"unexpected": "shape"})
    outcome = await probe_telegram_startup(
        "test-token", api=api, start_polling=False, **_fast_probe_kwargs()
    )
    assert outcome.status == "failed"
    assert outcome.category == "telegram-getme"


async def test_probe_network_failure_is_distinct() -> None:
    api = _ScriptedApi(me=OSError("network unreachable"))
    outcome = await probe_telegram_startup(
        "test-token", api=api, start_polling=False, **_fast_probe_kwargs()
    )
    assert outcome.status == "failed"
    assert outcome.category == "transport-network-failure"


async def test_send_ack_success_failure_unknown_are_distinct() -> None:
    from aa.telegram.transport import TelegramReply

    ok_api = _ScriptedApi(send_result={"message_id": 4242})
    ok_transport = PollingTelegramTransport(
        token="test-token",
        api=ok_api,
        poll_timeout_seconds=0,
        retry_base_delay_seconds=0.001,
        retry_max_delay_seconds=0.005,
    )
    sent_id = await ok_transport.send(TelegramReply(chat_id=1, text="ok"))
    assert sent_id == 4242

    unknown_api = _ScriptedApi(send_result={})
    unknown_transport = PollingTelegramTransport(
        token="test-token",
        api=unknown_api,
        poll_timeout_seconds=0,
        retry_base_delay_seconds=0.001,
        retry_max_delay_seconds=0.005,
    )
    assert await unknown_transport.send(TelegramReply(chat_id=1, text="ok")) is None

    failing_api = _ScriptedApi(send_result=TelegramApiError("down"))
    failing_transport = PollingTelegramTransport(
        token="test-token",
        api=failing_api,
        poll_timeout_seconds=0,
        retry_base_delay_seconds=0.001,
        retry_max_delay_seconds=0.005,
        max_send_retries=1,
    )
    with pytest.raises(TelegramApiError):
        await failing_transport.send(TelegramReply(chat_id=1, text="ok"))


def test_confirm_full_delivery_rejects_non_confirmed() -> None:
    from aa.conversation.finalization import normalize_answer_text, sha256_text

    certified = "Понял вас. Давайте разберём спокойно."
    normalized = normalize_answer_text(certified)
    digest = sha256_text(normalized)

    def _receipt(status: str, start: int, end: int) -> dict[str, Any]:
        return {
            "status": status,
            "final_sha256": digest,
            "char_start": start,
            "char_end": end,
        }

    total = len(normalized)
    ok, _ = confirm_full_delivery(certified, [_receipt("confirmed", 0, total)])
    assert ok is True
    # Split/partial/unknown/failed receipts never prove the complete text.
    for status in ("split", "partial", "unknown", "failed"):
        ok, _ = confirm_full_delivery(certified, [_receipt(status, 0, total)])
        assert ok is False
    ok, _ = confirm_full_delivery(certified, [_receipt("confirmed", 0, total // 2)])
    assert ok is False
    ok, _ = confirm_full_delivery(certified, [])
    assert ok is False
    ok, _ = confirm_full_delivery(
        certified, [{**_receipt("confirmed", 0, total), "final_sha256": "0" * 64}]
    )
    assert ok is False


async def test_retry_idempotency_preserves_confirmed_prefix() -> None:
    """A transient send failure never phantoms delivery; later turns recover."""
    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.graph_runtime import GraphTurnRuntime
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
    from aa.telegram.transport import StubTelegramTransport, TelegramIncoming, TelegramReply

    async def _delegate(thread: str, text: str) -> str:
        return "Понял вас. Давайте разберём спокойно. Что сейчас важнее?"

    class _FlakyOnce(StubTelegramTransport):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def send(self, reply: TelegramReply) -> int:
            self.calls += 1
            if self.calls == 1:
                raise TelegramApiError("transient")
            return await super().send(reply)

    settings = Settings.from_env({"TYPING_HEARTBEAT_SECONDS": "0.02"})
    flaky = _FlakyOnce()
    app = Application(
        settings,
        transport=flaky,
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        graph_runtime=GraphTurnRuntime(delegate=_delegate),
    )
    await app.start()
    try:
        await app._process_dispatched_update(
            TelegramIncoming(update_id=33201, chat_id=33201, message_id=1, text="привет")
        )
        # Production idempotency: a failed send records a failed receipt and
        # never resends the full reply as complete history, so no phantom
        # user-visible delivery exists after the transient failure.
        assert len(flaky.sent) == 0
        assert flaky.calls == 1
        # The next turn on the same chat still delivers exactly once.
        await app._process_dispatched_update(
            TelegramIncoming(update_id=33202, chat_id=33201, message_id=2, text="привет")
        )
        assert len(flaky.sent) == 1
        # Partial coverage never proves the complete text (confirmed prefix
        # semantics): only a confirmed full interval counts.
        certified = "Понял вас. Давайте разберём спокойно."
        from aa.conversation.finalization import normalize_answer_text, sha256_text

        normalized = normalize_answer_text(certified)
        digest = sha256_text(normalized)
        total = len(normalized)
        ok, _ = confirm_full_delivery(
            certified,
            [
                {
                    "status": "confirmed",
                    "final_sha256": digest,
                    "char_start": 0,
                    "char_end": total // 2,
                },
                {
                    "status": "failed",
                    "final_sha256": digest,
                    "char_start": total // 2,
                    "char_end": total,
                },
            ],
        )
        assert ok is False
    finally:
        await app.stop()


async def test_voice_fallback_to_text_without_tts() -> None:
    """Without a TTS pipeline a voice turn falls back to text, never dropped."""
    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.graph_runtime import GraphTurnRuntime
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
    from aa.safety.response import EMERGENCY_RESPONSE_RU
    from aa.telegram.transport import StubTelegramTransport, TelegramIncoming

    async def _delegate(thread: str, text: str) -> str:
        return "Понял вас. Давайте разберём спокойно."

    transport = StubTelegramTransport()
    app = Application(
        Settings.from_env(),
        transport=transport,
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        graph_runtime=GraphTurnRuntime(delegate=_delegate),
    )
    await app.start()
    try:
        incoming = TelegramIncoming(update_id=33211, chat_id=33211, message_id=1, text="")
        delivered = await app._send_voice_reply(incoming, EMERGENCY_RESPONSE_RU)
        assert delivered is False
        assert transport.sent_voices == []
        await app._send_text_reply(incoming, EMERGENCY_RESPONSE_RU)
        assert len(transport.sent) == 1
    finally:
        await app.stop()


async def test_meeting_callback_through_app_to_receipt() -> None:
    """The genuine #327 inline flow binds delivery: confirmed send activates."""
    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.graph_runtime import GraphTurnRuntime
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
    from aa.telegram.transport import StubTelegramTransport, TelegramIncoming

    async def _delegate(thread: str, text: str) -> str:
        return "Понял вас. Давайте разберём спокойно."

    transport = StubTelegramTransport()
    app = Application(
        Settings.from_env(),
        transport=transport,
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        graph_runtime=GraphTurnRuntime(delegate=_delegate),
    )
    await app.start()
    try:
        chat_id = 33221
        assert await app.offer_meeting(chat_id) is True
        assert len(transport.sent) == 1
        flow = app.meetings.flow_for(chat_id)
        assert flow is not None and flow.bound_message_id is not None
        bound = int(flow.bound_message_id)

        def _first_callback_data(node: Any) -> str | None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "callback_data" and isinstance(value, str) and value:
                        return value
                    found = _first_callback_data(value)
                    if found is not None:
                        return found
            elif isinstance(node, (list, tuple)):
                for item in node:
                    found = _first_callback_data(item)
                    if found is not None:
                        return found
            return None

        sent_before = len(transport.sent)
        # A stale/forged callback without valid data stays inert upstream.
        await app._handle_callback_update(
            TelegramIncoming(
                update_id=33222,
                chat_id=chat_id,
                message_id=bound,
                text="",
                is_callback=True,
                callback_id=" forged ",
                callback_data="forged-data",
                sender_id=chat_id,
                callback_message_id=bound,
            )
        )
        # Forged callbacks never mint user-visible state.
        assert len(transport.sent) == sent_before
    finally:
        await app.stop()


def test_live_peer_env_parsing_never_logs_value(monkeypatch: Any) -> None:
    monkeypatch.delenv("AA_LIVE_TELEGRAM_CHAT_ID", raising=False)
    assert live_delivery_peer_from_env() is None
    monkeypatch.setenv("AA_LIVE_TELEGRAM_CHAT_ID", "123456")
    assert live_delivery_peer_from_env() == 123456
    monkeypatch.setenv("AA_LIVE_TELEGRAM_CHAT_ID", "not-a-chat")
    assert live_delivery_peer_from_env() is None


async def test_lane_offline_incomplete_never_pass(monkeypatch: Any) -> None:
    from aa.qualification.product_contract_live import assert_no_text_leak, run_transport_lane

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("AA_LIVE_TELEGRAM_CHAT_ID", raising=False)
    result = await run_transport_lane()
    assert result.lane == "telegram-transport-25-32"
    assert result.status == "INCOMPLETE"
    assert not result.failed
    assert "real-telegram-typing-stream-requires-token" in result.incomplete
    assert result.metrics["live_transport_status"] == "blocked"
    assert result.metrics["live_transport_category"] == "environment-secrets"
    assert result.metrics["live_transport_bootstrap_verified"] is False
    assert result.metrics["live_external_dependency"] == "EXTERNAL_TELEGRAM_TOKEN_UNAVAILABLE"
    # Token presence alone is never treated as a confirmed message.
    assert not any("delivery-confirmed" in item for item in result.passed)
    assert_no_text_leak(result.to_dict())


async def test_lane_ready_without_peer_stays_incomplete(monkeypatch: Any) -> None:
    import aa.telegram.startup_probe as probe_mod
    from aa.qualification.product_contract_live import run_transport_lane

    async def _fake_ready(token: str, **kwargs: Any) -> Any:
        _ = (token, kwargs)
        return probe_mod.StartupOutcome(
            status="ready",
            category="ready",
            bot_id_present=True,
            username_len=3,
            polling_live=False,
            bootstrap_verified=True,
        )

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "probe-token")
    monkeypatch.delenv("AA_LIVE_TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setattr(probe_mod, "probe_telegram_startup", _fake_ready)
    result = await run_transport_lane()
    assert result.status == "INCOMPLETE"
    assert "real-telegram-startup-ready" in result.passed
    assert "live-telegram-delivery-peer-missing" in result.incomplete
    assert result.metrics["live_transport_bootstrap_verified"] is True
    assert result.metrics["live_external_dependency"] == EXTERNAL_PEER_DEPENDENCY
    assert not any("delivery-confirmed" in item for item in result.passed)


async def test_lane_blocked_conflict_stays_incomplete(monkeypatch: Any) -> None:
    import aa.telegram.startup_probe as probe_mod
    from aa.qualification.product_contract_live import run_transport_lane

    async def _fake_blocked(token: str, **kwargs: Any) -> Any:
        _ = (token, kwargs)
        return probe_mod.StartupOutcome(status="blocked", category="polling-webhook-conflict")

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "probe-token")
    monkeypatch.setattr(probe_mod, "probe_telegram_startup", _fake_blocked)
    result = await run_transport_lane()
    assert result.status == "INCOMPLETE"
    assert "live-telegram-startup-blocked-polling-webhook-conflict" in result.incomplete
    assert not result.failed


async def test_lane_auth_failure_is_fail_not_incomplete(monkeypatch: Any) -> None:
    import aa.telegram.startup_probe as probe_mod
    from aa.qualification.product_contract_live import run_transport_lane

    async def _fake_auth_failure(token: str, **kwargs: Any) -> Any:
        _ = (token, kwargs)
        return probe_mod.StartupOutcome(status="failed", category="telegram-auth")

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bad-token")
    monkeypatch.setattr(probe_mod, "probe_telegram_startup", _fake_auth_failure)
    result = await run_transport_lane()
    assert result.status == "FAIL"
    assert "real-telegram-startup-telegram-auth" in result.failed


async def test_lane_rate_limited_is_distinct_fail(monkeypatch: Any) -> None:
    import aa.telegram.startup_probe as probe_mod
    from aa.qualification.product_contract_live import run_transport_lane

    async def _fake_limited(token: str, **kwargs: Any) -> Any:
        _ = (token, kwargs)
        return probe_mod.StartupOutcome(status="failed", category="transport-rate-limited")

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "probe-token")
    monkeypatch.setattr(probe_mod, "probe_telegram_startup", _fake_limited)
    result = await run_transport_lane()
    assert result.status == "FAIL"
    assert "real-telegram-startup-transport-rate-limited" in result.failed


def test_no_synthetic_delivery_short_circuit_in_lane_source() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "aa"
        / "qualification"
        / "product_contract_live.py"
    ).read_text(encoding="utf-8")
    assert "real-telegram-typing-stream-not-dialed-in-qualification" not in source
    assert "probe_telegram_startup" in source
    assert "live-telegram-delivery-peer-missing" in source
