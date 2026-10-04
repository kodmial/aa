"""Telegram long-polling transport tests against a fake Bot API."""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

import pytest

from aa import logging as aa_logging
from aa.app import Application, create_application
from aa.config import Settings
from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
from aa.telegram.transport import (
    SUPPORTED_COMMANDS,
    PollingTelegramTransport,
    StubTelegramTransport,
    TelegramApi,
    TelegramApiError,
    TelegramAuthError,
    TelegramReply,
    parse_command,
)


def _private_message(update_id: int, chat_id: int, message_id: int, text: str) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "chat": {"id": chat_id, "type": "private"},
            "text": text,
        },
    }


class FakeTelegramApi(TelegramApi):
    """In-memory Bot API recording calls and serving scripted updates."""

    def __init__(
        self,
        *,
        me: dict[str, Any] | None = None,
        get_updates_scripts: list[list[dict[str, Any]]] | None = None,
        fail_send_times: int = 0,
        fail_descriptions: bool = False,
        unauthorized: bool = False,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.me = me if me is not None else {"id": 1, "is_bot": True, "username": "aabot"}
        self.get_updates_scripts = list(get_updates_scripts or [])
        self.fail_send_times = fail_send_times
        self.fail_descriptions = fail_descriptions
        self.unauthorized = unauthorized
        self.send_attempts = 0
        self.sent_payloads: list[dict[str, Any]] = []

    async def call(self, method: str, payload: dict[str, Any]) -> Any:
        self.calls.append((method, dict(payload)))
        if method == "getMe":
            if self.unauthorized:
                raise TelegramAuthError("telegram unauthorized: invalid bot token")
            return dict(self.me)
        if method in ("setMyDescription", "setMyShortDescription") and self.fail_descriptions:
            raise TelegramApiError(f"telegram {method} unsupported")
        if method == "sendMessage":
            self.send_attempts += 1
            self.sent_payloads.append(dict(payload))
            if self.send_attempts <= self.fail_send_times:
                raise TelegramApiError("telegram sendMessage transient failure")
            return {"message_id": 1000 + self.send_attempts}
        if method == "getUpdates":
            if self.get_updates_scripts:
                return [dict(u) for u in self.get_updates_scripts.pop(0)]
            await asyncio.sleep(0.01)
            return []
        return True

    def method_order(self) -> list[str]:
        return [method for method, _ in self.calls]

    def payloads_for(self, method: str) -> list[dict[str, Any]]:
        return [payload for name, payload in self.calls if name == method]


def _fast_transport(api: FakeTelegramApi, **overrides: Any) -> PollingTelegramTransport:
    kwargs: dict[str, Any] = {
        "retry_base_delay_seconds": 0.001,
        "retry_max_delay_seconds": 0.005,
        "poll_timeout_seconds": 0,
    }
    kwargs.update(overrides)
    return PollingTelegramTransport(token="123456:TEST-TOKEN", api=api, **kwargs)


async def _wait_for(predicate: Any, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


def test_parse_command_recognizes_start_and_new() -> None:
    assert parse_command("/start") == "start"
    assert parse_command("/start@MyBot") == "start"
    assert parse_command("/new") == "new"
    assert parse_command("/NEW extra args") == "new"
    assert parse_command("hello") is None
    assert parse_command("/") is None
    assert {name for name, _ in SUPPORTED_COMMANDS} >= {"start", "new"}


async def test_bootstrap_performs_startup_contract_in_order() -> None:
    api = FakeTelegramApi()
    transport = _fast_transport(api)
    await transport.start()
    try:
        assert transport.running
        await _wait_for(lambda: "getUpdates" in api.method_order())
        order = api.method_order()
        # Startup contract order must hold; polling follows bootstrap.
        assert order[:5] == [
            "getMe",
            "deleteWebhook",
            "setMyCommands",
            "setMyDescription",
            "setMyShortDescription",
        ]
        assert "getUpdates" in order[5:]
        delete_payloads = api.payloads_for("deleteWebhook")
        assert delete_payloads and delete_payloads[0].get("drop_pending_updates") is False
        commands_payloads = api.payloads_for("setMyCommands")
        assert commands_payloads
        configured = {c["command"] for c in commands_payloads[0]["commands"]}
        assert {"start", "new"} <= configured
        assert transport.bot_info is not None
        assert transport.bot_info.get("id") == 1
    finally:
        await transport.stop()


async def test_invalid_token_fails_closed() -> None:
    api = FakeTelegramApi(unauthorized=True)
    transport = _fast_transport(api)
    with pytest.raises(TelegramAuthError):
        await transport.start()
    assert not transport.running
    assert transport.offset is None
    # Must fail before any polling starts.
    assert "getUpdates" not in api.method_order()
    await transport.stop()
    assert not transport.running


async def test_optional_description_failures_do_not_fail_startup() -> None:
    api = FakeTelegramApi(fail_descriptions=True)
    transport = _fast_transport(api)
    await transport.start()
    try:
        assert transport.running
        await _wait_for(lambda: "getUpdates" in api.method_order())
        assert "getUpdates" in api.method_order()
    finally:
        await transport.stop()


async def test_long_polling_receives_private_text_and_sends_reply() -> None:
    updates = [
        _private_message(10, 7, 1, "hello bot"),
        _private_message(11, 7, 2, "/start"),
    ]
    api = FakeTelegramApi(get_updates_scripts=[updates])
    received: list[Any] = []

    async def _handler(incoming: Any) -> None:
        received.append(incoming)

    transport = _fast_transport(api, update_handler=_handler)
    await transport.start()
    try:
        await _wait_for(lambda: len(received) == 2)
        assert transport.offset == 12
        assert [m.command for m in received] == [None, "start"]
        assert [m.chat_id for m in received] == [7, 7]

        await transport.send(TelegramReply(chat_id=7, text="hi there"))
        assert api.sent_payloads
        assert api.sent_payloads[-1] == {"chat_id": 7, "text": "hi there"}
        assert transport.sent_messages[-1].text == "hi there"
    finally:
        await transport.stop()


async def test_duplicate_updates_are_idempotent() -> None:
    dup = _private_message(20, 9, 1, "once")
    api = FakeTelegramApi(get_updates_scripts=[[dup], [dup]])
    handled: list[Any] = []

    async def _handler(incoming: Any) -> None:
        handled.append(incoming)

    transport = _fast_transport(api, update_handler=_handler)
    await transport.start()
    try:
        await _wait_for(lambda: len(handled) == 1)
        await asyncio.sleep(0.05)
        assert len(handled) == 1
        assert len(transport.received) == 1
        # Offset still advances past the duplicate redelivery.
        assert transport.offset == 21
    finally:
        await transport.stop()


async def test_failed_handler_is_not_acknowledged_and_is_redelivered() -> None:
    update = _private_message(25, 9, 1, "retry-handler")
    api = FakeTelegramApi(get_updates_scripts=[[update], [update]])
    attempts = 0

    async def _handler(incoming: Any) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient handler failure")

    transport = _fast_transport(api, update_handler=_handler)
    await transport.start()
    try:
        await _wait_for(lambda: attempts >= 2)
        assert transport.offset == 26
        assert len(transport.received) == 1
    finally:
        await transport.stop()


async def test_get_updates_offset_handling() -> None:
    first = [_private_message(30, 1, 1, "one"), _private_message(31, 1, 2, "two")]
    second = [_private_message(32, 1, 3, "three")]
    api = FakeTelegramApi(get_updates_scripts=[first, second])
    seen: list[Any] = []

    async def _handler(incoming: Any) -> None:
        seen.append(incoming)

    transport = _fast_transport(api, update_handler=_handler)
    await transport.start()
    try:
        await _wait_for(lambda: len(seen) == 3)
        offsets = [p.get("offset") for p in api.payloads_for("getUpdates")]
        # First poll has no offset; later polls carry max(update_id) + 1.
        assert offsets[0] is None
        assert 32 in offsets
        assert transport.offset == 33
    finally:
        await transport.stop()


async def test_new_command_routing_hook() -> None:
    api = FakeTelegramApi(
        get_updates_scripts=[[_private_message(40, 5, 1, "/new"), _private_message(41, 5, 2, "hi")]]
    )
    routed: list[Any] = []
    generic: list[Any] = []

    async def _on_new(incoming: Any) -> None:
        routed.append(incoming)

    async def _on_update(incoming: Any) -> None:
        generic.append(incoming)

    transport = _fast_transport(api, update_handler=_on_update)
    transport.on_command("new", _on_new)
    await transport.start()
    try:
        await _wait_for(lambda: len(routed) == 1 and len(generic) == 1)
        assert routed[0].command == "new"
        assert generic[0].command is None
    finally:
        await transport.stop()


async def test_non_private_and_non_text_updates_are_ignored() -> None:
    api = FakeTelegramApi(
        get_updates_scripts=[
            [
                {
                    "update_id": 50,
                    "message": {
                        "message_id": 1,
                        "chat": {"id": -100, "type": "group"},
                        "text": "hello",
                    },
                },
                {
                    "update_id": 51,
                    "message": {
                        "message_id": 2,
                        "chat": {"id": 3, "type": "private"},
                    },
                },
                _private_message(52, 3, 3, "kept"),
            ]
        ]
    )
    seen: list[Any] = []

    async def _handler(incoming: Any) -> None:
        seen.append(incoming)

    transport = _fast_transport(api, update_handler=_handler)
    await transport.start()
    try:
        await _wait_for(lambda: len(seen) == 1)
        assert seen[0].text == "kept"
        assert transport.offset == 53
    finally:
        await transport.stop()


async def test_send_retries_are_bounded_and_succeed() -> None:
    api = FakeTelegramApi(fail_send_times=2)
    transport = _fast_transport(api)
    await transport.start()
    try:
        await transport.send(TelegramReply(chat_id=1, text="retry me"))
        assert api.send_attempts == 3
    finally:
        await transport.stop()


async def test_send_gives_up_after_bounded_retries() -> None:
    api = FakeTelegramApi(fail_send_times=100)
    transport = _fast_transport(api, max_send_retries=2)
    await transport.start()
    try:
        with pytest.raises(TelegramApiError):
            await transport.send(TelegramReply(chat_id=1, text="boom"))
        assert api.send_attempts == 3  # initial attempt + 2 retries
    finally:
        await transport.stop()


async def test_shutdown_interrupts_polling_cleanly() -> None:
    class BlockingApi(FakeTelegramApi):
        async def call(self, method: str, payload: dict[str, Any]) -> Any:
            if method == "getUpdates":
                self.calls.append((method, dict(payload)))
                await asyncio.sleep(30)
                return []
            return await super().call(method, payload)

    api = BlockingApi()
    transport = _fast_transport(api)
    await transport.start()
    assert transport.running
    await asyncio.wait_for(transport.stop(), timeout=2.0)
    assert not transport.running
    # Stopping twice stays clean.
    await transport.stop()
    assert not transport.running


async def test_logs_contain_no_token_or_message_bodies() -> None:
    stream = io.StringIO()
    aa_logging.configure_logging("INFO", stream=stream)
    secret_text = "super-secret-user-body-xyz-123"
    token = "123456:TEST-TOKEN-SECRET"
    api = FakeTelegramApi(get_updates_scripts=[[_private_message(60, 8, 1, secret_text)]])
    seen: list[Any] = []

    async def _handler(incoming: Any) -> None:
        seen.append(incoming)

    transport = PollingTelegramTransport(
        token=token,
        api=api,
        update_handler=_handler,
        retry_base_delay_seconds=0.001,
        retry_max_delay_seconds=0.005,
        poll_timeout_seconds=0,
    )
    try:
        await transport.start()
        await _wait_for(lambda: len(seen) == 1)
        await transport.send(TelegramReply(chat_id=8, text=secret_text))
        # Force a redacted-token-shaped log line through the privacy filter.
        logging.getLogger("aa.telegram.transport").info("raw %s", token)
    finally:
        await transport.stop()
    output = stream.getvalue()
    assert token not in output
    assert secret_text not in output
    assert "TEST-TOKEN-SECRET" not in output
    assert "super-secret-user-body" not in output


async def test_application_routes_polling_message_to_opencode_and_back() -> None:
    api = FakeTelegramApi(get_updates_scripts=[[_private_message(80, 42, 1, "hello")]])
    transport = _fast_transport(api)
    runtime = StubOpenCodeRuntime(
        OpenCodeConfig(
            base_url="http://127.0.0.1:4096",
            command="opencode",
            workdir=".",
        )
    )
    app = Application(Settings.from_env({}), transport=transport, opencode_runtime=runtime)
    await app.start()
    try:
        await _wait_for(lambda: len(api.sent_payloads) == 1)
        assert api.sent_payloads[0]["chat_id"] == 42
        assert api.sent_payloads[0]["text"] == "Фиктивный ответ 1"
        assert app.sessions.session_count() == 1
    finally:
        await app.stop()


async def test_application_new_command_resets_only_that_chat() -> None:
    api = FakeTelegramApi(
        get_updates_scripts=[
            [
                _private_message(90, 7, 1, "first"),
                _private_message(91, 8, 1, "other"),
                _private_message(92, 7, 2, "/new"),
                _private_message(93, 7, 3, "second"),
            ]
        ]
    )
    transport = _fast_transport(api)
    runtime = StubOpenCodeRuntime(
        OpenCodeConfig(
            base_url="http://127.0.0.1:4096",
            command="opencode",
            workdir=".",
        )
    )
    app = Application(Settings.from_env({}), transport=transport, opencode_runtime=runtime)
    await app.start()
    try:
        await _wait_for(lambda: len(api.sent_payloads) == 4)
        replies = [payload["text"] for payload in api.sent_payloads]
        assert replies[0] == "Фиктивный ответ 1"
        assert replies[1] == "Фиктивный ответ 1"
        assert "Новая беседа начата" in replies[2]
        # Chat 7 gets a fresh OpenCode session after /new.
        assert replies[3] == "Фиктивный ответ 1"
        assert app.sessions.get_opencode_session_id(7) != app.sessions.get_opencode_session_id(8)
    finally:
        await app.stop()


def test_application_wires_polling_transport_when_token_present() -> None:
    settings = Settings.from_env({"TELEGRAM_BOT_TOKEN": "123456:ABCDEF-test"})
    app = create_application(settings)
    assert isinstance(app.transport, PollingTelegramTransport)
    assert not isinstance(app.transport, StubTelegramTransport)


def test_application_uses_stub_without_token() -> None:
    app = create_application(Settings.from_env({}))
    assert isinstance(app.transport, StubTelegramTransport)
