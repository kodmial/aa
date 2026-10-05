"""Telegram/OpenCode per-chat isolation and concurrency tests (issue #5).

Covers the Definition of Done: distinct OpenCode sessions per chat, strict
per-chat FIFO with at most one active turn, concurrent execution across
chats without head-of-line blocking, serialized first-message create and
``/new`` resets, configurable global/per-chat bounds with safe
backpressure, bounded user-safe failures, safety precedence, and the steady
book-map/tool context contract.
"""

from __future__ import annotations

import asyncio
import pathlib
from typing import Any

import pytest

from aa.app import (
    _BUSY_REPLY,
    _NEW_REPLY,
    _START_REPLY,
    _TEMPORARY_ERROR_REPLY,
    Application,
)
from aa.config import Settings
from aa.conversation.orchestrator import FAIL_CLOSED_REPLY
from aa.conversation.output_limits import envelope_passes
from aa.opencode.client import FakeOpenCodeClient, OpenCodeClient
from aa.opencode.errors import OpenCodeTransientError
from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
from aa.sessions.coordinator import SessionCoordinator
from aa.telegram.dispatcher import ChatQueueFullError, ChatTurnDispatcher
from aa.telegram.transport import (
    PollingTelegramTransport,
    TelegramApi,
    TelegramIncoming,
)


def _incoming(
    update_id: int, chat_id: int, message_id: int, text: str, command: str | None = None
) -> TelegramIncoming:
    return TelegramIncoming(
        update_id=update_id,
        chat_id=chat_id,
        message_id=message_id,
        text=text,
        command=command,
    )


def _settings(**overrides: Any) -> Settings:
    base: dict[str, str] = {}
    base.update(overrides)
    return Settings.from_env(base)


def _stub_runtime(client: OpenCodeClient | None = None) -> StubOpenCodeRuntime:
    config = OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    if client is None:
        return StubOpenCodeRuntime(config)
    return StubOpenCodeRuntime(config, client=client)


async def _wait_for(predicate: Any, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


# ---------------------------------------------------------------------------
# Configuration contract.
# ---------------------------------------------------------------------------


def test_concurrency_defaults_are_conservative() -> None:
    settings = _settings()
    assert settings.max_concurrent_turns == 4
    assert settings.per_chat_queue_size == 8
    settings.validate()


def test_concurrency_settings_from_env() -> None:
    settings = Settings.from_env({"MAX_CONCURRENT_TURNS": "2", "PER_CHAT_QUEUE_SIZE": "3"})
    assert settings.max_concurrent_turns == 2
    assert settings.per_chat_queue_size == 3
    settings.validate()


def test_concurrency_settings_reject_non_positive() -> None:
    with pytest.raises(ValueError):
        Settings.from_env({"MAX_CONCURRENT_TURNS": "0"}).validate()
    with pytest.raises(ValueError):
        Settings.from_env({"PER_CHAT_QUEUE_SIZE": "0"}).validate()
    with pytest.raises(ValueError):
        Settings.from_env({"MAX_CONCURRENT_TURNS": "-1"}).validate()


def test_concurrency_settings_are_reserved_and_safe() -> None:
    reserved = set(Settings.RESERVED_ENV_NAMES)
    assert "MAX_CONCURRENT_TURNS" in reserved
    assert "PER_CHAT_QUEUE_SIZE" in reserved
    safe = Settings.from_env({"MAX_CONCURRENT_TURNS": "2"}).to_safe_dict()
    assert safe["max_concurrent_turns"] == 2
    assert safe["per_chat_queue_size"] == 8


def test_dispatcher_rejects_invalid_bounds() -> None:
    async def _noop(_: TelegramIncoming) -> None:
        return None

    with pytest.raises(ValueError):
        ChatTurnDispatcher(_noop, max_concurrent_turns=0, per_chat_queue_size=8)
    with pytest.raises(ValueError):
        ChatTurnDispatcher(_noop, max_concurrent_turns=2, per_chat_queue_size=0)


# ---------------------------------------------------------------------------
# Dispatcher: per-chat FIFO, cross-chat concurrency, global bound.
# ---------------------------------------------------------------------------


async def test_same_chat_turns_are_strict_fifo() -> None:
    order: list[str] = []

    async def _process(incoming: TelegramIncoming) -> None:
        order.append(incoming.text)

    dispatcher = ChatTurnDispatcher(_process, max_concurrent_turns=4, per_chat_queue_size=8)
    await dispatcher.start()
    try:
        for position, text in enumerate(("one", "two", "three")):
            await dispatcher.submit(_incoming(100 + position, 7, position + 1, text))
        await _wait_for(lambda: len(order) == 3)
        assert order == ["one", "two", "three"]
    finally:
        await dispatcher.stop()


async def test_same_chat_has_at_most_one_active_turn() -> None:
    active = 0
    max_active = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _process(_: TelegramIncoming) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        entered.set()
        await asyncio.wait_for(release.wait(), timeout=3.0)
        active -= 1

    dispatcher = ChatTurnDispatcher(_process, max_concurrent_turns=4, per_chat_queue_size=8)
    await dispatcher.start()
    try:
        await dispatcher.submit(_incoming(1, 7, 1, "first"))
        await _wait_for(entered.is_set)
        await dispatcher.submit(_incoming(2, 7, 2, "second"))
        await asyncio.sleep(0.05)
        # Second turn for the same chat waits: still exactly one active.
        assert max_active == 1
        assert dispatcher.pending_count(7) == 1
        release.set()
        await _wait_for(lambda: dispatcher.pending_count(7) == 0)
        assert max_active == 1
    finally:
        release.set()
        await dispatcher.stop()


async def test_different_chats_run_concurrently() -> None:
    started: dict[int, asyncio.Event] = {1: asyncio.Event(), 2: asyncio.Event()}
    release = asyncio.Event()
    active = 0
    max_active = 0

    async def _process(incoming: TelegramIncoming) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        started[incoming.chat_id].set()
        await asyncio.wait_for(release.wait(), timeout=3.0)
        active -= 1

    dispatcher = ChatTurnDispatcher(_process, max_concurrent_turns=4, per_chat_queue_size=8)
    await dispatcher.start()
    try:
        await dispatcher.submit(_incoming(1, 1, 1, "a"))
        await dispatcher.submit(_incoming(2, 2, 1, "b"))
        await _wait_for(lambda: started[1].is_set() and started[2].is_set())
        assert max_active == 2
    finally:
        release.set()
        await dispatcher.stop()


async def test_global_bound_serializes_across_chats() -> None:
    order: list[str] = []
    release_first = asyncio.Event()

    async def _process(incoming: TelegramIncoming) -> None:
        if incoming.chat_id == 1:
            await asyncio.wait_for(release_first.wait(), timeout=3.0)
        order.append(f"{incoming.chat_id}:{incoming.text}")

    dispatcher = ChatTurnDispatcher(_process, max_concurrent_turns=1, per_chat_queue_size=8)
    await dispatcher.start()
    try:
        await dispatcher.submit(_incoming(1, 1, 1, "slow"))
        await asyncio.sleep(0.05)
        await dispatcher.submit(_incoming(2, 2, 1, "fast"))
        await asyncio.sleep(0.05)
        # Global bound of 1: the second chat waits for the first.
        assert order == []
        release_first.set()
        await _wait_for(lambda: len(order) == 2)
        assert order == ["1:slow", "2:fast"]
    finally:
        release_first.set()
        await dispatcher.stop()


async def test_per_chat_queue_bound_backpressures() -> None:
    processed: list[str] = []
    block = asyncio.Event()

    async def _process(incoming: TelegramIncoming) -> None:
        await asyncio.wait_for(block.wait(), timeout=3.0)
        processed.append(incoming.text)

    dispatcher = ChatTurnDispatcher(_process, max_concurrent_turns=4, per_chat_queue_size=1)
    await dispatcher.start()
    try:
        await dispatcher.submit(_incoming(1, 9, 1, "active"))
        await asyncio.sleep(0.05)
        await dispatcher.submit(_incoming(2, 9, 2, "queued"))
        with pytest.raises(ChatQueueFullError):
            await dispatcher.submit(_incoming(3, 9, 3, "overflow"))
        # Another chat is unaffected by the full queue.
        await dispatcher.submit(_incoming(4, 10, 1, "other"))
        block.set()
        await _wait_for(lambda: len(processed) == 3)
        assert processed == ["active", "other", "queued"] or sorted(processed) == sorted(
            ["active", "other", "queued"]
        )
        # Same-chat order is preserved for the accepted turns.
        first_two = [text for text in processed if text in ("active", "queued")]
        assert first_two == ["active", "queued"]
    finally:
        block.set()
        await dispatcher.stop()


async def test_transient_retry_for_one_chat_does_not_block_other() -> None:
    """A slow/retrying chat A must not serialize chat B below the bound."""
    finished: list[int] = []
    release_a = asyncio.Event()

    async def _process(incoming: TelegramIncoming) -> None:
        if incoming.chat_id == 100:
            # Simulate bounded retries/transient work for chat A.
            for _ in range(3):
                await asyncio.sleep(0.01)
            await asyncio.wait_for(release_a.wait(), timeout=3.0)
        finished.append(incoming.chat_id)

    dispatcher = ChatTurnDispatcher(_process, max_concurrent_turns=4, per_chat_queue_size=8)
    await dispatcher.start()
    try:
        await dispatcher.submit(_incoming(1, 100, 1, "retrying"))
        await asyncio.sleep(0.05)
        await dispatcher.submit(_incoming(2, 200, 1, "fast"))
        await _wait_for(lambda: 200 in finished)
        assert 100 not in finished
        release_a.set()
        await _wait_for(lambda: 100 in finished)
    finally:
        release_a.set()
        await dispatcher.stop()


# ---------------------------------------------------------------------------
# Session coordinator: concurrent first messages create exactly one session.
# ---------------------------------------------------------------------------


async def test_concurrent_first_messages_create_one_session() -> None:
    client = FakeOpenCodeClient()
    coordinator = SessionCoordinator()
    await coordinator.start()
    results = await asyncio.gather(
        *(coordinator.ensure_opencode_session(55, client) for _ in range(10))
    )
    assert len(set(results)) == 1
    assert client._counter == 1
    await coordinator.stop()


async def test_concurrent_resets_do_not_compete() -> None:
    client = FakeOpenCodeClient()
    coordinator = SessionCoordinator()
    first = await coordinator.ensure_opencode_session(56, client)
    results = await asyncio.gather(
        *(coordinator.reset_opencode_session(56, client) for _ in range(5))
    )
    # Every reset replaces the session; all callers observe valid distinct
    # bindings and the coordinator ends on exactly one of them.
    assert len(set(results)) == 5
    assert first not in set(results)
    assert coordinator.get_opencode_session_id(56) in set(results)
    assert coordinator.get_or_create(56).generation == 5


# ---------------------------------------------------------------------------
# Application: correct session, isolation, /new ordering, no head-of-line
# blocking, safety precedence, bounded failures.
# ---------------------------------------------------------------------------


class _FakeApi(TelegramApi):
    def __init__(self, scripts: list[list[dict[str, Any]]] | None = None) -> None:
        self.scripts = [list(batch) for batch in (scripts or [])]
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.sent_payloads: list[dict[str, Any]] = []

    async def call(self, method: str, payload: dict[str, Any]) -> Any:
        self.calls.append((method, dict(payload)))
        if method == "getMe":
            return {"id": 1, "is_bot": True, "username": "aabot"}
        if method == "sendMessage":
            self.sent_payloads.append(dict(payload))
            return {"message_id": len(self.sent_payloads)}
        if method == "getUpdates":
            if self.scripts:
                return self.scripts.pop(0)
            await asyncio.sleep(0.01)
            return []
        return True


def _private(update_id: int, chat_id: int, message_id: int, text: str) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "chat": {"id": chat_id, "type": "private"},
            "text": text,
        },
    }


def _transport(api: _FakeApi) -> PollingTelegramTransport:
    return PollingTelegramTransport(
        token="123456:TEST-TOKEN",
        api=api,
        retry_base_delay_seconds=0.001,
        retry_max_delay_seconds=0.005,
        poll_timeout_seconds=0,
    )


async def test_telegram_text_reaches_correct_session_and_chats_isolated() -> None:
    api = _FakeApi(
        [
            [
                _private(1, 11, 1, "hello"),
                _private(2, 22, 1, "hi there"),
            ]
        ]
    )
    transport = _transport(api)
    app = Application(_settings(), transport=transport, opencode_runtime=_stub_runtime())
    await app.start()
    try:
        await _wait_for(lambda: len(api.sent_payloads) == 2)
        by_chat = {payload["chat_id"]: payload["text"] for payload in api.sent_payloads}
        assert by_chat[11] == "Фиктивный ответ 1"
        assert by_chat[22] == "Фиктивный ответ 1"
        session_a = app.sessions.get_opencode_session_id(11)
        session_b = app.sessions.get_opencode_session_id(22)
        assert session_a is not None and session_b is not None
        assert session_a != session_b
    finally:
        await app.stop()


class _SlowMarkerClient(FakeOpenCodeClient):
    """Sleeps only for prompts carrying the slow marker (chat A)."""

    async def send_message(
        self,
        session_id: str,
        text: str,
        *,
        timeout: float | None = None,
        agent: str = "",
        model: str = "",
        system: str = "",
        format: dict[str, object] | None = None,
    ) -> str:
        if "slow-marker" in text:
            await asyncio.sleep(0.4)
        return await super().send_message(
            session_id,
            text,
            timeout=timeout,
            agent=agent,
            model=model,
            system=system,
            format=format,
        )


async def test_slow_chat_does_not_head_of_line_block_other_chat() -> None:
    api = _FakeApi(
        [
            [
                _private(10, 101, 1, "hello slow-marker"),
                _private(11, 202, 1, "hello"),
            ]
        ]
    )
    transport = _transport(api)
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(_SlowMarkerClient()),
    )
    await app.start()
    try:
        await _wait_for(lambda: len(api.sent_payloads) == 2, timeout=5.0)
        # The fast unrelated chat must be served before the slow chat.
        assert api.sent_payloads[0]["chat_id"] == 202
        assert api.sent_payloads[1]["chat_id"] == 101
    finally:
        await app.stop()


async def test_new_is_serialized_with_in_flight_turn_and_resets_only_one_chat() -> None:
    api = _FakeApi(
        [
            [
                _private(20, 301, 1, "hello slow-marker"),
                _private(21, 302, 1, "hello"),
                _private(22, 301, 2, "/new"),
                _private(23, 301, 3, "hello again"),
            ]
        ]
    )
    transport = _transport(api)
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(_SlowMarkerClient()),
    )
    await app.start()
    try:
        await _wait_for(lambda: len(api.sent_payloads) == 4, timeout=5.0)
        by_chat: dict[int, list[str]] = {}
        for payload in api.sent_payloads:
            by_chat.setdefault(payload["chat_id"], []).append(payload["text"])
        # Chat 301 keeps strict FIFO: slow turn, then /new, then next turn.
        assert len(by_chat[301]) == 3
        assert by_chat[301][0] == "Фиктивный ответ 1"
        assert "Новая беседа начата" in by_chat[301][1]
        assert by_chat[301][2] == "Фиктивный ответ 1"
        assert by_chat[302] == ["Фиктивный ответ 1"]
        # Only the requesting chat was rebound.
        assert app.sessions.get_opencode_session_id(301) != app.sessions.get_opencode_session_id(
            302
        )
    finally:
        await app.stop()


async def test_queue_overflow_backpressures_safely() -> None:
    api = _FakeApi(scripts=[])
    transport = _transport(api)
    app = Application(
        Settings.from_env({"PER_CHAT_QUEUE_SIZE": "1", "MAX_CONCURRENT_TURNS": "4"}),
        transport=transport,
        opencode_runtime=_stub_runtime(),
    )
    await app.start()
    try:
        assert app.dispatcher.per_chat_queue_size == 1
        # Occupy the single worker with a blocked turn, fill the one pending
        # slot, then overflow: the overflow must raise instead of growing.
        block = asyncio.Event()
        processed: list[int] = []

        async def _blocking(incoming: TelegramIncoming) -> None:
            processed.append(incoming.update_id)
            await asyncio.wait_for(block.wait(), timeout=3.0)

        dispatcher = ChatTurnDispatcher(_blocking, max_concurrent_turns=4, per_chat_queue_size=1)
        await dispatcher.start()
        try:
            await dispatcher.submit(_incoming(1, 500, 1, "a"))
            await asyncio.sleep(0.05)
            await dispatcher.submit(_incoming(2, 500, 2, "b"))
            with pytest.raises(ChatQueueFullError):
                await dispatcher.submit(_incoming(3, 500, 3, "c"))
        finally:
            block.set()
            await dispatcher.stop()
        # The application busy reply is bounded and user-safe.
        assert envelope_passes(_BUSY_REPLY)
        assert _BUSY_REPLY.strip()
    finally:
        await app.stop()


async def test_safety_routing_precedes_opencode_work() -> None:
    app = Application(_settings(), opencode_runtime=_stub_runtime())
    await app.start()
    try:
        reply = await app.respond(600, "I want to kill myself tonight")
        assert "112" in reply
        assert envelope_passes(reply)
        # Emergency short-circuits before any OpenCode session is created.
        assert app.sessions.get_opencode_session_id(600) is None
    finally:
        await app.stop()


class _AlwaysFailingClient(FakeOpenCodeClient):
    async def send_message(
        self,
        session_id: str,
        text: str,
        *,
        timeout: float | None = None,
        agent: str = "",
        model: str = "",
        system: str = "",
        format: dict[str, object] | None = None,
    ) -> str:
        raise OpenCodeTransientError("provider down")


async def test_provider_failure_is_bounded_and_user_safe() -> None:
    app = Application(_settings(), opencode_runtime=_stub_runtime(_AlwaysFailingClient()))
    await app.start()
    try:
        reply = await app.respond(700, "hello")
        assert reply.strip()
        assert envelope_passes(reply)
        assert reply in (FAIL_CLOSED_REPLY, _TEMPORARY_ERROR_REPLY)
    finally:
        await app.stop()


async def test_substantive_turn_requires_source_support_and_fails_closed() -> None:
    # Without a provisioned RU index the substantive pipeline fails closed
    # (never an invented answer); trivial greetings still answer directly.
    app = Application(_settings(), opencode_runtime=_stub_runtime())
    await app.start()
    try:
        grounded = await app.respond(800, "я бухаю каждый вечер, что делать?")
        assert grounded == FAIL_CLOSED_REPLY
        assert envelope_passes(grounded)
    finally:
        await app.stop()


async def test_user_language_is_preserved_for_fixed_replies() -> None:
    assert "Бот готов" in _START_REPLY
    assert "Новая беседа" in _NEW_REPLY
    assert "Bot is ready" not in _START_REPLY
    assert "New conversation" not in _NEW_REPLY
    assert "Could not process" not in _TEMPORARY_ERROR_REPLY
    assert "The bot is busy" not in _BUSY_REPLY
    assert envelope_passes(_START_REPLY)
    assert envelope_passes(_NEW_REPLY)
    assert envelope_passes(_TEMPORARY_ERROR_REPLY)
    assert envelope_passes(_BUSY_REPLY)


def test_book_map_and_tools_available_in_every_session() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    map_text = (root / "corpus" / "book-map.md").read_text(encoding="utf-8")
    assert "navigation only, never evidence" in map_text
    for section_id in ("doctors-opinion", "chapter-1", "chapter-11"):
        assert section_id in map_text
    import json as _json

    declared = _json.loads((root / "opencode.json").read_text(encoding="utf-8"))
    agent = declared["agent"]["aa"]
    assert agent["mode"] == "primary"
    assert agent["model"] == "opencode/space-bunny-free"
    permissions = agent["permission"]
    assert permissions["*"] == "deny"
    for tool in ("book_search", "book_read", "book_expand", "book_section"):
        assert permissions[tool] == "allow"
    prompt = (root / "prompts" / "aa-agent-system.md").read_text(encoding="utf-8")
    assert "900" in prompt and "130" in prompt
    from aa.retrieval.book_tools import TOOL_NAMES

    assert set(TOOL_NAMES) == {"book_search", "book_read", "book_expand", "book_section"}
