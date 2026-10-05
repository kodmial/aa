from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from aa.app import Application
from aa.config import Settings
from aa.conversation.memory import thread_id_for_chat
from aa.telegram.transport import (
    PollingTelegramTransport,
    StubTelegramTransport,
    TelegramApi,
    TelegramIncoming,
    TelegramReply,
)


class _FakeConversationRuntime:
    def __init__(self, reply: str = "Готово.") -> None:
        self.reply = reply
        self.started = False
        self.calls: list[tuple[int, str]] = []
        self.resets: list[int] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.block = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def respond(self, chat_id: int, text: str) -> str:
        self.calls.append((chat_id, text))
        self.entered.set()
        if self.block:
            await self.release.wait()
        return self.reply

    async def reset(self, chat_id: int) -> None:
        self.resets.append(chat_id)


class _DelayedTransport(StubTelegramTransport):
    def __init__(self) -> None:
        super().__init__()
        self.send_started = asyncio.Event()
        self.allow_send = asyncio.Event()

    async def send(self, reply: TelegramReply) -> None:
        self.send_started.set()
        await self.allow_send.wait()
        await super().send(reply)


async def _wait_for(predicate: Any, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not reached")
        await asyncio.sleep(0.002)


def test_production_app_has_no_legacy_conversation_router_markers() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "aa" / "app.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        "from aa.conversation.meta import",
        "from aa.conversation.orchestrator import",
        "TurnRunner",
        "is_substantive",
        "FAIL_CLOSED_REPLY",
        "META_CAPABILITY_REPLY",
        "run_trivial_turn",
    ):
        assert marker not in source
    assert "ProductConversationRuntime" in source


def test_thread_identity_is_deterministic_and_private() -> None:
    first = thread_id_for_chat(123456789)
    assert first == thread_id_for_chat(123456789)
    assert first != thread_id_for_chat(123456788)
    assert "123456789" not in first


@pytest.mark.asyncio
async def test_typing_starts_immediately_and_lives_through_delivery() -> None:
    runtime = _FakeConversationRuntime()
    runtime.block = True
    transport = _DelayedTransport()
    app = Application(
        Settings(telegram_typing_interval_seconds=0.01),
        transport=transport,
        conversation_runtime=runtime,  # type: ignore[arg-type]
    )
    incoming = TelegramIncoming(
        update_id=1,
        chat_id=77,
        message_id=1,
        text="Привет",
    )

    task = asyncio.create_task(
        app._respond_and_deliver(incoming, text=incoming.text, voice_input=False)
    )
    await asyncio.wait_for(runtime.entered.wait(), timeout=1.0)
    assert transport.chat_actions and transport.chat_actions[0] == (77, "typing")

    runtime.release.set()
    await asyncio.wait_for(transport.send_started.wait(), timeout=1.0)
    before_delivery = len(transport.chat_actions)
    await _wait_for(lambda: len(transport.chat_actions) > before_delivery)

    transport.allow_send.set()
    await asyncio.wait_for(task, timeout=1.0)
    after_delivery = len(transport.chat_actions)
    await asyncio.sleep(0.03)
    assert len(transport.chat_actions) == after_delivery
    assert transport.sent[-1].text == "Готово."


@pytest.mark.asyncio
async def test_new_resets_only_requesting_langgraph_thread() -> None:
    runtime = _FakeConversationRuntime()
    transport = StubTelegramTransport()
    app = Application(
        Settings(),
        transport=transport,
        conversation_runtime=runtime,  # type: ignore[arg-type]
    )
    await app._handle_new_command(
        TelegramIncoming(update_id=1, chat_id=11, message_id=1, text="/new", command="new")
    )
    assert runtime.resets == [11]
    assert transport.sent[-1].chat_id == 11
    assert "Новая беседа" in transport.sent[-1].text


class _Api(TelegramApi):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, method: str, payload: dict[str, Any]) -> Any:
        self.calls.append((method, dict(payload)))
        return True


@pytest.mark.asyncio
async def test_polling_transport_uses_send_chat_action() -> None:
    api = _Api()
    transport = PollingTelegramTransport(
        token="123456:TEST",
        api=api,
        retry_base_delay_seconds=0.001,
        retry_max_delay_seconds=0.002,
    )
    await transport.send_chat_action(42, "typing")
    assert api.calls == [("sendChatAction", {"chat_id": 42, "action": "typing"})]


@pytest.mark.asyncio
async def test_text_and_voice_use_same_langgraph_boundary() -> None:
    runtime = _FakeConversationRuntime("Русский ответ.")
    transport = StubTelegramTransport()
    app = Application(
        Settings(telegram_typing_interval_seconds=0.01),
        transport=transport,
        conversation_runtime=runtime,  # type: ignore[arg-type]
    )
    text = TelegramIncoming(update_id=1, chat_id=5, message_id=1, text="Текст")
    voice = TelegramIncoming(update_id=2, chat_id=5, message_id=2, text="")

    await app._respond_and_deliver(text, text="Текст", voice_input=False)
    await app._respond_and_deliver(voice, text="Голос", voice_input=True)
    assert runtime.calls == [(5, "Текст"), (5, "Голос")]
