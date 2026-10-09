"""Production cutover regression suite: Telegram text+voice on LangGraph (issue #118).

Exercises the exact transport-independent boundary used by Telegram
(``Application.respond`` plus ``_respond_and_deliver`` with typing
heartbeat and delivery). No canonical book text is committed; all
wording is paraphrased fixture text.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from typing import Any

import pytest

from aa.app import Application
from aa.config import Settings
from aa.conversation.graph_runtime import GraphRuntimeError, GraphTurnRuntime
from aa.conversation.memory import thread_id_for_chat
from aa.conversation.output_limits import (
    aggregate_quote_chars,
    envelope_passes,
)
from aa.conversation.turn_pipeline import contains_cyrillic, leaks_internal_terms
from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
from aa.telegram.transport import (
    StubTelegramTransport,
    TelegramIncoming,
    TelegramReply,
)
from aa.telegram.typing import TypingHeartbeat


def _settings(**overrides: str) -> Settings:
    base: dict[str, str] = {"TYPING_HEARTBEAT_SECONDS": "0.02"}
    base.update(overrides)
    return Settings.from_env(base)


def _stub_runtime() -> StubOpenCodeRuntime:
    return StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )


async def _delegate(thread: str, text: str, *, runtime: GraphTurnRuntime) -> str:
    """Deterministic natural Russian delegate with thread-aware continuity."""
    hist = runtime.history_for_thread(thread)
    prior_users = hist[0::2]
    lowered = text.casefold()
    if "выведи" in lowered and "глав" in lowered:
        return "Кратко перескажу главное своими словами. Скажите, какая тема сейчас важнее."
    if "продолж" in lowered or "дальше" in lowered or "следующ" in lowered:
        return "Продолжим кратко своими словами. Уточните, что сейчас важнее всего."
    stripped = text.strip()
    if len(stripped) <= 12 or lowered in ("почему?", "почему", "а дальше?", "и что потом?"):
        topic = prior_users[-1].casefold() if prior_users else ""
        if "сон" in topic:
            return "Про сон: спокойный вечер и режим помогают. Что сейчас мешает отдыху?"
        if "тяга" in topic or "выпив" in topic or "буха" in topic or "ссор" in topic:
            return "Про тягу: поддержка рядом помогает пережить вечер. Что сейчас сильнее всего?"
        return "Уточните, что сейчас важнее всего?"
    if "сон" in lowered:
        return "Про сон: спокойный вечер и режим помогают. Расскажите, что мешает отдыху?"
    if "тяга" in lowered or "выпив" in lowered or "буха" in lowered:
        return "Понимаю, тяга тяжело переживается. Поддержка рядом помогает. Что сейчас важнее?"
    if "ссор" in lowered:
        return "Понимаю, ссора давит. Давайте разберём спокойно. Что сейчас важнее?"
    if "привет" in lowered:
        return "Привет! Расскажите, что сейчас беспокоит сильнее всего?"
    if "что ты можешь" in lowered or "зачем ты" in lowered or "ты кто" in lowered:
        return "Помогаю разбирать тягу и ближайшие шаги. Расскажите о своей ситуации."
    return "Понял вас. Давайте разберём это спокойно. Что сейчас важнее?"


def _runtime() -> GraphTurnRuntime:
    holder: dict[str, GraphTurnRuntime] = {}

    async def _run(thread: str, text: str) -> str:
        return await _delegate(thread, text, runtime=holder["rt"])

    runtime = GraphTurnRuntime(delegate=_run)
    holder["rt"] = runtime
    return runtime


def _app(
    runtime: GraphTurnRuntime | None = None,
    transport: StubTelegramTransport | None = None,
) -> Application:
    return Application(
        _settings(),
        transport=transport or StubTelegramTransport(),
        opencode_runtime=_stub_runtime(),
        graph_runtime=runtime or _runtime(),
    )


def _natural_ok(reply: str) -> None:
    assert contains_cyrillic(reply)
    assert not leaks_internal_terms(reply)
    assert envelope_passes(reply)


# ---------------------------------------------------------------------------
# Conversational semantics on the new runtime.
# ---------------------------------------------------------------------------


async def test_greeting_meta_followup_natural_russian() -> None:
    app = _app()
    await app.start()
    try:
        for text in ("привет", "А что ты можешь?", "почему?"):
            reply = await app.respond(101, text)
            _natural_ok(reply)
            assert "corpus" not in reply.casefold()
    finally:
        await app.stop()


async def test_ordinary_substantive_help_grounded_natural() -> None:
    app = _app()
    await app.start()
    try:
        reply = await app.respond(102, "тяга вечером, что делать?")
        _natural_ok(reply)
        assert "тяга" in reply.casefold() or "поддерж" in reply.casefold()
    finally:
        await app.stop()


async def test_short_followup_uses_context() -> None:
    app = _app()
    await app.start()
    try:
        first = await app.respond(103, "тяга вечером, что делать?")
        assert "тяга" in first.casefold() or "поддерж" in first.casefold()
        second = await app.respond(103, "почему?")
        _natural_ok(second)
        assert "тяга" in second.casefold() or "поддерж" in second.casefold()
    finally:
        await app.stop()


async def test_topic_shift_has_no_stale_lock_in() -> None:
    app = _app()
    await app.start()
    try:
        await app.respond(104, "тяга вечером, что делать?")
        shifted = await app.respond(104, "а теперь про сон, не могу уснуть")
        _natural_ok(shifted)
        assert "сон" in shifted.casefold()
    finally:
        await app.stop()


async def test_text_voice_text_continuity() -> None:
    # Issue #294: voice is transport only. An ASR transcript enters the
    # same respond() boundary as typed text with no voice flag; one
    # shared thread holds text -> voice -> text continuity.
    runtime = _runtime()
    app = _app(runtime)
    await app.start()
    try:
        first = await app.respond(105, "тяга вечером, что делать?")
        assert first.strip()
        voice_turn = await app.respond(105, "не могу уснуть после ссоры")
        _natural_ok(voice_turn)
        follow = await app.respond(105, "почему?")
        _natural_ok(follow)
        # Voice and text share one thread: history interleaves in one list.
        thread = thread_id_for_chat(105)
        hist = runtime.history_for_thread(thread)
        assert len(hist) == 6
        assert hist[0] == "тяга вечером, что делать?"
    finally:
        await app.stop()


async def test_two_chats_isolated() -> None:
    runtime = _runtime()
    app = _app(runtime)
    await app.start()
    try:
        await app.respond(201, "тяга вечером, что делать?")
        await app.respond(202, "не могу уснуть, про сон")
        follow_a = await app.respond(201, "почему?")
        follow_b = await app.respond(202, "почему?")
        assert "тяга" in follow_a.casefold() or "поддерж" in follow_a.casefold()
        assert "сон" in follow_b.casefold()
        assert thread_id_for_chat(201) != thread_id_for_chat(202)
        assert "201" not in thread_id_for_chat(201)
    finally:
        await app.stop()


async def test_three_chats_delayed_a_proves_bc_progress() -> None:
    async def _slow(thread: str, text: str) -> str:
        if text == "медленный маркер":
            await asyncio.sleep(0.3)
            return "Привет! Медленный ответ готов. Что сейчас важнее?"
        return "Понял вас. Давайте разберём спокойно. Что сейчас важнее?"

    runtime = GraphTurnRuntime(delegate=_slow)
    await runtime.start()
    app = _app(runtime)
    await app.start()
    try:
        slow_task = asyncio.create_task(app.respond(301, "медленный маркер"))
        await asyncio.sleep(0.05)
        fast_b = await app.respond(302, "привет")
        fast_c = await app.respond(303, "привет")
        _natural_ok(fast_b)
        _natural_ok(fast_c)
        slow = await slow_task
        _natural_ok(slow)
    finally:
        await app.stop()


async def test_rapid_same_chat_turns_fifo() -> None:
    order: list[str] = []

    async def _ordered(thread: str, text: str) -> str:
        order.append(text)
        await asyncio.sleep(0.01)
        return "Понял вас. Давайте разберём спокойно. Что сейчас важнее?"

    runtime = GraphTurnRuntime(delegate=_ordered)
    await runtime.start()
    transport = StubTelegramTransport()
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        graph_runtime=runtime,
    )
    await app.start()
    try:
        for position, text in enumerate(("первое", "второе", "третье")):
            await app.dispatcher.submit(
                TelegramIncoming(
                    update_id=500 + position,
                    chat_id=400,
                    message_id=position + 1,
                    text=text,
                )
            )
        deadline = asyncio.get_running_loop().time() + 3.0
        while len(order) < 3 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert order == ["первое", "второе", "третье"]
    finally:
        await app.stop()


async def test_new_race_is_deterministic() -> None:
    runtime = _runtime()
    transport = StubTelegramTransport()
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        graph_runtime=runtime,
    )
    await app.start()
    try:
        await app.respond(501, "тяга вечером, что делать?")
        await runtime.clear_chat(501)
        # Clearing is scoped: the other chat keeps its history.
        await app.respond(502, "не могу уснуть, про сон")
        thread_a = thread_id_for_chat(501)
        thread_b = thread_id_for_chat(502)
        assert runtime.history_for_thread(thread_a) == []
        assert len(runtime.history_for_thread(thread_b)) == 2
        # New turn after /new starts fresh continuity.
        fresh = await app.respond(501, "почему?")
        assert "Уточните" in fresh
    finally:
        await app.stop()


async def test_new_command_clears_only_requesting_chat() -> None:
    runtime = _runtime()
    transport = StubTelegramTransport()
    app = Application(
        _settings(),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        graph_runtime=runtime,
    )
    await app.start()
    try:
        await app._process_dispatched_update(
            TelegramIncoming(update_id=1, chat_id=601, message_id=1, text="тяга вечером")
        )
        await app._process_dispatched_update(
            TelegramIncoming(update_id=2, chat_id=602, message_id=1, text="про сон")
        )
        await app._process_dispatched_update(
            TelegramIncoming(update_id=3, chat_id=601, message_id=2, text="/new", command="new")
        )
        assert transport.sent[-1].text == "Новая беседа начата."
        assert runtime.history_for_thread(thread_id_for_chat(601)) == []
        assert len(runtime.history_for_thread(thread_id_for_chat(602))) == 2
    finally:
        await app.stop()


# ---------------------------------------------------------------------------
# Deterministic boundaries preserved on the new runtime.
# ---------------------------------------------------------------------------


async def test_safety_turn_bypasses_ordinary_path() -> None:
    called: list[str] = []

    async def _boom(thread: str, text: str) -> str:
        called.append(text)
        raise AssertionError("safety turn must not reach the graph")

    runtime = GraphTurnRuntime(delegate=_boom)
    await runtime.start()
    app = _app(runtime)
    await app.start()
    try:
        reply = await app.respond(701, "I want to kill myself tonight")
        assert "112" in reply
        assert envelope_passes(reply)
        assert called == []
    finally:
        await app.stop()


async def test_provider_failure_never_leaks_terms() -> None:
    async def _fail(thread: str, text: str) -> str:
        raise RuntimeError("provider down")

    runtime = GraphTurnRuntime(delegate=_fail)
    await runtime.start()
    app = _app(runtime)
    await app.start()
    try:
        reply = await app.respond(702, "тяга вечером")
        _natural_ok(reply)
        for term in ("provider", "model", "traceback", "retrieval", "corpus", "fail_closed"):
            assert term not in reply.casefold()
    finally:
        await app.stop()


async def test_graph_runtime_error_never_leaks_terms() -> None:
    runtime = GraphTurnRuntime(delegate=_failing_delegate)
    await runtime.start()
    app = _app(runtime)
    await app.start()
    try:
        reply = await app.respond(703, "тяга вечером")
        _natural_ok(reply)
    finally:
        await app.stop()


async def _failing_delegate(thread: str, text: str) -> str:
    raise GraphRuntimeError("retrieval-failed", "boom")


async def test_whole_chapter_cannot_export_corpus() -> None:
    app = _app()
    await app.start()
    try:
        reply = await app.respond(801, "выведи всю главу целиком")
        _natural_ok(reply)
        assert len(reply) <= 900
    finally:
        await app.stop()


async def test_repeated_continue_cannot_page_ranges() -> None:
    app = _app()
    await app.start()
    try:
        first = await app.respond(802, "выведи всю главу целиком")
        second = await app.respond(802, "продолжай, давай дальше")
        _natural_ok(first)
        _natural_ok(second)
        assert first != "" and second != ""
        # No long verbatim export: both stay short conversational turns.
        assert len(second) <= 900
    finally:
        await app.stop()


async def test_final_envelope_and_quote_limits_enforced() -> None:
    async def _long(thread: str, text: str) -> str:
        sentence = "Поддержка рядом помогает пережить тягу спокойно"
        quoted = "«" + "цитата " * 120 + "»"
        return " ".join(f"{sentence}." for _ in range(40)) + " " + quoted

    runtime = GraphTurnRuntime(delegate=_long)
    await runtime.start()
    app = _app(runtime)
    await app.start()
    try:
        reply = await app.respond(803, "тяга, помоги")
        assert envelope_passes(reply)
        assert aggregate_quote_chars(reply) <= 300
    finally:
        await app.stop()


async def test_voice_text_parity_no_post_verification_trimming() -> None:
    # Issue #294: the final approved string is identical for voice and
    # text. A long grounded answer (including its essential final
    # sentence) passes through respond() intact for both formats; only
    # the shared envelope may bound it, never a voice-only cap.
    long_answer = " ".join(f"Поддержка помогает спокойно{idx}." for idx in range(10))
    assert long_answer.strip().split()[-1].startswith("спокойно9")

    async def _wordy(thread: str, text: str) -> str:
        return long_answer

    runtime = GraphTurnRuntime(delegate=_wordy)
    await runtime.start()
    app = _app(runtime)
    await app.start()
    try:
        reply = await app.respond(804, "тяга вечером")
        _natural_ok(reply)
        assert reply == long_answer
        assert "спокойно9" in reply
    finally:
        await app.stop()


# ---------------------------------------------------------------------------
# Typing heartbeat lifecycle.
# ---------------------------------------------------------------------------


class _DelayTransport(StubTelegramTransport):
    def __init__(self, send_delay: float = 0.08) -> None:
        super().__init__()
        self.send_delay = send_delay
        self.send_calls = 0

    async def send(self, reply: TelegramReply) -> None:
        self.send_calls += 1
        await asyncio.sleep(self.send_delay)
        await super().send(reply)


async def test_typing_starts_immediately_and_stops_after_delivery() -> None:
    transport = _DelayTransport(send_delay=0.08)
    app = _app(_runtime(), transport)
    await app.start()
    try:
        assert transport.chat_actions == []
        await app._process_dispatched_update(
            TelegramIncoming(update_id=901, chat_id=901, message_id=1, text="привет")
        )
        # Heartbeat fired during processing and delivery succeeded once.
        assert len(transport.chat_actions) >= 2
        assert all(action == "typing" for _, action in transport.chat_actions)
        assert len(transport.sent) == 1
        assert transport.sent[0].chat_id == 901
    finally:
        await app.stop()


async def test_typing_keeps_heartbeat_across_slow_generation() -> None:
    async def _slow(thread: str, text: str) -> str:
        await asyncio.sleep(0.12)
        return "Понял вас. Давайте разберём спокойно. Что сейчас важнее?"

    transport = StubTelegramTransport()
    runtime = GraphTurnRuntime(delegate=_slow)
    await runtime.start()
    app = Application(
        Settings.from_env({"TYPING_HEARTBEAT_SECONDS": "0.02"}),
        transport=transport,
        opencode_runtime=_stub_runtime(),
        graph_runtime=runtime,
    )
    await app.start()
    try:
        await app._process_dispatched_update(
            TelegramIncoming(update_id=902, chat_id=902, message_id=1, text="тяга вечером")
        )
        # Slow generation spans several 20ms heartbeats.
        assert len(transport.chat_actions) >= 3
        assert len(transport.sent) == 1
    finally:
        await app.stop()


async def test_typing_survives_delivery_retry() -> None:
    class _Flaky(StubTelegramTransport):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def send(self, reply: TelegramReply) -> None:
            from aa.telegram.transport import TelegramApiError

            self.calls += 1
            if self.calls == 1:
                raise TelegramApiError("transient send failure")
            await super().send(reply)

    transport = _Flaky()
    app = _app(_runtime(), transport)
    await app.start()
    try:
        # Retryable delivery failure falls back without killing the turn;
        # the heartbeat already fired before the first send attempt.
        await app._process_dispatched_update(
            TelegramIncoming(update_id=903, chat_id=903, message_id=1, text="привет")
        )
        assert len(transport.chat_actions) >= 1
    finally:
        await app.stop()


async def test_heartbeat_cancel_is_clean() -> None:
    transport = StubTelegramTransport()
    beat = TypingHeartbeat(transport, 999, interval_seconds=0.01)
    await beat.start()
    assert beat.running
    assert len(transport.chat_actions) >= 1
    await beat.stop()
    await beat.stop()
    assert not beat.running
    count = len(transport.chat_actions)
    await asyncio.sleep(0.03)
    assert len(transport.chat_actions) == count


async def test_safety_turn_does_not_start_typing() -> None:
    transport = StubTelegramTransport()
    app = _app(_runtime(), transport)
    await app.start()
    try:
        await app._process_dispatched_update(
            TelegramIncoming(
                update_id=904, chat_id=904, message_id=1, text="I want to kill myself tonight"
            )
        )
        assert transport.chat_actions == []
        assert len(transport.sent) == 1
        assert "112" in transport.sent[0].text
    finally:
        await app.stop()


# ---------------------------------------------------------------------------
# Cutover hygiene: legacy path unreachable, thread mapping, privacy.
# ---------------------------------------------------------------------------


def test_legacy_semantic_router_not_invoked() -> None:
    source = (pathlib.Path(__file__).resolve().parents[1] / "src" / "aa" / "app.py").read_text(
        encoding="utf-8"
    )
    for snippet in (
        "is_substantive",
        "is_meta_capability_request",
        "META_CAPABILITY_REPLY",
        "FAIL_CLOSED_REPLY",
        "meets_russian_only",
        "TurnRunner",
        "run_trivial_turn",
        "from aa.conversation.orchestrator",
        "from aa.conversation.meta",
        "from aa.retrieval.planner",
        "book-QA",
        "book_qa",
    ):
        assert snippet not in source, snippet


def test_thread_mapping_deterministic_and_opaque() -> None:
    assert thread_id_for_chat(12345) == thread_id_for_chat(12345)
    assert thread_id_for_chat(12345) != thread_id_for_chat(54321)
    assert "12345" not in thread_id_for_chat(12345)


async def test_no_raw_chat_id_or_user_text_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = _app()
    await app.start()
    try:
        secret = "секретная фраза про вечернюю тягу девять"
        with caplog.at_level(logging.INFO, logger="aa.app"):
            await app.respond(31337, secret)
        assert secret not in caplog.text
        assert "31337" not in caplog.text
    finally:
        await app.stop()


async def test_real_graph_thread_persists_and_new_clears(tmp_path: pathlib.Path) -> None:
    from langchain_core.runnables import RunnableLambda

    from aa.conversation.graph import build_turn_graph, turn_input
    from aa.conversation.memory import MemoryConfig, SqliteCheckpointerFactory

    async def _plan(messages: Any) -> Any:
        return {"queries": []}

    factory = SqliteCheckpointerFactory(MemoryConfig(checkpoint_dir=tmp_path))
    async with factory.checkpointer() as saver:
        graph = build_turn_graph(planner_model=RunnableLambda(_plan), checkpointer=saver)
        thread = thread_id_for_chat(4242)
        config = {"configurable": {"thread_id": thread}}
        first = await graph.ainvoke(turn_input("первое сообщение"), config=config)  # type: ignore[call-overload]
        assert first["planner_invoked"] is True
        second = await graph.ainvoke(turn_input("почему?"), config=config)  # type: ignore[call-overload]
        humans = [str(item.content) for item in second["messages"] if item.type == "human"]
        assert "первое сообщение" in humans
        assert "почему?" in humans
        await saver.adelete_thread(thread)
        third = await graph.ainvoke(turn_input("почему?"), config=config)  # type: ignore[call-overload]
        cleared = [str(item.content) for item in third["messages"] if item.type == "human"]
        assert "первое сообщение" not in cleared
