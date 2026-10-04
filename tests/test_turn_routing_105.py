"""Turn-routing regression tests (issue #105).

Exercises the production ``Application.respond`` / Telegram adapter
boundary used in production, not helper-only units. Conversational/meta
turns must take the direct named ``aa`` agent path with no book-evidence
requirement; substantive recovery/book turns must take the full
grounded pipeline; true failures stay bounded and user-safe.
"""

from __future__ import annotations

import hashlib
import pathlib
import re

import pytest

from aa.app import Application
from aa.config import Settings
from aa.conversation.orchestrator import (
    AGENT_NAME,
    FAIL_CLOSED_REPLY,
    TurnRunner,
    is_substantive,
    meets_russian_only,
)
from aa.conversation.routing import TurnRoute, route_turn
from aa.corpus.structure import SECTION_IDS, build_full_structure
from aa.opencode.client import FakeOpenCodeClient
from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
from aa.retrieval.index import HybridIndex, build_hybrid_index
from aa.safety.router import SafetyDecision, SafetyRouter
from aa.telegram.transport import StubTelegramTransport, TelegramIncoming

PRIMARY = "opencode/muse-spark-1.3-contributor-free"
FALLBACK = "opencode/space-bunny-free"

INTERNAL_FAIL_WORDING = "имеющимся отрывкам"

RU_FIXTURES: dict[str, str] = {
    "doctors-opinion": (
        "Фиктивное мнение доктора о тяге и навязчивом желании выпить. "
        "Тяга приходит внезапно и требует внимания.\n\n"
        "Второй абзац мнения доктора. Наблюдение за тягой продолжается."
    ),
    "chapter-1": (
        "Фиктивный рассказ о первом глотке. Герой начал бухать каждый вечер "
        "и однажды тяпнул лишнего. Утром после выпивки было тяжело.\n\n"
        "Второй абзац рассказа. Нажрался в гостях и долго приходил в себя."
    ),
    "chapter-2": (
        "Фиктивный выход есть для пьющих. Надежда и поддержка рядом. "
        "Пьющие находят помощь в сообществе.\n\n"
        "Второй абзац выхода. Сообщество встречает новичков тепло."
    ),
    "chapter-3": (
        "Фиктивный алкоголизм как феномен тяги. Тяга к алкоголю и последствия "
        "алкоголизма обсуждаются подробно. Сорвался после долгой трезвости.\n\n"
        "Второй абзац об алкоголизме. Тянет выпить снова, страх срыва рядом."
    ),
    "chapter-4": (
        "Фиктивные размышления агностика. Готовность принять помощь растет.\n\n"
        "Второй абзац агностика. Сомнения обсуждаются открыто."
    ),
    "chapter-5": (
        "Фиктивная программа в действии требует честности. "
        "Практические шаги каждый день.\n\n"
        "Второй абзац программы. Утренний настрой задает тон."
    ),
    "chapter-6": (
        "Фиктивная работа по шагам продолжается. Утром делаем инвентаризацию. "
        "Срыв разбираем честно и спокойно.\n\n"
        "Второй абзац работы. Вечером подводим итоги дня."
    ),
    "chapter-7": (
        "Фиктивная работа с другими людьми. Несем весть тем кто страдает.\n\n"
        "Второй абзац помощи. Разговор ведется спокойно и честно."
    ),
    "chapter-8": (
        "Фиктивная жинка ругает из-за пьянки. Женушка переживает за семью. "
        "Жена ругает пьянство, доверие страдает.\n\n"
        "Второй абзац о семье. Пьянство разрушает доверие постепенно."
    ),
    "chapter-9": (
        "Фиктивные новые отношения в семье. Доверие возвращается постепенно. "
        "Семейный конфликт утихает.\n\n"
        "Второй абзац семьи. Разговоры становятся спокойнее."
    ),
    "chapter-10": (
        "Фиктивное обращение к работодателям. Трезвость на рабочем месте важна. "
        "Рабочий конфликт и страх увольнения обсуждаются.\n\n"
        "Второй абзац работодателям. Поддержка коллег помогает многим."
    ),
    "chapter-11": (
        "Фиктивный взгляд в будущее сообщества. Бухать больше не хочется, "
        "хочется жить трезво.\n\n"
        "Второй абзац будущего. Планы строятся на трезвую голову."
    ),
}


def _fixture_index(tmp_path: pathlib.Path) -> HybridIndex:
    en_sections: list[dict[str, object]] = []
    ru_sections: list[dict[str, object]] = []
    for section_id in SECTION_IDS:
        en_sections.append(
            {
                "id": section_id,
                "title": f"EN TITLE {section_id}",
                "text": f"Fixture EN {section_id} opening. Second sentence here.\n\n"
                f"Fixture EN {section_id} second paragraph.",
                "source_id": "core-pages-1-164",
                "source_file": "corpus/source/raw/AA.txt",
                "source_sha256": hashlib.sha256(b"en-source").hexdigest(),
            }
        )
        ru_sections.append(
            {
                "id": section_id,
                "title": f"RU TITLE {section_id}",
                "text": RU_FIXTURES[section_id],
                "source_id": "ru-fourth-edition-txt",
                "source_file": "corpus/source/raw-ru/aa-big-book.txt",
                "source_sha256": hashlib.sha256(b"ru-source").hexdigest(),
            }
        )
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
    )
    return build_hybrid_index(
        dict(full),
        ru_manifest={"format": "x", "artifact_sha256": "a" * 64},
        en_manifest={"format": "x", "artifact_sha256": "b" * 64},
        embedding_lock={"model_id": "m", "revision": "r" * 40},
        out_dir=tmp_path / "retrieval",
        backend="hashing",
    )


def _first_sentence(text: str) -> str:
    parts = [item.strip() for item in re.split(r"(?<=[.!?…])\s+", text) if item.strip()]
    return parts[0] if parts else text.strip()


class _RouteRecordingClient(FakeOpenCodeClient):
    """Fake OpenCode client recording prompts and separating paths."""

    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []

    async def send_message(
        self,
        session_id: str,
        text: str,
        *,
        timeout: float | None = None,
        agent: str = "",
        model: str = "",
    ) -> str:
        self.prompts.append(text)
        lowered = text.casefold()
        if "точные отрывки" in lowered:
            if "юпитер" in lowered or "спутник" in lowered:
                return "У Юпитера много спутников, это интересный факт."
            for line in text.splitlines():
                match = re.match(
                    r"\[([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+#[A-Za-z0-9_:.\-]+)\] (.*)",
                    line,
                )
                if match:
                    locator, passage = match.group(1), match.group(2).strip()
                    return f"{_first_sentence(passage)} [{locator}]"
            raise AssertionError("grounded prompt carries no evidence")
        assert agent == AGENT_NAME
        return (
            "Я помощник для поддержки трезвости. "
            "Могу выслушать и помочь разобраться. "
            "Расскажите, что вас беспокоит?"
        )


async def _make_app(
    tmp_path: pathlib.Path,
) -> tuple[Application, _RouteRecordingClient]:
    settings = Settings.from_env({})
    client = _RouteRecordingClient()
    runtime = StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir="."),
        client=client,
    )
    transport = StubTelegramTransport()
    app = Application(settings, opencode_runtime=runtime, transport=transport)
    await app.start()
    index = _fixture_index(tmp_path)
    app._index = index
    app._turn_runner = TurnRunner(
        index=index,
        ru_corpus_version="",
        agent=AGENT_NAME,
        primary_model=settings.opencode_model,
        fallback_model=settings.opencode_fallback_model,
    )
    return app, client


def _assert_useful_russian(reply: str) -> None:
    assert reply.strip()
    assert reply != FAIL_CLOSED_REPLY
    assert INTERNAL_FAIL_WORDING not in reply
    assert meets_russian_only(reply)
    assert re.search(r"[\u0400-\u04ff]", reply) is not None


# ---------------------------------------------------------------------------
# Routing contract invariants
# ---------------------------------------------------------------------------


def test_routing_contract_invariants() -> None:
    router = SafetyRouter()
    emergency = router.check("Я не могу дышать, задыхаюсь")
    assert emergency.decision is SafetyDecision.EMERGENCY
    decision = route_turn("Я не могу дышать, задыхаюсь", emergency)
    assert decision.route is TurnRoute.EMERGENCY
    assert decision.requires_grounding is False

    blocked = router.check("   ")
    assert blocked.decision is SafetyDecision.BLOCK
    assert route_turn("   ", blocked).route is TurnRoute.BLOCKED

    allow = router.check("Привет")
    assert allow.decision is SafetyDecision.ALLOW
    assert route_turn("/new", allow).route is TurnRoute.COMMAND
    assert route_turn("Привет", allow).route is TurnRoute.CONVERSATIONAL
    assert route_turn("как бросить пить", allow).route is TurnRoute.SUBSTANTIVE

    conversational = route_turn("Кто ты?", allow)
    assert conversational.requires_grounding is False
    assert conversational.agent == AGENT_NAME
    substantive = route_turn("как бросить пить", allow)
    assert substantive.requires_grounding is True
    assert substantive.agent == AGENT_NAME


def test_meta_with_question_mark_never_requires_grounding() -> None:
    router = SafetyRouter()
    for text in (
        "А что ты можешь?",
        "Тогда зачем ты?",
        "Кто ты?",
        "Привет",
        "что ты умеешь?",
        "зачем ты нужен?",
        "расскажи о себе",
        "как дела?",
    ):
        result = router.check(text)
        assert result.decision is SafetyDecision.ALLOW
        decision = route_turn(text, result)
        assert decision.route is TurnRoute.CONVERSATIONAL, text
        assert decision.requires_grounding is False
        assert is_substantive(text) is False


def test_substantive_recovery_still_requires_grounding() -> None:
    router = SafetyRouter()
    for text in (
        "как бросить пить",
        "Что говорит Большая книга о страхе?",
        "я бухаю каждый вечер, что делать",
        "Сколько спутников у Юпитера?",
    ):
        result = router.check(text)
        assert result.decision is SafetyDecision.ALLOW
        decision = route_turn(text, result)
        assert decision.route is TurnRoute.SUBSTANTIVE, text
        assert decision.requires_grounding is True
        assert is_substantive(text) is True


# ---------------------------------------------------------------------------
# Production boundary: Application.respond
# ---------------------------------------------------------------------------


META_CASES: tuple[str, ...] = (
    "А что ты можешь?",
    "Тогда зачем ты?",
    "Кто ты?",
    "Привет",
)


@pytest.mark.parametrize("text", META_CASES)
async def test_meta_turns_use_conversational_path(tmp_path: pathlib.Path, text: str) -> None:
    app, client = await _make_app(tmp_path)
    try:
        before = len(client.prompts)
        reply = await app.respond(1001, text)
        _assert_useful_russian(reply)
        new_prompts = client.prompts[before:]
        assert new_prompts, "conversational turn must call the agent once"
        assert all("точные отрывки" not in item.casefold() for item in new_prompts)
    finally:
        await app.stop()


async def test_substantive_recovery_uses_grounded_path(tmp_path: pathlib.Path) -> None:
    app, client = await _make_app(tmp_path)
    try:
        before = len(client.prompts)
        reply = await app.respond(2001, "как бросить пить")
        _assert_useful_russian(reply)
        assert "[" in reply
        new_prompts = client.prompts[before:]
        assert any("точные отрывки" in item.casefold() for item in new_prompts)
    finally:
        await app.stop()


async def test_book_question_uses_grounded_path(tmp_path: pathlib.Path) -> None:
    app, client = await _make_app(tmp_path)
    try:
        before = len(client.prompts)
        reply = await app.respond(2002, "Что говорит Большая книга о страхе?")
        _assert_useful_russian(reply)
        assert "[" in reply
        new_prompts = client.prompts[before:]
        assert any("точные отрывки" in item.casefold() for item in new_prompts)
    finally:
        await app.stop()


async def test_unsupported_out_of_corpus_fails_safe(tmp_path: pathlib.Path) -> None:
    app, client = await _make_app(tmp_path)
    try:
        before = len(client.prompts)
        reply = await app.respond(3001, "Сколько спутников у Юпитера?")
        assert reply == FAIL_CLOSED_REPLY
        assert meets_russian_only(reply)
        new_prompts = client.prompts[before:]
        assert any("точные отрывки" in item.casefold() for item in new_prompts)
    finally:
        await app.stop()


async def test_emergency_takes_safety_path_before_routing(
    tmp_path: pathlib.Path,
) -> None:
    app, client = await _make_app(tmp_path)
    try:
        reply = await app.respond(4001, "Я не могу дышать, задыхаюсь")
        assert "112" in reply
        assert INTERNAL_FAIL_WORDING not in reply
        assert len(client.prompts) == 0
        assert client._sessions == {}
    finally:
        await app.stop()


async def test_meta_turn_through_telegram_adapter_boundary(
    tmp_path: pathlib.Path,
) -> None:
    app, _client = await _make_app(tmp_path)
    try:
        transport = app.transport
        assert isinstance(transport, StubTelegramTransport)
        incoming = TelegramIncoming(
            update_id=1,
            chat_id=5001,
            message_id=1,
            text="А что ты можешь?",
            command=None,
        )
        await app._handle_telegram_update(incoming)
        assert len(transport.sent) == 1
        reply = transport.sent[0].text
        _assert_useful_russian(reply)
    finally:
        await app.stop()


async def test_conversational_and_substantive_share_agent_contract(
    tmp_path: pathlib.Path,
) -> None:
    app, client = await _make_app(tmp_path)
    try:
        await app.respond(6001, "Привет")
        await app.respond(6002, "как бросить пить")
        assert client.prompts
        router = SafetyRouter()
        assert route_turn("Привет", router.check("Привет")).agent == AGENT_NAME
        assert route_turn("как бросить пить", router.check("как бросить пить")).agent == AGENT_NAME
    finally:
        await app.stop()


def test_internal_wording_constant_is_fail_closed_only() -> None:
    assert INTERNAL_FAIL_WORDING in FAIL_CLOSED_REPLY
