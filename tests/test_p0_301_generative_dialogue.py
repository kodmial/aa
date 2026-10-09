"""P0 kodmial/aa#301: generative dialogue only, no hardcoded replies.

Every normal user-facing chat output (greeting, identity/capabilities,
clarification, follow-up, disambiguation, safety-recovered answers) is
composed by the AA model from the user message and conversational
state. Model/provider/retrieval/timeout/verifier failures are typed
unsuccessful outcomes (:class:`TurnFailed`), surfaced at the transport
boundary only as the clearly marked service error, never as synthetic
successful AA conversation and never qualifying as a substantive
answer.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aa.conversation.failures import (
    SERVICE_ERROR_MARKER,
    SERVICE_ERROR_REPLY,
    TurnFailed,
    is_service_error,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "aa"

# Legitimate fixed Russian strings that are NOT AA conversation:
# Telegram command/control replies, emergency service signalling,
# voice-transport errors, and the marked service-error signal itself.
ALLOWED_FIXED_RUSSIAN_OWNERS = (
    "src/aa/app.py",  # _START_REPLY/_NEW_REPLY/_BUSY_REPLY + service error
    "src/aa/safety/response.py",  # emergency templates
    "src/aa/telegram/voice.py",  # voice transport errors
    "src/aa/conversation/failures.py",  # marked service error
)

BANNED_IDENTIFIERS = (
    "SAFE_UNAVAILABLE_REPLY",
    "NATURAL_CLARIFICATION_REPLY",
    "NATURAL_RETRY_REPLY",
    "NATURAL_RETRY_VARIANTS",
    "CONVERSATIONAL_FALLBACK_REPLY",
    "META_CAPABILITY_REPLY",
    "META_CAPABILITY_PATTERNS",
    "FAIL_CLOSED_REPLY",
    "ENVELOPE_FALLBACK_REPLY",
    "select_retry_reply",
)


def _iter_src_files() -> list[pathlib.Path]:
    return sorted((SRC).rglob("*.py"))


def test_no_hardcoded_conversational_replies_in_src() -> None:
    """Static inventory: no canned AA conversational reply may reappear."""
    violations: list[str] = []
    for path in _iter_src_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
        for banned in BANNED_IDENTIFIERS:
            if banned in names:
                violations.append(f"{path.relative_to(ROOT)}: {banned}")
    assert violations == [], f"hardcoded conversational replies reintroduced: {violations}"


def test_fixed_strings_outside_allowlist_are_not_conversation() -> None:
    """No new fixed Russian reply constants outside protocol/error owners."""
    offenders: list[str] = []
    for path in _iter_src_files():
        rel = path.as_posix()
        if any(rel.endswith(owner) for owner in ALLOWED_FIXED_RUSSIAN_OWNERS):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            targets: list[str] = []
            value: Any = None
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                value = node.value.value
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        targets.append(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant):
                value = node.value.value
                if isinstance(node.target, ast.Name):
                    targets.append(node.target.id)
            else:
                continue
            if not isinstance(value, str):
                continue
            cyrillic = sum(1 for ch in value if "\u0400" <= ch <= "\u04ff")
            if cyrillic < 8:
                continue
            for name in targets:
                upper = name.upper()
                if upper.endswith(("REPLY", "REPLIES", "FALLBACK", "VARIANTS")):
                    offenders.append(f"{path.relative_to(ROOT)}:{name}")
    assert offenders == [], f"new fixed conversational replies: {offenders}"


def test_removed_names_are_not_importable() -> None:
    import importlib

    for module_name in (
        "aa.conversation.turn_pipeline",
        "aa.conversation.output_limits",
        "aa.conversation.meta",
        "aa.conversation.orchestrator",
        "aa.safety.outbound",
    ):
        module = importlib.import_module(module_name)
        for banned in BANNED_IDENTIFIERS:
            assert not hasattr(module, banned), f"{module_name}.{banned}"


def _pack_entry(
    text: str = "Фиктивная поддержка рядом помогает пережить тягу сегодня.",
) -> dict[str, Any]:
    return {
        "passage_id": "chapter-3#exp0000",
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
    }


class _StaticAnswer:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    async def ainvoke(self, messages: Any) -> AIMessage:
        _ = messages
        self.calls += 1
        return AIMessage(content=self._text)


class _PassVerifier:
    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        return {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["chapter-3#exp0000"],
            "addresses_intent": True,
        }


class _GluePassVerifier:
    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        return {
            "requires_book_evidence": False,
            "supported": True,
            "evidence_passage_ids": [],
            "addresses_intent": True,
        }


class _FailVerifier:
    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        return {
            "requires_book_evidence": True,
            "supported": False,
            "evidence_passage_ids": [],
            "addresses_intent": False,
        }


async def test_substantive_help_comes_from_generative_path() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    model_text = "Поддержка рядом помогает пережить тягу сегодня спокойно."
    outcome = await run_v2_answer_turn(
        user_message="К вечеру тянет выпить, как быть?",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_StaticAnswer(model_text),
        verifier_model=_PassVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
    )
    assert outcome["text"] == model_text
    assert not is_service_error(outcome["text"])
    assert outcome["telemetry"]["qualified"] is True


async def test_multiturn_context_flows_to_generation() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    seen: list[str] = []

    class _ContextAnswer:
        async def ainvoke(self, messages: Any) -> AIMessage:
            from langchain_core.messages import BaseMessage

            if isinstance(messages, list):
                for item in messages:
                    if isinstance(item, BaseMessage) and item.type == "human":
                        content = item.content
                        if isinstance(content, str):
                            seen.append(content)
            return AIMessage(content="Поддержка рядом помогает пережить тягу сегодня спокойно.")

    prior_user = HumanMessage(content="Вечером тяжело без выпивки, как справляться?")
    prior_ai = AIMessage(content="Поддержка рядом помогает пережить тягу сегодня.")
    outcome = await run_v2_answer_turn(
        user_message="А если тяга вернётся позже?",
        summary="Обсуждали вечернюю тягу и поддержку рядом.",
        recent=[prior_user, prior_ai],
        evidence_pack=[_pack_entry()],
        answer_model=_ContextAnswer(),
        verifier_model=_PassVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=8,
    )
    assert not is_service_error(outcome["text"])
    assert any("тягу" in chunk or "вечер" in chunk for chunk in seen)


async def test_meta_hello_is_generative() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    model_text = "Привет! Я помощник, поддерживаю разговор о трезвости."
    outcome = await run_v2_answer_turn(
        user_message="привет",
        summary="",
        recent=[],
        evidence_pack=[],
        answer_model=_StaticAnswer(model_text),
        verifier_model=_GluePassVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=0,
        planner_reason="legitimate-glue",
        planner_mode="conversational",
        resolved_intent="",
    )
    assert outcome["text"] == model_text
    assert not is_service_error(outcome["text"])


async def test_generative_clarification_for_glue() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    model_text = "Расскажите, что сейчас важнее всего разобрать?"
    outcome = await run_v2_answer_turn(
        user_message="А что ты можешь?",
        summary="",
        recent=[],
        evidence_pack=[],
        answer_model=_StaticAnswer(model_text),
        verifier_model=_GluePassVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=0,
        planner_reason="legitimate-glue",
        planner_mode="conversational",
        resolved_intent="",
    )
    assert outcome["text"] == model_text
    assert not is_service_error(outcome["text"])


async def test_unsafe_generation_recovered_safely() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.safety.outbound import is_outbound_safe

    harmful = "Попробуйте начать пить и резко прекратить, повторите несколько раз."
    safe = "Поддержка рядом помогает пережить тягу сегодня спокойно."

    class _HarmThenSafe:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            self.calls += 1
            return AIMessage(content=harmful if self.calls == 1 else safe)

    outcome = await run_v2_answer_turn(
        user_message="Как мне бросить пить?",
        summary="",
        recent=[],
        evidence_pack=[_pack_entry()],
        answer_model=_HarmThenSafe(),
        verifier_model=_PassVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
    )
    assert outcome["text"] == safe
    assert is_outbound_safe(outcome["text"])
    assert harmful not in outcome["text"]


async def test_impossible_generation_is_typed_failure() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    with pytest.raises(TurnFailed):
        await run_v2_answer_turn(
            user_message="Как обходиться с тягой вечером?",
            summary="",
            recent=[],
            evidence_pack=[_pack_entry()],
            answer_model=_StaticAnswer("Неподтверждённая мысль без опоры."),
            verifier_model=_FailVerifier(),
            planner_model=None,
            retrieval_index=None,
            initial_query_count=12,
        )
    assert _is_grounded_substantive_reply({}, SERVICE_ERROR_REPLY) is False


async def test_provider_outage_is_typed_failure_with_service_error() -> None:
    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.graph_runtime import GraphTurnRuntime
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime

    async def _boom(thread: str, text: str) -> str:
        raise RuntimeError("provider down")

    runtime = GraphTurnRuntime(delegate=_boom)
    await runtime.start()
    app = Application(
        Settings.from_env({}),
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        graph_runtime=runtime,
    )
    await app.start()
    try:
        reply = await app.respond(424201, "тяга вечером, что делать?")
        assert is_service_error(reply)
        assert SERVICE_ERROR_MARKER in reply
    finally:
        await app.stop()
        await runtime.stop()


async def test_repeated_failures_stay_stable_service_error() -> None:
    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.graph_runtime import GraphTurnRuntime
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    async def _boom(thread: str, text: str) -> str:
        raise RuntimeError("provider down")

    runtime = GraphTurnRuntime(delegate=_boom)
    await runtime.start()
    app = Application(
        Settings.from_env({}),
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        graph_runtime=runtime,
    )
    await app.start()
    try:
        replies = [await app.respond(424202, f"сообщение {i}") for i in range(4)]
        assert len(set(replies)) == 1
        assert all(is_service_error(item) for item in replies)
        assert all(_is_grounded_substantive_reply({}, item) is False for item in replies)
    finally:
        await app.stop()
        await runtime.stop()


async def test_voice_text_identity() -> None:
    """Voice transcripts enter the exact text boundary (no voice mode)."""
    from aa.app import Application
    from aa.config import Settings
    from aa.conversation.graph_runtime import GraphTurnRuntime
    from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
    from aa.telegram.transport import (
        StubTelegramTransport,
        TelegramIncoming,
        VoiceAttachment,
    )

    async def _echo(thread: str, text: str) -> str:
        return f"Эхо: {text}"

    class _FakeRecognizer:
        @property
        def available(self) -> bool:
            return True

    class _FakeVoicePipeline:
        """Minimal ASR stand-in exercising voice bytes handling."""

        def __init__(self, transcript: str) -> None:
            self._transcript = transcript
            self.calls = 0
            self.recognizer = _FakeRecognizer()

        async def transcribe_voice(
            self,
            *,
            file_id: str,
            file_size_bytes: int | None = None,
            duration_seconds: int | None = None,
        ) -> str:
            _ = (file_id, file_size_bytes, duration_seconds)
            self.calls += 1
            return self._transcript

    transport = StubTelegramTransport()
    voice_pipeline = _FakeVoicePipeline("не могу уснуть")
    runtime = GraphTurnRuntime(delegate=_echo)
    await runtime.start()
    app = Application(
        Settings.from_env({}),
        transport=transport,
        opencode_runtime=StubOpenCodeRuntime(
            OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
        ),
        voice_pipeline=voice_pipeline,  # type: ignore[arg-type]
        graph_runtime=runtime,
    )
    await app.start()
    try:
        text_reply = await app.respond(424203, "не могу уснуть")
        await app._process_dispatched_update(
            TelegramIncoming(
                update_id=1,
                chat_id=424204,
                message_id=1,
                text="",
                command=None,
                voice=VoiceAttachment(file_id="f1", duration_seconds=5, file_size_bytes=100),
            )
        )
        assert voice_pipeline.calls == 1
        assert len(transport.sent) == 1
        voice_reply = transport.sent[0].text
        assert text_reply == voice_reply
        assert not is_service_error(text_reply)
    finally:
        await app.stop()
        await runtime.stop()


async def test_explicit_drink_self_test_never_serves_harm() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.safety.outbound import is_outbound_safe

    harmful = "Попробуйте начать пить и резко прекратить, повторите несколько раз."

    with pytest.raises(TurnFailed) as exc:
        await run_v2_answer_turn(
            user_message="Можно ли мне проверить себя, выпив немного?",
            summary="",
            recent=[],
            evidence_pack=[_pack_entry()],
            answer_model=_StaticAnswer(harmful),
            verifier_model=_PassVerifier(),
            planner_model=None,
            retrieval_index=None,
            initial_query_count=12,
        )
    assert exc.value.category == "safety-blocked"
    assert is_outbound_safe(harmful) is False
