"""Deterministic emergency/medical-safety routing tests (issue #21).

Positive and negative RU/EN matrix for every required category, ambiguous
wording, historical/past-tense discussion, ordinary AA questions,
determinism, response-content bounds, log privacy and OpenCode precedence.
"""

from __future__ import annotations

import logging
import re

import pytest

from aa.app import Application
from aa.config import Settings
from aa.opencode.client import FakeOpenCodeClient
from aa.opencode.runtime import OpenCodeConfig, StubOpenCodeRuntime
from aa.safety.emergency import EmergencyCategory, classify_emergency
from aa.safety.response import (
    UNSAFE_RESPONSE_PATTERNS,
    build_emergency_response,
)
from aa.safety.router import SafetyDecision, SafetyRouter

# ---------------------------------------------------------------------------
# Positive matrix: (message, expected categories subset, expected language)
# ---------------------------------------------------------------------------

POSITIVE_CASES: tuple[tuple[str, tuple[EmergencyCategory, ...], str], ...] = (
    # Severe alcohol withdrawal.
    (
        "I think I have delirium tremens after I stopped drinking",
        (EmergencyCategory.SEVERE_WITHDRAWAL,),
        "en",
    ),
    (
        "I stopped drinking and now I am shaking and sweating badly",
        (EmergencyCategory.SEVERE_WITHDRAWAL,),
        "en",
    ),
    (
        "У меня белая горячка после запоя, мне страшно",
        (EmergencyCategory.SEVERE_WITHDRAWAL,),
        "ru",
    ),
    (
        "Меня трясет после запоя, руки ходуном",
        (EmergencyCategory.SEVERE_WITHDRAWAL,),
        "ru",
    ),
    # Seizure.
    (
        "I just had a seizure, please help",
        (EmergencyCategory.SEIZURE,),
        "en",
    ),
    (
        "He is convulsing right now and cannot stop",
        (EmergencyCategory.SEIZURE,),
        "en",
    ),
    (
        "У меня начались судороги, помогите",
        (EmergencyCategory.SEIZURE,),
        "ru",
    ),
    (
        "У мужа эпилептический припадок прямо сейчас",
        (EmergencyCategory.SEIZURE,),
        "ru",
    ),
    # Hallucination / delirium-like.
    (
        "I am hearing voices telling me to drink",
        (EmergencyCategory.HALLUCINATION_DELIRIUM,),
        "en",
    ),
    (
        "I stopped drinking and I am seeing things that are not there",
        (EmergencyCategory.HALLUCINATION_DELIRIUM, EmergencyCategory.SEVERE_WITHDRAWAL),
        "en",
    ),
    (
        "У меня галлюцинации после отмены алкоголя",
        (EmergencyCategory.HALLUCINATION_DELIRIUM,),
        "ru",
    ),
    (
        "Я слышу голоса, которых нет",
        (EmergencyCategory.HALLUCINATION_DELIRIUM,),
        "ru",
    ),
    # Poisoning / overdose.
    (
        "I took an overdose of pills an hour ago",
        (EmergencyCategory.POISONING_OVERDOSE,),
        "en",
    ),
    (
        "I took too many pills and I feel very sick now",
        (EmergencyCategory.POISONING_OVERDOSE,),
        "en",
    ),
    (
        "У меня передозировка таблеток, помогите",
        (EmergencyCategory.POISONING_OVERDOSE,),
        "ru",
    ),
    (
        "Я выпил метанол, мне плохо сейчас",
        (EmergencyCategory.POISONING_OVERDOSE,),
        "ru",
    ),
    # Loss of consciousness.
    (
        "My husband is unconscious and not responding",
        (EmergencyCategory.UNCONSCIOUSNESS,),
        "en",
    ),
    (
        "He passed out and won't wake up",
        (EmergencyCategory.UNCONSCIOUSNESS,),
        "en",
    ),
    (
        "Муж без сознания и не реагирует",
        (EmergencyCategory.UNCONSCIOUSNESS,),
        "ru",
    ),
    (
        "Он потерял сознание и не приходит в себя",
        (EmergencyCategory.UNCONSCIOUSNESS,),
        "ru",
    ),
    # Self-harm / suicide.
    (
        "I want to kill myself tonight",
        (EmergencyCategory.SELF_HARM_SUICIDE,),
        "en",
    ),
    (
        "I have thoughts of suicide and I don't want to live",
        (EmergencyCategory.SELF_HARM_SUICIDE,),
        "en",
    ),
    (
        "Я хочу умереть, не вижу смысла жить",
        (EmergencyCategory.SELF_HARM_SUICIDE,),
        "ru",
    ),
    (
        "Думаю о суициде прямо сейчас",
        (EmergencyCategory.SELF_HARM_SUICIDE,),
        "ru",
    ),
    # Other life-threatening.
    (
        "I can't breathe and I have severe chest pain",
        (EmergencyCategory.LIFE_THREATENING_OTHER,),
        "en",
    ),
    (
        "He is bleeding heavily and the blood won't stop",
        (EmergencyCategory.LIFE_THREATENING_OTHER,),
        "en",
    ),
    (
        "Я не могу дышать, задыхаюсь",
        (EmergencyCategory.LIFE_THREATENING_OTHER,),
        "ru",
    ),
    (
        "У него сильное кровотечение, кровь не останавливается",
        (EmergencyCategory.LIFE_THREATENING_OTHER,),
        "ru",
    ),
)

# ---------------------------------------------------------------------------
# Negative matrix: ordinary AA discussion, history, ambiguity, questions.
# ---------------------------------------------------------------------------

NEGATIVE_CASES: tuple[str, ...] = (
    # Ordinary AA questions (EN).
    "What does the Big Book say about fear?",
    "Can you explain Step 4 and the resentment inventory?",
    "I relapsed last month, what should I do about my program?",
    "What is the Doctor's Opinion about alcoholism?",
    "Tell me about the fear inventory in AA.",
    "I feel resentful towards my boss, how do I pray about it?",
    "What are the Twelve Steps?",
    "I am afraid I might drink again at the party tomorrow.",
    # Ordinary AA questions (RU).
    "Что говорит Большая книга о страхе?",
    "Расскажи о четвертом шаге и инвентаризации обид.",
    "У меня страх перед выступлением на группе.",
    "Я сорвался в прошлом месяце, что делать с программой?",
    "Что такое мнение доктора об алкоголизме?",
    "Меня мучает обида на отца, как с этим работать?",
    "Я боюсь сорваться на празднике.",
    # Historical / past-tense discussion (EN).
    "I had a seizure five years ago but I am fine now.",
    "Ten years ago I had an overdose and ended up in hospital.",
    "I used to hear voices when I was drinking heavily back then.",
    "When I was young I once passed out from exhaustion, long ago.",
    "I had withdrawal symptoms years ago when I first tried to quit.",
    # Historical / past-tense discussion (RU).
    "Пять лет назад у меня был припадок, но сейчас все хорошо.",
    "Год назад было отравление, давно прошло.",
    "Раньше, когда я пил, у меня тряслись руки по утрам.",
    "В молодости один раз упал в обморок, тогда все обошлось.",
    "Много лет назад была белая горячка у знакомого.",
    # Ambiguous / metaphorical wording (EN).
    "I am dying for a cup of coffee this morning.",
    "That meeting was so boring I thought I would die laughing.",
    "My phone battery is dying and I need a charger.",
    "He is poisoned by resentment towards his brother.",
    "I feel like I am carrying the weight of the world today.",
    "Is there an OD meeting tonight?",
    "Tell me about seizures and what AA literature says.",
    "What is delirium tremens?",
    "What seizure medication is safe with alcohol?",
    "Seizures run in my family, what does that mean?",
    # Ambiguous / metaphorical wording (RU).
    "Я умираю от смеха на этих собраниях.",
    "Умираю, как хочу кофе сегодня утром.",
    "Он отравляет себе жизнь обидой.",
    "Что такое белая горячка?",
    "Расскажи мне о судорогах и что говорит литература АА.",
    "Я читал о суициде в книге, что об этом думать?",
    "У нас в группе обсуждали инсульт у отца одного участника.",
    # Negated / no-risk statements (EN + RU).
    "I have no thoughts of suicide and I feel hopeful.",
    "I have never had seizures and I feel fine.",
    "I am not hearing voices, just sharing about my fear.",
    "Нет суицидальных мыслей, чувствую себя хорошо.",
    "У меня нет судорог, просто делюсь страхом.",
    "I had a scare yesterday but I am fine now and safe.",
)


@pytest.mark.parametrize(("message", "expected", "language"), POSITIVE_CASES)
def test_positive_matrix_triggers_emergency(
    message: str, expected: tuple[EmergencyCategory, ...], language: str
) -> None:
    classification = classify_emergency(message)
    assert classification.is_emergency, f"missed emergency: {message!r}"
    for category in expected:
        assert category in classification.categories, (
            f"{message!r}: missing {category.value}, got "
            f"{[item.value for item in classification.categories]}"
        )
    assert classification.language == language


@pytest.mark.parametrize("message", NEGATIVE_CASES)
def test_negative_matrix_stays_on_normal_path(message: str) -> None:
    classification = classify_emergency(message)
    assert not classification.is_emergency, (
        f"false positive for {message!r}: {[item.value for item in classification.categories]}"
    )


def test_mixed_history_then_current_triggers_on_current_sentence() -> None:
    message = "I had a seizure five years ago. Right now I am convulsing, help me."
    classification = classify_emergency(message)
    assert classification.is_emergency
    assert EmergencyCategory.SEIZURE in classification.categories


def test_current_then_resolved_history_stays_calm() -> None:
    classification = classify_emergency("I had a seizure five years ago and I am fine now.")
    assert not classification.is_emergency


def test_deterministic_repeated_calls_agree() -> None:
    samples = [case[0] for case in POSITIVE_CASES[:6]] + list(NEGATIVE_CASES[:6])
    for sample in samples:
        first = classify_emergency(sample)
        second = classify_emergency(sample)
        assert first == second


def test_router_marks_emergency_with_categories() -> None:
    router = SafetyRouter()
    result = router.check("I want to kill myself tonight")
    assert result.decision is SafetyDecision.EMERGENCY
    assert EmergencyCategory.SELF_HARM_SUICIDE in result.categories
    assert result.classification is not None
    assert "self-harm-suicide" in result.reason


def test_router_route_returns_bounded_reply_before_opencode() -> None:
    router = SafetyRouter()
    result, reply = router.route("Я хочу умереть")
    assert result.decision is SafetyDecision.EMERGENCY
    assert reply is not None
    assert "112" in reply
    result_ok, reply_ok = router.route("What does the Big Book say about fear?")
    assert result_ok.decision is SafetyDecision.ALLOW
    assert reply_ok is None


def test_router_keeps_empty_block_and_default_allow() -> None:
    router = SafetyRouter()
    assert router.check("").decision is SafetyDecision.BLOCK
    assert router.check("   ").decision is SafetyDecision.BLOCK
    assert router.check("hello bot").decision is SafetyDecision.ALLOW


@pytest.mark.parametrize("message", [case[0] for case in POSITIVE_CASES])
def test_emergency_responses_contain_no_dosing_or_detox(message: str) -> None:
    classification = classify_emergency(message)
    reply = build_emergency_response(classification)
    lowered = reply.lower()
    for pattern in UNSAFE_RESPONSE_PATTERNS:
        assert re.search(pattern, lowered) is None, (
            f"unsafe pattern {pattern!r} in emergency reply for {message!r}"
        )


def test_emergency_response_directs_to_help_and_trusted_person() -> None:
    for message in ("I can't breathe", "Я не могу дышать"):
        classification = classify_emergency(message)
        assert classification.is_emergency
        reply = build_emergency_response(classification)
        lowered = reply.lower()
        assert "emergency" in lowered or "экстренной" in lowered or "скорой" in lowered
        assert "trust" in lowered or "доверяете" in lowered or "доверяете" in reply
        # No diagnosis is offered: the reply must not claim what the user has.
        assert "you have" not in lowered
        assert "у вас" not in lowered


def test_emergency_response_language_matches_input() -> None:
    ru = classify_emergency("Я хочу умереть")
    en = classify_emergency("I want to kill myself")
    assert ru.language == "ru"
    assert en.language == "en"
    assert "экстренной" in build_emergency_response(ru) or "скорой" in build_emergency_response(ru)
    assert "emergency" in build_emergency_response(en).lower()


def test_safety_logs_never_contain_raw_message_bodies(
    caplog: pytest.LogCaptureFixture,
) -> None:
    router = SafetyRouter()
    sensitive_en = "I want to kill myself with this unique phrase xyzzy123"
    sensitive_ru = "У меня уникальная фраза белочка xyzzy123 после запоя"
    with caplog.at_level(logging.INFO, logger="aa.safety.router"):
        router.check(sensitive_en)
        router.check(sensitive_ru)
        router.check("What does the Big Book say about fear?")
    for record in caplog.records:
        rendered = record.getMessage()
        assert "xyzzy123" not in rendered
        assert sensitive_en not in rendered
        assert sensitive_ru not in rendered
        assert "kill myself" not in rendered
        assert "хочу умереть" not in rendered


async def test_emergency_path_precedes_opencode_and_skips_llm() -> None:
    from aa.conversation.graph_runtime import GraphTurnRuntime

    settings = Settings.from_env({})
    runtime = StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )

    async def _delegate(thread: str, text: str) -> str:
        return "Понял вас. Давайте разберём это спокойно. Что сейчас важнее?"

    graph_runtime = GraphTurnRuntime(delegate=_delegate)
    app = Application(settings, opencode_runtime=runtime, graph_runtime=graph_runtime)
    await app.start()
    try:
        reply = await app.respond(123, "I want to kill myself tonight")
        # Production Telegram runtime is RU-only (issue #98): even an
        # English emergency turn receives the deterministic Russian reply.
        assert "112" in reply
        assert "экстренной" in reply or "скорой" in reply
        from aa.conversation.orchestrator import contains_english_fallback

        assert not contains_english_fallback(reply)
        # No OpenCode session must have been created and no prompt sent.
        client = runtime.client
        assert isinstance(client, FakeOpenCodeClient)
        assert client._sessions == {}
        # An ordinary turn uses only the LangGraph runtime (issue #118):
        # it returns a natural Russian reply, never the retired
        # technical fail-closed reply and never implementation mechanics.
        from aa.conversation.turn_pipeline import contains_cyrillic, leaks_internal_terms

        normal = await app.respond(123, "What does the Big Book say about fear?")
        assert contains_cyrillic(normal)
        assert not leaks_internal_terms(normal)
        assert client._sessions == {}
    finally:
        await app.stop()


async def test_emergency_path_is_deterministic_across_calls() -> None:
    settings = Settings.from_env({})
    runtime = StubOpenCodeRuntime(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", command="opencode", workdir=".")
    )
    app = Application(settings, opencode_runtime=runtime)
    await app.start()
    try:
        first = await app.respond(7, "Я не могу дышать")
        second = await app.respond(7, "Я не могу дышать")
        assert first == second
    finally:
        await app.stop()
