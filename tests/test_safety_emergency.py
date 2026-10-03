"""Deterministic emergency-routing safety tests (EN + RU)."""

from __future__ import annotations

from aa.safety.emergency import (
    build_emergency_response,
    contains_prohibited_medical_advice,
    detect_acute_category,
    is_emergency,
)
from aa.safety.router import SafetyDecision, SafetyRouter


def test_ordinary_messages_are_allowed() -> None:
    router = SafetyRouter()
    assert router.check("hello bot").decision is SafetyDecision.ALLOW
    assert router.check("мне грустно, поговорим о книге?").decision is SafetyDecision.ALLOW


def test_empty_messages_are_blocked() -> None:
    router = SafetyRouter()
    assert router.check("").decision is SafetyDecision.BLOCK
    assert router.check("   ").decision is SafetyDecision.BLOCK


EMERGENCY_CASES = [
    ("I have severe withdrawal and shaking, what do I do?", "en"),
    ("I think I am having a seizure right now", "en"),
    ("I keep having convulsions after drinking stopped", "en"),
    ("I am hallucinating and hearing voices", "en"),
    ("I think I took too much and may have overdosed", "en"),
    ("I drank surrogate alcohol and feel poisoned", "en"),
    ("He passed out and I can't wake him", "en"),
    ("She is unconscious after drinking", "en"),
    ("I want to kill myself tonight", "en"),
    ("I don't want to live anymore", "en"),
    ("У меня сильная ломка и трясет, что делать?", "ru"),
    ("Белая горячка началась, вижу то, чего нет", "ru"),
    ("У него судороги и припадок после запоя", "ru"),
    ("Слышу голоса в голове после запоя", "ru"),
    ("Кажется, у меня передоз, выпил слишком много", "ru"),
    ("Отравился суррогатом, что делать?", "ru"),
    ("Он без сознания, не приходит в себя", "ru"),
    ("Я потерял сознание после выпивки", "ru"),
    ("Хочу покончить с собой сегодня", "ru"),
    ("Не хочу больше жить", "ru"),
]


def test_emergency_routing_is_deterministic() -> None:
    router = SafetyRouter()
    for text, _ in EMERGENCY_CASES:
        first = router.check(text)
        second = router.check(text)
        assert first.decision is SafetyDecision.EMERGENCY, text
        assert second.decision is SafetyDecision.EMERGENCY, text
        assert first.reason == second.reason, text
        assert first.emergency_category is not None, text
        assert first.emergency_response, text


def test_emergency_category_detection_covers_all_classes() -> None:
    categories = {detect_acute_category(text) for text, _ in EMERGENCY_CASES}
    assert "seizure" in categories
    assert "hallucination" in categories
    assert "poisoning-overdose" in categories
    assert "loss-of-consciousness" in categories
    assert "self-harm" in categories
    assert "withdrawal" in categories or "severe-withdrawal" in categories
    assert None not in categories
    assert not is_emergency("hello bot")
    assert not is_emergency("")


def test_emergency_response_is_bilingual_and_precedes_ordinary_reply() -> None:
    router = SafetyRouter()
    english = router.check("He passed out and I can't wake him")
    russian = router.check("Он без сознания, не приходит в себя")
    assert "112" in english.emergency_response
    assert "112" in russian.emergency_response
    assert "скорую" in russian.emergency_response or "112" in russian.emergency_response
    # Fixed templates match the builders (routing precedes any AA response).
    assert english.emergency_response == build_emergency_response(lang="en")
    assert russian.emergency_response == build_emergency_response(lang="ru")


def test_emergency_template_contains_no_dosing_or_detox() -> None:
    for lang in ("en", "ru"):
        response = build_emergency_response(lang=lang)
        assert not contains_prohibited_medical_advice(response)
        lowered = response.lower()
        assert "mg" not in lowered or "medical" in lowered or "emergency" in lowered
    # Spot-check the guard itself catches real dosing language.
    assert contains_prohibited_medical_advice("Take 10 mg of diazepam now")
    assert contains_prohibited_medical_advice("Here is a detox protocol for you")
