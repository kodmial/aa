"""Bounded emergency response path (Russian production, English retained).

The templates below are the only text the deterministic safety layer may
produce. The production Telegram runtime always serves the Russian
template (RU-only user-facing contract); the English template exists for
language detection coverage and non-production callers only.
They deliberately contain:

- direction to immediate local emergency/medical help;
- a prompt to involve a trusted nearby person;
- a statement that the bot cannot diagnose, prescribe, or schedule detox;
- an optional brief AA-oriented supportive follow-up after urgent guidance.

They never contain medication dosing, detox schedules, diagnoses, or
instructions for unsupervised withdrawal.
"""

from __future__ import annotations

from aa.safety.emergency import EmergencyClassification

EMERGENCY_RESPONSE_EN = (
    "I am concerned about what you describe. "
    "What you describe can be dangerous, so please act right now: "
    "call your local emergency number immediately "
    "(for example 112, 911, or your country's emergency number) "
    "or contact your local emergency medical service. "
    "If possible, ask a trusted person nearby to stay with you and help you call. "
    "I cannot diagnose you, prescribe medication, give doses, "
    "or plan detox or withdrawal treatment, "
    "and you should not try to manage this alone or unsupervised. "
    "Once you are safe and in contact with medical help, "
    "AA fellowship support can be there for you too."
)

EMERGENCY_RESPONSE_RU = (
    "Меня беспокоит то, что вы описываете. "
    "Описанное может быть опасно для жизни, поэтому действуйте прямо сейчас: "
    "немедленно позвоните в местную службу экстренной помощи "
    "(например, по номеру 112 или по номеру экстренной помощи вашей страны) "
    "или обратитесь в ближайшую службу скорой медицинской помощи. "
    "Если рядом есть человек, которому вы доверяете, "
    "попросите его остаться с вами и помочь вызвать помощь. "
    "Я не могу ставить диагноз, назначать лекарства, указывать дозы "
    "или составлять план выхода из запоя или снятия абстиненции, "
    "и не пытайтесь справиться с этим самостоятельно или без наблюдения. "
    "Когда вы будете в безопасности и на связи с медицинской помощью, "
    "поддержка сообщества АА тоже сможет быть рядом с вами."
)

# Patterns that must never appear in an emergency response. The test suite
# asserts their absence so future template edits cannot silently introduce
# dosing or unsupervised-detox instructions.
UNSAFE_RESPONSE_PATTERNS = (
    r"\b\d+\s*mg\b",
    r"\b\d+\s*мг\b",
    r"\b\d+\s*ml\b",
    r"\b\d+\s*мл\b",
    r"detox\s+schedule",
    r"withdrawal\s+schedule",
    r"take\s+\d+",
    r"принимайте?\s+\d+",
    r"выпейте?\s+\d+",
    r"доза",
    r"\bdose\b",
    r"\bdosage\b",
    r"дозировка",
    r"taper\s+off",
    r"похмелиться",
    r"похмелитесь",
    r"reduce\s+drinking\s+gradually",
    r"пейте\s+меньше\s+каждый\s+день",
)


def build_emergency_response(classification: EmergencyClassification, *, language: str = "") -> str:
    """Return the bounded safe reply for an emergency classification."""
    resolved = language or classification.language
    if resolved == "ru":
        return EMERGENCY_RESPONSE_RU
    return EMERGENCY_RESPONSE_EN


__all__ = [
    "EMERGENCY_RESPONSE_EN",
    "EMERGENCY_RESPONSE_RU",
    "UNSAFE_RESPONSE_PATTERNS",
    "build_emergency_response",
]
