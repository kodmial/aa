"""Conversational meta/capability turn boundary (issues #105/#106).

Meta/capability/identity questions (``А что ты можешь?``,
``Тогда зачем ты?``) are conversational turns, not corpus-grounded AA
turns. Routing them through the book-grounded pipeline fails closed with
the generic unsupported-answer message and breaks the user-facing
contract. This module detects them deterministically so the application
serves a bounded Russian capability reply without retrieval, grounding,
or model dependence.
"""

from __future__ import annotations

from aa.retrieval.normalize import normalize_ru

# Normalized substring patterns for meta/capability/identity questions.
# Matched against ``normalize_ru`` text (casefolded, ``ё`` -> ``е``).
META_CAPABILITY_PATTERNS: tuple[str, ...] = (
    "что ты можешь",
    "что ты умеешь",
    "что умеешь",
    "зачем ты",
    "зачем ты нужен",
    "зачем ты нужна",
    "кто ты",
    "что ты такое",
    "твои возможности",
    "твои функции",
    "что ты делаешь",
    "чем ты можешь помочь",
    "чем можешь помочь",
    "как ты можешь помочь",
    "расскажи о себе",
    "что ты за бот",
    "что за бот",
    "твоя помощь",
)

META_CAPABILITY_REPLY = (
    "Я помощник по материалам сообщества: поддерживаю разговор о трезвости, "
    "помогаю разобрать тягу, срыв и отношения, подсказываю ближайшие шаги. "
    "Спросите о конкретной ситуации."
)


def is_meta_capability_request(text: str) -> bool:
    """Return whether ``text`` is a meta/capability/identity question."""
    normalized = normalize_ru(text.strip())
    if not normalized:
        return False
    return any(pattern in normalized for pattern in META_CAPABILITY_PATTERNS)


__all__ = [
    "META_CAPABILITY_PATTERNS",
    "META_CAPABILITY_REPLY",
    "is_meta_capability_request",
]
