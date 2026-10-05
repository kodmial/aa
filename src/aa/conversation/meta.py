"""Retired conversational meta/capability turn boundary (issues #105/#106).

Retired by the issue #118 production cutover: ordinary turns no longer
route through ``is_substantive``/meta/punctuation heuristics in
production. The LangGraph planner owns conversational continuity and
meta questions are answered as natural turns. This module remains for
offline qualification history only; production code must not import it.
"""

from __future__ import annotations

from aa.retrieval.normalize import normalize_ru, ru_tokens

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


# Token sequences for each capability pattern (word/phrase boundaries).
# Matching is token-contiguous so ``что ты можешь`` inside
# ``Что ты можешь сказать о страхе ... по книге?`` does not misroute a
# grounded turn: the pattern must cover nearly the whole utterance.
_META_PATTERN_TOKEN_SEQS: tuple[tuple[str, ...], ...] = tuple(
    tuple(ru_tokens(pattern)) for pattern in META_CAPABILITY_PATTERNS
)

# Extra non-pattern tokens tolerated around a capability question
# (politeness wrappers such as ``а``, ``тогда``, ``пожалуйста``).
# Composite capability turns (``Расскажи о себе, чем ты можешь помочь?``)
# are covered by several patterns at once, so coverage is unioned.
_MAX_META_RESIDUAL_TOKENS = 3


def is_meta_capability_request(text: str) -> bool:
    """Return whether ``text`` is a meta/capability/identity question."""
    normalized = normalize_ru(text.strip())
    if not normalized:
        return False
    tokens = ru_tokens(normalized)
    if not tokens:
        return False
    covered: set[int] = set()
    for seq in _META_PATTERN_TOKEN_SEQS:
        if not seq:
            continue
        width = len(seq)
        if width > len(tokens):
            continue
        for start in range(len(tokens) - width + 1):
            if tuple(tokens[start : start + width]) == seq:
                covered.update(range(start, start + width))
    if not covered:
        return False
    return len(tokens) - len(covered) <= _MAX_META_RESIDUAL_TOKENS


__all__ = [
    "META_CAPABILITY_PATTERNS",
    "META_CAPABILITY_REPLY",
    "is_meta_capability_request",
]
