"""Gate C live repair for kodmial/aa#236 (recurrence 2), superseded by #240.

Live run 37732467481 on exact main d4cb46f spread slow-tail turns
across a SHA-256-selected pool sized above the diversity floor, so an
all-fallback run could still satisfy ``len(set(replies)) >= 8``.
Manual Telegram evidence on 2026-10-08 (kodmial/aa#240) proves that
variety is not help: real drinking/recovery requests were served
hash-selected filler with zero verified book units while Gate C
reported PASS on diversity.

kodmial/aa#240 keeps the pool frozen (byte-identical, in order) so
qualification keeps counting every template retry as a failed
non-answer, but selection is now stable and single: degraded turns
collapse visibly to one string instead of mimicking helpful variety.
A retry without verified substantive material is an explicit product
failure, never completion. Product Contract #110 is unchanged.
"""

from __future__ import annotations

from aa.conversation.output_limits import envelope_passes
from aa.conversation.turn_pipeline import (
    NATURAL_CLARIFICATION_REPLY,
    NATURAL_RETRY_REPLY,
    NATURAL_RETRY_VARIANTS,
    contains_cyrillic,
    leaks_internal_terms,
    select_retry_reply,
)


def test_retry_variants_contract() -> None:
    # The pool must clear the live ``len(set(replies)) >= 8`` diversity
    # floor even in an all-fallback worst case.
    assert len(NATURAL_RETRY_VARIANTS) >= 8
    assert NATURAL_RETRY_VARIANTS[0] == NATURAL_RETRY_REPLY
    assert len(set(NATURAL_RETRY_VARIANTS)) == len(NATURAL_RETRY_VARIANTS)
    for variant in NATURAL_RETRY_VARIANTS:
        assert variant.strip()
        assert variant != NATURAL_CLARIFICATION_REPLY
        assert contains_cyrillic(variant)
        assert not leaks_internal_terms(variant)
        assert envelope_passes(variant)


def test_retry_selection_is_stable_single_without_content_matching() -> None:
    # kodmial/aa#240: every message selects the same single retry;
    # selection never matches on wording, family, or keywords, and never
    # uses a hash to mimic helpful variety.
    first = select_retry_reply("К вечеру тянет выпить, как быть?")
    assert first == NATURAL_RETRY_REPLY
    assert select_retry_reply("К вечеру тянет выпить, как быть?") == first
    assert select_retry_reply("  К ВЕЧЕРУ ТЯНЕТ ВЫПИТЬ, КАК БЫТЬ?  ") == first
    # Empty input keeps the historical single retry.
    assert select_retry_reply("") == NATURAL_RETRY_REPLY
    assert select_retry_reply("   ") == NATURAL_RETRY_REPLY


def test_retry_selection_never_spreads_unrelated_prompts() -> None:
    # Unrelated prompts (including same-length ones) collapse to the one
    # stable retry by design: a pool of generic replies must never pass
    # for grounded help regardless of diversity.
    seen = {select_retry_reply(f"сообщение {i} {'x' * i}") for i in range(32)}
    assert seen == {NATURAL_RETRY_REPLY}


def test_live_families_collapse_visibly_to_one_retry() -> None:
    # Representative prompts across the live lane families: retries for
    # unrelated turns collapse to one string instead of spreading, so a
    # fallback run is visibly bookless and fails the grounded-answer
    # checks.
    prompts = [
        "Чем ты вообще можешь быть полезен здесь?",
        "К вечеру очень тянет выпить, как с этим обходиться?",
        "Дома снова ссора из-за моей выпивки, как мне на это посмотреть?",
        "А почему это вообще важно?",
        "А теперь другое: ночью не могу успокоиться и уснуть",
        "Стоит ли мне сейчас покупать акции?",
        "Мне трудно признать, что одному не получается",
        "И что из этого следует для меня прямо сейчас?",
    ]
    retries = [select_retry_reply(prompt) for prompt in prompts]
    assert set(retries) == {NATURAL_RETRY_REPLY}
    assert all(reply != NATURAL_CLARIFICATION_REPLY for reply in retries)


def test_all_fallback_worst_case_cannot_clear_diversity_floor() -> None:
    # Worst case: every turn in a 16-turn live lane falls back, so the
    # retry alone yields exactly one distinct string and can never clear
    # the ``len(set(replies)) >= 8`` floor. Only verified grounded
    # answers clear it.
    prompts = [
        f"тестовое сообщение номер {i} " + "длинный хвост " * (i % 5) + "конец" for i in range(16)
    ]
    retries = [select_retry_reply(prompt) for prompt in prompts]
    assert len(set(retries)) == 1
    assert all(reply != NATURAL_CLARIFICATION_REPLY for reply in retries)
