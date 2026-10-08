"""Regression guard: the Gate C repair must not game reply diversity.

A formerly shipped SHA-256 pool of 10 polite non-answers met the old
8-distinct-reply test while real Telegram users received no book-grounded
help (manual Telegram evidence on 2026-10-08, kodmial/aa#240: real
drinking/recovery requests were served hash-selected filler with zero
verified book units while Gate C reported PASS on diversity). Technical
failures must now be identified as unqualified and counted as product
failures, not disguised as valid AA advice: selection is stable and
single, so fallback runs collapse visibly to one honest unqualified
status instead of mimicking helpful variety.
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
from aa.qualification.product_contract_live import _is_grounded_substantive_reply


def test_unqualified_retry_is_honest_bounded_russian_status() -> None:
    assert NATURAL_RETRY_VARIANTS == (NATURAL_RETRY_REPLY,)
    assert NATURAL_RETRY_REPLY != NATURAL_CLARIFICATION_REPLY
    assert "не удалось" in NATURAL_RETRY_REPLY
    assert "книге" in NATURAL_RETRY_REPLY
    assert contains_cyrillic(NATURAL_RETRY_REPLY)
    assert not leaks_internal_terms(NATURAL_RETRY_REPLY)
    assert envelope_passes(NATURAL_RETRY_REPLY)


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


def test_retry_is_not_shuffled_to_fake_diversity() -> None:
    prompts = (
        "Как бросить пить?",
        "К вечеру опять тянет выпить",
        "Я думал ты дашь рекомендации",
        "Что делать с зависимостью?",
        "Стоит ли купить акции?",
        "",
        "   ",
    )
    assert {select_retry_reply(prompt) for prompt in prompts} == {NATURAL_RETRY_REPLY}
    assert select_retry_reply("КАК БРОСИТЬ ПИТЬ?") == NATURAL_RETRY_REPLY


def test_bookless_retries_never_prove_grounded_help() -> None:
    # Even optimistic stage numbers must not qualify a known non-answer.
    snapshot = {
        "answer_outcome": "served",
        "planner_query_count": 12,
        "retrieval_passages": 5,
        "verified_book_units": 2,
    }
    assert not _is_grounded_substantive_reply(snapshot, NATURAL_RETRY_REPLY)
    assert not _is_grounded_substantive_reply(snapshot, NATURAL_CLARIFICATION_REPLY)
    assert not _is_grounded_substantive_reply(
        {**snapshot, "verified_book_units": 0}, "Неподтверждённый ответ."
    )


def test_all_fallback_worst_case_must_fail_grounded_gate() -> None:
    # The earlier test expected >=8 distinct strings from 16 bookless
    # replies; that was the defect. Any all-fallback sample must fail.
    retries = [
        select_retry_reply(f"тестовое сообщение номер {i} " + "длинный хвост " * (i % 5))
        for i in range(16)
    ]
    assert len(set(retries)) == 1
    assert all(
        not _is_grounded_substantive_reply(
            {
                "answer_outcome": "retry-turn-budget",
                "planner_query_count": 12,
                "retrieval_passages": 5,
                "verified_book_units": 0,
            },
            text,
        )
        for text in retries
    )
