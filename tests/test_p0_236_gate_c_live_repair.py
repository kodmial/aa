"""Regression guard: the Gate C repair must not game reply diversity.

A formerly shipped SHA-256 pool of 10 polite non-answers met the old
8-distinct-reply test while real Telegram users received no book-grounded
help. Technical failures must now be identified as unqualified and counted
as product failures, not disguised as valid AA advice.
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
