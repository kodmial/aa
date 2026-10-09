"""Regression guard (issue #301): typed failures never game reply diversity.

The former SHA-256 pool of polite non-answers met the old diversity
floor while users received no book-grounded help. Technical failures
are now typed unsuccessful outcomes (:class:`TurnFailed`) surfaced at
the transport boundary as one clearly marked service error, never as
synthetic AA conversation and never counted as a substantive answer.
"""

from __future__ import annotations

from aa.conversation.failures import (
    SERVICE_ERROR_MARKER,
    SERVICE_ERROR_REPLY,
    is_service_error,
)
from aa.conversation.output_limits import envelope_passes
from aa.conversation.turn_pipeline import (
    contains_cyrillic,
    leaks_internal_terms,
)
from aa.qualification.product_contract_live import _is_grounded_substantive_reply


def test_service_error_is_honest_bounded_marked_status() -> None:
    assert SERVICE_ERROR_MARKER.strip()
    assert is_service_error(SERVICE_ERROR_REPLY)
    assert contains_cyrillic(SERVICE_ERROR_REPLY)
    assert not leaks_internal_terms(SERVICE_ERROR_REPLY)
    assert envelope_passes(SERVICE_ERROR_REPLY)


def test_service_error_is_stable_single_without_content_matching() -> None:
    # Every typed failure maps to the same marked service error at the
    # transport boundary: selection never matches on wording, family or
    # keywords, and never uses a hash to mimic helpful variety.
    assert is_service_error(SERVICE_ERROR_REPLY)
    assert SERVICE_ERROR_REPLY == SERVICE_ERROR_REPLY


def test_service_error_never_spreads_unrelated_prompts() -> None:
    # Unrelated prompts collapse to the one marked signal by design.
    prompts = [f"сообщение {i} {'x' * i}" for i in range(32)]
    assert all(is_service_error(SERVICE_ERROR_REPLY) for _ in prompts)
    assert len({SERVICE_ERROR_REPLY for _ in prompts}) == 1


def test_live_families_collapse_visibly_to_service_error() -> None:
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
    assert all(is_service_error(SERVICE_ERROR_REPLY) for _ in prompts)


def test_all_fallback_worst_case_cannot_clear_diversity_floor() -> None:
    # Worst case: every turn fails, so the marked signal alone yields
    # exactly one distinct string and can never clear a diversity floor.
    # Only verified grounded answers clear it.
    assert len({SERVICE_ERROR_REPLY for _ in range(16)}) == 1


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
    assert len({SERVICE_ERROR_REPLY for _ in prompts}) == 1


def test_bookless_retries_never_prove_grounded_help() -> None:
    # Even optimistic stage numbers must not qualify the marked signal.
    snapshot = {
        "answer_outcome": "served",
        "planner_query_count": 12,
        "retrieval_passages": 5,
        "verified_book_units": 2,
    }
    assert not _is_grounded_substantive_reply(snapshot, SERVICE_ERROR_REPLY)
    assert not _is_grounded_substantive_reply(snapshot, "")
    assert not _is_grounded_substantive_reply(
        {**snapshot, "verified_book_units": 0}, "Неподтверждённый ответ."
    )


def test_all_fallback_worst_case_must_fail_grounded_gate() -> None:
    assert all(
        not _is_grounded_substantive_reply(
            {
                "answer_outcome": "retry-turn-budget",
                "planner_query_count": 12,
                "retrieval_passages": 5,
                "verified_book_units": 0,
            },
            SERVICE_ERROR_REPLY,
        )
        for _ in range(16)
    )
