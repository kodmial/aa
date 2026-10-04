"""Runtime context budget tests (offline, no model access)."""

from __future__ import annotations

import pytest

from aa.corpus.budget import (
    BOOK_MAP_BUDGET_TOKENS,
    CHARS_PER_TOKEN,
    CONVERSATION_HISTORY_BUDGET_TOKENS,
    EFFECTIVE_CONTEXT_TOKENS_DEFAULT,
    RESPONSE_HEADROOM_TOKENS,
    RETRIEVED_PASSAGES_BUDGET_TOKENS,
    RU_CHARS_PER_TOKEN,
    SYSTEM_POLICY_BUDGET_TOKENS,
    BudgetExceededError,
    ContextBudget,
    TruncationRefusedError,
    default_budget,
    estimate_text_tokens,
    estimate_tokens,
    estimate_tokens_ru,
    fit_passages,
    resolve_effective_context,
)


def test_effective_context_floor_and_override() -> None:
    assert EFFECTIVE_CONTEXT_TOKENS_DEFAULT == 200_000
    assert resolve_effective_context(0) == 200_000
    assert resolve_effective_context(500_000) == 500_000
    with pytest.raises(ValueError, match=">= 0"):
        resolve_effective_context(-1)


def test_token_estimators_are_measured_ceilings() -> None:
    # Calibrated by docs/context-cost-measurement.md (issue #49): English
    # budgets 3 chars/token (measured floor 3.31), Russian budgets 2
    # chars/token (measured floor 2.66 on the fallback spot-check).
    assert CHARS_PER_TOKEN == 3
    assert RU_CHARS_PER_TOKEN == 2
    assert estimate_tokens(0) == 0
    assert estimate_tokens(1) == 1
    assert estimate_tokens(3) == 1
    assert estimate_tokens(4) == 2
    assert estimate_tokens_ru(1) == 1
    assert estimate_tokens_ru(2) == 1
    assert estimate_tokens_ru(3) == 2
    with pytest.raises(ValueError, match=">= 0"):
        estimate_tokens(-1)
    with pytest.raises(ValueError, match=">= 0"):
        estimate_tokens_ru(-1)


def test_mixed_text_estimator_splits_cyrillic() -> None:
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcd") == estimate_tokens(4)
    assert estimate_text_tokens("абвг") == estimate_tokens_ru(4)
    mixed = "abc абв"
    assert estimate_text_tokens(mixed) == estimate_tokens(4) + estimate_tokens_ru(3)
    # Dense English planner JSON from the measurement stays a ceiling.
    assert estimate_tokens(295) >= 89
    # Dense Russian evidence from the measurement stays a ceiling.
    assert estimate_tokens_ru(296) >= 98


def test_default_budget_reserves_and_validates() -> None:
    budget = default_budget()
    assert (
        SYSTEM_POLICY_BUDGET_TOKENS,
        BOOK_MAP_BUDGET_TOKENS,
        RETRIEVED_PASSAGES_BUDGET_TOKENS,
        CONVERSATION_HISTORY_BUDGET_TOKENS,
        RESPONSE_HEADROOM_TOKENS,
    ) == (
        budget.system_policy_tokens,
        budget.book_map_tokens,
        budget.retrieved_passages_tokens,
        budget.conversation_history_tokens,
        budget.response_headroom_tokens,
    )
    assert budget.total_reserved_tokens == 76_000
    assert budget.spare_tokens == 124_000
    budget.validate()
    with pytest.raises(BudgetExceededError):
        ContextBudget(effective_context_tokens=10_000).validate()


def test_full_book_must_not_be_always_loaded() -> None:
    # Measured canonical size (~272K chars ~= ~91K tokens at the calibrated
    # 3 chars/token) exceeds the whole retrieved-passages budget, so dynamic
    # per-need retrieval is required.
    full_book_chars = 272_432
    assert estimate_tokens(full_book_chars) == 90_811
    assert estimate_tokens(full_book_chars) > RETRIEVED_PASSAGES_BUDGET_TOKENS
    with pytest.raises(TruncationRefusedError):
        fit_passages(["x" * full_book_chars], RETRIEVED_PASSAGES_BUDGET_TOKENS)


def test_fit_passages_accepts_exact_fit() -> None:
    passages = ["a" * 4000, "b" * 4000]
    accepted = fit_passages(passages, 2668)
    assert accepted == passages
    assert accepted is not passages


def test_fit_passages_refuses_overflow_without_truncation() -> None:
    passages = ["a" * 8000, "b" * 8000, "c" * 8000]
    with pytest.raises(TruncationRefusedError):
        fit_passages(passages, 4000)
    assert passages == ["a" * 8000, "b" * 8000, "c" * 8000]


def test_fit_passages_refuses_single_oversize_passage() -> None:
    with pytest.raises(TruncationRefusedError, match="passage 0"):
        fit_passages(["a" * 8000], 1000)


def test_fit_passages_rejects_negative_budget() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        fit_passages(["a"], -1)


def test_fit_passages_russian_estimator_is_stricter() -> None:
    from aa.corpus.budget import RU_CHARS_PER_TOKEN

    passage = "я" * 300
    assert fit_passages([passage], 150) == [passage]
    with pytest.raises(TruncationRefusedError):
        fit_passages([passage], 149, chars_per_token=RU_CHARS_PER_TOKEN)
    with pytest.raises(ValueError, match="> 0"):
        fit_passages([passage], 150, chars_per_token=0)
