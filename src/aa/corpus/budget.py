"""Runtime context budgeting for the OpenCode/Zen session.

The steady LLM context contains only bounded, explicitly budgeted parts:

1. behavior/system policy;
2. compact book map (navigation only, never evidence);
3. source-exact passages retrieved for the current need (dynamic retrieval,
   never an always-loaded full book);
4. conversation history;
5. response headroom.

Retrieved passages are packed atomically: a passage either fits entirely or
the call fails with :class:`TruncationRefusedError`. Silent truncation or
silent dropping of passages is never allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

EFFECTIVE_CONTEXT_TOKENS_DEFAULT = 200_000
"""Conservative effective context in tokens.

Zen serves models whose input tiers are explicitly split at 200K tokens
(Claude Sonnet/Opus and Gemini families price ``<= 200K`` versus ``> 200K``
separately), so 200K is the largest input budget that is safe regardless of
which Zen model backs the session. A larger deployment-specific limit can be
provided through ``OPENCODE_CONTEXT_LIMIT_TOKENS``; see
:func:`resolve_effective_context`.
"""

CHARS_PER_TOKEN = 4
"""Conservative English-prose estimator: four characters per token."""

SYSTEM_POLICY_BUDGET_TOKENS = 6_000
BOOK_MAP_BUDGET_TOKENS = 6_000
RETRIEVED_PASSAGES_BUDGET_TOKENS = 16_000
CONVERSATION_HISTORY_BUDGET_TOKENS = 32_000
RESPONSE_HEADROOM_TOKENS = 16_000


class BudgetExceededError(ValueError):
    """Raised when a context budget allocation does not fit."""


class TruncationRefusedError(BudgetExceededError):
    """Raised instead of silently truncating or dropping a passage."""


def estimate_tokens(char_count: int) -> int:
    """Estimate tokens for ``char_count`` characters (conservative ceiling)."""
    if char_count < 0:
        raise ValueError("char_count must be >= 0")
    return -(-char_count // CHARS_PER_TOKEN)


def resolve_effective_context(configured_limit_tokens: int) -> int:
    """Resolve the effective context from an ``OPENCODE_*`` limit hint.

    A positive configured limit (owned by the OpenCode runtime side) wins;
    otherwise the conservative Zen floor applies. Negative values are a
    configuration error.
    """
    if configured_limit_tokens < 0:
        raise ValueError("configured context limit must be >= 0")
    if configured_limit_tokens > 0:
        return configured_limit_tokens
    return EFFECTIVE_CONTEXT_TOKENS_DEFAULT


@dataclass(frozen=True)
class ContextBudget:
    """Explicit token budgets for every steady-context category."""

    effective_context_tokens: int
    system_policy_tokens: int = SYSTEM_POLICY_BUDGET_TOKENS
    book_map_tokens: int = BOOK_MAP_BUDGET_TOKENS
    retrieved_passages_tokens: int = RETRIEVED_PASSAGES_BUDGET_TOKENS
    conversation_history_tokens: int = CONVERSATION_HISTORY_BUDGET_TOKENS
    response_headroom_tokens: int = RESPONSE_HEADROOM_TOKENS

    @property
    def total_reserved_tokens(self) -> int:
        """Sum of all category budgets (must stay within effective)."""
        return (
            self.system_policy_tokens
            + self.book_map_tokens
            + self.retrieved_passages_tokens
            + self.conversation_history_tokens
            + self.response_headroom_tokens
        )

    @property
    def spare_tokens(self) -> int:
        """Unreserved tokens left for tool wrappers and model overhead."""
        return self.effective_context_tokens - self.total_reserved_tokens

    def validate(self) -> None:
        """Fail closed when any budget is negative or the sum overflows."""
        for name in (
            "system_policy_tokens",
            "book_map_tokens",
            "retrieved_passages_tokens",
            "conversation_history_tokens",
            "response_headroom_tokens",
        ):
            if getattr(self, name) < 0:
                raise BudgetExceededError(f"budget {name} must be >= 0")
        if self.effective_context_tokens <= 0:
            raise BudgetExceededError("effective context must be > 0")
        if self.total_reserved_tokens > self.effective_context_tokens:
            raise BudgetExceededError(
                f"budgets reserve {self.total_reserved_tokens} tokens "
                f"but effective context is {self.effective_context_tokens}"
            )


def default_budget(configured_limit_tokens: int = 0) -> ContextBudget:
    """Build the default budget for an effective context (validated)."""
    budget = ContextBudget(
        effective_context_tokens=resolve_effective_context(configured_limit_tokens)
    )
    budget.validate()
    return budget


def fit_passages(passages: list[str], budget_tokens: int) -> list[str]:
    """Accept ``passages`` only when they all fit atomically.

    Every passage is either carried in full or the whole call fails with
    :class:`TruncationRefusedError`. Callers that need fewer passages must
    explicitly re-rank and re-request a smaller set; this function never
    truncates a passage and never silently drops trailing passages.
    """
    if budget_tokens < 0:
        raise ValueError("budget_tokens must be >= 0")
    for index, passage in enumerate(passages):
        if estimate_tokens(len(passage)) > budget_tokens:
            raise TruncationRefusedError(
                f"passage {index} needs {estimate_tokens(len(passage))} tokens "
                f"but the retrieved-passages budget is {budget_tokens}"
            )
    total = sum(estimate_tokens(len(passage)) for passage in passages)
    if total > budget_tokens:
        raise TruncationRefusedError(
            f"{len(passages)} passages need {total} tokens "
            f"but the retrieved-passages budget is {budget_tokens}"
        )
    return list(passages)
