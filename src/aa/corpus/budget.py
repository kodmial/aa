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

Token estimates are calibrated by the pinned-runtime measurement in
``docs/context-cost-measurement.md`` (issue #49): English text budgets
three characters per token, Cyrillic (Russian) text budgets two. Both
ceilings cover every measured payload on the pinned primary model, and the
Russian ceiling additionally covers the fallback model, where dense
Russian evidence measured 2.66 chars/token.
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

CHARS_PER_TOKEN = 3
"""Conservative estimator: three characters per token.

Measured floor on the pinned runtime is 3.31 chars/token for English
(structured planner JSON) and 3.02 for Russian, so ``ceil(chars / 3)``
is a true ceiling for every primary-model payload in
``docs/context-cost-measurement.md``. Cyrillic text must use
:const:`RU_CHARS_PER_TOKEN` instead.
"""

RU_CHARS_PER_TOKEN = 2
"""Conservative estimator for Cyrillic (Russian) text: two chars per token.

Russian measured 3.02-4.62 chars/token on the pinned primary model but
2.66 on the fallback spot-check (dense translated evidence), so Russian
budgets use ``ceil(chars / 2)`` to stay a ceiling on both models.
"""

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


def estimate_tokens_ru(char_count: int) -> int:
    """Estimate tokens for Cyrillic (Russian) ``char_count`` (ceiling)."""
    if char_count < 0:
        raise ValueError("char_count must be >= 0")
    return -(-char_count // RU_CHARS_PER_TOKEN)


def _is_cyrillic(char: str) -> bool:
    """Whether ``char`` falls in a Cyrillic Unicode block."""
    code = ord(char)
    return (
        0x0400 <= code <= 0x04FF
        or 0x0500 <= code <= 0x052F
        or 0x2DE0 <= code <= 0x2DFF
        or 0xA640 <= code <= 0xA69F
    )


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens for mixed-language ``text`` (conservative ceiling).

    Cyrillic characters budget :const:`RU_CHARS_PER_TOKEN`; every other
    character budgets :const:`CHARS_PER_TOKEN`. Pure-English input matches
    :func:`estimate_tokens` exactly, pure-Cyrillic matches
    :func:`estimate_tokens_ru`, and mixed turns (Russian history plus
    English evidence) budget each span at its measured rate. Conversation
    history, which arrives in the user's language, must be budgeted with
    this function rather than :func:`estimate_tokens`.
    """
    cyrillic = sum(1 for char in text if _is_cyrillic(char))
    return estimate_tokens(len(text) - cyrillic) + estimate_tokens_ru(cyrillic)


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


def fit_passages(
    passages: list[str], budget_tokens: int, *, chars_per_token: int = CHARS_PER_TOKEN
) -> list[str]:
    """Accept ``passages`` only when they all fit atomically.

    Every passage is either carried in full or the whole call fails with
    :class:`TruncationRefusedError`. Callers that need fewer passages must
    explicitly re-rank and re-request a smaller set; this function never
    truncates a passage and never silently drops trailing passages.

    ``chars_per_token`` is retained for backward compatibility and only
    tightens the budget: the effective estimate for each passage is
    ``max(ceil(len / chars_per_token), estimate_text_tokens(passage))``,
    so the default (English rate) can no longer undercount Russian or
    mixed-language text. New callers should omit it and rely on the
    language-aware :func:`estimate_text_tokens` ceiling.
    """
    if budget_tokens < 0:
        raise ValueError("budget_tokens must be >= 0")
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be > 0")

    def estimate(passage: str) -> int:
        legacy = -(-len(passage) // chars_per_token)
        return max(legacy, estimate_text_tokens(passage))

    for index, passage in enumerate(passages):
        if estimate(passage) > budget_tokens:
            raise TruncationRefusedError(
                f"passage {index} needs {estimate(passage)} tokens "
                f"but the retrieved-passages budget is {budget_tokens}"
            )
    total = sum(estimate(passage) for passage in passages)
    if total > budget_tokens:
        raise TruncationRefusedError(
            f"{len(passages)} passages need {total} tokens "
            f"but the retrieved-passages budget is {budget_tokens}"
        )
    return list(passages)
