"""Deterministic context-budget contract for the canonical AA corpus.

The complete validated corpus lives in the stable OpenCode prompt/session
prefix (or equivalent cached context) and is never silently truncated.
Conversation history is compacted first; the source corpus is never compacted
away.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: Effective model context must be >= 200k tokens.
MIN_EFFECTIVE_CONTEXT_TOKENS = 200_000

#: Deterministic headroom reservations (tokens).
DEFAULT_RESERVED_SYSTEM_TOKENS = 4_000
DEFAULT_RESERVED_CONVERSATION_TOKENS = 16_000
DEFAULT_RESERVED_OUTPUT_TOKENS = 8_000


class ContextBudgetError(ValueError):
    """Raised when the corpus cannot fit without truncation (fail-closed)."""


@dataclass(frozen=True)
class ContextBudget:
    """Resolved token budget for one startup validation."""

    effective_context_tokens: int
    corpus_tokens: int
    reserved_system_tokens: int = DEFAULT_RESERVED_SYSTEM_TOKENS
    reserved_conversation_tokens: int = DEFAULT_RESERVED_CONVERSATION_TOKENS
    reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS

    @property
    def reserved_total(self) -> int:
        """Total reserved headroom outside the corpus."""
        return (
            self.reserved_system_tokens
            + self.reserved_conversation_tokens
            + self.reserved_output_tokens
        )

    @property
    def required_total(self) -> int:
        """Corpus plus all reserved headroom."""
        return self.corpus_tokens + self.reserved_total

    @property
    def remaining_tokens(self) -> int:
        """Free tokens after corpus plus headroom (may be negative)."""
        return self.effective_context_tokens - self.required_total

    def fits(self) -> bool:
        """Whether the full corpus fits without truncation."""
        return (
            self.effective_context_tokens >= MIN_EFFECTIVE_CONTEXT_TOKENS
            and self.remaining_tokens >= 0
        )

    def validate(self) -> None:
        """Fail closed when the corpus would need truncation."""
        if self.effective_context_tokens < MIN_EFFECTIVE_CONTEXT_TOKENS:
            raise ContextBudgetError(
                "effective context "
                f"{self.effective_context_tokens} is below the "
                f"minimum {MIN_EFFECTIVE_CONTEXT_TOKENS}"
            )
        if self.remaining_tokens < 0:
            raise ContextBudgetError(
                f"corpus ({self.corpus_tokens}) + headroom ({self.reserved_total}) "
                f"= {self.required_total} exceeds effective context "
                f"{self.effective_context_tokens}; refusing to truncate"
            )

    def to_safe_dict(self) -> dict[str, object]:
        """Return budget metadata (counts only, no corpus contents)."""
        return {
            "effective_context_tokens": self.effective_context_tokens,
            "corpus_tokens": self.corpus_tokens,
            "reserved_system_tokens": self.reserved_system_tokens,
            "reserved_conversation_tokens": self.reserved_conversation_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "required_total": self.required_total,
            "remaining_tokens": self.remaining_tokens,
            "fits": self.fits(),
            "min_effective_context_tokens": MIN_EFFECTIVE_CONTEXT_TOKENS,
        }


def validate_context_budget(
    *,
    effective_context_tokens: int,
    corpus_tokens: int,
    reserved_system_tokens: int = DEFAULT_RESERVED_SYSTEM_TOKENS,
    reserved_conversation_tokens: int = DEFAULT_RESERVED_CONVERSATION_TOKENS,
    reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
) -> ContextBudget:
    """Build and validate a :class:`ContextBudget` (raises on overflow)."""
    budget = ContextBudget(
        effective_context_tokens=effective_context_tokens,
        corpus_tokens=corpus_tokens,
        reserved_system_tokens=reserved_system_tokens,
        reserved_conversation_tokens=reserved_conversation_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    budget.validate()
    return budget


def compact_conversation_tail(
    messages: Sequence[str],
    *,
    max_conversation_tokens: int,
    estimate: Callable[[str], int] | None = None,
) -> list[str]:
    """Keep the newest messages fitting ``max_conversation_tokens``.

    The source corpus is never passed here and therefore can never be
    compacted away; only old conversation turns are dropped, newest-first
    retention. Never raises for empty input; returns ``[]`` when nothing fits.
    """
    from aa.corpus.canonical import estimate_tokens as default_estimate

    measure = estimate or default_estimate
    kept: list[str] = []
    used = 0
    for message in reversed(list(messages)):
        cost = measure(message)
        if kept and used + cost > max_conversation_tokens:
            break
        if not kept and cost > max_conversation_tokens:
            # Even the newest single message overflows: drop everything
            # rather than partially truncating a message.
            return []
        used += cost
        kept.append(message)
    kept.reverse()
    return kept
