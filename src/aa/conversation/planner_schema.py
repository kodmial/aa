"""Minimal structured planner schema for the v2 turn graph (issue #113).

The ``#112`` planner contract is intentionally tiny: the output schema is
only ``QueryPlan { queries: list[str] }``. Valid lengths are exactly 0 or
10-16 distinct queries after whitespace/case normalization. Zero is allowed
only when a natural answer can contain no substantive claim; that judgment
is model-driven inside the structured output, never a hand-written router.

Validation here is structural, never semantic: no intent labels,
categories, substantive flags, confidence scores, aspect taxonomies,
ambiguity taxonomies or hand-written keyword/slang/theme dictionaries.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

MIN_NONEMPTY_QUERIES = 10
MAX_QUERIES = 16
PLANNER_MAX_ATTEMPTS = 2


class QueryPlan(BaseModel):
    """Structured retrieval plan produced by the hidden planner node."""

    queries: list[str] = Field(default_factory=list)


class QueryPlanValidationError(ValueError):
    """Structural rejection of a planner output (cardinality/duplicates)."""


def normalize_queries(raw: list[str]) -> list[str]:
    """Collapse whitespace and drop empty/exact-duplicate entries.

    Dedup is case-insensitive on the whitespace-collapsed form; the first
    spelling is kept. No semantic judgment is applied.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw:
        collapsed = " ".join(str(item).split())
        if not collapsed:
            continue
        key = collapsed.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(collapsed)
    return cleaned


def validate_query_plan(plan: QueryPlan) -> QueryPlan:
    """Enforce the 0-or-10..16 structural contract on a parsed plan."""
    queries = normalize_queries(list(plan.queries))
    if not queries:
        return QueryPlan(queries=[])
    if not MIN_NONEMPTY_QUERIES <= len(queries) <= MAX_QUERIES:
        raise QueryPlanValidationError(
            "non-empty plans require "
            f"{MIN_NONEMPTY_QUERIES}-{MAX_QUERIES} distinct queries, "
            f"got {len(queries)}"
        )
    return QueryPlan(queries=queries)


__all__ = [
    "MAX_QUERIES",
    "MIN_NONEMPTY_QUERIES",
    "PLANNER_MAX_ATTEMPTS",
    "QueryPlan",
    "QueryPlanValidationError",
    "normalize_queries",
    "validate_query_plan",
]
