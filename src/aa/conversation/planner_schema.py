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
# Bounded native structured retry budget (Gate C+E live repair,
# kodmial/aa#217 recurrence 4 on exact main 7a9c907 run 37697730282:
# C:live-answer-no-generic-collapse plus E:latency-budget-exceeded
# p50 20.4s / p95 33.3s / max 34.3s with planner p50 7.0s / p95 16.4s /
# max 28.7s, retrieval p50 0.5s, repair_turns=0). Prior repairs bounded
# per-message/per-passage display tokens at the planner/answer/verifier
# boundaries and saved ~9s p50 across runs, but the persistent planner
# tail is server-side structured retries, not display tokens: each
# planner call pays up to 3 sequential model invocations inside OpenCode
# (initial + 2 validation retries) before the AA-level text fallback,
# so one weak-model invalid plan costs ~8-16s before serving anything.
# The verifier precedent (run 37538518277) already halved its worst case
# to a single server retry with unchanged strict Pydantic validation.
# This changes strategy at the responsible OpenCode-request boundary
# (retry budget, not another token-display patch): one server retry
# bounds the planner tail while grounding stays strict (only a fully
# validated 0 or 10-16 distinct-query plan is accepted; anything else
# fails closed to the bounded text fallback or raises). Turn-
# independent, never an exact-question special case.
PLANNER_MAX_ATTEMPTS = 1


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
