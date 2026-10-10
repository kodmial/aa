"""Model-driven turn planner schema for the v2 turn graph (issues #112, #268, #295).

The planner receives the current user turn plus managed conversation
state and returns a typed structured result:

- ``mode``: ``conversational`` | ``retrieval``
- ``resolved_intent``: standalone semantic formulation of what the user
  means now (empty only for purely conversational turns)
- ``queries``: [] for conversational turns, otherwise 1..16 distinct
  useful Russian queries (issue #295: meaning-driven count, never padded
  to 10 or truncated at 16 by fiat)

Application code validates schema/cardinality only. It never infers
meaning from words, punctuation, step numbers, greeting lists, recovery
stems, first-person markers or manually curated intents.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PlannerMode = Literal["conversational", "retrieval"]

MIN_NONEMPTY_QUERIES = 1
MAX_QUERIES = 16
MAX_RESOLVED_INTENT_CHARS = 2000
# Bounded native structured retry budget: OpenCode owns the retry, AA code
# performs exactly one Pydantic validation afterwards.
PLANNER_MAX_ATTEMPTS = 1


class QueryPlan(BaseModel):
    """Structured turn-understanding plan produced by the hidden planner."""

    mode: PlannerMode = Field(default="retrieval")
    resolved_intent: str = Field(default="")
    queries: list[str] = Field(default_factory=list)
    # Optional trusted planner-side query->need association (#311): parallel
    # to ``queries``; each inner list holds InformationNeed ids served by
    # that query (empty = unknown/unmapped). ``None`` means the planner did
    # not emit a mapping and the structural plan-order derivation applies.
    query_need_ids: list[list[str]] | None = Field(default=None)


class QueryPlanValidationError(ValueError):
    """Structural rejection of a planner output (mode/cardinality/duplicates)."""


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
    """Enforce the structural contract on a parsed plan.

    - ``mode`` must be ``conversational`` or ``retrieval``;
    - ``conversational`` requires zero queries;
    - ``retrieval`` requires 1..16 distinct useful non-empty queries and a
      non-empty ``resolved_intent`` (issue #295: flexible meaning-driven
      count; padding to a fixed minimum is rejected upstream by prompt
      contract, not here);
    - no lexical, semantic or domain judgment is applied here.

    There is deliberately no semantic migration leniency here.
    An explicit retrieval plan with empty queries or empty resolved_intent
    is invalid and must fail closed into the generic retrieval fallback;
    it must never be reinterpreted as conversational glue.
    """
    mode = plan.mode
    if mode not in ("conversational", "retrieval"):
        raise QueryPlanValidationError(
            f"planner mode must be conversational|retrieval, got {mode!r}"
        )
    intent = " ".join(str(plan.resolved_intent or "").split())
    if len(intent) > MAX_RESOLVED_INTENT_CHARS:
        raise QueryPlanValidationError("resolved_intent exceeds the length budget")
    queries = normalize_queries(list(plan.queries))
    links = plan.query_need_ids
    if mode == "conversational":
        if queries:
            raise QueryPlanValidationError("conversational plans must carry zero queries")
        if links is not None and links not in (None, []):
            raise QueryPlanValidationError("conversational plans must carry no query mapping")
        return QueryPlan(mode="conversational", resolved_intent="", queries=[])
    if not intent:
        raise QueryPlanValidationError("retrieval plans require a non-empty resolved_intent")
    if not MIN_NONEMPTY_QUERIES <= len(queries) <= MAX_QUERIES:
        raise QueryPlanValidationError(
            "non-empty plans require "
            f"{MIN_NONEMPTY_QUERIES}-{MAX_QUERIES} distinct queries, "
            f"got {len(queries)}"
        )
    if links is None:
        return QueryPlan(mode="retrieval", resolved_intent=intent, queries=queries)
    if not isinstance(links, list) or len(links) != len(queries):
        raise QueryPlanValidationError("query_need_ids must parallel queries when present")
    cleaned_links: list[list[str]] = []
    for inner in links:
        if not isinstance(inner, list):
            raise QueryPlanValidationError("query_need_ids entries must be lists")
        kept: list[str] = []
        seen: set[str] = set()
        for raw_need in inner:
            if not isinstance(raw_need, str) or not raw_need.strip():
                raise QueryPlanValidationError("query_need_ids entries must be strings")
            nid = raw_need.strip()
            if nid in seen:
                continue
            seen.add(nid)
            kept.append(nid)
        cleaned_links.append(kept)
    return QueryPlan(
        mode="retrieval",
        resolved_intent=intent,
        queries=queries,
        query_need_ids=cleaned_links,
    )


__all__ = [
    "MAX_QUERIES",
    "MAX_RESOLVED_INTENT_CHARS",
    "MIN_NONEMPTY_QUERIES",
    "PLANNER_MAX_ATTEMPTS",
    "PlannerMode",
    "QueryPlan",
    "QueryPlanValidationError",
    "normalize_queries",
    "validate_query_plan",
]
