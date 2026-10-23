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

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

PlannerMode = Literal["conversational", "retrieval"]

MIN_NONEMPTY_QUERIES = 1
MAX_QUERIES = 16
MAX_RESOLVED_INTENT_CHARS = 2000
# Issue #295: the planner emits the typed set of independently answerable
# semantic information needs IN THE SAME planning result, separate from
# ``queries`` (which stay search variants for recall). The emission is
# bounded, never padded; AA code only validates structure/cardinality.
MAX_INFORMATION_NEEDS = 8
MAX_NEED_TEXT_CHARS = 1000
MAX_NEED_PURPOSE_CHARS = 300
# Bounded native structured retry budget: OpenCode owns the retry, AA code
# performs exactly one Pydantic validation afterwards.
PLANNER_MAX_ATTEMPTS = 1


class PlanInformationNeed(BaseModel):
    """One planner-emitted semantic information need (issue #295).

    ``text`` is a standalone semantic formulation of one genuinely
    independently answerable part of the user's request, conditions and
    qualifiers preserved. ``need_id`` is a stable identifier for the
    current turn; when the planner omits it, AA assigns a deterministic
    fresh ``need-N`` id during normalization. ``purpose`` is optional.
    Carrying an id never claims the need is fulfilled.
    """

    need_id: str = Field(default="")
    text: str = Field(default="")
    purpose: str = Field(default="")


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
    # Issue #295: the planner's own typed information-need emission,
    # validated in the same planning result. Empty list means the planner
    # did not (or could not) emit needs and the conservative compatibility
    # path applies; it never means pad needs from separate search
    # formulations again.
    information_needs: list[PlanInformationNeed] = Field(default_factory=list)


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


def normalize_planner_needs(
    raw: list[PlanInformationNeed] | list[dict[str, Any]],
) -> list[PlanInformationNeed]:
    """Structurally normalize one planner need emission (issue #295).

    Whitespace collapses; empty texts drop; an id/text repeated later in
    the emission collapses to its first occurrence (same need). A missing
    or unknown-format id gets a deterministic fresh ``need-N`` id (first
    unused integer), so emitted need ids stay stable for the turn. No
    semantic judgment is applied here.
    """
    used: set[str] = set()
    seen_texts: set[str] = set()
    auto_index = 1
    out: list[PlanInformationNeed] = []
    for item in list(raw or []):
        if isinstance(item, dict):
            try:
                entry = PlanInformationNeed.model_validate(item)
            except ValidationError:
                continue
        elif isinstance(item, PlanInformationNeed):
            entry = item
        else:
            continue
        text = " ".join(str(entry.text or "").split()).strip()
        if not text:
            continue
        text_key = text.casefold()
        if text_key in seen_texts:
            continue
        nid = " ".join(str(entry.need_id or "").split()).strip()
        if nid and nid in used:
            continue
        if not nid:
            while f"need-{auto_index}" in used:
                auto_index += 1
            nid = f"need-{auto_index}"
        used.add(nid)
        seen_texts.add(text_key)
        purpose = " ".join(str(entry.purpose or "").split()).strip()
        out.append(PlanInformationNeed(need_id=nid, text=text, purpose=purpose))
    return out


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
    intent = " ".join(str(plan.resolved_intent or "").split()).strip()
    if len(intent) > MAX_RESOLVED_INTENT_CHARS:
        raise QueryPlanValidationError("resolved_intent exceeds the length budget")
    needs = normalize_planner_needs(list(plan.information_needs))
    if len(needs) > MAX_INFORMATION_NEEDS:
        raise QueryPlanValidationError(
            f"planner emitted more than {MAX_INFORMATION_NEEDS} typed information needs"
        )
    for need in needs:
        if len(need.text) > MAX_NEED_TEXT_CHARS:
            raise QueryPlanValidationError("information-need text exceeds the length budget")
        if len(need.purpose) > MAX_NEED_PURPOSE_CHARS:
            raise QueryPlanValidationError("information-need purpose exceeds the length budget")
    queries = normalize_queries(list(plan.queries))
    links = plan.query_need_ids
    if mode == "conversational":
        if queries:
            raise QueryPlanValidationError("conversational plans must carry zero queries")
        if links is not None and links not in (None, []):
            raise QueryPlanValidationError("conversational plans must carry no query mapping")
        if needs:
            raise QueryPlanValidationError("conversational plans must carry no information needs")
        return QueryPlan(
            mode="conversational", resolved_intent="", queries=[], information_needs=[]
        )
    if not intent:
        raise QueryPlanValidationError("retrieval plans require a non-empty resolved_intent")
    if not MIN_NONEMPTY_QUERIES <= len(queries) <= MAX_QUERIES:
        raise QueryPlanValidationError(
            "non-empty plans require "
            f"{MIN_NONEMPTY_QUERIES}-{MAX_QUERIES} distinct queries, "
            f"got {len(queries)}"
        )
    if links is None:
        return QueryPlan(
            mode="retrieval",
            resolved_intent=intent,
            queries=queries,
            information_needs=needs,
        )
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
        information_needs=needs,
    )


__all__ = [
    "MAX_INFORMATION_NEEDS",
    "MAX_NEED_PURPOSE_CHARS",
    "MAX_NEED_TEXT_CHARS",
    "MAX_QUERIES",
    "MAX_RESOLVED_INTENT_CHARS",
    "MIN_NONEMPTY_QUERIES",
    "PLANNER_MAX_ATTEMPTS",
    "PlanInformationNeed",
    "PlannerMode",
    "QueryPlan",
    "QueryPlanValidationError",
    "normalize_planner_needs",
    "normalize_queries",
    "validate_query_plan",
]
