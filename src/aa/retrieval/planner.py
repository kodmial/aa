"""Validated slang-aware Russian query planner contract (issues #17, #46).

Authoritative schema ``ru-query-plan-v1`` (fixed 2026-10-04):

- the original user text is always preserved and always participates in
  retrieval (at least the dense branch);
- the planner only adds same-language Russian rewrites and decomposed
  aspect queries; it never replaces the original query;
- ``lexical_query_en`` stays null unless the selected architecture has an
  English lexical branch (it does not: RU-first hybrid is fixed);
- planner output is navigation metadata only and can never enter the
  evidence pack as source authority.

This module validates planner JSON and exposes the per-aspect fused
query lists consumed by the hybrid index. Meaning preservation
(slang/diminutives/morphology/typos without strengthening uncertain
claims) is enforced structurally: aspects carry explicit
``forbidden_inferences`` and material ambiguity must be marked.
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "ru-query-plan-v1"
PLANNER_LANGUAGE = "ru"

MAX_ASPECTS = 8
MAX_QUERIES_PER_FIELD = 8
MAX_QUERY_CHARS = 500
MAX_MEANING_CHARS = 500

AMBIGUITY_LEVELS = ("none", "low", "material")


class PlannerError(ValueError):
    """Raised when planner output fails validation (fails closed)."""


@dataclass(frozen=True)
class PlannedAspect:
    """One validated retrieval aspect."""

    aspect_id: str
    meaning: str
    semantic_queries_ru: tuple[str, ...]
    lexical_queries_ru: tuple[str, ...]
    lexical_query_en: str | None
    ambiguity: str
    forbidden_inferences: tuple[str, ...]


@dataclass(frozen=True)
class QueryPlan:
    """Validated planner output plus the preserved original query."""

    utterance_id: str
    language: str
    aspects: tuple[PlannedAspect, ...]
    original_query: str


def _require_str(value: object, *, owner: str, field: str, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlannerError(f"{owner}: {field} must be a non-empty string")
    text = value.strip()
    if len(text) > max_chars:
        raise PlannerError(f"{owner}: {field} exceeds {max_chars} chars")
    return text


def _require_str_list(
    value: object, *, owner: str, field: str, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise PlannerError(f"{owner}: {field} must be a list of strings")
    if not value and not allow_empty:
        raise PlannerError(f"{owner}: {field} must not be empty")
    if len(value) > MAX_QUERIES_PER_FIELD:
        raise PlannerError(f"{owner}: {field} exceeds {MAX_QUERIES_PER_FIELD} queries")
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise PlannerError(f"{owner}: {field} holds an empty query")
        query = item.strip()
        if len(query) > MAX_QUERY_CHARS:
            raise PlannerError(f"{owner}: {field} query exceeds {MAX_QUERY_CHARS} chars")
        cleaned.append(query)
    return tuple(cleaned)


def validate_plan(data: object, *, original_query: str) -> QueryPlan:
    """Validate planner JSON against ``ru-query-plan-v1`` (fails closed)."""
    if not isinstance(original_query, str) or not original_query.strip():
        raise PlannerError("original_query must be a non-empty string")
    if not isinstance(data, dict):
        raise PlannerError("planner output must be a JSON object")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise PlannerError(f"planner schema_version must be {SCHEMA_VERSION!r}")
    utterance_id = _require_str(
        data.get("utterance_id"), owner="plan", field="utterance_id", max_chars=128
    )
    if data.get("language") != PLANNER_LANGUAGE:
        raise PlannerError("planner language must be 'ru'")
    raw_aspects = data.get("aspects")
    if not isinstance(raw_aspects, list) or not raw_aspects:
        raise PlannerError("planner aspects must be a non-empty list")
    if len(raw_aspects) > MAX_ASPECTS:
        raise PlannerError(f"planner aspects exceed {MAX_ASPECTS}")
    aspects: list[PlannedAspect] = []
    seen_ids: set[str] = set()
    for position, raw in enumerate(raw_aspects):
        owner = f"aspect[{position}]"
        if not isinstance(raw, dict):
            raise PlannerError(f"{owner} must be an object")
        aspect_id = _require_str(raw.get("aspect_id"), owner=owner, field="aspect_id", max_chars=64)
        if aspect_id in seen_ids:
            raise PlannerError(f"{owner}: duplicate aspect_id {aspect_id!r}")
        seen_ids.add(aspect_id)
        meaning = _require_str(
            raw.get("meaning"), owner=owner, field="meaning", max_chars=MAX_MEANING_CHARS
        )
        semantic = _require_str_list(
            raw.get("semantic_queries_ru"), owner=owner, field="semantic_queries_ru"
        )
        lexical = _require_str_list(
            raw.get("lexical_queries_ru"), owner=owner, field="lexical_queries_ru"
        )
        lexical_en = raw.get("lexical_query_en")
        if lexical_en is not None:
            raise PlannerError(f"{owner}: lexical_query_en must stay null in the RU-first baseline")
        ambiguity = raw.get("ambiguity")
        if ambiguity not in AMBIGUITY_LEVELS:
            raise PlannerError(f"{owner}: ambiguity must be one of {AMBIGUITY_LEVELS}")
        forbidden_raw = raw.get("forbidden_inferences")
        if not isinstance(forbidden_raw, list):
            raise PlannerError(f"{owner}: forbidden_inferences must be a list")
        forbidden: list[str] = []
        for item in forbidden_raw:
            if not isinstance(item, str) or not item.strip():
                raise PlannerError(f"{owner}: forbidden_inferences holds an empty entry")
            forbidden.append(item.strip())
        aspects.append(
            PlannedAspect(
                aspect_id=aspect_id,
                meaning=meaning,
                semantic_queries_ru=semantic,
                lexical_queries_ru=lexical,
                lexical_query_en=None,
                ambiguity=str(ambiguity),
                forbidden_inferences=tuple(forbidden),
            )
        )
    return QueryPlan(
        utterance_id=utterance_id,
        language=PLANNER_LANGUAGE,
        aspects=tuple(aspects),
        original_query=original_query.strip(),
    )


def aspect_search_queries(aspect: PlannedAspect, *, original_query: str) -> list[str]:
    """Return fused queries for one aspect: original first, then rewrites.

    The original user wording is always retained in position 0; planner
    rewrites are additive and deduplicated case-insensitively.
    """
    if not original_query.strip():
        raise PlannerError("original_query must be a non-empty string")
    fused: list[str] = [original_query.strip()]
    seen = {original_query.strip().casefold()}
    for query in (*aspect.lexical_queries_ru, *aspect.semantic_queries_ru):
        key = query.casefold()
        if key not in seen:
            seen.add(key)
            fused.append(query)
    return fused
