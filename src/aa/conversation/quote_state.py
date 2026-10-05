"""Minimal deterministic recent quote-range state for anti-corpus-export.

This control metadata prevents contiguous multi-turn extraction (a user
asking to "continue" or for the "next part" after a prior quote cannot
serially page through adjacent canonical source ranges). It carries only
provenance and offsets, never corpus text, and it is not conversational
knowledge.
"""

from __future__ import annotations

from typing import Any

MAX_STORED_RANGES = 8
ADJACENCY_GAP_CHARS = 400


def _as_int(value: object, default: int = 0) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def ranges_from_pack(pack_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract minimal quote-range metadata from an Evidence Pack."""
    ranges: list[dict[str, Any]] = []
    for item in pack_dicts:
        passage_id = item.get("passage_id")
        source_id = item.get("source_id", item.get("source", ""))
        section_id = item.get("section_id", item.get("section", ""))
        if not isinstance(passage_id, str) or not passage_id:
            continue
        if not isinstance(source_id, str) or not source_id:
            continue
        if not isinstance(section_id, str) or not section_id:
            continue
        ranges.append(
            {
                "passage_id": passage_id,
                "source_id": source_id,
                "section_id": section_id,
                "char_start": _as_int(item.get("char_start", 0)),
                "char_end": _as_int(item.get("char_end", 0)),
            }
        )
    return ranges


def merge_recent_ranges(
    previous: list[dict[str, Any]], new_ranges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge new ranges into bounded recent history (dedup by passage)."""
    merged: list[dict[str, Any]] = [dict(item) for item in previous if isinstance(item, dict)]
    seen = {str(item.get("passage_id", "")) for item in merged}
    for item in new_ranges:
        key = str(item.get("passage_id", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(dict(item))
    return merged[-MAX_STORED_RANGES:]


def is_adjacent_to_recent(candidate: dict[str, Any], recent: list[dict[str, Any]]) -> bool:
    """Whether ``candidate`` is contiguous with any recent quoted range."""
    source = str(candidate.get("source_id", ""))
    section = str(candidate.get("section_id", ""))
    start = _as_int(candidate.get("char_start", 0))
    end = _as_int(candidate.get("char_end", 0))
    if not source or not section or end <= start:
        return False
    for item in recent:
        if not isinstance(item, dict):
            continue
        if str(item.get("source_id", "")) != source:
            continue
        if str(item.get("section_id", "")) != section:
            continue
        prev_start = _as_int(item.get("char_start", 0))
        prev_end = _as_int(item.get("char_end", 0))
        if prev_end <= prev_start:
            continue
        # Overlap or near-contiguity in either direction pages the source.
        if start <= prev_end + ADJACENCY_GAP_CHARS and end >= prev_start - ADJACENCY_GAP_CHARS:
            return True
    return False


def pack_pages_recent(pack_dicts: list[dict[str, Any]], recent: list[dict[str, Any]]) -> bool:
    """Whether the new pack would continue a recent quoted range."""
    if not recent:
        return False
    for item in pack_dicts:
        if not isinstance(item, dict):
            continue
        candidate = {
            "source_id": str(item.get("source_id", item.get("source", ""))),
            "section_id": str(item.get("section_id", item.get("section", ""))),
            "char_start": _as_int(item.get("char_start", 0)),
            "char_end": _as_int(item.get("char_end", 0)),
        }
        if is_adjacent_to_recent(candidate, recent):
            return True
    return False


__all__ = [
    "ADJACENCY_GAP_CHARS",
    "MAX_STORED_RANGES",
    "is_adjacent_to_recent",
    "merge_recent_ranges",
    "pack_pages_recent",
    "ranges_from_pack",
]
