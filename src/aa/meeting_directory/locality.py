"""Deterministic offline locality normalization and resolution."""

from __future__ import annotations

import logging
import re

from aa.meeting_directory.models import (
    DirectorySnapshot,
    LocationCandidate,
    LocationResolution,
    PlaceRecord,
)

logger = logging.getLogger("aa.meeting_directory")

MAX_SUGGESTIONS = 4


def normalize_locality_text(raw: str) -> str:
    """Normalize user-supplied locality text without guessing.

    Case/yo-ye/spacing/punctuation variants and known abbreviations fold
    to one key. Uncertain fuzziness is never collapsed here; the caller
    surfaces a disambiguation choice instead.
    """
    text = raw.strip().casefold().replace("ё", "е")
    text = text.replace("-", " ").replace(".", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _place_keys(place: PlaceRecord) -> set[str]:
    keys = {normalize_locality_text(place.display_name)}
    for alias in place.aliases:
        keys.add(normalize_locality_text(alias))
    return {key for key in keys if key}


def _sorted_candidates(places: list[PlaceRecord]) -> tuple[LocationCandidate, ...]:
    ordered = sorted(places, key=lambda item: item.place_id)
    return tuple(
        LocationCandidate(
            place_id=item.place_id,
            display_name=item.display_name,
            place_type=item.place_type,
            region_hint=item.region_hint,
        )
        for item in ordered[:MAX_SUGGESTIONS]
    )


def resolve_locality(
    snapshot: DirectorySnapshot,
    query: str,
    region_hint: str | None = None,
) -> LocationResolution:
    """Resolve a city/town/region query to stable place IDs, offline.

    No external geocoder, Telegram profile or IP is consulted. Only exact
    normalized matches and known aliases resolve; typos and near-collisions
    return ``unknown`` or ``ambiguous`` rather than an arbitrary best guess.
    No user text is logged.
    """
    normalized = normalize_locality_text(query)
    if not normalized:
        return LocationResolution(
            status="unknown",
            place_id=None,
            candidates=(),
            display_note="Name a city, e.g. Moscow or Saint Petersburg.",
        )
    exact = [place for place in snapshot.places if normalized in _place_keys(place)]
    if region_hint:
        hint_key = normalize_locality_text(region_hint)
        narrowed = [
            place
            for place in exact
            if place.region_hint and hint_key == normalize_locality_text(place.region_hint)
        ]
        if len(narrowed) == 1:
            return LocationResolution(
                status="exact_unique",
                place_id=narrowed[0].place_id,
                candidates=(),
                display_note=narrowed[0].display_name,
            )
        if len(narrowed) > 1:
            return LocationResolution(
                status="ambiguous",
                place_id=None,
                candidates=_sorted_candidates(narrowed),
                display_note="Several places share this name; choose one.",
            )
        if exact:
            return LocationResolution(
                status="needs_region",
                place_id=None,
                candidates=_sorted_candidates(exact),
                display_note="Several places share this name; name the region.",
            )
    if len(exact) == 1:
        return LocationResolution(
            status="exact_unique",
            place_id=exact[0].place_id,
            candidates=(),
            display_note=exact[0].display_name,
        )
    if len(exact) > 1:
        regions = {place.region_hint for place in exact}
        if len(regions) > 1:
            return LocationResolution(
                status="needs_region",
                place_id=None,
                candidates=_sorted_candidates(exact),
                display_note="Several places share this name; name the region.",
            )
        return LocationResolution(
            status="ambiguous",
            place_id=None,
            candidates=_sorted_candidates(exact),
            display_note="Several places share this name; choose one.",
        )
    # Region-only input: match a region name to its directory pointer.
    region_matches = [
        place
        for place in snapshot.places
        if place.region_hint and normalized == normalize_locality_text(place.region_hint)
    ]
    if region_matches:
        return LocationResolution(
            status="ambiguous",
            place_id=None,
            candidates=_sorted_candidates(region_matches),
            display_note="Region matched; choose a city or use the regional directory.",
        )
    logger.info("locality resolution finished", extra={"resolved": False})
    return LocationResolution(
        status="unknown",
        place_id=None,
        candidates=(),
        display_note="Place not found in the Russia directory; try the official catalog.",
    )
