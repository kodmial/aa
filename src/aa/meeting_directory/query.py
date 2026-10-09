"""Structured offline directory queries for consumer #315 (no LLM)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from aa.meeting_directory.loader import (
    DirectoryUnavailableError,
    get_cached_snapshot,
)
from aa.meeting_directory.locality import resolve_locality as _resolve_impl
from aa.meeting_directory.models import (
    DirectorySearchResult,
    DirectorySnapshot,
    GroupRecord,
    LocationResolution,
    MeetingResource,
    QueryFormat,
    SourceRecord,
    UpcomingMeetingResult,
)
from aa.meeting_directory.schedule import (
    STALE_THRESHOLD_DAYS,
    InvalidCursorError,
    rank_upcoming_meetings,
)
from aa.meeting_directory.validator import is_allowed_url

logger = logging.getLogger("aa.meeting_directory")

ROOT_CATALOG_URL = "https://aarussia.ru/aagroups/"
UNCONFIRMED_NOTE = "Published schedule, unconfirmed: verify with organizers."


def _freshness_note(verified_at: datetime | None, now: datetime | None = None) -> str:
    if verified_at is None:
        return "unknown"
    if now is not None:
        try:
            if (now - verified_at).days > STALE_THRESHOLD_DAYS:
                return "stale"
        except TypeError:
            return "unknown"
    return "fresh"


def _slot_verified_at(group_id: str, snapshot: DirectorySnapshot) -> datetime | None:
    stamps = [
        slot.schedule_verified_at
        for slot in snapshot.slots
        if slot.group_id == group_id and slot.schedule_verified_at is not None
    ]
    if not stamps:
        return None
    return max(stamps)


def _resource_for_group(group: GroupRecord, snapshot: DirectorySnapshot) -> MeetingResource:
    verified_at = _slot_verified_at(group.group_id, snapshot)
    freshness = _freshness_note(verified_at)
    if verified_at is not None:
        now_utc = datetime.now(UTC)
        freshness = _freshness_note(verified_at, now_utc)
    return MeetingResource(
        record_id=group.group_id,
        group_id=group.group_id,
        slot_id=None,
        name=group.name,
        group_format=group.group_format,
        city=group.city,
        venue_or_url=group.venue_or_url,
        timezone=group.timezone,
        source_url=group.source_url,
        freshness=freshness,  # type: ignore[arg-type]
        access=group.access,
        display_note=UNCONFIRMED_NOTE,
    )


def _region_link_for(region_id: str | None, snapshot: DirectorySnapshot) -> str | None:
    if region_id is None:
        return None
    for link in snapshot.region_links:
        if link.region_id == region_id and is_allowed_url(link.url):
            return link.url
    return None


def _unavailable_result(reason: str) -> DirectorySearchResult:
    return DirectorySearchResult(
        outcome="directory_unavailable",
        items=(),
        region_link=None,
        source_urls=(),
        display_note=reason,
    )


def get_sources(snapshot: DirectorySnapshot | None = None) -> list[SourceRecord]:
    """Return the registered official sources (offline, validated snapshot)."""
    directory = snapshot if snapshot is not None else get_cached_snapshot()
    return list(directory.sources)


def search(
    country: str = "RU",
    region: str | None = None,
    city: str | None = None,
    place_id: str | None = None,
    region_id: str | None = None,
    format: QueryFormat | None = None,  # noqa: A002 - consumer contract name
    weekday: int | None = None,
    limit: int = 3,
    snapshot: DirectorySnapshot | None = None,
) -> DirectorySearchResult:
    """Structured typed-filter search over the validated snapshot.

    Only records already present in the snapshot are returned. A region
    link never satisfies a request for an exact place/time as a matched
    meeting. Never fabricates groups, times, contacts, or distances.
    """
    try:
        directory = snapshot if snapshot is not None else get_cached_snapshot()
    except DirectoryUnavailableError:
        return _unavailable_result("Meeting navigation is temporarily unavailable.")
    if country != "RU":
        return DirectorySearchResult(
            outcome="no_coverage",
            items=(),
            region_link=None,
            source_urls=(ROOT_CATALOG_URL,),
            display_note="The directory covers Russia only.",
        )
    if weekday is not None and (weekday < 0 or weekday > 6):
        raise ValueError("weekday must be in 0..6 (Monday=0)")

    resolved_place_id = place_id
    if resolved_place_id is None and city:
        resolution = _resolve_impl(directory, city, region)
        if resolution.status == "exact_unique":
            resolved_place_id = resolution.place_id
        elif resolution.status in ("ambiguous", "needs_region"):
            return DirectorySearchResult(
                outcome="unsupported_city",
                items=(),
                region_link=None,
                source_urls=(ROOT_CATALOG_URL,),
                display_note=resolution.display_note,
            )
        else:
            return DirectorySearchResult(
                outcome="no_coverage",
                items=(),
                region_link=None,
                source_urls=(ROOT_CATALOG_URL,),
                display_note=resolution.display_note,
            )

    candidates = list(directory.groups)
    if resolved_place_id is not None:
        candidates = [item for item in candidates if item.place_id == resolved_place_id]
    elif region_id is not None:
        place_ids = {place.place_id for place in directory.places if place.region_id == region_id}
        candidates = [item for item in candidates if item.place_id in place_ids]
    elif region is not None:
        wanted = region.strip().casefold()
        candidates = [item for item in candidates if (item.region or "").casefold() == wanted]
    if format == "online":
        candidates = [item for item in candidates if item.group_format in ("online", "hybrid")]
    elif format == "in_person":
        candidates = [item for item in candidates if item.group_format in ("in_person", "hybrid")]
    if weekday is not None:
        slot_groups = {slot.group_id for slot in directory.slots if weekday in slot.weekdays}
        candidates = [item for item in candidates if item.group_id in slot_groups]
    candidates = sorted(candidates, key=lambda item: item.group_id)
    bounded = candidates[: max(1, limit)]

    if bounded:
        items = tuple(_resource_for_group(group, directory) for group in bounded)
        stale = [item for item in items if item.freshness == "stale"]
        if stale and len(stale) == len(items):
            link = _region_link_for(_place_region(resolved_place_id, directory), directory)
            return DirectorySearchResult(
                outcome="stale_details",
                items=items,
                region_link=link,
                source_urls=tuple(dict.fromkeys([*(i.source_url for i in items)])),
                display_note="Details may be outdated; verify via the official directory.",
            )
        return DirectorySearchResult(
            outcome="matched_group",
            items=items,
            region_link=None,
            source_urls=tuple(dict.fromkeys([*(i.source_url for i in items)])),
            display_note=UNCONFIRMED_NOTE,
        )

    # No verified rows: fall back to official regional navigation, never a
    # fictional meeting.
    link_region = region_id or _place_region(resolved_place_id, directory)
    link = _region_link_for(link_region, directory)
    if link is not None:
        return DirectorySearchResult(
            outcome="region_directory_only",
            items=(),
            region_link=link,
            source_urls=(link,),
            display_note="No verified group rows for this place; official directory link.",
        )
    if format == "online":
        return DirectorySearchResult(
            outcome="region_directory_only",
            items=(),
            region_link="https://aarussia.ru/online-groups/",
            source_urls=("https://aarussia.ru/online-groups/",),
            display_note="Online catalog link; no city-specific rows.",
        )
    return DirectorySearchResult(
        outcome="no_coverage",
        items=(),
        region_link=None,
        source_urls=(ROOT_CATALOG_URL,),
        display_note="No directory coverage for this place; see the official catalog.",
    )


def _place_region(place_id: str | None, snapshot: DirectorySnapshot) -> str | None:
    if place_id is None:
        return None
    for place in snapshot.places:
        if place.place_id == place_id:
            return place.region_id
    return None


def get_upcoming_meetings(
    place_id: str | None = None,
    region_id: str | None = None,
    format: QueryFormat | None = None,  # noqa: A002 - consumer contract name
    now_utc: datetime | None = None,
    limit: int = 3,
    cursor: str | None = None,
    lookahead_days: int = 35,
    snapshot: DirectorySnapshot | None = None,
) -> UpcomingMeetingResult:
    """Return the chronologically nearest verified future occurrences.

    Pure calendar computation with zero LLM calls and zero network access
    on the hot path. ``now_utc`` must be an aware datetime; callers pass
    ``datetime.now(timezone.utc)`` once at query start for live queries.
    """

    try:
        directory = snapshot if snapshot is not None else get_cached_snapshot()
    except DirectoryUnavailableError:
        return UpcomingMeetingResult(
            occurrences=(),
            ongoing=(),
            fallback="directory_unavailable",
            next_cursor=None,
            directory_version="unknown",
            display_note="Meeting navigation is temporarily unavailable.",
        )
    moment: datetime
    if now_utc is None:
        moment = datetime.now(UTC)
    elif now_utc.tzinfo is None:
        raise ValueError("now_utc must be a timezone-aware datetime")
    else:
        moment = now_utc.astimezone(UTC)

    groups_by_id = {group.group_id: group for group in directory.groups}
    if place_id is not None:
        wanted_groups = {group.group_id for group in directory.groups if group.place_id == place_id}
        scope = f"place:{place_id}:{format or 'any'}"
    elif region_id is not None:
        place_ids = {place.place_id for place in directory.places if place.region_id == region_id}
        wanted_groups = {
            group.group_id for group in directory.groups if group.place_id in place_ids
        }
        scope = f"region:{region_id}:{format or 'any'}"
    elif format == "online":
        wanted_groups = {
            group.group_id for group in directory.groups if group.group_format == "online"
        }
        scope = "online:any"
    else:
        wanted_groups = set(groups_by_id)
        scope = f"all:{format or 'any'}"
    slots = [slot for slot in directory.slots if slot.group_id in wanted_groups]
    if format == "online":
        slots = [slot for slot in slots if slot.slot_format in ("online", "hybrid")]
    elif format == "in_person":
        slots = [slot for slot in slots if slot.slot_format in ("in_person", "hybrid")]

    if not slots:
        link = _region_link_for(_place_region(place_id, directory) or region_id, directory)
        if link is not None or place_id is not None or region_id is not None:
            return UpcomingMeetingResult(
                occurrences=(),
                ongoing=(),
                fallback="directory_only",
                next_cursor=None,
                directory_version=directory.version,
                display_note="Directory link only; no verified timetable here.",
            )
        return UpcomingMeetingResult(
            occurrences=(),
            ongoing=(),
            fallback="no_coverage",
            next_cursor=None,
            directory_version=directory.version,
            display_note="No directory coverage; see the official catalog.",
        )

    rankable = [
        slot
        for slot in slots
        if slot.parsing_status == "verified"
        and slot.timezone != "unknown"
        and slot.start_local is not None
        and slot.recurrence != "unknown"
    ]
    if not rankable:
        return UpcomingMeetingResult(
            occurrences=(),
            ongoing=(),
            fallback="schedule_unknown",
            next_cursor=None,
            directory_version=directory.version,
            display_note="No trustworthy timetable; official directory link instead.",
        )
    fresh = [
        slot
        for slot in rankable
        if slot.schedule_verified_at is not None
        and (moment - slot.schedule_verified_at).days <= STALE_THRESHOLD_DAYS
    ]
    if not fresh:
        return UpcomingMeetingResult(
            occurrences=(),
            ongoing=(),
            fallback="schedule_stale",
            next_cursor=None,
            directory_version=directory.version,
            display_note="Schedules are outdated; verify via the official directory.",
        )
    try:
        result = rank_upcoming_meetings(
            directory,
            fresh,
            groups_by_id,
            moment,
            limit=limit,
            lookahead_days=lookahead_days,
            cursor=cursor,
            cursor_scope=scope,
        )
    except InvalidCursorError:
        return UpcomingMeetingResult(
            occurrences=(),
            ongoing=(),
            fallback="no_upcoming_in_window",
            next_cursor=None,
            directory_version=directory.version,
            display_note="Cursor expired; start a new search.",
        )
    if not result.occurrences:
        return UpcomingMeetingResult(
            occurrences=(),
            ongoing=result.ongoing,
            fallback="no_upcoming_in_window",
            next_cursor=None,
            directory_version=directory.version,
            display_note="No verified meetings in the lookup window.",
        )
    return result


def coverage_summary(
    snapshot: DirectorySnapshot | None = None,
) -> dict[str, int]:
    """Report separate region-navigation vs verified group/slot coverage."""
    directory = snapshot if snapshot is not None else get_cached_snapshot()
    rankable = sum(
        1
        for slot in directory.slots
        if slot.parsing_status == "verified"
        and slot.timezone != "unknown"
        and slot.start_local is not None
        and slot.recurrence != "unknown"
    )
    return {
        "places": len(directory.places),
        "region_directory_links": len(directory.region_links),
        "verified_groups": len(directory.groups),
        "slots": len(directory.slots),
        "rankable_slots": rankable,
        "sources": len(directory.sources),
    }


def resolve_locality(
    query: str,
    region_hint: str | None = None,
    snapshot: DirectorySnapshot | None = None,
) -> LocationResolution:
    """Offline type-a-city resolution over the validated snapshot."""
    directory = snapshot if snapshot is not None else get_cached_snapshot()
    return _resolve_impl(directory, query, region_hint)


# Re-exported for tests and tooling.
__all__ = [
    "ROOT_CATALOG_URL",
    "coverage_summary",
    "get_sources",
    "get_upcoming_meetings",
    "resolve_locality",
    "search",
]
