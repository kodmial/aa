"""Typed entities for the Russia-only AA meeting directory (issue #316).

Offline navigation/reference data plane, separate from the canonical book
corpus. Meeting times/addresses/URLs are not authority for recovery advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

GroupFormat = Literal["in_person", "online", "hybrid"]
QueryFormat = Literal["in_person", "online"]
Recurrence = Literal["daily", "weekly", "monthly_nth", "dated", "unknown"]
ParsingStatus = Literal["verified", "unparsed", "stale"]
CoverageKind = Literal["verified_groups", "directory_only"]
PlaceType = Literal["city", "town", "settlement", "region"]
AccessKind = Literal["open", "closed", "unknown"]

SearchOutcome = Literal[
    "matched_group",
    "region_directory_only",
    "stale_details",
    "unsupported_city",
    "no_coverage",
    "source_unavailable",
    "directory_unavailable",
]

UpcomingFallback = Literal[
    "directory_only",
    "schedule_unknown",
    "schedule_stale",
    "no_coverage",
    "no_upcoming_in_window",
    "source_unavailable",
    "directory_unavailable",
]

ResolutionStatus = Literal["exact_unique", "ambiguous", "unknown", "needs_region"]


@dataclass(frozen=True)
class SourceRecord:
    """One registered official source."""

    source_id: str
    title: str
    url: str
    kind: str
    robots: str
    terms: str
    permission_status: str
    checked_at: str


@dataclass(frozen=True)
class PlaceRecord:
    """A sourced locality usable for type-a-city navigation."""

    place_id: str
    display_name: str
    place_type: PlaceType
    region_id: str | None
    region_hint: str | None
    aliases: tuple[str, ...]
    coverage_kind: CoverageKind
    source_id: str


@dataclass(frozen=True)
class RegionLink:
    """A verified official regional directory pointer (navigation only)."""

    link_id: str
    region_id: str
    region_name: str
    display_name: str
    url: str
    source_id: str
    support: str


@dataclass(frozen=True)
class GroupRecord:
    """A verified public group entity (identity stable across schedule edits)."""

    group_id: str
    name: str
    country: str
    region: str | None
    city: str | None
    place_id: str | None
    group_format: GroupFormat
    venue_or_url: str
    timezone: str
    access: AccessKind
    source_id: str
    source_url: str


@dataclass(frozen=True)
class SlotRecord:
    """One independently sourced recurring meeting slot of a group."""

    slot_id: str
    group_id: str
    slot_format: GroupFormat
    recurrence: Recurrence
    weekdays: tuple[int, ...]
    start_local: str | None
    end_local: str | None
    timezone: str
    date: str | None
    month_week: int | None
    month_weekday: int | None
    valid_from: str | None
    valid_until: str | None
    cancellations: tuple[str, ...]
    access: AccessKind
    source_url: str
    schedule_verified_at: datetime | None
    parsing_status: ParsingStatus


@dataclass(frozen=True)
class MeetingResource:
    """Consumer-facing verified directory entry for #315."""

    record_id: str
    group_id: str | None
    slot_id: str | None
    name: str
    group_format: GroupFormat
    city: str | None
    venue_or_url: str
    timezone: str
    source_url: str
    freshness: Literal["fresh", "stale", "unknown"]
    access: AccessKind
    display_note: str


@dataclass(frozen=True)
class DirectorySearchResult:
    """Typed result of a structured directory search."""

    outcome: SearchOutcome
    items: tuple[MeetingResource, ...]
    region_link: str | None
    source_urls: tuple[str, ...]
    display_note: str


@dataclass(frozen=True)
class LocationCandidate:
    """One stable disambiguation candidate for a locality query."""

    place_id: str
    display_name: str
    place_type: PlaceType
    region_hint: str | None


@dataclass(frozen=True)
class LocationResolution:
    """Typed result of offline locality resolution."""

    status: ResolutionStatus
    place_id: str | None
    candidates: tuple[LocationCandidate, ...]
    display_note: str


@dataclass(frozen=True)
class MeetingOccurrence:
    """One resolved future meeting instance with UTC + source-local time."""

    group_id: str
    slot_id: str
    group_name: str
    group_format: GroupFormat
    city: str | None
    venue_or_url: str
    timezone: str
    start_local: str
    end_local: str | None
    start_utc: datetime
    day_label: Literal["today", "tomorrow", "date"]
    source_url: str
    freshness_note: str
    access: AccessKind


@dataclass(frozen=True)
class UpcomingMeetingResult:
    """Typed ranked upcoming-meeting response with pagination cursor."""

    occurrences: tuple[MeetingOccurrence, ...]
    ongoing: tuple[MeetingOccurrence, ...]
    fallback: UpcomingFallback | None
    next_cursor: str | None
    directory_version: str
    display_note: str


@dataclass(frozen=True)
class DirectorySnapshot:
    """The validated in-RAM directory (loaded once at startup)."""

    version: str
    digest: str
    places: tuple[PlaceRecord, ...] = field(default_factory=tuple)
    groups: tuple[GroupRecord, ...] = field(default_factory=tuple)
    slots: tuple[SlotRecord, ...] = field(default_factory=tuple)
    region_links: tuple[RegionLink, ...] = field(default_factory=tuple)
    sources: tuple[SourceRecord, ...] = field(default_factory=tuple)
