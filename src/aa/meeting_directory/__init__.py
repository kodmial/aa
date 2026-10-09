"""Russia-only AA meeting directory: offline snapshot lookup for #315.

Public consumer surface (typed filters only, no natural-language routing)::

    from aa.meeting_directory import (
        get_sources,
        get_upcoming_meetings,
        resolve_locality,
        search,
    )

All hot-path calls are offline with zero LLM invocations. A missing or
invalid snapshot yields typed ``directory_unavailable`` outcomes and never
disrupts normal AA/book conversation.
"""

from __future__ import annotations

from aa.meeting_directory.loader import (
    DirectoryUnavailableError,
    default_snapshot_dir,
    get_cached_snapshot,
    load_snapshot,
    reset_cache,
)
from aa.meeting_directory.models import (
    DirectorySearchResult,
    DirectorySnapshot,
    LocationResolution,
    MeetingOccurrence,
    MeetingResource,
    UpcomingMeetingResult,
)
from aa.meeting_directory.query import (
    coverage_by_place,
    coverage_summary,
    fixture_find_next,
    get_sources,
    get_upcoming_meetings,
    resolve_locality,
    search,
    snapshot_identity,
)
from aa.meeting_directory.schedule import (
    DEFAULT_LOOKAHEAD_DAYS,
    MAX_LOOKAHEAD_DAYS,
    STALE_THRESHOLD_DAYS,
)

__all__ = [
    "DEFAULT_LOOKAHEAD_DAYS",
    "MAX_LOOKAHEAD_DAYS",
    "STALE_THRESHOLD_DAYS",
    "DirectorySearchResult",
    "DirectorySnapshot",
    "DirectoryUnavailableError",
    "LocationResolution",
    "MeetingOccurrence",
    "MeetingResource",
    "UpcomingMeetingResult",
    "coverage_by_place",
    "coverage_summary",
    "default_snapshot_dir",
    "fixture_find_next",
    "get_cached_snapshot",
    "get_sources",
    "get_upcoming_meetings",
    "load_snapshot",
    "reset_cache",
    "resolve_locality",
    "search",
    "snapshot_identity",
]
