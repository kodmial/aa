"""Deterministic calendar ranking for verified meeting slots (no LLM)."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aa.meeting_directory.models import (
    DirectorySnapshot,
    GroupRecord,
    MeetingOccurrence,
    SlotRecord,
    UpcomingMeetingResult,
)

logger = logging.getLogger("aa.meeting_directory")

DEFAULT_LOOKAHEAD_DAYS = 35
MAX_LOOKAHEAD_DAYS = 60
STALE_THRESHOLD_DAYS = 30


class InvalidCursorError(ValueError):
    """A pagination cursor is tampered, stale, or foreign to this search."""


def _slot_is_fresh(slot: SlotRecord, now_utc: datetime) -> bool:
    if slot.parsing_status != "verified":
        return False
    if slot.schedule_verified_at is None:
        return False
    return (now_utc - slot.schedule_verified_at).days <= STALE_THRESHOLD_DAYS


def _slot_is_structurally_rankable(slot: SlotRecord) -> bool:
    if slot.parsing_status != "verified":
        return False
    if slot.timezone == "unknown":
        return False
    if slot.start_local is None:
        return False
    if slot.recurrence == "unknown":
        return False
    if slot.recurrence == "dated" and slot.date is None:
        return False
    return True


def _resolve_zone(name: str) -> ZoneInfo | None:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def _parse_hhmm(raw: str) -> tuple[int, int]:
    hour = int(raw.split(":")[0])
    minute = int(raw.split(":")[1])
    if hour > 23 or minute > 59:
        raise ValueError(f"invalid wall time: {raw!r}")
    return hour, minute


def _local_to_utc(day: date, hhmm: str, zone: ZoneInfo) -> datetime | None:
    """Convert a wall-clock instance to UTC, rejecting DST gaps explicitly.

    Nonexistent local times (spring-forward gap) yield ``None`` and are
    never silently shifted. Ambiguous times (fall-back overlap) use
    ``fold=0`` exactly once and are never double-counted.
    """
    hour, minute = _parse_hhmm(hhmm)
    local = datetime(day.year, day.month, day.day, hour, minute, fold=0, tzinfo=zone)
    as_utc = local.astimezone(UTC)
    roundtrip = as_utc.astimezone(zone)
    if (roundtrip.hour, roundtrip.minute) != (hour, minute):
        return None
    if (roundtrip.year, roundtrip.month, roundtrip.day) != (day.year, day.month, day.day):
        return None
    return as_utc


def _monthly_nth_date(year: int, month: int, weekday: int, week: int) -> date | None:
    """Resolve the nth weekday of a month (week 5 means last such weekday)."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    day = 1 + offset + (week - 1) * 7
    if week >= 5:
        # Last occurrence: step back while the month overflows.
        while True:
            try:
                candidate = date(year, month, day)
            except ValueError:
                day -= 7
                continue
            break
        while candidate.month == month:
            try:
                following = date(year, month, candidate.day + 7)
            except ValueError:
                break
            if following.month != month:
                break
            candidate = following
        return candidate
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _slot_dates_in_window(slot: SlotRecord, start_day: date, end_day: date) -> list[date]:
    """Enumerate candidate local calendar dates for one slot (bounded)."""
    if slot.recurrence == "dated" and slot.date is not None:
        try:
            single = date.fromisoformat(slot.date)
        except ValueError:
            return []
        return [single] if start_day <= single <= end_day else []
    if slot.recurrence == "daily":
        days: list[date] = []
        current = start_day
        while current <= end_day:
            days.append(current)
            current += timedelta(days=1)
        return days
    if slot.recurrence == "weekly":
        wanted = set(slot.weekdays)
        days = []
        current = start_day
        while current <= end_day:
            if current.weekday() in wanted:
                days.append(current)
            current += timedelta(days=1)
        return days
    if (
        slot.recurrence == "monthly_nth"
        and slot.month_week is not None
        and slot.month_weekday is not None
    ):
        days = []
        year, month = start_day.year, start_day.month
        stop = (end_day.year, end_day.month)
        while (year, month) <= stop:
            resolved = _monthly_nth_date(year, month, slot.month_weekday, slot.month_week)
            if resolved is not None and start_day <= resolved <= end_day:
                days.append(resolved)
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
        return days
    return []


def _in_validity(day: date, slot: SlotRecord) -> bool:
    if slot.valid_from:
        try:
            if day < date.fromisoformat(slot.valid_from):
                return False
        except ValueError:
            return False
    if slot.valid_until:
        try:
            if day > date.fromisoformat(slot.valid_until):
                return False
        except ValueError:
            return False
    return day.isoformat() not in slot.cancellations


def _day_label(start_utc: datetime, zone: ZoneInfo, now_utc: datetime) -> str:
    meeting_day = start_utc.astimezone(zone).date()
    today = now_utc.astimezone(zone).date()
    if meeting_day == today:
        return "today"
    if meeting_day == today + timedelta(days=1):
        return "tomorrow"
    return "date"


def _encode_cursor(payload: dict[str, str], version: str, digest: str) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    token = base64.urlsafe_b64encode(body).decode("ascii")
    signature = hashlib.sha256(f"{version}|{digest}|{token}".encode()).hexdigest()[:32]
    return f"{token}.{signature}"


def _decode_cursor(cursor: str, version: str, digest: str) -> dict[str, str]:
    try:
        token, signature = cursor.split(".", 1)
    except ValueError as exc:
        raise InvalidCursorError("malformed cursor") from exc
    expected = hashlib.sha256(f"{version}|{digest}|{token}".encode()).hexdigest()[:32]
    if signature != expected:
        raise InvalidCursorError("cursor signature mismatch")
    try:
        payload = json.loads(base64.urlsafe_b64decode(token.encode("ascii")))
    except (ValueError, TypeError) as exc:
        raise InvalidCursorError("malformed cursor payload") from exc
    if not isinstance(payload, dict):
        raise InvalidCursorError("malformed cursor payload")
    if payload.get("v") != version:
        raise InvalidCursorError("stale cursor version")
    return {str(key): str(value) for key, value in payload.items()}


def _occurrence_from_instance(
    group: GroupRecord,
    slot: SlotRecord,
    start_utc: datetime,
    zone: ZoneInfo,
    now_utc: datetime,
) -> MeetingOccurrence:
    local_start = start_utc.astimezone(zone)
    start_local = local_start.strftime("%A %Y-%m-%d %H:%M")
    end_local: str | None = None
    if slot.end_local is not None:
        try:
            end_utc = _local_to_utc(local_start.date(), slot.end_local, zone)
        except ValueError:
            end_utc = None
        if end_utc is not None and end_utc > start_utc:
            end_local = end_utc.astimezone(zone).strftime("%H:%M")
    return MeetingOccurrence(
        group_id=group.group_id,
        slot_id=slot.slot_id,
        group_name=group.name,
        group_format=slot.slot_format,
        city=group.city,
        venue_or_url=group.venue_or_url,
        timezone=slot.timezone,
        start_local=f"{start_local} ({slot.timezone})",
        end_local=end_local,
        start_utc=start_utc,
        day_label=_day_label(start_utc, zone, now_utc),  # type: ignore[arg-type]
        source_url=slot.source_url,
        freshness_note="Published schedule, unconfirmed: verify with organizers.",
        access=slot.access,
    )


def rank_upcoming_meetings(
    snapshot: DirectorySnapshot,
    slots: list[SlotRecord],
    groups_by_id: dict[str, GroupRecord],
    now_utc: datetime,
    limit: int = 3,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    cursor: str | None = None,
    cursor_scope: str = "",
) -> UpcomingMeetingResult:
    """Rank verified future occurrences by UTC instant (no LLM, no network)."""
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be a timezone-aware datetime")
    now = now_utc.astimezone(UTC)
    window = max(1, min(lookahead_days, MAX_LOOKAHEAD_DAYS))
    boundary_start: datetime | None = None
    boundary_slot = ""
    if cursor is not None:
        payload = _decode_cursor(cursor, snapshot.version, snapshot.digest)
        if payload.get("scope", "") != cursor_scope:
            raise InvalidCursorError("cursor does not belong to this search")
        if payload.get("now") != now.isoformat():
            raise InvalidCursorError("cursor expired for a new search time")
        boundary_start = datetime.fromisoformat(payload["after"])
        if boundary_start.tzinfo is None:
            raise InvalidCursorError("malformed cursor boundary")
        boundary_start = boundary_start.astimezone(UTC)
        boundary_slot = payload.get("slot", "")

    upcoming: list[MeetingOccurrence] = []
    ongoing: list[MeetingOccurrence] = []
    for slot in sorted(slots, key=lambda item: item.slot_id):
        if not _slot_is_structurally_rankable(slot):
            continue
        if not _slot_is_fresh(slot, now):
            continue
        group = groups_by_id.get(slot.group_id)
        if group is None:
            continue
        zone = _resolve_zone(slot.timezone)
        if zone is None or slot.start_local is None:
            continue
        # Scan from yesterday so a known in-progress meeting (verified start
        # and end) can be labeled separately without entering the future list.
        scan_start = now.astimezone(zone).date() - timedelta(days=1)
        scan_end = now.astimezone(zone).date() + timedelta(days=window)
        for day in _slot_dates_in_window(slot, scan_start, scan_end):
            if not _in_validity(day, slot):
                continue
            try:
                start_utc = _local_to_utc(day, slot.start_local, zone)
            except ValueError:
                continue
            if start_utc is None:
                continue
            occurrence = _occurrence_from_instance(group, slot, start_utc, zone, now)
            end_utc: datetime | None = None
            if slot.end_local is not None:
                try:
                    local_end = _local_to_utc(day, slot.end_local, zone)
                except ValueError:
                    local_end = None
                if local_end is not None and local_end > start_utc:
                    end_utc = local_end
            if start_utc <= now < end_utc if end_utc is not None else False:
                ongoing.append(occurrence)
            elif start_utc > now:
                upcoming.append(occurrence)
    upcoming.sort(key=lambda item: (item.start_utc, item.slot_id))
    ongoing.sort(key=lambda item: (item.start_utc, item.slot_id))
    # Deduplicate one actual session shown once (same slot+instant).
    deduped: list[MeetingOccurrence] = []
    seen: set[tuple[str, str]] = set()
    for occurrence in upcoming:
        key = (occurrence.slot_id, occurrence.start_utc.isoformat())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(occurrence)
    if boundary_start is not None:
        deduped = [
            item
            for item in deduped
            if (item.start_utc, item.slot_id) > (boundary_start, boundary_slot)
        ]
    page = deduped[: max(1, limit)]
    next_cursor: str | None = None
    if len(deduped) > len(page):
        last = page[-1]
        next_cursor = _encode_cursor(
            {
                "v": snapshot.version,
                "now": now.isoformat(),
                "after": last.start_utc.isoformat(),
                "slot": last.slot_id,
                "scope": cursor_scope,
            },
            snapshot.version,
            snapshot.digest,
        )
    logger.info(
        "upcoming ranking finished",
        extra={"candidates": len(deduped), "shown": len(page)},
    )
    return UpcomingMeetingResult(
        occurrences=tuple(page),
        ongoing=tuple(ongoing[:3]),
        fallback=None,
        next_cursor=next_cursor,
        directory_version=snapshot.version,
        display_note="Nearest verified meetings first. Schedules are published, not live.",
    )
