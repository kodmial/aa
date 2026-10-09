"""Calendar regression tests for nearest-meeting ranking (issue #316).

All queries use fixed injected UTC instants with zero LLM calls and zero
network access on the hot path.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aa.meeting_directory import (
    get_upcoming_meetings,
    load_snapshot,
    search,
)
from aa.meeting_directory.loader import default_snapshot_dir, reset_cache
from aa.meeting_directory.models import (
    DirectorySnapshot,
    GroupRecord,
    SlotRecord,
)
from aa.meeting_directory.schedule import rank_upcoming_meetings

UTC = UTC
TUESDAY_NOON = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _reset_directory_cache() -> None:
    reset_cache()


def _snapshot() -> DirectorySnapshot:
    return load_snapshot(default_snapshot_dir())


def test_weekly_slot_before_start_shows_today() -> None:
    upcoming = get_upcoming_meetings(place_id="place-moscow-city", now_utc=TUESDAY_NOON, limit=1)
    assert upcoming.fallback is None
    first = upcoming.occurrences[0]
    assert first.slot_id == "s-msk-vozrozhdenie-tue-1900"
    assert first.start_utc == datetime(2026, 10, 6, 16, 0, tzinfo=UTC)
    assert first.day_label == "today"
    assert "Europe/Moscow" in first.start_local


def test_weekly_slot_after_start_rolls_to_next_week() -> None:
    now = datetime(2026, 10, 6, 17, 0, tzinfo=UTC)
    upcoming = get_upcoming_meetings(place_id="place-moscow-city", now_utc=now, limit=1)
    assert upcoming.fallback is None
    first = upcoming.occurrences[0]
    assert first.slot_id == "s-msk-nadezhda-wed-1930"
    assert first.start_utc == datetime(2026, 10, 7, 16, 30, tzinfo=UTC)
    assert first.day_label == "tomorrow"


def test_saturday_to_sunday_and_month_boundary() -> None:
    saturday = datetime(2026, 10, 10, 9, 0, tzinfo=UTC)
    upcoming = get_upcoming_meetings(place_id="place-spb-city", now_utc=saturday, limit=2)
    assert upcoming.fallback is None
    first = upcoming.occurrences[0]
    assert first.slot_id == "s-spb-severnaya-sun-2000-online"
    assert first.start_utc == datetime(2026, 10, 11, 17, 0, tzinfo=UTC)


def test_year_boundary_rolls_into_january() -> None:
    group = _group("g-ny")
    slot = _slot(
        "s-ny-tue",
        "g-ny",
        "19:00",
        (1,),
        tz="Europe/Moscow",
        verified_at=datetime(2026, 12, 20, 12, 0, tzinfo=UTC),
    )
    snapshot = _synthetic_snapshot([slot], [group])
    new_year_eve = datetime(2026, 12, 31, 20, 0, tzinfo=UTC)
    result = rank_upcoming_meetings(
        snapshot, [slot], {"g-ny": group}, new_year_eve, limit=1, cursor_scope="t"
    )
    assert result.occurrences
    assert result.occurrences[0].start_utc > new_year_eve
    assert result.occurrences[0].start_utc.year == 2027
    assert result.occurrences[0].start_utc.date().isoformat() == "2027-01-05"


def test_same_wall_clock_orders_by_utc_not_strings() -> None:
    # 19:00 in Moscow (UTC+3), Yekaterinburg (UTC+5) and Vladivostok (UTC+10)
    # must order by real UTC instants.
    wednesday = datetime(2026, 10, 7, 8, 0, tzinfo=UTC)
    online = get_upcoming_meetings(format="online", now_utc=wednesday, limit=10)
    assert online.fallback is None
    same_day = [
        occurrence
        for occurrence in online.occurrences
        if occurrence.start_utc.date().isoformat() == "2026-10-07"
    ]
    assert same_day, "expected Wednesday online occurrences"
    instants = [occurrence.start_utc for occurrence in same_day]
    assert instants == sorted(instants)
    zones = {occurrence.timezone for occurrence in same_day}
    assert len(zones) >= 2


def test_unknown_online_timezone_is_never_ranked() -> None:
    now = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
    upcoming = get_upcoming_meetings(format="online", now_utc=now, limit=50)
    assert upcoming.fallback is None
    assert all(
        occurrence.slot_id != "s-online-vecherniy-unparsed" for occurrence in upcoming.occurrences
    )
    assert all(occurrence.timezone != "unknown" for occurrence in upcoming.occurrences)


def test_explicit_cancellation_skips_that_date() -> None:
    # Yekaterinburg Monday 19:00 is cancelled on 2026-10-12.
    upcoming = get_upcoming_meetings(
        place_id="place-yekaterinburg-city", now_utc=TUESDAY_NOON, limit=5
    )
    assert upcoming.fallback is None
    days = [occurrence.start_utc.date().isoformat() for occurrence in upcoming.occurrences]
    assert "2026-10-12" not in days
    assert days[0] == "2026-10-19"


def test_dated_event_never_rolls_into_next_week() -> None:
    after_event = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
    upcoming = get_upcoming_meetings(place_id="place-moscow-city", now_utc=after_event, limit=50)
    assert all(
        occurrence.slot_id != "s-msk-vozrozhdenie-speaker-20261015"
        for occurrence in upcoming.occurrences
    )


def test_known_ongoing_meeting_is_labeled_separately() -> None:
    during = datetime(2026, 10, 6, 16, 15, tzinfo=UTC)
    upcoming = get_upcoming_meetings(place_id="place-moscow-city", now_utc=during, limit=5)
    assert upcoming.fallback is None
    assert all(occurrence.start_utc > during for occurrence in upcoming.occurrences)
    assert any(item.slot_id == "s-msk-vozrozhdenie-tue-1900" for item in upcoming.ongoing)


def _synthetic_snapshot(slots: list[SlotRecord], groups: list[GroupRecord]) -> DirectorySnapshot:
    return DirectorySnapshot(
        version="test-v1", digest="test", groups=tuple(groups), slots=tuple(slots)
    )


def _group(group_id: str, name: str = "Synthetic") -> GroupRecord:
    return GroupRecord(
        group_id=group_id,
        name=name,
        country="RU",
        region=None,
        city=None,
        place_id=None,
        group_format="online",
        venue_or_url="https://aarussia.ru/online-groups/",
        timezone="America/New_York",
        access="open",
        source_id="aarussia-online",
        source_url="https://aarussia.ru/online-groups/",
    )


def _slot(
    slot_id: str,
    group_id: str,
    start_local: str,
    weekdays: tuple[int, ...],
    tz: str = "America/New_York",
    verified_at: datetime | None = None,
) -> SlotRecord:
    return SlotRecord(
        slot_id=slot_id,
        group_id=group_id,
        slot_format="online",
        recurrence="weekly",
        weekdays=weekdays,
        start_local=start_local,
        end_local=None,
        timezone=tz,
        date=None,
        month_week=None,
        month_weekday=None,
        valid_from=None,
        valid_until=None,
        cancellations=(),
        access="open",
        source_url="https://aarussia.ru/online-groups/",
        schedule_verified_at=verified_at or datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        parsing_status="verified",
    )


def test_nonexistent_dst_time_is_skipped_not_shifted() -> None:
    # 2026-03-08 02:30 does not exist in America/New_York (spring forward).
    group = _group("g-dst")
    slot = _slot("s-dst-gap", "g-dst", "02:30", (6,))
    snapshot = _synthetic_snapshot([slot], [group])
    now = datetime(2026, 3, 7, 12, 0, tzinfo=UTC)
    result = rank_upcoming_meetings(
        snapshot, [slot], {"g-dst": group}, now, limit=5, cursor_scope="t"
    )
    days = [occurrence.start_utc.date().isoformat() for occurrence in result.occurrences]
    assert "2026-03-08" not in days
    assert days[0] == "2026-03-15"


def test_ambiguous_dst_time_emits_single_occurrence() -> None:
    # 2026-11-01 01:30 is ambiguous in America/New_York (fall back).
    group = _group("g-dst-fall")
    slot = _slot(
        "s-dst-fold",
        "g-dst-fall",
        "01:30",
        (6,),
        verified_at=datetime(2026, 10, 20, 12, 0, tzinfo=UTC),
    )
    snapshot = _synthetic_snapshot([slot], [group])
    now = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
    result = rank_upcoming_meetings(
        snapshot, [slot], {"g-dst-fall": group}, now, limit=5, cursor_scope="t"
    )
    same_day = [
        occurrence
        for occurrence in result.occurrences
        if occurrence.start_utc.date().isoformat() == "2026-11-01"
    ]
    assert len(same_day) == 1


def test_two_times_same_weekday_both_ranked() -> None:
    group = _group("g-twice", "Twice")
    morning = _slot(
        "s-twice-0900",
        "g-twice",
        "09:00",
        (0,),
        tz="Europe/Moscow",
        verified_at=datetime(2026, 10, 5, 12, 0, tzinfo=UTC),
    )
    evening = _slot(
        "s-twice-1900",
        "g-twice",
        "19:00",
        (0,),
        tz="Europe/Moscow",
        verified_at=datetime(2026, 10, 5, 12, 0, tzinfo=UTC),
    )
    snapshot = _synthetic_snapshot([morning, evening], [group])
    sunday = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
    result = rank_upcoming_meetings(
        snapshot,
        [morning, evening],
        {"g-twice": group},
        sunday,
        limit=2,
        cursor_scope="t",
    )
    assert [item.slot_id for item in result.occurrences] == ["s-twice-0900", "s-twice-1900"]
    assert result.occurrences[0].start_utc < result.occurrences[1].start_utc


def test_stable_tie_break_and_pagination() -> None:
    first = get_upcoming_meetings(place_id="place-moscow-city", now_utc=TUESDAY_NOON, limit=2)
    assert first.fallback is None
    assert len(first.occurrences) == 2
    assert first.next_cursor is not None
    second = get_upcoming_meetings(
        place_id="place-moscow-city",
        now_utc=TUESDAY_NOON,
        limit=2,
        cursor=first.next_cursor,
    )
    assert second.fallback is None
    first_keys = {(item.slot_id, item.start_utc) for item in first.occurrences}
    second_keys = {(item.slot_id, item.start_utc) for item in second.occurrences}
    assert not first_keys & second_keys
    combined = [*first.occurrences, *second.occurrences]
    assert [item.start_utc for item in combined] == sorted(item.start_utc for item in combined)


def test_new_search_reflects_newer_now_and_stale_cursor_fails_clean() -> None:
    first = get_upcoming_meetings(place_id="place-moscow-city", now_utc=TUESDAY_NOON, limit=1)
    assert first.next_cursor is not None
    later = get_upcoming_meetings(
        place_id="place-moscow-city",
        now_utc=datetime(2026, 10, 7, 12, 0, tzinfo=UTC),
        limit=1,
    )
    assert later.fallback is None
    assert later.occurrences[0].start_utc != first.occurrences[0].start_utc
    tampered = first.next_cursor + "x"
    failed = get_upcoming_meetings(
        place_id="place-moscow-city",
        now_utc=TUESDAY_NOON,
        limit=1,
        cursor=tampered,
    )
    assert failed.occurrences == ()
    assert "new search" in failed.display_note.lower()


def test_first_three_are_chronologically_earliest() -> None:
    upcoming = get_upcoming_meetings(place_id="place-moscow-city", now_utc=TUESDAY_NOON, limit=3)
    assert upcoming.fallback is None
    assert [item.start_utc for item in upcoming.occurrences] == sorted(
        item.start_utc for item in upcoming.occurrences
    )
    assert upcoming.occurrences[0].slot_id == "s-msk-vozrozhdenie-tue-1900"
    assert all(item.start_utc > TUESDAY_NOON for item in upcoming.occurrences)
    for item in upcoming.occurrences:
        assert item.source_url.startswith("https://aarussia.ru/")
        assert item.group_id
        assert item.slot_id


def test_naive_now_utc_is_rejected() -> None:
    with pytest.raises(ValueError):
        get_upcoming_meetings(place_id="place-moscow-city", now_utc=datetime(2026, 10, 6, 12, 0))


def test_consumer_contract_keys_are_stable() -> None:
    result = search(place_id="place-moscow-city", limit=1)
    assert result.outcome == "matched_group"
    item = result.items[0]
    assert isinstance(item.record_id, str) and item.record_id
    assert isinstance(item.source_url, str) and item.source_url
    assert item.freshness in ("fresh", "stale", "unknown")


def test_no_results_vs_no_coverage_distinction() -> None:
    directory_only = search(place_id="place-vladivostok-city")
    assert directory_only.outcome == "region_directory_only"
    unknown = search(city="Atlantis")
    assert unknown.outcome == "no_coverage"
    assert unknown.items == ()


def test_leap_day_dated_event() -> None:
    group = _group("g-leap")
    slot = SlotRecord(
        slot_id="s-leap-20280229",
        group_id="g-leap",
        slot_format="online",
        recurrence="dated",
        weekdays=(),
        start_local="20:00",
        end_local=None,
        timezone="Europe/Moscow",
        date="2028-02-29",
        month_week=None,
        month_weekday=None,
        valid_from=None,
        valid_until=None,
        cancellations=(),
        access="open",
        source_url="https://aarussia.ru/online-groups/",
        schedule_verified_at=datetime(2028, 2, 1, 12, 0, tzinfo=UTC),
        parsing_status="verified",
    )
    snapshot = _synthetic_snapshot([slot], [group])
    now = datetime(2028, 2, 28, 12, 0, tzinfo=UTC)
    result = rank_upcoming_meetings(
        snapshot, [slot], {"g-leap": group}, now, limit=3, cursor_scope="t"
    )
    assert len(result.occurrences) == 1
    assert result.occurrences[0].start_utc.date().isoformat() == "2028-02-29"
