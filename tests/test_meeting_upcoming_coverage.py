"""Issue #317: real source-verified upcoming-meeting coverage (not link-only).

Deterministic fixture-backed acceptance over the versioned snapshot with a
fixed aware UTC instant. No LLM calls, no runtime HTTP/geocoder access.
"""

from __future__ import annotations

import hashlib
import json
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from aa.meeting_directory import (
    coverage_by_place,
    coverage_summary,
    fixture_find_next,
    get_upcoming_meetings,
    load_snapshot,
    search,
    snapshot_identity,
)
from aa.meeting_directory.loader import default_snapshot_dir, reset_cache
from aa.meeting_directory.validator import FORBIDDEN_PATTERNS, is_allowed_url

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
VERSION = "ru-2026-10-09-v2"
NATIONAL_URL = "https://aarussia.ru/aagroups/russiagroups/"
ONLINE_URL = "https://aarussia.ru/online-groups/"
ROOT_URL = "https://aarussia.ru/aagroups/"


@pytest.fixture(autouse=True)
def _reset_directory_cache() -> None:
    reset_cache()


def _snapshot_dir() -> Path:
    return default_snapshot_dir()


def test_snapshot_version_pinned_for_315() -> None:
    snapshot = load_snapshot(_snapshot_dir())
    assert snapshot.version == VERSION
    manifest = json.loads((_snapshot_dir() / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    assert manifest["snapshot_version"] == VERSION
    digests = manifest["data_digests"]
    assert isinstance(digests, dict)
    for name in ("groups.json", "sources.json"):
        expected = digests[name]
        actual = "sha256:" + hashlib.sha256((_snapshot_dir() / name).read_bytes()).hexdigest()
        assert actual == expected
    identity = snapshot_identity(snapshot)
    assert identity == {"version": VERSION, "digest": digests["groups.json"]}


def test_coverage_floor_three_cities_two_timezones_plus_online() -> None:
    snapshot = load_snapshot(_snapshot_dir())
    by_place = coverage_by_place(snapshot, NOW)
    upcoming_places = [
        place_id for place_id, row in by_place.items() if row["status"] == "upcoming"
    ]
    in_person = [
        place_id
        for place_id in upcoming_places
        if any(
            group.group_format in ("in_person", "hybrid") and group.place_id == place_id
            for group in snapshot.groups
        )
    ]
    assert len(in_person) >= 3
    zones = {
        slot.timezone
        for slot in snapshot.slots
        if slot.timezone != "unknown"
        and slot.group_id in {g.group_id for g in snapshot.groups if g.place_id in in_person}
    }
    assert len(zones) >= 2
    online = fixture_find_next(now_utc=NOW, format="online", limit=3)
    assert online.fallback is None
    assert len(online.occurrences) >= 1
    assert all(item.timezone != "unknown" for item in online.occurrences)


def test_manifest_counts_separate_verified_vs_navigation() -> None:
    snapshot = load_snapshot(_snapshot_dir())
    summary = coverage_summary(snapshot, NOW)
    assert summary["places"] == 11
    assert summary["verified_groups"] == 12
    assert summary["slots"] == 17
    assert summary["rankable_slots"] == 16
    assert summary["unrankable_slots"] == 1
    assert summary["directory_only_places"] == 4
    assert summary["stale_slots"] == 1
    assert summary["fresh_rankable_slots"] == 15
    by_place = coverage_by_place(snapshot, NOW)
    assert by_place["place-kazan-city"]["status"] == "schedule_stale"
    assert by_place["place-novosibirsk-city"]["status"] == "directory_only"
    assert by_place["place-vladivostok-city"]["status"] == "directory_only"
    assert by_place["place-blagoveshchensk-city"]["status"] == "upcoming"
    assert by_place["place-arkhangelsk-city"]["status"] == "upcoming"
    assert by_place["place-belgorod-city"]["status"] == "upcoming"
    # A regional directory link never counts as a future meeting.
    assert by_place["place-novosibirsk-city"]["future_occurrences_probe"] == 0


def test_blagoveshchensk_next_is_wednesday_yakutsk_evening() -> None:
    result = fixture_find_next(place_id="place-blagoveshchensk-city", now_utc=NOW, limit=3)
    assert result.fallback is None
    first = result.occurrences[0]
    assert first.slot_id == "s-blg-edinstvo-wed-1900"
    assert first.start_utc == datetime(2026, 10, 7, 10, 0, tzinfo=UTC)
    assert first.timezone == "Asia/Yakutsk"
    assert "19:00" in first.start_local
    # Monday 19:00 Yakutsk (10:00 UTC) already passed relative to NOW.
    assert all(item.start_utc > NOW for item in result.occurrences)


def test_arkhangelsk_next_is_sunday_moscow_evening() -> None:
    result = fixture_find_next(place_id="place-arkhangelsk-city", now_utc=NOW, limit=2)
    assert result.fallback is None
    first = result.occurrences[0]
    assert first.slot_id == "s-arkh-istok-sun-1900"
    assert first.start_utc == datetime(2026, 10, 11, 16, 0, tzinfo=UTC)
    assert first.timezone == "Europe/Moscow"


def test_belgorod_daily_next_is_today() -> None:
    result = fixture_find_next(place_id="place-belgorod-city", now_utc=NOW, limit=2)
    assert result.fallback is None
    assert result.occurrences[0].slot_id == "s-bel-12-daily-1830"
    assert result.occurrences[0].start_utc == datetime(2026, 10, 6, 15, 30, tzinfo=UTC)
    assert result.occurrences[1].start_utc == datetime(2026, 10, 7, 15, 30, tzinfo=UTC)


def test_partizan_online_daily_next_is_today_msk() -> None:
    result = fixture_find_next(now_utc=NOW, format="online", limit=5)
    assert result.fallback is None
    assert result.occurrences
    partizan = [item for item in result.occurrences if item.slot_id.startswith("s-online-")]
    assert partizan
    assert result.occurrences[0].start_utc == datetime(2026, 10, 6, 17, 0, tzinfo=UTC)
    assert "20:00" in result.occurrences[0].start_local
    # Same wall clock in different zones orders by true UTC instant.
    instants = [item.start_utc for item in result.occurrences]
    assert instants == sorted(instants)


def test_top_three_sorted_by_utc_with_stable_ids_and_cursor() -> None:
    first = fixture_find_next(now_utc=NOW, limit=3)
    assert first.fallback is None
    assert len(first.occurrences) == 3
    assert [item.start_utc for item in first.occurrences] == sorted(
        item.start_utc for item in first.occurrences
    )
    assert all(item.slot_id and item.group_id for item in first.occurrences)
    assert first.next_cursor is not None
    second = get_upcoming_meetings(now_utc=NOW, limit=3, cursor=first.next_cursor)
    assert second.fallback is None
    first_keys = {(item.slot_id, item.start_utc) for item in first.occurrences}
    second_keys = {(item.slot_id, item.start_utc) for item in second.occurrences}
    assert not first_keys & second_keys
    combined = [*first.occurrences, *second.occurrences]
    assert [item.start_utc for item in combined] == sorted(item.start_utc for item in combined)


def test_past_starts_excluded_and_no_fake_meetings() -> None:
    snapshot = load_snapshot(_snapshot_dir())
    known_slots = {slot.slot_id for slot in snapshot.slots}
    result = fixture_find_next(now_utc=NOW, limit=50)
    assert result.fallback is None
    assert result.occurrences
    for item in result.occurrences:
        assert item.start_utc > NOW
        assert item.slot_id in known_slots
        assert is_allowed_url(item.source_url)
        assert item.source_url.startswith("https://aarussia.ru/")


def test_35_day_window_bounds_future() -> None:
    far = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)
    result = fixture_find_next(place_id="place-moscow-city", now_utc=far, limit=50)
    for item in result.occurrences:
        assert item.start_utc <= far + timedelta(days=36)
    assert all(item.slot_id != "s-msk-vozrozhdenie-speaker-20261015" for item in result.occurrences)


def test_stale_kazan_yields_schedule_stale() -> None:
    result = fixture_find_next(place_id="place-kazan-city", now_utc=NOW)
    assert result.fallback == "schedule_stale"
    assert result.occurrences == ()
    search_result = search(place_id="place-kazan-city")
    assert search_result.outcome == "stale_details"


def test_unknown_vecherniy_never_ranked() -> None:
    result = fixture_find_next(now_utc=NOW, format="online", limit=50)
    assert result.fallback is None
    assert all(item.slot_id != "s-online-vecherniy-unparsed" for item in result.occurrences)
    assert all(item.timezone != "unknown" for item in result.occurrences)


def test_directory_only_places_yield_link_only() -> None:
    for place_id in ("place-novosibirsk-city", "place-vladivostok-city"):
        result = fixture_find_next(place_id=place_id, now_utc=NOW)
        assert result.fallback == "directory_only"
        assert result.occurrences == ()
        found = search(place_id=place_id)
        assert found.outcome == "region_directory_only"
        assert found.region_link == NATIONAL_URL
    unknown = search(city="Atlantis")
    assert unknown.outcome == "no_coverage"
    assert ROOT_URL in unknown.source_urls


def test_source_verified_at_fresh_for_all_ranked() -> None:
    from aa.meeting_directory.schedule import STALE_THRESHOLD_DAYS

    snapshot = load_snapshot(_snapshot_dir())
    by_slot = {slot.slot_id: slot for slot in snapshot.slots}
    result = fixture_find_next(now_utc=NOW, limit=50)
    assert result.occurrences
    for item in result.occurrences:
        verified_at = by_slot[item.slot_id].schedule_verified_at
        assert verified_at is not None
        assert (NOW - verified_at).days <= STALE_THRESHOLD_DAYS


def test_month_boundary_rolls_into_november() -> None:
    friday = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
    result = fixture_find_next(place_id="place-blagoveshchensk-city", now_utc=friday, limit=1)
    assert result.fallback is None
    first = result.occurrences[0]
    assert first.start_utc > friday
    assert (first.start_utc.year, first.start_utc.month) == (2026, 11)
    assert first.start_utc == datetime(2026, 11, 2, 10, 0, tzinfo=UTC)


def test_hot_path_offline_no_llm_no_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _blocked(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access on the directory hot path")

    monkeypatch.setattr(socket, "socket", _blocked)
    import aa.opencode.runtime as opencode_runtime

    def _raise(*args: object, **kwargs: object) -> object:
        raise AssertionError("model client must not be used on the directory hot path")

    monkeypatch.setattr(opencode_runtime.LocalOpenCodeRuntime, "start", _raise)
    monkeypatch.setattr(opencode_runtime.LocalOpenCodeRuntime, "ensure_ready", _raise)
    snapshot = load_snapshot(_snapshot_dir())
    assert (
        fixture_find_next(
            place_id="place-blagoveshchensk-city", now_utc=NOW, snapshot=snapshot
        ).fallback
        is None
    )
    assert fixture_find_next(now_utc=NOW, format="online", snapshot=snapshot).fallback is None


def test_loader_has_no_cwd_assumptions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    reset_cache()
    snapshot = load_snapshot(default_snapshot_dir())
    assert snapshot.version == VERSION
    assert fixture_find_next(place_id="place-belgorod-city", now_utc=NOW).fallback is None


def test_tzdata_covers_all_snapshot_zones() -> None:
    snapshot = load_snapshot(_snapshot_dir())
    zones = {slot.timezone for slot in snapshot.slots if slot.timezone != "unknown"}
    assert len(zones) >= 3
    for name in zones:
        assert ZoneInfo(name) is not None
    assert ZoneInfo("Asia/Yakutsk") is not None


def test_meeting_data_absent_from_book_corpus() -> None:
    repo = Path(__file__).resolve().parents[1]
    assert (repo / "data" / "meeting_directory").is_dir()
    assert not (repo / "corpus" / "meeting_directory").exists()
    for name in ("canonical.manifest.json", "canonical.ru.manifest.json"):
        blob = (repo / "corpus" / name).read_text(encoding="utf-8")
        assert "meeting_directory" not in blob
        assert "g-blg-edinstvo" not in blob
    groups_blob = (
        (repo / "data" / "meeting_directory" / "ru" / "groups.json")
        .read_text(encoding="utf-8")
        .lower()
    )
    for marker in FORBIDDEN_PATTERNS:
        assert marker not in groups_blob


def test_refresh_check_enforces_floor() -> None:
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "refresh_meeting_directory.py"
    spec = importlib.util.spec_from_file_location("refresh_meeting_directory", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check_snapshot(_snapshot_dir()) == 0
    ok, detail = module.check_coverage_floor(load_snapshot(_snapshot_dir()))
    assert ok
    assert "met" in detail
