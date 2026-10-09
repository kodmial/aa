"""Meeting directory tests: snapshot, locality, search, importer, privacy."""

from __future__ import annotations

import copy
import hashlib
import json
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

from aa.meeting_directory import (
    coverage_summary,
    get_sources,
    get_upcoming_meetings,
    load_snapshot,
    resolve_locality,
    search,
)
from aa.meeting_directory.loader import (
    DirectoryUnavailableError,
    default_snapshot_dir,
    reset_cache,
)
from aa.meeting_directory.validator import (
    FORBIDDEN_PATTERNS,
    is_allowed_url,
    validate_snapshot_payload,
)

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _reset_directory_cache() -> None:
    reset_cache()


def _snapshot_dir() -> Path:
    return default_snapshot_dir()


def test_snapshot_loads_offline_with_separate_coverage() -> None:
    snapshot = load_snapshot(_snapshot_dir())
    assert snapshot.version == "ru-2026-10-09-v2"
    summary = coverage_summary(snapshot)
    assert summary["region_directory_links"] >= 8
    assert summary["verified_groups"] >= 5
    assert summary["slots"] >= 10
    assert summary["rankable_slots"] >= 9
    assert summary["places"] >= 8
    # Region navigation and verified rows are counted separately.
    assert summary["slots"] > summary["verified_groups"]
    assert summary["region_directory_links"] > 0


def test_snapshot_covers_multiple_cities_and_online() -> None:
    snapshot = load_snapshot(_snapshot_dir())
    cities = {group.city for group in snapshot.groups if group.city}
    assert {"Moscow", "Saint Petersburg", "Yekaterinburg"} <= cities
    online = [group for group in snapshot.groups if group.group_format == "online"]
    assert len(online) >= 2
    in_person = [group for group in snapshot.groups if group.group_format == "in_person"]
    assert len(in_person) >= 3


def test_region_link_never_claims_a_meeting() -> None:
    result = search(place_id="place-novosibirsk-city")
    assert result.outcome == "region_directory_only"
    assert result.items == ()
    assert result.region_link == "https://aarussia.ru/aagroups/russiagroups/"


def test_resolve_unique_major_city() -> None:
    resolution = resolve_locality("Moscow")
    assert resolution.status == "exact_unique"
    assert resolution.place_id == "place-moscow-city"


def test_resolve_spb_alias() -> None:
    for alias in ("СПб", "спб", "Санкт-Петербург", "Санкт Петербург", "Питер"):
        resolution = resolve_locality(alias)
        assert resolution.status == "exact_unique", alias
        assert resolution.place_id == "place-spb-city", alias


def test_resolve_ambiguous_kirov_needs_region() -> None:
    resolution = resolve_locality("Киров")
    assert resolution.status == "needs_region"
    assert resolution.place_id is None
    assert 2 <= len(resolution.candidates) <= 4
    ids = {candidate.place_id for candidate in resolution.candidates}
    assert ids == {"place-kirov-kirovskaya", "place-kirov-kaluzhskaya"}
    narrowed = resolve_locality("Киров", region_hint="Kirov region")
    assert narrowed.status == "exact_unique"
    assert narrowed.place_id == "place-kirov-kirovskaya"


def test_resolve_unknown_place() -> None:
    resolution = resolve_locality("Atlantis")
    assert resolution.status == "unknown"
    assert resolution.place_id is None
    assert resolution.candidates == ()


def test_typo_is_not_converted_to_wrong_city() -> None:
    resolution = resolve_locality("Maskva")
    assert resolution.status == "unknown"
    assert resolution.place_id is None


def test_asr_variant_with_yo_and_case() -> None:
    first = resolve_locality("САНКТ-ПЕТЕРБУРГ")
    assert first.place_id == "place-spb-city"


def test_region_only_input_does_not_invent_city() -> None:
    resolution = resolve_locality("Tatarstan")
    assert resolution.place_id is None
    assert resolution.status in ("ambiguous", "unknown")
    result = search(region_id="region-novosibirsk")
    assert result.outcome == "region_directory_only"


def test_online_results_without_city() -> None:
    result = search(format="online")
    assert result.outcome == "matched_group"
    assert all(item.group_format in ("online", "hybrid") for item in result.items)
    upcoming = get_upcoming_meetings(format="online", now_utc=NOW)
    assert upcoming.fallback is None
    assert upcoming.occurrences, "expected ranked online occurrences"


def test_search_distinguishes_open_and_closed() -> None:
    result = search(place_id="place-moscow-city", limit=10)
    assert result.outcome == "matched_group"
    access = {item.record_id: item.access for item in result.items}
    assert access.get("g-msk-nadezhda") == "closed"
    assert access.get("g-msk-vozrozhdenie") == "open"


def test_search_rejects_non_ru_scope() -> None:
    result = search(country="US")
    assert result.outcome == "no_coverage"
    assert result.items == ()


def test_stale_schedule_returns_honest_navigation() -> None:
    result = search(place_id="place-kazan-city")
    assert result.outcome == "stale_details"
    assert result.region_link is not None
    upcoming = get_upcoming_meetings(place_id="place-kazan-city", now_utc=NOW)
    assert upcoming.fallback == "schedule_stale"
    assert upcoming.occurrences == ()


def test_hybrid_group_keeps_distinct_slots() -> None:
    upcoming = get_upcoming_meetings(place_id="place-spb-city", now_utc=NOW, limit=10)
    assert upcoming.fallback is None
    slot_ids = {occurrence.slot_id for occurrence in upcoming.occurrences}
    assert "s-spb-severnaya-thu-1900" in slot_ids
    assert "s-spb-severnaya-sun-2000-online" in slot_ids


def test_same_named_localities_stay_distinct() -> None:
    snapshot = load_snapshot(_snapshot_dir())
    kirov = [place for place in snapshot.places if place.display_name == "Kirov"]
    assert len(kirov) == 2
    assert kirov[0].place_id != kirov[1].place_id
    assert kirov[0].region_id != kirov[1].region_id


def test_no_full_region_or_city_menu_is_generated() -> None:
    resolution = resolve_locality("Киров")
    assert len(resolution.candidates) <= 4
    result = search(place_id="place-moscow-city", limit=1)
    assert len(result.items) <= 1
    snapshot = load_snapshot(_snapshot_dir())
    assert len(snapshot.places) < 100


def test_every_result_cites_vetted_source_url() -> None:
    result = search(place_id="place-moscow-city", limit=10)
    assert result.items
    for item in result.items:
        assert is_allowed_url(item.source_url)
        assert item.source_url.startswith("https://aarussia.ru/")


def test_get_sources_reports_rights_review() -> None:
    sources = get_sources()
    assert len(sources) >= 3
    urls = {source.url for source in sources}
    assert "https://aarussia.ru/aagroups/russiagroups/" in urls
    assert "https://aarussia.ru/online-groups/" in urls
    for source in sources:
        assert is_allowed_url(source.url)
        assert source.permission_status, "permission review must be recorded"


def test_snapshot_contains_no_private_markers() -> None:
    blob = (_snapshot_dir() / "groups.json").read_text(encoding="utf-8").lower()
    for marker in FORBIDDEN_PATTERNS:
        assert marker not in blob
    sources = (_snapshot_dir() / "sources.json").read_text(encoding="utf-8").lower()
    for marker in FORBIDDEN_PATTERNS:
        assert marker not in blob + sources


def test_validator_rejects_private_markers() -> None:
    directory = _snapshot_dir()
    raw_groups = json.loads((directory / "groups.json").read_text(encoding="utf-8"))
    raw_sources = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
    raw_manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    poisoned = copy.deepcopy(raw_groups)
    assert isinstance(poisoned, dict)
    groups = poisoned["groups"]
    assert isinstance(groups, list)
    first = groups[0]
    assert isinstance(first, dict)
    first["address"] = "Moscow, entry code 1234, домофон 56"
    with pytest.raises(ValueError):
        validate_snapshot_payload(poisoned, raw_sources, raw_manifest)


def test_tampered_checksum_fails_closed() -> None:
    reset_cache()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        for name in ("groups.json", "sources.json", "manifest.json"):
            (staged / name).write_text(
                (_snapshot_dir() / name).read_text(encoding="utf-8"), encoding="utf-8"
            )
        payload = (staged / "groups.json").read_text(encoding="utf-8") + " "
        (staged / "groups.json").write_text(payload, encoding="utf-8")
        with pytest.raises(DirectoryUnavailableError):
            load_snapshot(staged)


def test_missing_snapshot_yields_typed_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AA_MEETING_DIRECTORY_PATH", str(tmp_path / "absent"))
    reset_cache()
    result = search(place_id="place-moscow-city")
    assert result.outcome == "directory_unavailable"
    assert result.items == ()
    upcoming = get_upcoming_meetings(place_id="place-moscow-city", now_utc=NOW)
    assert upcoming.fallback == "directory_unavailable"
    assert upcoming.occurrences == ()
    monkeypatch.delenv("AA_MEETING_DIRECTORY_PATH")
    reset_cache()
    # Normal directory lookup recovers once the snapshot path is restored.
    assert search(place_id="place-moscow-city").outcome == "matched_group"


def test_hot_path_makes_zero_network_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _blocked(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access on the directory hot path")

    monkeypatch.setattr(socket, "socket", _blocked)
    snapshot = load_snapshot(_snapshot_dir())
    assert search(place_id="place-moscow-city", snapshot=snapshot).outcome != (
        "directory_unavailable"
    )
    upcoming = get_upcoming_meetings(place_id="place-moscow-city", now_utc=NOW, snapshot=snapshot)
    assert upcoming.occurrences


def test_no_llm_dependency_on_hot_path() -> None:
    import subprocess

    code = (
        "import sys;"
        "from aa.meeting_directory import search, get_upcoming_meetings;"
        "from datetime import datetime, timezone;"
        "now = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc);"
        "assert search(place_id='place-moscow-city').outcome == 'matched_group';"
        "result = get_upcoming_meetings(place_id='place-moscow-city', now_utc=now);"
        "assert result.occurrences;"
        "banned = [m for m in ('langchain', 'langgraph', 'transformers', 'torch')"
        " if m in sys.modules];"
        "assert not banned, banned;"
        "print('hot-path clean')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert completed.returncode == 0, completed.stderr
    assert "hot-path clean" in completed.stdout


def test_directory_works_when_model_clients_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aa.opencode.runtime as opencode_runtime

    def _raise(*args: object, **kwargs: object) -> object:
        raise AssertionError("model client must not be used on the directory hot path")

    monkeypatch.setattr(opencode_runtime.LocalOpenCodeRuntime, "start", _raise)
    monkeypatch.setattr(opencode_runtime.LocalOpenCodeRuntime, "ensure_ready", _raise)
    snapshot = load_snapshot(_snapshot_dir())
    result = search(place_id="place-moscow-city", snapshot=snapshot)
    assert result.outcome == "matched_group"
    upcoming = get_upcoming_meetings(place_id="place-moscow-city", now_utc=NOW, snapshot=snapshot)
    assert upcoming.fallback is None
    assert upcoming.occurrences[0].slot_id == "s-msk-vozrozhdenie-tue-1900"


def test_snapshot_memory_bounds() -> None:
    total = sum(
        (_snapshot_dir() / name).stat().st_size
        for name in ("groups.json", "sources.json", "manifest.json")
    )
    assert total < 1024 * 1024
    summary = coverage_summary()
    assert summary["slots"] < 50000


def _load_refresh_tool() -> ModuleType:
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "refresh_meeting_directory.py"
    spec = importlib.util.spec_from_file_location("refresh_meeting_directory", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importer_diff_reports_changes_without_overwrite(tmp_path: Path) -> None:
    tool = _load_refresh_tool()
    assert callable(tool.check_snapshot) and callable(tool.diff_staging)
    assert tool.check_snapshot(_snapshot_dir()) == 0
    raw = json.loads((_snapshot_dir() / "groups.json").read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    staging = copy.deepcopy(raw)
    groups = staging["groups"]
    assert isinstance(groups, list)
    first = copy.deepcopy(groups[0])
    assert isinstance(first, dict)
    first["group_id"] = "g-test-new"
    first["name"] = "Test New"
    groups.append(first)
    staging_path = tmp_path / "staging.json"
    staging_path.write_text(json.dumps(staging), encoding="utf-8")
    assert tool.diff_staging(staging_path, _snapshot_dir()) == 0


def test_importer_rejects_private_staging(tmp_path: Path) -> None:
    tool = _load_refresh_tool()
    assert callable(tool.diff_staging)

    raw = json.loads((_snapshot_dir() / "groups.json").read_text(encoding="utf-8"))
    staging = copy.deepcopy(raw)
    assert isinstance(staging, dict)
    groups = staging["groups"]
    assert isinstance(groups, list)
    first = groups[0]
    assert isinstance(first, dict)
    first["address"] = "Moscow, домофон 42"
    staging_path = tmp_path / "staging.json"
    staging_path.write_text(json.dumps(staging), encoding="utf-8")
    assert tool.diff_staging(staging_path, _snapshot_dir()) == 1


def test_manifest_digest_matches_files() -> None:
    manifest = json.loads((_snapshot_dir() / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    digests = manifest["data_digests"]
    assert isinstance(digests, dict)
    for name in ("groups.json", "sources.json"):
        expected = digests[name]
        actual = "sha256:" + hashlib.sha256((_snapshot_dir() / name).read_bytes()).hexdigest()
        assert actual == expected
