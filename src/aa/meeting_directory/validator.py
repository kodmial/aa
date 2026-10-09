"""Validation for the meeting-directory snapshot: schema, links, privacy."""

from __future__ import annotations

import html
import logging
import re
from datetime import UTC, datetime
from urllib.parse import urlparse

from aa.meeting_directory.models import (
    DirectorySnapshot,
    GroupRecord,
    PlaceRecord,
    RegionLink,
    SlotRecord,
    SourceRecord,
)

logger = logging.getLogger("aa.meeting_directory")

SCHEMA_VERSION = 1
MAX_RECORDS = 50000
ALLOWED_HOSTS = ("aarussia.ru", "www.aarussia.ru")

# Privacy gate: patterns that must never be stored in the snapshot.
FORBIDDEN_PATTERNS = (
    "домофон",
    "код подъезда",
    "код_подъезда",
    "intercom",
    "парадная",
)


def sanitize_text(value: str) -> str:
    """Strip HTML/script markup from source-derived display text."""
    text = re.sub(r"<[^>]*>", "", value)
    text = html.unescape(text)
    return " ".join(text.split())


def is_allowed_url(url: str, extra_hosts: tuple[str, ...] = ()) -> bool:
    """Check an outgoing URL against the strict https allowlist."""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    if parsed.scheme != "https":
        return False
    if parsed.username or parsed.password:
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in ALLOWED_HOSTS or host in extra_hosts:
        return True
    return False


def _parse_optional_dt(raw: object) -> datetime | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"expected ISO datetime string, got {type(raw).__name__}")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid ISO datetime: {raw!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"datetime must carry a timezone offset: {raw!r}")
    return parsed.astimezone(UTC)


def _require_str(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"field {key!r} must be a non-empty string")
    return value


def validate_snapshot_payload(
    raw_groups: object,
    raw_sources: object,
    raw_manifest: object,
) -> DirectorySnapshot:
    """Validate raw JSON payloads and build the in-RAM snapshot.

    Raises ``ValueError`` with a bounded category message on any problem;
    callers translate this into a typed ``directory_unavailable`` outcome
    without disrupting normal AA conversation.
    """
    if not isinstance(raw_groups, dict):
        raise ValueError("groups payload must be a JSON object")
    if not isinstance(raw_sources, dict):
        raise ValueError("sources payload must be a JSON object")
    if not isinstance(raw_manifest, dict):
        raise ValueError("manifest payload must be a JSON object")

    schema_version = raw_manifest.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema_version: {schema_version!r}")
    if raw_groups.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported groups schema_version")
    if raw_sources.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported sources schema_version")
    version_raw = raw_manifest.get("snapshot_version")
    if not isinstance(version_raw, str) or not version_raw.strip():
        raise ValueError("manifest snapshot_version must be a non-empty string")
    version = version_raw.strip()
    digest_raw = raw_manifest.get("data_digests")
    digest = ""
    if isinstance(digest_raw, dict):
        groups_digest = digest_raw.get("groups.json")
        if isinstance(groups_digest, str):
            digest = groups_digest.strip()

    sources = _parse_sources(raw_sources)
    places = _parse_places(raw_groups)
    region_links = _parse_region_links(raw_groups)
    groups = _parse_groups(raw_groups)
    slots = _parse_slots(raw_groups)
    _check_identity_and_links(places, groups, slots, region_links, sources)
    _check_privacy_gate(groups, slots)

    total = len(places) + len(groups) + len(slots) + len(region_links)
    if total > MAX_RECORDS:
        raise ValueError(f"snapshot exceeds record bound ({total} rows)")
    logger.info(
        "meeting directory snapshot validated",
        extra={"groups": len(groups), "slots": len(slots), "places": len(places)},
    )
    return DirectorySnapshot(
        version=version,
        digest=digest,
        places=tuple(places),
        groups=tuple(groups),
        slots=tuple(slots),
        region_links=tuple(region_links),
        sources=tuple(sources),
    )


def _parse_sources(raw_sources: dict[str, object]) -> list[SourceRecord]:
    raw_list = raw_sources.get("sources")
    if not isinstance(raw_list, list) or not raw_list:
        raise ValueError("sources list must be non-empty")
    sources: list[SourceRecord] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            raise ValueError("source entry must be an object")
        source_id = _require_str(entry, "source_id")
        url = _require_str(entry, "url")
        if not is_allowed_url(url):
            raise ValueError(f"source {source_id!r} URL is not allowlisted")
        sources.append(
            SourceRecord(
                source_id=source_id,
                title=sanitize_text(str(entry.get("title", ""))),
                url=url.strip(),
                kind=str(entry.get("kind", "")),
                robots=str(entry.get("robots", "unknown")),
                terms=str(entry.get("terms", "unknown")),
                permission_status=str(entry.get("permission_status", "unknown")),
                checked_at=str(entry.get("checked_at", "")),
            )
        )
    return sources


def _parse_places(raw_groups: dict[str, object]) -> list[PlaceRecord]:
    raw_list = raw_groups.get("places")
    if not isinstance(raw_list, list):
        raise ValueError("places must be a list")
    places: list[PlaceRecord] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            raise ValueError("place entry must be an object")
        place_type = str(entry.get("place_type", ""))
        if place_type not in ("city", "town", "settlement", "region"):
            raise ValueError(f"unsupported place_type: {place_type!r}")
        coverage = str(entry.get("coverage_kind", ""))
        if coverage not in ("verified_groups", "directory_only"):
            raise ValueError(f"unsupported coverage_kind: {coverage!r}")
        aliases_raw = entry.get("aliases", [])
        if not isinstance(aliases_raw, list):
            raise ValueError("place aliases must be a list")
        aliases = tuple(sanitize_text(str(alias)) for alias in aliases_raw if str(alias).strip())
        region_id = entry.get("region_id")
        region_hint = entry.get("region_hint")
        places.append(
            PlaceRecord(
                place_id=_require_str(entry, "place_id"),
                display_name=sanitize_text(_require_str(entry, "display_name")),
                place_type=place_type,  # type: ignore[arg-type]
                region_id=str(region_id) if isinstance(region_id, str) else None,
                region_hint=str(region_hint) if isinstance(region_hint, str) else None,
                aliases=aliases,
                coverage_kind=coverage,  # type: ignore[arg-type]
                source_id=_require_str(entry, "source_id"),
            )
        )
    return places


def _parse_region_links(raw_groups: dict[str, object]) -> list[RegionLink]:
    raw_list = raw_groups.get("region_links")
    if not isinstance(raw_list, list):
        raise ValueError("region_links must be a list")
    links: list[RegionLink] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            raise ValueError("region link entry must be an object")
        url = _require_str(entry, "url")
        if not is_allowed_url(url):
            raise ValueError("region link URL is not allowlisted")
        links.append(
            RegionLink(
                link_id=_require_str(entry, "link_id"),
                region_id=_require_str(entry, "region_id"),
                region_name=sanitize_text(_require_str(entry, "region_name")),
                display_name=sanitize_text(_require_str(entry, "display_name")),
                url=url.strip(),
                source_id=_require_str(entry, "source_id"),
                support=sanitize_text(str(entry.get("support", ""))),
            )
        )
    return links


def _parse_groups(raw_groups: dict[str, object]) -> list[GroupRecord]:
    raw_list = raw_groups.get("groups")
    if not isinstance(raw_list, list):
        raise ValueError("groups must be a list")
    groups: list[GroupRecord] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            raise ValueError("group entry must be an object")
        group_format = str(entry.get("format", ""))
        if group_format not in ("in_person", "online", "hybrid"):
            raise ValueError(f"unsupported group format: {group_format!r}")
        access = str(entry.get("access", "unknown"))
        if access not in ("open", "closed", "unknown"):
            raise ValueError(f"unsupported access value: {access!r}")
        country = _require_str(entry, "country")
        if country != "RU":
            raise ValueError("directory scope is RU-only")
        venue = sanitize_text(str(entry.get("address") or entry.get("online_url") or ""))
        if not venue:
            raise ValueError("group must carry a public venue or online URL")
        region = entry.get("region")
        city = entry.get("city")
        place_id = entry.get("place_id")
        source_url = _require_str(entry, "source_url")
        if not is_allowed_url(source_url):
            raise ValueError("group source URL is not allowlisted")
        groups.append(
            GroupRecord(
                group_id=_require_str(entry, "group_id"),
                name=sanitize_text(_require_str(entry, "name")),
                country=country,
                region=sanitize_text(str(region)) if isinstance(region, str) else None,
                city=sanitize_text(str(city)) if isinstance(city, str) else None,
                place_id=str(place_id) if isinstance(place_id, str) else None,
                group_format=group_format,  # type: ignore[arg-type]
                venue_or_url=venue,
                timezone=str(entry.get("timezone", "unknown")),
                access=access,  # type: ignore[arg-type]
                source_id=_require_str(entry, "source_id"),
                source_url=source_url.strip(),
            )
        )
    return groups


def _parse_slots(raw_groups: dict[str, object]) -> list[SlotRecord]:
    raw_list = raw_groups.get("slots")
    if not isinstance(raw_list, list):
        raise ValueError("slots must be a list")
    slots: list[SlotRecord] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            raise ValueError("slot entry must be an object")
        recurrence = str(entry.get("recurrence", "unknown"))
        if recurrence not in ("daily", "weekly", "monthly_nth", "dated", "unknown"):
            raise ValueError(f"unsupported recurrence: {recurrence!r}")
        weekdays_raw = entry.get("weekdays", [])
        if not isinstance(weekdays_raw, list):
            raise ValueError("slot weekdays must be a list")
        weekdays: list[int] = []
        for day in weekdays_raw:
            if not isinstance(day, int) or day < 0 or day > 6:
                raise ValueError(f"invalid weekday: {day!r}")
            weekdays.append(day)
        if recurrence == "weekly" and not weekdays:
            raise ValueError("weekly slot must list weekdays")
        start_raw = entry.get("start_local")
        start_local = str(start_raw) if isinstance(start_raw, str) else None
        if start_local is not None and not re.fullmatch(r"[0-2]\d:[0-5]\d", start_local):
            raise ValueError(f"invalid start_local: {start_local!r}")
        end_raw = entry.get("end_local")
        end_local = str(end_raw) if isinstance(end_raw, str) else None
        if end_local is not None and not re.fullmatch(r"[0-2]\d:[0-5]\d", end_local):
            raise ValueError(f"invalid end_local: {end_local!r}")
        status = str(entry.get("parsing_status", "unparsed"))
        if status not in ("verified", "unparsed", "stale"):
            raise ValueError(f"unsupported parsing_status: {status!r}")
        slot_format = str(entry.get("format", "in_person"))
        if slot_format not in ("in_person", "online", "hybrid"):
            raise ValueError(f"unsupported slot format: {slot_format!r}")
        access = str(entry.get("access", "unknown"))
        if access not in ("open", "closed", "unknown"):
            raise ValueError(f"unsupported slot access: {access!r}")
        cancellations_raw = entry.get("cancellations", [])
        if not isinstance(cancellations_raw, list):
            raise ValueError("cancellations must be a list")
        cancellations = tuple(str(item) for item in cancellations_raw)
        for stamp in cancellations:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", stamp):
                raise ValueError(f"invalid cancellation date: {stamp!r}")
        date_raw = entry.get("date")
        if recurrence == "dated" and not isinstance(date_raw, str):
            raise ValueError("dated slot must carry a date")
        month_week = entry.get("month_week")
        month_weekday = entry.get("month_weekday")
        source_url = _require_str(entry, "source_url")
        if not is_allowed_url(source_url):
            raise ValueError("slot source URL is not allowlisted")
        slots.append(
            SlotRecord(
                slot_id=_require_str(entry, "slot_id"),
                group_id=_require_str(entry, "group_id"),
                slot_format=slot_format,  # type: ignore[arg-type]
                recurrence=recurrence,  # type: ignore[arg-type]
                weekdays=tuple(weekdays),
                start_local=start_local,
                end_local=end_local,
                timezone=str(entry.get("timezone", "unknown")),
                date=str(date_raw) if isinstance(date_raw, str) else None,
                month_week=int(month_week) if isinstance(month_week, int) else None,
                month_weekday=int(month_weekday) if isinstance(month_weekday, int) else None,
                valid_from=str(entry.get("valid_from"))
                if isinstance(entry.get("valid_from"), str)
                else None,
                valid_until=str(entry.get("valid_until"))
                if isinstance(entry.get("valid_until"), str)
                else None,
                cancellations=cancellations,
                access=access,  # type: ignore[arg-type]
                source_url=source_url.strip(),
                schedule_verified_at=_parse_optional_dt(entry.get("schedule_verified_at")),
                parsing_status=status,  # type: ignore[arg-type]
            )
        )
    return slots


def _check_identity_and_links(
    places: list[PlaceRecord],
    groups: list[GroupRecord],
    slots: list[SlotRecord],
    region_links: list[RegionLink],
    sources: list[SourceRecord],
) -> None:
    """Enforce stable identity and cross-reference integrity."""
    source_ids = {source.source_id for source in sources}
    place_ids = [place.place_id for place in places]
    if len(set(place_ids)) != len(place_ids):
        raise ValueError("duplicate place_id in snapshot")
    group_ids = [group.group_id for group in groups]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("duplicate group_id in snapshot")
    slot_ids = [slot.slot_id for slot in slots]
    if len(set(slot_ids)) != len(slot_ids):
        raise ValueError("duplicate slot_id in snapshot")
    link_ids = [link.link_id for link in region_links]
    if len(set(link_ids)) != len(link_ids):
        raise ValueError("duplicate link_id in snapshot")
    group_id_set = set(group_ids)
    place_id_set = set(place_ids)
    region_ids = {link.region_id for link in region_links}
    for group in groups:
        if group.source_id not in source_ids:
            raise ValueError(f"group {group.group_id!r} has unknown source")
        if group.place_id is not None and group.place_id not in place_id_set:
            raise ValueError(f"group {group.group_id!r} has unknown place")
    for slot in slots:
        if slot.group_id not in group_id_set:
            raise ValueError(f"slot {slot.slot_id!r} has unknown group")
    for place in places:
        if place.source_id not in source_ids:
            raise ValueError(f"place {place.place_id!r} has unknown source")
        if place.region_id is not None and place.region_id not in region_ids:
            raise ValueError(f"place {place.place_id!r} has unknown region")
    # Same group name in two places must not share one ID: IDs are checked
    # unique above; names may repeat across places without collapsing.


def _check_privacy_gate(groups: list[GroupRecord], slots: list[SlotRecord]) -> None:
    """Reject snapshots carrying access codes or personal data markers."""
    for group in groups:
        haystack = f"{group.name} {group.venue_or_url}".lower()
        for marker in FORBIDDEN_PATTERNS:
            if marker in haystack:
                raise ValueError(f"privacy gate: forbidden marker in {group.group_id!r}")
    for slot in slots:
        haystack = f"{slot.slot_id} {slot.source_url}".lower()
        for marker in FORBIDDEN_PATTERNS:
            if marker in haystack:
                raise ValueError(f"privacy gate: forbidden marker in {slot.slot_id!r}")
