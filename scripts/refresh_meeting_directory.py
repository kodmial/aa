"""Offline maintenance for the Russia-only AA meeting directory (issue #316).

Maintenance-only importer/diff generator. Never runs in the runtime chat
path or on every bot restart. Default mode validates the committed
snapshot offline; ``--stage-from`` validates a staging payload and prints
a reviewable additions/changes/removals diff without overwriting main.
No destructive bulk-delete on fetch/parser failure: a failed parse keeps
the last-known-good snapshot untouched.

Usage:
    python3 scripts/refresh_meeting_directory.py --check
    python3 scripts/refresh_meeting_directory.py --diff --stage-from /tmp/staging.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = REPO_ROOT / "data" / "meeting_directory" / "ru"

sys.path.insert(0, str(REPO_ROOT / "src"))

from aa.meeting_directory.validator import (  # noqa: E402
    FORBIDDEN_PATTERNS,
    is_allowed_url,
    validate_snapshot_payload,
)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def check_snapshot(snapshot_dir: Path = SNAPSHOT_DIR) -> int:
    """Validate the committed snapshot and verify manifest digests."""
    try:
        raw_groups = _read_json(snapshot_dir / "groups.json")
        raw_sources = _read_json(snapshot_dir / "sources.json")
        raw_manifest = _read_json(snapshot_dir / "manifest.json")
    except (OSError, ValueError) as exc:
        print(f"Invalid snapshot JSON: {exc}", file=sys.stderr)
        return 1
    if not isinstance(raw_manifest, dict):
        print("Invalid manifest payload", file=sys.stderr)
        return 1
    digests = raw_manifest.get("data_digests")
    if isinstance(digests, dict):
        for filename in ("groups.json", "sources.json"):
            expected = digests.get(filename)
            if isinstance(expected, str) and expected.strip():
                actual = (
                    "sha256:" + hashlib.sha256((snapshot_dir / filename).read_bytes()).hexdigest()
                )
                if actual != expected.strip():
                    print(f"Checksum mismatch: {filename}", file=sys.stderr)
                    return 1
    try:
        snapshot = validate_snapshot_payload(raw_groups, raw_sources, raw_manifest)
    except ValueError as exc:
        print(f"Snapshot validation failed: {exc}", file=sys.stderr)
        return 1
    stale_sources = [
        source.source_id
        for source in snapshot.sources
        if source.robots == "unknown" or source.terms == "unknown"
    ]
    print(
        f"ok version={snapshot.version} "
        f"places={len(snapshot.places)} groups={len(snapshot.groups)} "
        f"slots={len(snapshot.slots)} links={len(snapshot.region_links)} "
        f"sources={len(snapshot.sources)} "
        f"rights_unknown={len(stale_sources)}"
    )
    return 0


def _index_groups(raw_groups: dict[str, object]) -> dict[str, dict[str, object]]:
    entries = raw_groups.get("groups")
    if not isinstance(entries, list):
        return {}
    index: dict[str, dict[str, object]] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("group_id"), str):
            index[str(entry["group_id"])] = entry
    return index


def _index_slots(raw_groups: dict[str, object]) -> dict[str, dict[str, object]]:
    entries = raw_groups.get("slots")
    if not isinstance(entries, list):
        return {}
    index: dict[str, dict[str, object]] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("slot_id"), str):
            index[str(entry["slot_id"])] = entry
    return index


def diff_staging(staging_path: Path, snapshot_dir: Path = SNAPSHOT_DIR) -> int:
    """Diff a staging payload against the committed snapshot (read-only)."""
    try:
        raw_current = _read_json(snapshot_dir / "groups.json")
        raw_staging = _read_json(staging_path)
    except (OSError, ValueError) as exc:
        print(f"Cannot read payloads: {exc}", file=sys.stderr)
        return 1
    if not isinstance(raw_current, dict) or not isinstance(raw_staging, dict):
        print("Payloads must be JSON objects", file=sys.stderr)
        return 1
    # Privacy gate on staging content before any further review.
    try:
        blob = json.dumps(raw_staging, ensure_ascii=False).lower()
    except (TypeError, ValueError) as exc:
        print(f"Cannot serialize staging payload: {exc}", file=sys.stderr)
        return 1
    for marker in FORBIDDEN_PATTERNS:
        if marker in blob:
            print(f"Staging rejected by privacy gate: {marker!r}", file=sys.stderr)
            return 1
    current_groups = _index_groups(raw_current)
    staging_groups = _index_groups(raw_staging)
    current_slots = _index_slots(raw_current)
    staging_slots = _index_slots(raw_staging)
    added_groups = sorted(set(staging_groups) - set(current_groups))
    removed_groups = sorted(set(current_groups) - set(staging_groups))
    changed_groups = sorted(
        key
        for key in set(staging_groups) & set(current_groups)
        if staging_groups[key] != current_groups[key]
    )
    added_slots = sorted(set(staging_slots) - set(current_slots))
    removed_slots = sorted(set(current_slots) - set(staging_slots))
    changed_slots = sorted(
        key
        for key in set(staging_slots) & set(current_slots)
        if staging_slots[key] != current_slots[key]
    )
    if removed_groups or removed_slots:
        print(
            "Refusing destructive bulk view: removals require explicit human review; "
            "last-known-good snapshot is preserved."
        )
    report = {
        "added_groups": added_groups,
        "changed_groups": changed_groups,
        "removed_groups_needing_review": removed_groups,
        "added_slots": added_slots,
        "changed_slots": changed_slots,
        "removed_slots_needing_review": removed_slots,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    # Validate outgoing URLs in staging additions without fetching anything.
    bad_urls: list[str] = []
    for key in [*added_groups, *changed_groups]:
        for field in ("source_url", "online_url", "address"):
            value = staging_groups[key].get(field)
            if isinstance(value, str) and value.startswith("http") and not is_allowed_url(value):
                bad_urls.append(f"{key}.{field}")
    for key in [*added_slots, *changed_slots]:
        value = staging_slots[key].get("source_url")
        if isinstance(value, str) and value.startswith("http") and not is_allowed_url(value):
            bad_urls.append(f"{key}.source_url")
    if bad_urls:
        print(f"Staging has non-allowlisted URLs: {sorted(bad_urls)}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the maintenance CLI parser."""
    parser = argparse.ArgumentParser(description="Meeting directory maintenance.")
    parser.add_argument("--check", action="store_true", help="Validate the snapshot.")
    parser.add_argument("--diff", action="store_true", help="Diff staging vs snapshot.")
    parser.add_argument("--stage-from", default="", help="Staging JSON payload path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (maintenance only, never the runtime chat path)."""
    args = build_parser().parse_args(argv)
    if args.diff:
        if not args.stage_from:
            print("--diff requires --stage-from PATH", file=sys.stderr)
            return 2
        return diff_staging(Path(args.stage_from))
    return check_snapshot()


if __name__ == "__main__":
    raise SystemExit(main())
