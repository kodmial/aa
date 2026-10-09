"""Offline snapshot loading with checksum validation and fail-closed errors."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from aa.meeting_directory.models import DirectorySnapshot
from aa.meeting_directory.validator import validate_snapshot_payload

logger = logging.getLogger("aa.meeting_directory")

SNAPSHOT_SUBPATH = Path("data") / "meeting_directory" / "ru"


class DirectoryUnavailableError(RuntimeError):
    """The directory snapshot cannot be used; navigation is unavailable."""


def default_snapshot_dir() -> Path:
    """Resolve the snapshot directory from a deterministic configured path.

    ``AA_MEETING_DIRECTORY_PATH`` overrides the location. Otherwise the
    directory is discovered from the repository/package path that contains
    this module, so behavior does not depend on the developer CWD inside
    the GitHub Actions runner.
    """
    override = os.environ.get("AA_MEETING_DIRECTORY_PATH", "").strip()
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / SNAPSHOT_SUBPATH
        if (candidate / "manifest.json").exists():
            return candidate
        # ``src`` layout: <root>/src/aa/meeting_directory/loader.py means
        # the data dir may sit two levels above ``src``.
        if parent.name == "src":
            candidate = parent.parent / SNAPSHOT_SUBPATH
            if (candidate / "manifest.json").exists():
                return candidate
    return Path.cwd() / SNAPSHOT_SUBPATH


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DirectoryUnavailableError(f"snapshot file missing: {path.name}") from exc
    except (OSError, ValueError) as exc:
        raise DirectoryUnavailableError(f"snapshot file unreadable: {path.name}") from exc


def _digest_file(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DirectoryUnavailableError(f"snapshot file missing: {path.name}") from exc
    return "sha256:" + hashlib.sha256(data).hexdigest()


def load_snapshot(snapshot_dir: Path | None = None) -> DirectorySnapshot:
    """Load and validate the snapshot entirely offline (no network)."""
    directory = snapshot_dir or default_snapshot_dir()
    raw_groups = _read_json(directory / "groups.json")
    raw_sources = _read_json(directory / "sources.json")
    raw_manifest = _read_json(directory / "manifest.json")
    if not isinstance(raw_manifest, dict):
        raise DirectoryUnavailableError("manifest payload must be a JSON object")
    digests = raw_manifest.get("data_digests")
    if isinstance(digests, dict):
        for filename in ("groups.json", "sources.json"):
            expected = digests.get(filename)
            if isinstance(expected, str) and expected.strip():
                actual = _digest_file(directory / filename)
                if actual != expected.strip():
                    raise DirectoryUnavailableError(f"snapshot checksum mismatch: {filename}")
    try:
        return validate_snapshot_payload(raw_groups, raw_sources, raw_manifest)
    except ValueError as exc:
        raise DirectoryUnavailableError(str(exc)) from exc


_cached_snapshot: DirectorySnapshot | None = None
_cached_error: str | None = None


def get_cached_snapshot() -> DirectorySnapshot:
    """Return the process-RAM snapshot loaded once at startup."""
    global _cached_snapshot
    global _cached_error
    if _cached_snapshot is not None:
        return _cached_snapshot
    if _cached_error is not None:
        raise DirectoryUnavailableError(_cached_error)
    try:
        _cached_snapshot = load_snapshot()
        return _cached_snapshot
    except DirectoryUnavailableError as exc:
        _cached_error = str(exc)
        logger.warning("meeting directory unavailable")
        raise


def reset_cache() -> None:
    """Clear the cached snapshot (tests only)."""
    global _cached_snapshot
    global _cached_error
    _cached_snapshot = None
    _cached_error = None
