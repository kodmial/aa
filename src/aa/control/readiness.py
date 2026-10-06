"""Authoritative runtime readiness control plane (issue #144).

The live runtime has an internal readiness barrier (OpenCode health,
Telegram bootstrap, long polling, bounded controller), but historically
no authoritative readiness state was published outside the long-running
GitHub Actions step. As a result ``/bot status`` could not distinguish
bootstrap from a poller that is actually ready for manual testing.

This module owns the privacy-safe machine-readable READY / STARTUP_FAILED
markers published to control issue #31:

- READY is emitted only after ``Application.start()`` has completed all
  of: local OpenCode health/readiness, Telegram ``getMe`` bootstrap,
  webhook/commands bootstrap, long-polling task start, and the bounded
  application controller start.
- STARTUP_FAILED is emitted when startup exits before READY, with a
  bounded failure category; the existing bounded recovery path (notably
  the OpenCode 429 runner-restart exit 75) is preserved.
- Markers carry only run id, exact main SHA, timestamp and campaign
  start ordinal. No token, username, chat id, user text, prompt, corpus
  text, model output or secret ever enters a marker.

A GitHub Actions step that is merely ``in_progress`` is NEVER readiness:
only a trusted READY marker for the exact active run counts as ready.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

CONTROL_ISSUE_NUMBER = 31

READY_MARKER_KIND = "aa-runtime-ready"
FAILED_MARKER_KIND = "aa-runtime-startup-failed"

_SHA_RE = re.compile(r"[0-9a-f]{40}")

_READY_RE = re.compile(
    r"<!--\s*aa-runtime-ready\s+"
    r"run=(?P<run>\d+)\s+"
    r"sha=(?P<sha>[0-9a-f]{40})\s+"
    r"ready_at=(?P<ready_at>\d+)\s+"
    r"ordinal=(?P<ordinal>\d+)"
    r"\s*-->",
)

_FAILED_RE = re.compile(
    r"<!--\s*aa-runtime-startup-failed\s+"
    r"run=(?P<run>\d+)\s+"
    r"sha=(?P<sha>[0-9a-f]{40})\s+"
    r"failed_at=(?P<failed_at>\d+)\s+"
    r"ordinal=(?P<ordinal>\d+)\s+"
    r"category=(?P<category>[a-z0-9-]+)"
    r"\s*-->",
)

# Bounded failure taxonomy for STARTUP_FAILED markers. Categories are
# slugs only (no messages, no tracebacks, no secrets).
FAILURE_CATEGORIES = frozenset(
    {
        "opencode-not-ready",
        "opencode-429",
        "opencode-startup",
        "telegram-auth",
        "telegram-bootstrap",
        "controller-start",
        "config-invalid",
        "corpus-unavailable",
        "unknown",
    }
)

# Authoritative poller states reported by `/bot status`.
POLLER_STARTING = "starting"
POLLER_READY = "ready"
POLLER_FAILED = "failed"
POLLER_STOPPED = "stopped"
POLLER_COMPLETED = "completed"
POLLER_IDLE = "idle"


@dataclass(frozen=True)
class ReadyMarker:
    """Parsed READY marker (ids/counters/timestamps only)."""

    run_id: int
    sha: str
    ready_at: int
    ordinal: int


@dataclass(frozen=True)
class StartupFailedMarker:
    """Parsed STARTUP_FAILED marker (ids/counters/category only)."""

    run_id: int
    sha: str
    failed_at: int
    ordinal: int
    category: str


@dataclass(frozen=True)
class RuntimeIdentity:
    """Identity of the current runtime run for marker publication."""

    run_id: int
    sha: str
    ordinal: int


def validate_sha(sha: str) -> str:
    """Normalize an exact 40-hex main SHA or raise."""
    normalized = (sha or "").strip().lower()
    if not _SHA_RE.fullmatch(normalized):
        raise ValueError("sha must be an exact 40-hex main SHA")
    return normalized


def validate_ordinal(ordinal: int) -> int:
    """Validate a campaign start ordinal (1..4)."""
    value = int(ordinal)
    if value < 1 or value > 4:
        raise ValueError("ordinal must be a campaign start 1..4")
    return value


def validate_category(category: str) -> str:
    """Validate a STARTUP_FAILED failure category slug."""
    normalized = (category or "").strip().lower()
    if normalized not in FAILURE_CATEGORIES:
        raise ValueError(f"unknown startup failure category: {category!r}")
    return normalized


def format_ready_marker(*, run_id: int, sha: str, ready_at: int, ordinal: int) -> str:
    """Return the machine-readable READY marker for control issue #31."""
    clean_sha = validate_sha(sha)
    clean_ordinal = validate_ordinal(ordinal)
    run = int(run_id)
    stamp = int(ready_at)
    if run <= 0 or stamp <= 0:
        raise ValueError("run_id and ready_at must be positive")
    return (
        f"<!-- {READY_MARKER_KIND} run={run} sha={clean_sha} "
        f"ready_at={stamp} ordinal={clean_ordinal} -->"
    )


def parse_ready_marker(body: str) -> ReadyMarker | None:
    """Parse one READY marker from a comment body, if present."""
    match = _READY_RE.search(body or "")
    if match is None:
        return None
    try:
        ordinal = validate_ordinal(int(match.group("ordinal")))
    except ValueError:
        return None
    return ReadyMarker(
        run_id=int(match.group("run")),
        sha=validate_sha(match.group("sha")),
        ready_at=int(match.group("ready_at")),
        ordinal=ordinal,
    )


def format_startup_failed_marker(
    *, run_id: int, sha: str, failed_at: int, ordinal: int, category: str
) -> str:
    """Return the machine-readable STARTUP_FAILED marker for issue #31."""
    clean_sha = validate_sha(sha)
    clean_ordinal = validate_ordinal(ordinal)
    clean_category = validate_category(category)
    run = int(run_id)
    stamp = int(failed_at)
    if run <= 0 or stamp <= 0:
        raise ValueError("run_id and failed_at must be positive")
    return (
        f"<!-- {FAILED_MARKER_KIND} run={run} sha={clean_sha} "
        f"failed_at={stamp} ordinal={clean_ordinal} category={clean_category} -->"
    )


def parse_startup_failed_marker(body: str) -> StartupFailedMarker | None:
    """Parse one STARTUP_FAILED marker from a comment body, if present."""
    match = _FAILED_RE.search(body or "")
    if match is None:
        return None
    try:
        ordinal = validate_ordinal(int(match.group("ordinal")))
        category = validate_category(match.group("category"))
    except ValueError:
        return None
    return StartupFailedMarker(
        run_id=int(match.group("run")),
        sha=validate_sha(match.group("sha")),
        failed_at=int(match.group("failed_at")),
        ordinal=ordinal,
        category=category,
    )


def find_ready_for_run(bodies: list[str], run_id: int) -> ReadyMarker | None:
    """Return the READY marker for ``run_id``, if any body carries it."""
    wanted = int(run_id)
    for body in bodies:
        parsed = parse_ready_marker(body)
        if parsed is not None and parsed.run_id == wanted:
            return parsed
    return None


def find_failure_for_run(bodies: list[str], run_id: int) -> StartupFailedMarker | None:
    """Return the STARTUP_FAILED marker for ``run_id``, if any."""
    wanted = int(run_id)
    for body in bodies:
        parsed = parse_startup_failed_marker(body)
        if parsed is not None and parsed.run_id == wanted:
            return parsed
    return None


def has_ready_for_run(bodies: list[str], run_id: int) -> bool:
    """Whether a trusted READY marker exists for ``run_id``."""
    return find_ready_for_run(bodies, run_id) is not None


def assert_marker_privacy_safe(marker: str) -> None:
    """Fail closed when a marker carries anything beyond its allowlist."""
    lowered = marker.lower()
    for probe in (
        "token",
        "secret",
        "username",
        "chat",
        "prompt",
        "corpus",
        "transcript",
        "password",
        "bearer",
    ):
        if probe in lowered:
            raise ValueError(f"readiness marker must not carry {probe!r}")


def resolve_poller_state(
    *,
    run_active: bool,
    run_conclusion: str | None,
    has_ready: bool,
    has_failed: bool,
) -> str:
    """Resolve the authoritative poller state for one runtime run.

    ``run_active`` mirrors the GitHub Actions notion (conclusion is null
    and status is queued/in_progress/waiting/requested/pending).
    Crucially, an active step alone resolves to ``starting``, never to
    ``ready``: only a trusted READY marker for the exact active run
    yields ``ready``. A STARTUP_FAILED marker (or a concluded run)
    yields ``failed``/``stopped``/``completed`` and never readiness.
    """
    if has_ready and run_active:
        return POLLER_READY
    if has_failed:
        return POLLER_FAILED
    if run_active:
        return POLLER_STARTING
    if run_conclusion in ("failure", "timed_out"):
        return POLLER_FAILED
    if run_conclusion == "cancelled":
        return POLLER_STOPPED
    if run_conclusion == "success":
        return POLLER_COMPLETED
    if run_conclusion is None:
        return POLLER_IDLE
    return POLLER_STOPPED


def is_usable_poller(*, run_active: bool, has_ready: bool) -> bool:
    """Whether the run is an active usable poller (READY only).

    Duplicate prevention is unchanged: callers must still suppress a
    second dispatch while ``run_active`` is true even when ``has_ready``
    is false (the ``starting`` window). This helper answers only the
    usability question (ready to test), never the dispatch question.
    """
    return bool(run_active and has_ready)


def _git_head_sha(repo_root: Path | None = None) -> str | None:
    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(root),
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    candidate = proc.stdout.strip().lower()
    if not _SHA_RE.fullmatch(candidate):
        return None
    return candidate


def resolve_runtime_identity(
    *,
    run_id: str | int | None = None,
    sha: str | None = None,
    ordinal: str | int | None = None,
    environ: dict[str, str] | None = None,
) -> RuntimeIdentity | None:
    """Resolve the current runtime identity from explicit args or env.

    Reads ``GITHUB_RUN_ID`` / ``AA_RUNTIME_RUN_ID``, ``GITHUB_SHA`` /
    ``AA_RUNTIME_SHA`` (falling back to ``git rev-parse HEAD``), and
    ``AA_CAMPAIGN_SEQ`` / ``INPUT_SEQ`` for the campaign start ordinal.
    Returns ``None`` when offline (no run id or no valid SHA), in which
    case marker publication is a no-op and tests stay hermetic.
    """
    source: dict[str, str] = dict(os.environ) if environ is None else dict(environ)
    raw_run = (
        str(run_id)
        if run_id is not None
        else (source.get("AA_RUNTIME_RUN_ID") or source.get("GITHUB_RUN_ID") or "")
    ).strip()
    raw_sha = (
        (sha or "").strip()
        or (source.get("AA_RUNTIME_SHA") or "").strip()
        or (source.get("GITHUB_SHA") or "").strip()
    )
    raw_ordinal = (
        str(ordinal)
        if ordinal is not None
        else (
            source.get("AA_CAMPAIGN_SEQ")
            or source.get("AA_CAMPAIGN_ORDINAL")
            or source.get("INPUT_SEQ")
            or "1"
        )
    ).strip()
    if not raw_run.isdigit() or int(raw_run) <= 0:
        return None
    clean_sha = raw_sha.strip().lower()
    if not _SHA_RE.fullmatch(clean_sha):
        fallback = _git_head_sha()
        if fallback is None:
            return None
        clean_sha = fallback
    try:
        clean_ordinal = validate_ordinal(int(raw_ordinal))
    except (ValueError, TypeError):
        clean_ordinal = 1
    return RuntimeIdentity(run_id=int(raw_run), sha=clean_sha, ordinal=clean_ordinal)


def current_timestamp_seconds() -> int:
    """Return the current epoch seconds for READY/FAILED markers."""
    return int(time.time())


__all__ = [
    "CONTROL_ISSUE_NUMBER",
    "FAILED_MARKER_KIND",
    "FAILURE_CATEGORIES",
    "POLLER_COMPLETED",
    "POLLER_FAILED",
    "POLLER_IDLE",
    "POLLER_READY",
    "POLLER_STARTING",
    "POLLER_STOPPED",
    "READY_MARKER_KIND",
    "ReadyMarker",
    "RuntimeIdentity",
    "StartupFailedMarker",
    "assert_marker_privacy_safe",
    "current_timestamp_seconds",
    "find_failure_for_run",
    "find_ready_for_run",
    "format_ready_marker",
    "format_startup_failed_marker",
    "has_ready_for_run",
    "is_usable_poller",
    "parse_ready_marker",
    "parse_startup_failed_marker",
    "resolve_poller_state",
    "resolve_runtime_identity",
    "validate_category",
    "validate_ordinal",
    "validate_sha",
]
