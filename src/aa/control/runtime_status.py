"""Durable runtime readiness marker (issue #146, Gate D).

The production Telegram runtime publishes exactly one durable phase
marker per run::

    STARTING -> READY -> STOPPED | FAILED

Each marker carries the exact run id and main SHA. ``/bot status`` reports
that marker, never a generic workflow ``in_progress`` string.

Helpers here are pure and privacy-safe (run ids, SHAs, phases and
timestamps only; never message text, prompts, corpus text or credentials).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

PHASES: tuple[str, ...] = ("STARTING", "READY", "STOPPED", "FAILED")
TERMINAL_PHASES: tuple[str, ...] = ("STOPPED", "FAILED")

MARKER_KIND = "aa-runtime-status"
SCHEMA_VERSION = "aa-runtime-status/1"

_SHA_RE = re.compile(r"[0-9a-f]{40}")


class RuntimeStatusError(ValueError):
    """Raised when runtime status invariants fail (fail-closed)."""


@dataclass(frozen=True)
class RuntimeMarker:
    """Durable readiness marker for one runtime run."""

    run_id: str
    sha: str
    phase: str
    timestamp_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": MARKER_KIND,
            "run_id": self.run_id,
            "sha": self.sha,
            "phase": self.phase,
            "timestamp_s": self.timestamp_s,
        }


def validate_run_id(run_id: str) -> str:
    """Normalize a run id token or raise fail-closed."""
    normalized = (run_id or "").strip()
    if not normalized or any(c.isspace() for c in normalized):
        raise RuntimeStatusError("run_id must be a non-empty token")
    return normalized


def validate_sha(sha: str) -> str:
    """Normalize an exact 40-hex SHA or raise fail-closed."""
    normalized = (sha or "").strip().lower()
    if not _SHA_RE.fullmatch(normalized):
        raise RuntimeStatusError("sha must be an exact 40-hex main SHA")
    return normalized


def validate_phase(phase: str) -> str:
    """Normalize a lifecycle phase or raise fail-closed."""
    normalized = (phase or "").strip().upper()
    if normalized not in PHASES:
        raise RuntimeStatusError(f"unknown runtime phase {phase!r}")
    return normalized


def transition_allowed(previous: str | None, nxt: str) -> bool:
    """Return whether ``previous -> nxt`` is a legal lifecycle transition."""
    target = validate_phase(nxt)
    if previous is None:
        return target == "STARTING"
    current = validate_phase(previous)
    if current == "STARTING":
        return target in ("READY", "FAILED")
    if current == "READY":
        return target in ("STOPPED", "FAILED")
    # Terminal phases never transition again within one run.
    return False


def format_marker(marker: RuntimeMarker) -> str:
    """Format one marker as a machine-readable HTML comment."""
    validate_run_id(marker.run_id)
    validate_sha(marker.sha)
    validate_phase(marker.phase)
    return f"<!-- {MARKER_KIND} run={marker.run_id} sha={marker.sha} phase={marker.phase} -->"


def parse_marker(body: str) -> RuntimeMarker | None:
    """Parse one marker from a comment body, if present."""
    match = re.search(
        r"<!--\s*aa-runtime-status\s+"
        r"run=(?P<run>\S+)\s+"
        r"sha=(?P<sha>[0-9a-f]{40})\s+"
        r"phase=(?P<phase>STARTING|READY|STOPPED|FAILED)\s*-->",
        body or "",
    )
    if match is None:
        return None
    return RuntimeMarker(
        run_id=match.group("run"),
        sha=match.group("sha"),
        phase=match.group("phase"),
        timestamp_s=0.0,
    )


def status_report(marker: RuntimeMarker | None, *, run_id: str = "") -> str:
    """Render the authoritative ``/bot status`` body from the marker.

    The report always names the durable phase with the exact run id/SHA.
    It never substitutes a generic workflow ``in_progress`` string.
    """
    if marker is None:
        return (
            "No authoritative runtime marker. No Telegram poller is READY. "
            "Post `/run` (owner only) after exact-main qualification PASS."
        )
    return (
        f"Authoritative runtime marker: phase={marker.phase} "
        f"run={marker.run_id} sha={marker.sha[:12]}. "
        + (
            "Telegram long polling is READY."
            if marker.phase == "READY"
            else ("Telegram long polling is not READY; manual testing must wait for READY.")
        )
        + (f" (request {run_id})" if run_id.strip() else "")
    )


def write_marker(path: Path, marker: RuntimeMarker) -> None:
    """Persist one marker as JSON (durable artifact for workflows)."""
    payload = dict(marker.to_dict())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read_marker(path: Path) -> RuntimeMarker:
    """Read one durable marker file or raise fail-closed."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeStatusError(f"missing runtime marker: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeStatusError(f"runtime marker is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeStatusError("runtime marker must be a JSON object")
    try:
        return RuntimeMarker(
            run_id=validate_run_id(str(payload.get("run_id", ""))),
            sha=validate_sha(str(payload.get("sha", ""))),
            phase=validate_phase(str(payload.get("phase", ""))),
            timestamp_s=float(payload.get("timestamp_s", 0.0)),
        )
    except (ValueError, TypeError) as exc:
        raise RuntimeStatusError(f"runtime marker is invalid: {exc}") from exc


def now_s() -> float:
    """Return the current epoch seconds (single seam for tests)."""
    return time.time()


__all__ = [
    "MARKER_KIND",
    "PHASES",
    "SCHEMA_VERSION",
    "TERMINAL_PHASES",
    "RuntimeMarker",
    "RuntimeStatusError",
    "format_marker",
    "now_s",
    "parse_marker",
    "read_marker",
    "status_report",
    "transition_allowed",
    "validate_phase",
    "validate_run_id",
    "validate_sha",
    "write_marker",
]
