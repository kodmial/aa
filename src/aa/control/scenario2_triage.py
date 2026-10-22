"""Deterministic scenario #2 diagnostic triage decisions (kodmial/aa#425).

Pure, network-free decision core for the AA-owned post-run reconciliation
of the single Gate C scenario #2 diagnostic
(``.github/workflows/aa-gate-c-scenario2-diagnostic.yml``, built by
kodmial/aa#418/#424). The reconciler consumes only AA-owned evidence (the
diagnostic run inventory and that run's uploaded artifacts) and produces:

- truthful typed outcomes for operations issue kodmial/aa#419:
  ``DELIVERED``, ``ROOT_CAUSE_PROVED``, ``BOUNDARY_LOCALIZED_CAUSE_UNKNOWN``,
  ``HARNESS_FAILURE``, ``BOOTSTRAP_BLOCKED``, ``INCONCLUSIVE``,
  ``RUN_MISSING``;
- a privacy-safe factual timeline read from the sanitized artifact
  JSONL/schemas (whitelisted numeric/identifier fields only);
- explicit separation between a proven root cause and a last-seen
  boundary (a slow-but-progressing turn is NEVER called a hang);
- idempotent plans: ONE triage comment per ``(sha, run)`` on #419, ONE
  repair issue per evidence-harness fingerprint, ONE bounded-careful
  diagnostic ``workflow_dispatch`` (``target_sha`` only) when a run is
  missing.

No external LLM/ChatGPT judgement exists anywhere in this control plane,
and nothing here patches product semantic logic: escalation tasks repair
the diagnostic instrumentation only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Ownership / identity constants (AA-owned single-scenario diagnostic only)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "aa-scenario2-triage/1"
DIAGNOSTIC_WORKFLOW_NAME = "AA Gate C scenario2 diagnostic"
DIAGNOSTIC_WORKFLOW_FILE = "aa-gate-c-scenario2-diagnostic.yml"
DIAGNOSTIC_SCHEMA_VERSION = "aa-gate-c-scenario2-diagnostic/2"
TRIAGE_ISSUE_NUMBER = 419


def artifact_name_for_run(run_id: int | str) -> str:
    """Canonical AA-owned artifact upload name of the diagnostic run."""
    return f"aa-gate-c-scenario2-diagnostic-{run_id}-scenario2"


# Artifact members consumed by the reconciler. ``final-report.md`` is an
# expected pointer target; the other three are REQUIRED telemetry.
REQUIRED_RESULT_MEMBER = "result.json"
REQUIRED_SUPERVISOR_MEMBER = "supervisor-events.jsonl"
REQUIRED_SCENARIO_EVENTS_MEMBER = "probe-scenario2-events.jsonl"
REQUIRED_MEMBERS: tuple[str, ...] = (
    REQUIRED_RESULT_MEMBER,
    REQUIRED_SUPERVISOR_MEMBER,
    REQUIRED_SCENARIO_EVENTS_MEMBER,
)
OPTIONAL_REPORT_MEMBER = "final-report.md"

# Files that change what the diagnostic machinery measures. A diagnostic run
# built from an older commit is still accepted evidence for the current main
# head ONLY while none of these differ between the two trees.
MACHINERY_FILES: tuple[str, ...] = (
    ".github/workflows/aa-gate-c-scenario2-diagnostic.yml",
    "scripts/diagnose_gate_c_scenario2.py",
)

# ---------------------------------------------------------------------------
# Truthful typed outcomes (kodmial/aa#425 Definition of Done)
# ---------------------------------------------------------------------------

OUTCOME_DELIVERED = "DELIVERED"
OUTCOME_ROOT_CAUSE_PROVED = "ROOT_CAUSE_PROVED"
OUTCOME_BOUNDARY_LOCALIZED_CAUSE_UNKNOWN = "BOUNDARY_LOCALIZED_CAUSE_UNKNOWN"
OUTCOME_HARNESS_FAILURE = "HARNESS_FAILURE"
OUTCOME_BOOTSTRAP_BLOCKED = "BOOTSTRAP_BLOCKED"
OUTCOME_INCONCLUSIVE = "INCONCLUSIVE"
OUTCOME_RUN_MISSING = "RUN_MISSING"

ALL_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_DELIVERED,
        OUTCOME_ROOT_CAUSE_PROVED,
        OUTCOME_BOUNDARY_LOCALIZED_CAUSE_UNKNOWN,
        OUTCOME_HARNESS_FAILURE,
        OUTCOME_BOOTSTRAP_BLOCKED,
        OUTCOME_INCONCLUSIVE,
        OUTCOME_RUN_MISSING,
    }
)

# Blocked-evidence codes (typed; never a fabricated product PASS).
BLOCK_ARTIFACT_MISSING = "EVIDENCE_ARTIFACT_MISSING"
BLOCK_ARTIFACT_CORRUPT = "EVIDENCE_ARTIFACT_CORRUPT"
BLOCK_RESULT_MISSING = "EVIDENCE_RESULT_JSON_MISSING"
BLOCK_SCHEMA_INVALID = "EVIDENCE_RESULT_SCHEMA_INVALID"
BLOCK_IDENTITY_MISMATCH = "EVIDENCE_IDENTITY_MISMATCH"
BLOCK_TELEMETRY_MISSING = "EVIDENCE_TELEMETRY_MISSING"
BLOCK_SANITIZATION_REJECTED = "EVIDENCE_SANITIZATION_REJECTED"
BLOCK_HARNESS_TRUNCATED = "HARNESS_PROBE_TRUNCATED"
BLOCK_UNKNOWN_ARTIFACT_GROUP = "EVIDENCE_ARTIFACT_UNREADABLE"

# Diagnostic classifications produced by the scenario2 diagnostic itself.
CLASS_DELIVERED = "delivered"
CLASS_PROVIDER_ERROR_PROVEN = "provider-error-proven"
CLASS_SERVER_ERROR_PROVEN = "server-error-proven"
CLASS_HARNESS_ERROR = "harness-error"
CLASS_UNRESOLVED = "unresolved-at-campaign-end"
CLASS_BOUNDARY_LOCALIZED = "boundary-localized-root-unknown"
CLASS_STAGE_UNOBSERVED = "stage-unobserved"

# Legacy answer windows are EVIDENCE checkpoints only (kodmial/aa#424):
# crossing them with the turn undelivered is recorded, never judged a hang.
SLOW_WORK_THRESHOLDS_S: tuple[float, ...] = (45.0, 75.0, 130.0, 300.0, 900.0)

# GitHub Actions run statuses that mean the diagnostic is not finished yet.
ACTIVE_RUN_STATUSES: frozenset[str] = frozenset(
    {"queued", "in_progress", "waiting", "requested", "pending"}
)

# Dispatch-bound retry hygiene (idempotent backoff, never terminal pause).
MAX_DISPATCH_ATTEMPTS = 3
DISPATCH_RETRY_BACKOFF_S = 1800.0

MAX_TIMELINE_ENTRIES = 60
MAX_FIELD_CHARS = 96
MAX_JSONL_LINES = 20000

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# Deliberately noisy unparsed child snapshot events (already capped by the
# diagnostic itself) are excluded from the human timeline.
TIMELINE_SKIPPED_EVENTS: frozenset[str] = frozenset({"snapshot"})

# Source ordering for the deterministic timeline render.
TIMELINE_SOURCE_ORDER: tuple[str, ...] = (
    "supervisor",
    "probe-direct-model",
    "probe-scenario2",
)


# Whitelisted (already-sanitized by the diagnostic) detail fields per event
# kind. Unknown kinds and unknown fields NEVER travel into the report.
_EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "supervisor-start": ("sha", "policy", "effective_campaign_allowance_s"),
    "supervisor-child-spawn": ("probe", "has_group"),
    "supervisor-probe-start": ("probe", "campaign_time_left_s"),
    "supervisor-probe-end": (
        "probe",
        "status",
        "exit_code",
        "exited_by_signal",
        "terminated_at_campaign_end",
        "killed",
        "delivered",
        "snapshots",
        "active_stage",
        "last_completed_stage",
        "last_boundary_stage",
    ),
    "supervisor-campaign-budget-check": ("remaining_s",),
    "supervisor-report-published": ("target_issue",),
    "supervisor-stray-check": ("opencode_processes_leftover",),
    "campaign-end": ("probe", "elapsed_s", "delivered", "last_boundary_stage"),
    "server-health": ("ok", "elapsed_ms", "attached"),
    "server-health-phases": ("connect_ms", "write_ms", "read_ms", "status"),
    "session-create-start": ("probe",),
    "session-create-end": ("ok", "connect_ms", "write_ms", "read_ms", "status"),
    "model-send-start": ("probe", "model", "read_inactivity_s"),
    "model-send-end": (
        "probe",
        "ok",
        "provider_error",
        "elapsed_ms",
        "connect_ms",
        "write_ms",
        "read_ms",
        "status",
        "served_model",
        "fail_category",
    ),
    "model-in-flight-liveness": (
        "elapsed_ms",
        "server_healthy",
        "sessions_responsive",
        "sessions_seen",
        "messages_seen",
        "part_kinds",
    ),
    "probe-start": ("probe", "family"),
    "bootstrap-start": ("probe",),
    "bootstrap-end": ("ok", "elapsed_ms", "opencode_ready", "graph_running"),
    "scenario-start": (),
    "stage-start": ("stage",),
    "stage-end": ("stage", "elapsed_ms"),
    "delivery": ("elapsed_ms", "reply_len", "typing_sends"),
    "probe-end": ("probe", "status", "fail_category"),
    "scenario-end": ("family", "stage_values"),
    "observation-continues": (
        "elapsed_s",
        "threshold_s",
        "last_stage",
        "last_real_event",
        "seconds_since_last_real",
        "delivered",
    ),
    "server-nonresponsive-sustained": ("consecutive_failures", "last_health_ok_at_s"),
    "child-exit": ("probe", "exit_code", "cleanup_done"),
    "child-forced-exit": ("probe", "exit_code"),
    "child-unexpected": ("category",),
    "stack-handler-failed": (),
    "opencode-internal": (),
}


class TriageSanitizationError(ValueError):
    """A rendered artifact-derived text tripped a secret-bearing pattern."""


@dataclass(frozen=True)
class DiagnosticRun:
    """One AA-owned diagnostic workflow run (sanitized, Actions-shaped)."""

    run_id: int
    head_sha: str
    event: str
    status: str
    conclusion: str | None
    created_at: str | None
    updated_at: str | None
    run_number: int
    html_url: str = ""

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_RUN_STATUSES

    @property
    def is_completed(self) -> bool:
        return self.status == "completed" and self.conclusion is not None


@dataclass(frozen=True)
class SelectionDecision:
    """Outcome of selecting the run to reconcile against allowed SHAs."""

    action: str  # "reconcile" | "wait" | "missing"
    run: DiagnosticRun | None
    reason: str


@dataclass(frozen=True)
class TriageOutcome:
    """Typed, truthful verdict rendered to issue kodmial/aa#419."""

    outcome: str
    classification: str
    blocked_code: str | None = None
    missing_files: tuple[str, ...] = ()
    detail: str = ""

    @property
    def is_success(self) -> bool:
        return self.outcome == OUTCOME_DELIVERED

    @property
    def requires_repair_issue(self) -> bool:
        """Only proven harness/evidence gaps spawn one escalation issue."""
        return self.outcome == OUTCOME_HARNESS_FAILURE


@dataclass(frozen=True)
class TimelineEntry:
    """One privacy-safe timeline fact (identifier/numeric fields only)."""

    source: str
    mono: float
    event: str
    detail: str

    def render(self) -> str:
        stamp = f"{self.mono:8.1f}s"
        return f"- [`{self.source}` +{stamp}] `{self.event}`{self.detail}"


@dataclass(frozen=True)
class DispatchMarker:
    """Durable in-#419 record of one diagnostic dispatch attempt."""

    sha: str
    created_epoch: float
    attempt: int


@dataclass(frozen=True)
class DispatchPlan:
    """Decision whether to dispatch the existing diagnostic workflow."""

    should_dispatch: bool
    attempt: int
    reason: str
    marker_body: str = ""


@dataclass(frozen=True)
class CommentAction:
    """Idempotent #419 comment update/create decision."""

    kind: str  # "create" | "update" | "none"
    comment_id: int | None
    reason: str


@dataclass(frozen=True)
class RepairIssuePlan:
    """One-minimal-issue escalation plan for a proven harness/evidence gap."""

    should_create: bool
    fingerprint: str
    fingerprint_key: str
    title: str
    body: str
    labels: tuple[str, ...]
    reason: str


# ---------------------------------------------------------------------------
# Small parsing/formatting helpers
# ---------------------------------------------------------------------------


def parse_iso_epoch(value: str | None) -> float | None:
    if not value or not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.timestamp()


def format_elapsed(iso_start: str | None, iso_end: str | None) -> str:
    """Human wall-time between two Actions timestamps ('unknown' on gaps)."""
    start = parse_iso_epoch(iso_start)
    end = parse_iso_epoch(iso_end)
    if start is None or end is None or end < start:
        return "unknown"
    return f"{round(end - start, 1)}s"


def normalize_sha(value: Any) -> str:
    """Validate/normalize one 40-hex commit SHA (fail closed otherwise)."""
    sha = str(value or "").strip().lower()
    if not _SHA_RE.fullmatch(sha):
        raise ValueError("sha must be a 40-hex commit SHA")
    return sha


def normalize_allowed_shas(values: Sequence[Any]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for value in values:
        try:
            sha = normalize_sha(value)
        except ValueError:
            continue
        seen.setdefault(sha, None)
    return tuple(seen)


# ---------------------------------------------------------------------------
# Run inventory + selection
# ---------------------------------------------------------------------------


def parse_runs(payload: Any) -> list[DiagnosticRun]:
    """Parse unordered Actions workflow-run inventory into typed runs."""
    raw_items: Any = payload
    if isinstance(payload, Mapping):
        raw_items = payload.get("workflow_runs", [])
    if not isinstance(raw_items, Sequence):
        raise ValueError("run inventory must be a list of run objects")
    out: list[DiagnosticRun] = []
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        run_id = item.get("id")
        if not isinstance(run_id, int) or run_id <= 0:
            try:
                run_id = int(str(run_id))
            except (TypeError, ValueError):
                continue
        sha = str(item.get("head_sha", "")).lower()
        if not _SHA_RE.fullmatch(sha):
            continue
        conclusion = item.get("conclusion")
        if conclusion is not None and not isinstance(conclusion, str):
            conclusion = str(conclusion)
        run_number_raw = item.get("run_number", 0)
        out.append(
            DiagnosticRun(
                run_id=run_id,
                head_sha=sha,
                event=str(item.get("event", "")),
                status=str(item.get("status", "")),
                conclusion=conclusion,
                created_at=str(item.get("created_at") or "") or None,
                updated_at=str(item.get("updated_at") or "") or None,
                run_number=(int(run_number_raw) if isinstance(run_number_raw, int) else 0),
                html_url=str(item.get("html_url") or ""),
            )
        )
    return out


def _run_sort_key(run: DiagnosticRun) -> tuple[float, int]:
    return (parse_iso_epoch(run.created_at) or 0.0, run.run_id)


def select_diagnostic_run(
    runs: Sequence[DiagnosticRun], *, allowed_shas: Sequence[str]
) -> SelectionDecision:
    """Select the run to reconcile for the allowed exact SHAs.

    Order: any still-active diagnostic run for an allowed SHA wins a
    ``wait`` (never a duplicate dispatch); otherwise the newest completed
    run for an allowed SHA is selected; otherwise the run is missing.
    """
    allowed = normalize_allowed_shas(allowed_shas)
    if not allowed:
        return SelectionDecision("missing", None, "no allowed SHA given: cannot reconcile evidence")
    matching = [run for run in runs if run.head_sha in allowed]
    for run in sorted(matching, key=_run_sort_key, reverse=True):
        if run.is_active:
            return SelectionDecision(
                "wait",
                run,
                f"diagnostic run {run.run_id} still {run.status}; "
                "reconcile once it completes (never dispatch a duplicate)",
            )
    for run in sorted(matching, key=_run_sort_key, reverse=True):
        if run.is_completed:
            return SelectionDecision(
                "reconcile",
                run,
                f"newest completed diagnostic run for allowed SHA: run {run.run_id} "
                f"({run.conclusion})",
            )
    return SelectionDecision(
        "missing",
        None,
        "no completed or active diagnostic run for any allowed SHA",
    )


# ---------------------------------------------------------------------------
# Typed outcome mapping (never fabricates a PASS)
# ---------------------------------------------------------------------------


def _bootstrap_from_probes(result: Mapping[str, Any]) -> bool:
    """True when a probe died before the turn could be observed."""
    for key in ("probe_direct_model", "probe_scenario2"):
        probe = result.get(key)
        if not isinstance(probe, Mapping):
            continue
        status = str(probe.get("status", ""))
        if status == "bootstrap-failed" or status.startswith("bootstrap-"):
            return True
    return False


def map_typed_outcome(
    result: Mapping[str, Any] | None,
    *,
    missing_files: Sequence[str] = (),
    blocked_code: str | None = None,
) -> TriageOutcome:
    """Map the diagnostic verdict onto a typed reconciler outcome.

    Precedence: evidence/harness defects outrank any diagnostic verdict —
    a run whose required telemetry is missing/expired/corrupt is NEVER a
    product success.
    """
    missing = tuple(missing_files)
    if missing:
        return TriageOutcome(
            OUTCOME_HARNESS_FAILURE,
            "",
            blocked_code=BLOCK_TELEMETRY_MISSING,
            missing_files=missing,
            detail="required AA-owned telemetry files absent from the artifact",
        )
    if blocked_code is not None:
        return TriageOutcome(
            OUTCOME_HARNESS_FAILURE,
            "",
            blocked_code=blocked_code,
            detail="artifact/evidence defect; diagnostic verdict not evaluable",
        )
    if not isinstance(result, Mapping):
        return TriageOutcome(
            OUTCOME_HARNESS_FAILURE,
            "",
            blocked_code=BLOCK_SCHEMA_INVALID,
            detail="result.json missing or not a JSON object",
        )
    classification = str(result.get("classification", ""))
    # A probe that died before the scenario could be observed is a bootstrap
    # block regardless of the leftover classification text.
    if _bootstrap_from_probes(result) and classification != CLASS_DELIVERED:
        return TriageOutcome(
            OUTCOME_BOOTSTRAP_BLOCKED,
            classification,
            detail="a probe failed before the scenario turn could be observed",
        )
    if not classification:
        unknown_schema = TriageOutcome(
            OUTCOME_HARNESS_FAILURE,
            "",
            blocked_code=BLOCK_SCHEMA_INVALID,
            detail="result.json lacks a classification field",
        )
        return unknown_schema
    if classification == CLASS_DELIVERED:
        return TriageOutcome(OUTCOME_DELIVERED, classification)
    if classification in (CLASS_PROVIDER_ERROR_PROVEN, CLASS_SERVER_ERROR_PROVEN):
        return TriageOutcome(
            OUTCOME_ROOT_CAUSE_PROVED,
            classification,
            detail="concrete error evidence recorded by the diagnostic itself",
        )
    if classification == CLASS_HARNESS_ERROR:
        return TriageOutcome(
            OUTCOME_HARNESS_FAILURE,
            classification,
            blocked_code=BLOCK_HARNESS_TRUNCATED,
            detail="the diagnostic harness truncated its own probes(s)",
        )
    if classification == CLASS_UNRESOLVED:
        return TriageOutcome(
            OUTCOME_INCONCLUSIVE,
            classification,
            detail=(
                "campaign window ended live-but-uncertain: evidence cannot "
                "yet discriminate slow productive work from a stall"
            ),
        )
    if classification in (CLASS_BOUNDARY_LOCALIZED, CLASS_STAGE_UNOBSERVED):
        return TriageOutcome(
            OUTCOME_BOUNDARY_LOCALIZED_CAUSE_UNKNOWN,
            classification,
            detail=(
                "BOUNDARY LOCALIZED / PROVIDER CAUSE UNPROVEN — the last "
                "observed boundary is recorded; no component is attributed"
            ),
        )
    # Unknown classification: truthful unknown, never synthetic success.
    return TriageOutcome(
        OUTCOME_INCONCLUSIVE,
        classification,
        detail=f"unrecognized classification {classification!r}; refuses attribution",
    )


def validate_result_identity(result: Mapping[str, Any], run: DiagnosticRun) -> str | None:
    """Cross-check result.json identity against the selected Actions run."""
    if not isinstance(result, Mapping):
        return "result.json is not an object"
    run_id = result.get("run_id")
    if str(run_id) != str(run.run_id):
        return f"result.json run_id {run_id!r} != selected run {run.run_id}"
    sha = str(result.get("main_sha", "")).lower()
    if sha and sha != run.head_sha:
        return f"result.json main_sha {sha} != run head_sha {run.head_sha}"
    schema = str(result.get("schema", ""))
    if schema and schema != DIAGNOSTIC_SCHEMA_VERSION:
        return f"unexpected result.json schema {schema!r}"
    return None


def slow_work_crossings(result: Mapping[str, Any]) -> tuple[str, ...]:
    """Record that observed slow work crossed legacy windows and DELIVERED."""
    friendly = {
        "probe_scenario2": "scenario #2 turn",
        "probe_direct_model": "direct baseline call",
    }
    crossings: list[str] = []
    for probe_key, elapsed_key in (
        ("probe_scenario2", "delivered_elapsed_ms"),
        ("probe_direct_model", "model_request_total_ms"),
    ):
        probe = result.get(probe_key)
        if not isinstance(probe, Mapping):
            continue
        if probe.get("delivered") is not True and probe_key == "probe_scenario2":
            continue
        elapsed_ms = probe.get(elapsed_key)
        if not isinstance(elapsed_ms, (int, float)) or elapsed_ms <= 0:
            continue
        elapsed_s = elapsed_ms / 1000.0
        crossed = [
            f"{threshold:.0f}s" for threshold in SLOW_WORK_THRESHOLDS_S if elapsed_s > threshold
        ]
        if crossed:
            crossings.append(
                f"{friendly[probe_key]} finished OK after {round(elapsed_s, 1)}s — "
                f"crossed legacy {', '.join(crossed)} windows: NORMAL OBSERVED SLOW "
                "WORK, NOT a provider hang (no per-answer cutoff exists, "
                "kodmial/aa#424)"
            )
    return tuple(crossings)


# ---------------------------------------------------------------------------
# Sanitized artifact parsing
# ---------------------------------------------------------------------------


def parse_jsonl(raw: bytes | str, *, max_lines: int = MAX_JSONL_LINES) -> list[dict[str, Any]]:
    """Parse sanitized JSONL evidence, skipping malformed lines."""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            decoded = json.loads(stripped)
        except ValueError:
            continue
        if isinstance(decoded, dict):
            events.append(decoded)
        if len(events) >= max_lines:
            break
    return events


def _format_detail(event: str, payload: Mapping[str, Any]) -> str:
    allowed = _EVENT_FIELDS.get(event, ())
    parts: list[str] = []
    for field_name in allowed:
        if field_name not in payload:
            continue
        value = payload[field_name]
        if isinstance(value, (bool, int, float)) or value is None:
            rendered = "true" if value is True else "false" if value is False else str(value)
        elif isinstance(value, str):
            rendered = value.strip()[:MAX_FIELD_CHARS]
        else:
            rendered = "<skipped>"
        if rendered == "" or rendered == "None":
            continue
        parts.append(f"{field_name}={rendered}")
    return (" " + " ".join(parts)) if parts else ""


def build_timeline(
    events_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_entries: int = MAX_TIMELINE_ENTRIES,
) -> list[TimelineEntry]:
    """Build the deterministic sanitized timeline across evidence sources."""
    entries: list[TimelineEntry] = []
    ordered_sources = [source for source in TIMELINE_SOURCE_ORDER if source in events_by_source]
    ordered_sources += [
        source for source in events_by_source if source not in TIMELINE_SOURCE_ORDER
    ]
    for source in ordered_sources:
        raw_events = events_by_source[source]
        collected: list[TimelineEntry] = []
        for payload in raw_events:
            if not isinstance(payload, Mapping):
                continue
            event = str(payload.get("event", ""))
            if not event or event in TIMELINE_SKIPPED_EVENTS:
                continue
            mono = payload.get("mono")
            mono_value = float(mono) if isinstance(mono, (int, float)) else 0.0
            collected.append(
                TimelineEntry(
                    source=source,
                    mono=mono_value,
                    event=event,
                    detail=_format_detail(event, payload),
                )
            )
        collected.sort(key=lambda entry: (entry.mono, entry.event, entry.detail))
        entries.extend(collected)
    return entries[:max_entries]


def render_timeline(entries: Sequence[TimelineEntry]) -> list[str]:
    return [entry.render() for entry in entries]


# ---------------------------------------------------------------------------
# Sanitization guard for anything rendered to GitHub
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"gh[pousr]_[A-Za-z0-9_]{20,}",
        r"github_pat_[A-Za-z0-9_]{20,}",
        r"AKIA[0-9A-Z]{16}",
        r"xox[baprs]-[A-Za-z0-9-]{10,}",
        r"(?i)authorization[: ]*bearer\s+[A-Za-z0-9\-._~+/=]{10,}",
        r"(?i)[?&](key|token|access_token|api_key)=[^&\s]{8,}",
        (
            r"(?<![A-Za-z0-9_-])(?:bot)?\d{8,10}:"
            r"(?=[A-Za-z0-9_-]{35,}(?![A-Za-z0-9_-]))"
            r"(?=[A-Za-z0-9_-]*[A-Z])(?=[A-Za-z0-9_-]*[a-z])"
            r"[A-Za-z0-9_-]{35,}(?![A-Za-z0-9_-])"
        ),
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    )
)


def assert_sanitized(text: str, *, owner: str = "triage") -> None:
    """Reject secret-shaped content before anything reaches GitHub."""
    for pattern in _SENSITIVE_PATTERNS:
        match = pattern.search(text)
        if match:
            raise TriageSanitizationError(
                f"{owner}: secret-shaped content rejected ({pattern.pattern})"
            )


# ---------------------------------------------------------------------------
# Issue comment identity + idempotency
# ---------------------------------------------------------------------------


def triage_marker(sha: str, run_id: int | str) -> str:
    """Deterministic ONE-per-run marker on issue kodmial/aa#419."""
    return f"<!-- aa-scenario2-triage sha={sha} run={run_id} -->"


def parse_triage_markers(bodies: Sequence[str]) -> list[tuple[str, str]]:
    """Return every (sha, run_id) already triaged on the issue."""
    out: list[tuple[str, str]] = []
    pattern = re.compile(r"<!--\s*aa-scenario2-triage\s+sha=([0-9a-f]{40})\s+run=([^\s>]+)\s*-->")
    for body in bodies:
        match = pattern.search(body or "")
        if match:
            out.append((match.group(1), match.group(2)))
    return out


def decide_triage_comment(
    comments: Sequence[tuple[int, str]], *, sha: str, run_id: int | str
) -> CommentAction:
    """Idempotent action for the single per-run triage comment."""
    marker = triage_marker(sha, run_id)
    matching = [
        (comment_id, body)
        for comment_id, body in comments
        if (body or "").strip().startswith(marker)
    ]
    if not matching:
        return CommentAction("create", None, "no triage comment for this (sha, run) yet")
    # Retried runs update the same comment in place; never duplicate.
    target = min(matching, key=lambda item: item[0])
    extra = len(matching) - 1
    return CommentAction(
        "update",
        target[0],
        "triage comment already exists: update in place"
        + (f" ({extra} retried duplicate(s) tolerated)" if extra else ""),
    )


# ---------------------------------------------------------------------------
# Root-cause/boundary rendering
# ---------------------------------------------------------------------------

ROOT_STATUS_BY_OUTCOME: dict[str, str] = {
    OUTCOME_DELIVERED: (
        "no fault proven: the scenario delivered with real recorded evidence; "
        "observed slow completions are normal productive work, not a hang"
    ),
    OUTCOME_ROOT_CAUSE_PROVED: (
        "PROVEN from a concrete error payload/trace recorded by the "
        "diagnostic itself (provider or local-server layer)"
    ),
    OUTCOME_BOUNDARY_LOCALIZED_CAUSE_UNKNOWN: (
        "BOUNDARY LOCALIZED / PROVIDER CAUSE UNPROVEN: only the last "
        "observed boundary is recorded; no component or provider fault is "
        "claimed without evidence from its own layer"
    ),
    OUTCOME_HARNESS_FAILURE: (
        "instrumentation/harness defect: this run cannot prove a product "
        "root cause; one evidence-repair escalation is required"
    ),
    OUTCOME_BOOTSTRAP_BLOCKED: (
        "a probe never started observing the turn; local harness/startup boundary, root unproven"
    ),
    OUTCOME_INCONCLUSIVE: (
        "inconclusive by design-boundary: the recorded evidence does not "
        "yet discriminate slow productive work from a stall; treated as "
        "NOT done, never as a success"
    ),
    OUTCOME_RUN_MISSING: ("no diagnostic evidence exists for the exact SHA under test"),
}

_DONE_BY_OUTCOME = {
    OUTCOME_DELIVERED: True,
    OUTCOME_ROOT_CAUSE_PROVED: True,
    OUTCOME_BOUNDARY_LOCALIZED_CAUSE_UNKNOWN: False,
    OUTCOME_HARNESS_FAILURE: False,
    OUTCOME_BOOTSTRAP_BLOCKED: False,
    OUTCOME_INCONCLUSIVE: False,
    OUTCOME_RUN_MISSING: False,
}


def is_done(outcome: str) -> bool:
    """Typed truthfulness: only DELIVERED ever counts done-for-that-run."""
    return outcome == OUTCOME_DELIVERED


def product_gate_fulfilled(outcome: str) -> bool:
    """A green Actions workflow with an inconclusive result is NOT done."""
    return bool(_DONE_BY_OUTCOME.get(outcome, False))


# ---------------------------------------------------------------------------
# Comment rendering
# ---------------------------------------------------------------------------


def build_triage_comment(
    *,
    run: DiagnosticRun,
    artifact_name: str,
    triage: TriageOutcome,
    timeline_lines: Sequence[str],
    crossings: Sequence[str],
    evidence_gaps: Sequence[str] = (),
    repair_issue_note: str = "",
) -> str:
    """Privacy-safe factual triage report published on kodmial/aa#419."""
    marker = triage_marker(run.head_sha, run.run_id)
    root_status = ROOT_STATUS_BY_OUTCOME.get(triage.outcome) or triage.detail
    lines: list[str] = [
        marker,
        "",
        "**Scenario #2 diagnostic triage (AA-owned post-run reconciler, "
        "kodmial/aa#425 — zero ChatGPT dependencies)**",
        "",
        f"- Typed outcome: `{triage.outcome}`"
        + (f" (blocked code `{triage.blocked_code}`)" if triage.blocked_code else ""),
        f"- Diagnostic classification: `{triage.classification or 'unavailable'}` "
        f"(from `{DIAGNOSTIC_SCHEMA_VERSION}` result.json)",
        f"- Root-cause status: {root_status}",
    ]
    duration = format_elapsed(run.created_at, run.updated_at)
    lines.append(
        f"- Run: #{run.run_id} ({run.event}, main) — {run.html_url} — "
        f"started {run.created_at or 'unknown'} — wall time {duration}"
    )
    lines.append(f"- Exact SHA: `{run.head_sha}`")
    conclusion = run.conclusion or "unknown"
    lines.append(
        f"- Actions conclusion: `{conclusion}` — GitHub green is NOT evidence "
        "of product success; the typed verdict above comes only from "
        "result.json evidence."
    )
    lines.append(f"- Artifact: `{artifact_name}`")
    lines.append("")
    lines.append("### Factual timeline (sanitized identifiers/numbers only)")
    if timeline_lines:
        lines.extend(timeline_lines)
    else:
        lines.append("- (no sanitized timeline events recovered from the artifact)")
    if crossings:
        lines.append("")
        lines.append("### Slow-work observation")
        lines.extend(f"- {line}" for line in crossings)
    lines.append("")
    if evidence_gaps:
        lines.append("### Evidence gaps")
        lines.extend(f"- {line}" for line in evidence_gaps)
        lines.append("")
    if repair_issue_note:
        lines.append("### Repair escalation")
        lines.append(f"- {repair_issue_note}")
        lines.append("")
    lines.append("")
    lines.append(
        f"- Triaged issue: #{TRIAGE_ISSUE_NUMBER}; policy: no per-answer "
        "cutoff; one finite campaign allowance (kodmial/aa#424)."
    )
    lines.append(
        "_Deterministic AA-owned reconciliation output. This comment "
        "represents exactly the run named in the marker above and is "
        "re-published in place on retried runs. Not qualification evidence._"
    )
    text = "\n".join(lines)
    assert_sanitized(text, owner="triage-comment")
    return text


# ---------------------------------------------------------------------------
# Repair issue planning (evidence/instrumentation gaps only)
# ---------------------------------------------------------------------------

_FINGERPRINT_MARKER_RE = re.compile(r"<!--\s*aa-scenario2-triage-fp\s+fp=([0-9a-f]{12})\s*-->")


def parse_issue_fingerprints(bodies: Sequence[str]) -> set[str]:
    """Fingerprint markers already carried by OPEN repository issues."""
    out: set[str] = set()
    for body in bodies:
        match = _FINGERPRINT_MARKER_RE.search(body or "")
        if match:
            out.add(match.group(1))
    return out


def repair_fingerprint(blocked_code: str, detail_key: str) -> str:
    """Stable fingerprint for one actionable instrumentation defect.

    Deliberately excludes the run id and timing: identical defects on
    later runs must NOT spawn duplicate escalation issues.
    """
    digest = hashlib.sha256(
        f"aa-scenario2-triage-fp/1|{blocked_code}|{detail_key}".encode()
    ).hexdigest()
    return digest[:12]


def fingerprint_key_for_triage(triage: TriageOutcome, result: Mapping[str, Any]) -> str:
    """Deterministic key naming the defective probe/stage (no free text)."""
    if triage.blocked_code == BLOCK_TELEMETRY_MISSING:
        return "+".join(sorted(triage.missing_files)) or "unknown-members"
    if triage.blocked_code in (
        BLOCK_ARTIFACT_MISSING,
        BLOCK_ARTIFACT_CORRUPT,
        BLOCK_RESULT_MISSING,
        BLOCK_SCHEMA_INVALID,
        BLOCK_IDENTITY_MISMATCH,
    ):
        return triage.blocked_code
    if triage.blocked_code == BLOCK_HARNESS_TRUNCATED:
        keys: list[str] = []
        for probe_key in ("probe_direct_model", "probe_scenario2"):
            probe = result.get(probe_key)
            if not isinstance(probe, Mapping):
                keys.append(f"{probe_key}:absent")
                continue
            status = str(probe.get("status", "absent"))
            exit_code = probe.get("exit_code")
            keys.append(f"{probe_key}:{status}:{exit_code}")
        return "|".join(keys)
    return triage.blocked_code or triage.classification or "unknown"


def build_repair_issue_plan(
    *,
    triage: TriageOutcome,
    run: DiagnosticRun,
    artifact_name: str,
    fingerprint_key: str,
    existing_fingerprints: set[str],
) -> RepairIssuePlan:
    """Plan ONE minimal Continuum escalation issue per proven gap.

    Never fires for slow-but-delivered work, for boundary/inconclusive
    classifications (those are recorded, not repaired by code), or when
    the same fingerprint already has an open issue.
    """
    fingerprint = repair_fingerprint(
        triage.blocked_code or triage.classification or "unknown", fingerprint_key
    )
    if not triage.requires_repair_issue:
        return RepairIssuePlan(
            False,
            fingerprint,
            fingerprint_key,
            "",
            "",
            (),
            f"outcome {triage.outcome} is not a proven instrumentation defect: "
            "recorded on #419 only",
        )
    if fingerprint in existing_fingerprints:
        return RepairIssuePlan(
            False,
            fingerprint,
            fingerprint_key,
            "",
            "",
            (),
            f"fingerprint {fingerprint} already has an open repair issue: "
            "never loop an identical fingerprint",
        )
    defect = triage.blocked_code or triage.classification
    title = f"P0: aa-scenario2-triage {defect} — proven diagnostic instrumentation/evidence gap"
    detail_line = triage.detail or "proven by the reconciler from AA-owned evidence"
    body_lines = [
        "<!-- continuum-task:implementation -->",
        f"<!-- aa-scenario2-triage-fp fp={fingerprint} -->",
        "",
        "**Proven AA-owned diagnostic instrumentation/evidence gap "
        "(kodmial/aa#425 reconciler escalation)**",
        "",
        "- Source run: #" + str(run.run_id) + f" — {run.html_url}",
        f"- Exact SHA: `{run.head_sha}`",
        f"- Failing stage/code: `{defect}`"
        + (
            f" — members `{', '.join(f'`{m}`' for m in triage.missing_files)}`"
            if triage.missing_files
            else ""
        ),
        f"- Detail: {detail_line}",
        f"- Artifact examined: `{artifact_name}`",
        f"- Operations tracker: #{TRIAGE_ISSUE_NUMBER} "
        "(triage comment for this run is published there)",
        "",
        "### Scope guard",
        "",
        "- This task repairs the DIAGNOSTIC instrumentation/evidence path "
        "only. It must never patch product semantic logic based on opaque "
        "provider state, never weaken Gate A-F, and never introduce an "
        "arbitrary answer cutoff (kodmial/aa#424).",
        "",
        "### Reproduction",
        "",
        "- Rerun the AA-owned post-run reconciler against this exact run "
        "(`gh workflow run aa-scenario2-diagnostic-reconcile.yml --ref main`) "
        "or regenerate evidence with "
        "`gh workflow run aa-gate-c-scenario2-diagnostic.yml --ref main "
        f"-f target_sha={run.head_sha}` and re-reconcile the new run.",
        "",
        "### Definition of Done",
        "",
        "- Add a deterministic regression test covering this exact defect "
        f"fingerprint (`{fingerprint_key}`), including the blocked-code "
        "classification and the no-duplicate-issue idempotency.",
        "- After merge the diagnostic push trigger (or the reconciler's "
        "missing-run dispatch) produces a fresh real run whose triage "
        "comment on #" + str(TRIAGE_ISSUE_NUMBER) + " shows the gap resolved.",
        "- The reconciler must classify the same defect as fixed given new "
        "healthy evidence and must not re-open this issue for new runs of "
        "the same fingerprint.",
        "",
        "_One minimal Continuum implementation issue; scheduler + PR-Agent + "
        "CI implement/merge it; no ChatGPT dependency._",
    ]
    body = "\n".join(body_lines)
    assert_sanitized(body, owner="repair-issue")
    return RepairIssuePlan(
        True,
        fingerprint,
        fingerprint_key,
        title,
        body,
        ("priority:p0", "automation:ready"),
        f"proven instrumentation gap: create exactly one issue (fingerprint {fingerprint})",
    )


# ---------------------------------------------------------------------------
# Missing-run dispatch planning
# ---------------------------------------------------------------------------

_DISPATCH_MARKER_RE = re.compile(
    r"<!--\s*aa-scenario2-diagnostic-dispatch\s+sha=([0-9a-f]{40})"
    r"\s+at=(\d+)\s+attempt=(\d+)\s*-->"
)


def parse_dispatch_markers(bodies: Sequence[str]) -> list[DispatchMarker]:
    """Durable dispatch attempts recorded as sanitized #419 comments."""
    out: list[DispatchMarker] = []
    for body in bodies:
        match = _DISPATCH_MARKER_RE.search(body or "")
        if match:
            out.append(
                DispatchMarker(
                    sha=match.group(1),
                    created_epoch=float(match.group(2)),
                    attempt=int(match.group(3)),
                )
            )
    return out


def dispatch_marker_body(sha: str, *, at_epoch: float, attempt: int) -> str:
    sha = normalize_sha(sha)
    body = (
        f"<!-- aa-scenario2-diagnostic-dispatch sha={sha} "
        f"at={int(at_epoch)} attempt={attempt} -->\n"
        "Scheduled reconcile controller: dispatched the existing "
        "single-scenario diagnostic workflow via workflow_dispatch for this "
        "exact SHA (deduplicated; no equivalent run existed for it)."
    )
    assert_sanitized(body, owner="dispatch-marker")
    return body


def build_dispatch_inputs(target_sha: str) -> dict[str, str]:
    """The diagnostic dispatch schema accepts ONLY ``target_sha``.

    (It must never receive ``issue_number`` — that signature belongs to
    qualification workflows and would create a competing coder.)
    """
    sha = normalize_sha(target_sha)
    inputs = {"target_sha": sha}
    assert_sanitized(f"target_sha={sha}", owner="dispatch-inputs")
    return inputs


def decide_dispatch(
    runs: Sequence[DiagnosticRun],
    *,
    target_sha: str,
    markers: Sequence[DispatchMarker],
    now_epoch: float,
) -> DispatchPlan:
    """Decide whether the reconciler must trigger the diagnostic itself.

    Triggers ONLY for the exact SHA with no equivalent queued/running or
    completed run, honoring durable attempt markers with a bounded count
    and a backoff between attempts (idempotent retry, never a terminal
    pause and never an unbounded loop).
    """
    sha = normalize_sha(target_sha)
    for_target = [run for run in runs if run.head_sha == sha]
    if any(run.is_active for run in for_target):
        return DispatchPlan(
            False,
            0,
            "an equivalent diagnostic run is already active for this SHA; "
            "never dispatch a duplicate",
        )
    if any(run.is_completed for run in for_target):
        return DispatchPlan(
            False,
            0,
            "a completed diagnostic run already exists for this SHA: "
            "reconcile it instead of dispatching",
        )
    relevant = sorted(
        [marker for marker in markers if marker.sha == sha],
        key=lambda marker: marker.created_epoch,
    )
    attempts = max((marker.attempt for marker in relevant), default=0)
    if attempts >= MAX_DISPATCH_ATTEMPTS:
        return DispatchPlan(
            False,
            attempts,
            f"bounded retry cap reached ({attempts}/{MAX_DISPATCH_ATTEMPTS}) for "
            "this SHA; a manual workflow_dispatch remains available",
        )
    last_epoch = relevant[-1].created_epoch if relevant else 0.0
    if relevant and now_epoch < last_epoch + DISPATCH_RETRY_BACKOFF_S:
        remaining = round(last_epoch + DISPATCH_RETRY_BACKOFF_S - now_epoch, 1)
        return DispatchPlan(
            False,
            attempts,
            f"backing off before retry attempt {attempts + 1} "
            f"(remaining {remaining}s): Actions may still be starting the "
            "dispatched run",
        )
    attempt = attempts + 1
    return DispatchPlan(
        True,
        attempt,
        f"no equivalent diagnostic run for this SHA: dispatch attempt "
        f"{attempt}/{MAX_DISPATCH_ATTEMPTS} (target_sha only)",
        dispatch_marker_body(sha, at_epoch=now_epoch, attempt=attempt),
    )
