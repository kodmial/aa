"""Mandatory exact-main qualification lifecycle (issue #336).

Binds the existing Continuum mandatory-qualification protocol
(``<!-- automation-qualification: #7 -->`` + ``automation:qualifying`` +
exact-SHA dispatch + trusted result) to the actual AA final A-F tracker
(#7) without creating a second proprietary scheduler.

Observed gap fixed here: implementation PRs #333/#334 merged and GitHub
auto-closed #331/#332 while the mandatory exact-main Gate B/C/E evidence
had not been posted on #7. A merge or green CI is never qualification.

Lifecycle enforced by this module (all helpers are pure, no network):

- a product capability whose Definition of Done specifies an exact-main
  live gate must carry an explicit typed qualification policy
  (``automation-qualification: #7``); a missing marker is an admission
  setup error reported before the first issue snapshot, never a silent
  close;
- such a capability references its PR with ``Relates to`` (never
  ``Closes``/``Fixes``) and stays OPEN as ``automation:qualifying`` after
  merge until a trusted exact-main PASS from #7;
- post-merge dispatch is observable and idempotent as the immutable tuple
  (capability issue, qualification issue #7, exact merged main SHA,
  contract version); the same SHA is never qualified twice and an old SHA
  is never accepted as new;
- the reusable #7 ``automation:in-progress`` reservation is classified as
  genuinely active vs orphaned/stale; orphaned leases recover through a
  deterministic re-dispatch plan, never by blindly stripping labels or
  posting repeated ``/oc``;
- only a trusted exact-SHA PASS for the exact current main may complete
  the capability; FAIL/INCOMPLETE/UNKNOWN keeps it open machine-blocked
  with a typed reason (external Telegram/book/model/service blockers are
  reported, never fabricated as PASS);
- the #290 convergence breaker stays engaged for its exhausted repeated
  fingerprint: no infinite fix loops are re-enabled.

Product Contract #110 and Product Implementation #112 are immutable and
are never rewritten here. Issue #7 itself is never rewritten here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QUALIFICATION_ISSUE_NUMBER = 7

CONTRACT_VERSION = "product-contract-110/immutable"

QUALIFYING_LABEL = "automation:qualifying"
BLOCKED_LABEL = "automation:blocked"
IN_PROGRESS_LABEL = "automation:in-progress"

# Machine relation required by the Continuum mandatory-qualification
# protocol (Continuum #187/#278). Reuse this marker; do not invent a
# second proprietary scheduler marker for the same lifecycle.
_AUTOMATION_QUALIFICATION_RE = re.compile(
    r"<!--\s*automation-qualification\s*:\s*#(?P<issue>\d+)\s*-->",
    re.IGNORECASE,
)

# Immutable post-merge dispatch record posted on the qualification tracker.
_DISPATCH_MARKER_RE = re.compile(
    r"<!--\s*continuum-qualification-required\s+"
    r"capability=(?P<capability>\d+)\s+"
    r"qualification=(?P<qualification>\d+)\s+"
    r"sha=(?P<sha>[0-9a-f]{40})\s+"
    r"contract=(?P<contract>\S+?)\s*-->",
    re.IGNORECASE,
)

# Trusted final evidence posted by the repository-owned A-F workflow.
_RESULT_MARKER_RE = re.compile(
    r"<!--\s*continuum-qualification-result\s+"
    r"issue=(?P<issue>\d+)\s+"
    r"sha=(?P<sha>[0-9a-f]{40})\s+"
    r"result=(?P<result>pass|fail)\s*-->",
    re.IGNORECASE,
)

_SHA_RE = re.compile(r"[0-9a-f]{40}")

_CLOSE_KEYWORDS_RE = re.compile(r"\b(?:closes?|fix(?:es)?|resolves?)\s+#\d+", re.IGNORECASE)

TRUSTED_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
TRUSTED_BOT_LOGINS = frozenset({"github-actions[bot]"})

# Typed external blockers: report the blocker, never fabricate PASS.
BLOCKER_TELEGRAM_TOKEN = "EXTERNAL_TELEGRAM_TOKEN_UNAVAILABLE"
BLOCKER_TELEGRAM_PEER = "EXTERNAL_TELEGRAM_PEER_UNAVAILABLE"
BLOCKER_TELEGRAM_TRANSPORT = "EXTERNAL_TELEGRAM_TRANSPORT_UNAVAILABLE"
BLOCKER_BOOK_SOURCE = "EXTERNAL_BOOK_SOURCE_UNAVAILABLE"
BLOCKER_MODEL_PROVIDER = "EXTERNAL_MODEL_PROVIDER_UNAVAILABLE"
BLOCKER_SERVICE = "EXTERNAL_SERVICE_UNAVAILABLE"

TYPED_BLOCKERS = frozenset(
    {
        BLOCKER_TELEGRAM_TOKEN,
        BLOCKER_TELEGRAM_PEER,
        BLOCKER_TELEGRAM_TRANSPORT,
        BLOCKER_BOOK_SOURCE,
        BLOCKER_MODEL_PROVIDER,
        BLOCKER_SERVICE,
    }
)

# Fingerprints whose repeated-repair budget is exhausted by the #290
# convergence breaker. Repair routing must stay refused for these even
# when a fresh qualification FAIL arrives for the same scope.
EXHAUSTED_BREAKER_FINGERPRINTS = frozenset(
    {
        # Stable defect identity from the self-proving repair publisher.
        "C:live-book-grounding-substantive-drinking-2:live-production-path",
    }
)
EXHAUSTED_BREAKER_SCOPES = frozenset({"gate:C"})


class MandatoryLifecycleError(ValueError):
    """Raised when lifecycle inputs are malformed (fail-closed)."""


def normalize_sha(value: str) -> str:
    """Normalize an exact 40-hex main SHA or raise fail-closed."""
    normalized = (value or "").strip().lower()
    if not _SHA_RE.fullmatch(normalized):
        raise MandatoryLifecycleError("sha must be an exact 40-hex main SHA")
    return normalized


def parse_automation_qualification(body: str) -> int | None:
    """Return the mandatory qualification issue number, if declared."""
    match = _AUTOMATION_QUALIFICATION_RE.search(body or "")
    if match is None:
        return None
    return int(match.group("issue"))


def requires_exact_main_qualification(body: str) -> bool:
    """Return whether a task body demands exact-main live qualification.

    A product capability specifies an exact-main live gate when its text
    names the exact-main invariant together with live qualification
    vocabulary (qualification/Gate/QUALIFIED/live A-F evidence). Ordinary
    code-only tasks carry neither and stay on the plain close-on-merge
    path.
    """
    text = body or ""
    lowered = text.lower()
    if "exact-main" not in lowered and "exact main" not in lowered:
        return False
    return any(
        token in lowered
        for token in (
            "qualification",
            "qualifying",
            "qualified",
            "gate ",
            "gates ",
            "gate-",
            "live",
        )
    )


def validate_task_admission(body: str) -> tuple[bool, str]:
    """Validate the typed qualification policy before the first snapshot.

    Returns ``(True, reason)`` when the task may enter the scheduler and
    ``(False, setup-error)`` when a product capability demanding
    exact-main live qualification omits the mandatory machine relation.
    The missing marker is a setup error, never a silent close.
    """
    qualification = parse_automation_qualification(body)
    if not requires_exact_main_qualification(body):
        return True, "code-only task: no exact-main live gate declared"
    if qualification is None:
        return (
            False,
            "setup-error: exact-main live gate declared but "
            "<!-- automation-qualification: #N --> is missing",
        )
    if qualification != QUALIFICATION_ISSUE_NUMBER:
        return (
            False,
            f"setup-error: qualification tracker must be #{QUALIFICATION_ISSUE_NUMBER}, "
            f"got #{qualification}",
        )
    return True, f"mandatory qualification via #{QUALIFICATION_ISSUE_NUMBER}"


def pr_reference_kind(body: str) -> str:
    """Return the required PR reference keyword family for a task."""
    if parse_automation_qualification(body) is not None:
        return "Relates to"
    return "Closes"


def pr_body_closes_declared_capability(pr_body: str, capability: int) -> bool:
    """Return whether a PR body would auto-close the capability issue."""
    for match in _CLOSE_KEYWORDS_RE.finditer(pr_body or ""):
        numbers = re.findall(r"#(\d+)", match.group(0))
        if any(int(item) == capability for item in numbers):
            return True
    return False


def validate_pr_body_for_capability(
    task_body: str, pr_body: str, capability: int
) -> tuple[bool, str]:
    """Validate that a PR body cannot bypass mandatory qualification.

    A task carrying the mandatory relation must use a non-closing
    reference; mutable PR text containing ``Closes/Fixes #<capability>``
    is rejected so GitHub auto-close can never substitute for exact-main
    evidence.
    """
    if parse_automation_qualification(task_body) is None:
        return True, "code-only task: closing reference permitted"
    if pr_body_closes_declared_capability(pr_body, capability):
        return (
            False,
            f"bypass-rejected: PR for mandatory capability #{capability} "
            "must use 'Relates to', never 'Closes/Fixes'",
        )
    return True, "non-closing PR reference verified"


def post_merge_state(task_body: str) -> str:
    """Return the required post-merge label state for a merged capability."""
    if parse_automation_qualification(task_body) is not None:
        return QUALIFYING_LABEL
    return "closed"


@dataclass(frozen=True)
class MandatoryQualification:
    """Immutable post-merge qualification binding."""

    capability: int
    qualification: int
    sha: str
    contract: str


def build_dispatch_tuple(
    capability: int, sha: str, *, contract: str = CONTRACT_VERSION
) -> MandatoryQualification:
    """Build the immutable (capability, #7, exact SHA, contract) tuple."""
    if capability <= 0:
        raise MandatoryLifecycleError("capability issue number must be positive")
    return MandatoryQualification(
        capability=capability,
        qualification=QUALIFICATION_ISSUE_NUMBER,
        sha=normalize_sha(sha),
        contract=(contract or "").strip() or CONTRACT_VERSION,
    )


def format_dispatch_marker(item: MandatoryQualification) -> str:
    """Format the observable immutable dispatch marker comment fragment."""
    return (
        f"<!-- continuum-qualification-required capability={item.capability} "
        f"qualification={item.qualification} sha={item.sha} contract={item.contract} -->"
    )


def parse_dispatch_markers(bodies: list[str]) -> list[MandatoryQualification]:
    """Parse immutable dispatch markers from comment bodies in order."""
    out: list[MandatoryQualification] = []
    for body in bodies:
        for match in _DISPATCH_MARKER_RE.finditer(body or ""):
            out.append(
                MandatoryQualification(
                    capability=int(match.group("capability")),
                    qualification=int(match.group("qualification")),
                    sha=match.group("sha").lower(),
                    contract=match.group("contract"),
                )
            )
    return out


@dataclass(frozen=True)
class TrustedResult:
    """One trusted exact-SHA qualification verdict."""

    sha: str
    result: str


def is_trusted_author(*, association: str, login: str, owner_login: str = "") -> bool:
    """Return whether a result comment author is trusted evidence."""
    normalized_association = (association or "").strip().upper()
    normalized_login = (login or "").strip()
    if normalized_association in TRUSTED_ASSOCIATIONS:
        return True
    if normalized_login in TRUSTED_BOT_LOGINS:
        return True
    return bool(owner_login) and normalized_login == owner_login


def parse_trusted_results(
    comments: list[tuple[str, str, str]],
    *,
    owner_login: str = "",
    issue: int = QUALIFICATION_ISSUE_NUMBER,
) -> list[TrustedResult]:
    """Parse trusted #7 result markers, ignoring forged untrusted markers.

    ``comments`` carries ``(body, association, login)`` triples in order.
    Only markers from trusted authors are evidence; anything else is not
    qualification and must never close a capability.
    """
    out: list[TrustedResult] = []
    for body, association, login in comments:
        if not is_trusted_author(association=association, login=login, owner_login=owner_login):
            continue
        for match in _RESULT_MARKER_RE.finditer(body or ""):
            if int(match.group("issue")) != issue:
                continue
            out.append(
                TrustedResult(
                    sha=match.group("sha").lower(),
                    result=match.group("result").lower(),
                )
            )
    return out


@dataclass(frozen=True)
class DispatchDecision:
    """Idempotent post-merge dispatch verdict."""

    dispatch: bool
    reason: str


def should_dispatch_qualification(
    *,
    capability: int,
    current_main_sha: str,
    dispatches: list[MandatoryQualification],
    results: list[TrustedResult],
) -> DispatchDecision:
    """Decide exactly one correct new-SHA dispatch (idempotent, fail-closed).

    - no duplicate qualification when the same SHA already has a dispatch
      record or trustworthy final evidence;
    - no acceptance of an old SHA as new: results for any other SHA never
      satisfy the current main.
    """
    current = normalize_sha(current_main_sha)
    if capability <= 0:
        raise MandatoryLifecycleError("capability issue number must be positive")
    for result in results:
        if result.sha == current and result.result in ("pass", "fail"):
            return DispatchDecision(
                dispatch=False,
                reason=f"no duplicate: SHA {current[:12]} already has trusted {result.result}",
            )
    for item in dispatches:
        if item.capability == capability and item.sha == current:
            short = current[:12]
            return DispatchDecision(
                dispatch=False,
                reason=(f"no duplicate: capability #{capability} already dispatched for {short}"),
            )
    return DispatchDecision(
        dispatch=True,
        reason=f"dispatch capability #{capability} on exact main {current[:12]}",
    )


@dataclass(frozen=True)
class CompletionDecision:
    """Capability completion verdict gated by actual evidence."""

    may_close: bool
    reason: str


def gate_capability_completion(
    *,
    capability: int,
    current_main_sha: str,
    results: list[TrustedResult],
) -> CompletionDecision:
    """Gate capability completion by trusted exact-SHA evidence.

    Only a real trusted PASS for the exact current main SHA may close the
    declared capability. FAIL, INCOMPLETE (no result), UNKNOWN, out-of-date
    PASS, or forged untrusted markers (already excluded by
    :func:`parse_trusted_results`) leave it open machine-blocked.
    """
    current = normalize_sha(current_main_sha)
    latest: TrustedResult | None = results[-1] if results else None
    if latest is None:
        return CompletionDecision(
            may_close=False,
            reason=f"capability #{capability} INCOMPLETE: no trusted #7 result for {current[:12]}",
        )
    if latest.sha != current:
        return CompletionDecision(
            may_close=False,
            reason=f"capability #{capability} blocked: latest trusted result "
            f"is for stale {latest.sha[:12]}, current main is {current[:12]}",
        )
    if latest.result == "pass":
        return CompletionDecision(
            may_close=True,
            reason=f"capability #{capability} QUALIFIED on exact main {current[:12]}",
        )
    return CompletionDecision(
        may_close=False,
        reason=f"capability #{capability} blocked: trusted #7 {latest.result} on {current[:12]}",
    )


@dataclass(frozen=True)
class TrackerLease:
    """Classification of the reusable #7 reservation label."""

    state: str
    reason: str


def classify_tracker_lease(
    *,
    has_in_progress_label: bool,
    has_active_run: bool,
    age_s: float,
    lease_s: float = 3600.0,
) -> TrackerLease:
    """Distinguish a genuinely active A-F run from a stale reservation.

    - ``active``: the label is backed by a live A-F run right now;
    - ``orphaned``: the label persists with no live run (stale
      reservation or a previous FAIL left behind); deterministic recovery
      must re-dispatch, not deadlock;
    - ``idle``: no reservation is held.
    """
    if not has_in_progress_label:
        return TrackerLease(state="idle", reason="no in-progress reservation held")
    if has_active_run:
        return TrackerLease(state="active", reason="live A-F run owns the reservation")
    if age_s < 0:
        raise MandatoryLifecycleError("lease age must be non-negative")
    if age_s >= lease_s:
        return TrackerLease(
            state="orphaned",
            reason=f"reservation age {age_s:.0f}s exceeds lease {lease_s:.0f}s with no live run",
        )
    return TrackerLease(
        state="orphaned",
        reason="reservation held with no live run: orphaned, eligible for deterministic recovery",
    )


@dataclass(frozen=True)
class LeaseRecovery:
    """Deterministic orphaned-lease recovery plan."""

    action: str
    reason: str


def orphaned_lease_recovery(lease: TrackerLease) -> LeaseRecovery:
    """Recover an orphaned #7 lease through documented deterministic code.

    Never blindly strips labels and never posts repeated ``/oc``: an
    orphaned lease re-dispatches exactly one new-SHA qualification and
    keeps the reservation until trusted evidence lands. Active and idle
    leases require no recovery.
    """
    if lease.state == "active":
        return LeaseRecovery(action="retain", reason="live run owns the lease; no recovery")
    if lease.state == "idle":
        return LeaseRecovery(action="none", reason="no reservation to recover")
    if lease.state == "orphaned":
        return LeaseRecovery(
            action="re-dispatch-exact-main",
            reason=f"orphaned lease: {lease.reason}; re-dispatch one exact-main run, keep label",
        )
    raise MandatoryLifecycleError(f"unknown lease state {lease.state!r}")


def typed_blocker_reason(code: str, *, detail: str = "") -> str:
    """Format a typed external blocker without fabricating PASS."""
    normalized = (code or "").strip()
    if normalized not in TYPED_BLOCKERS:
        raise MandatoryLifecycleError(f"unknown blocker code {code!r}")
    suffix = f": {detail.strip()}" if detail.strip() else ""
    return f"blocked: {normalized}{suffix}; qualification stays FAIL/BLOCKED, never PASS"


def is_repair_allowed(fingerprint: str, *, scope: str = "") -> bool:
    """Preserve the #290 convergence breaker for exhausted fingerprints.

    Returns False when the stable fingerprint (or its convergence scope)
    already exhausted its bounded repair budget, so no further automatic
    fix loop may be re-enabled for the same repeated failure.
    """
    normalized = (fingerprint or "").strip()
    if normalized in EXHAUSTED_BREAKER_FINGERPRINTS:
        return False
    if scope.strip() in EXHAUSTED_BREAKER_SCOPES and "live-book-grounding" in normalized:
        return False
    return True


# ---------------------------------------------------------------------------
# Issue #345: ARG_MAX-safe paginated inventory, full tracker history,
# shared-SHA dispatch, runner-state handling, durable records, recovery.
# All helpers below stay pure (no network) so unit tests exercise them
# without GitHub access. Live pagination lives in the workflow, which only
# passes small scalar args and chmod-600 file paths to Python.
# ---------------------------------------------------------------------------

# Typed blocked reasons for dispatch/record paths. These describe
# infrastructure or history problems; they never fabricate PASS.
BLOCKED_INVENTORY_INCOMPLETE = "BLOCKED_INVENTORY_INCOMPLETE"
BLOCKED_HISTORY_INCOMPLETE = "BLOCKED_TRACKER_HISTORY_INCOMPLETE"
BLOCKED_RUNNER_INFRA = "BLOCKED_RUNNER_INFRA_RETRYABLE"
BLOCKED_RUNNER_TERMINAL = "BLOCKED_RUNNER_TERMINAL"
BLOCKED_LEASE_ACTIVE = "BLOCKED_LEASE_ACTIVE_RETAIN"
BLOCKED_NO_CAPABILITIES = "BLOCKED_NO_WAITING_CAPABILITIES"

# Shared dispatch marker records every waiting capability for one SHA in a
# single durable identity, so duplicate events cannot mint a second run.
_SHARED_DISPATCH_MARKER_RE = re.compile(
    r"<!--\s*continuum-qualification-required-shared\s+"
    r"qualification=(?P<qualification>\d+)\s+"
    r"sha=(?P<sha>[0-9a-f]{40})\s+"
    r"contract=(?P<contract>\S+?)\s+"
    r"capabilities=(?P<capabilities>[\d,\s]+?)\s*-->",
    re.IGNORECASE,
)

# Bounded idempotent retry budget for infrastructure failures.
MAX_INFRA_DISPATCH_ATTEMPTS = 3

# Runner failure kinds that are infrastructure (retryable, bounded) rather
# than terminal qualification evidence.
INFRA_RETRYABLE_KINDS = frozenset(
    {
        "rate_limited_429",
        "postponed",
        "transient_403",
        "transient_5xx",
        "runner_error",
        "canceled_same_sha",
        "api_unavailable",
    }
)


def is_pull_request_record(item: Mapping[str, Any]) -> bool:
    """Return whether a REST issue object is actually a pull request."""
    if not isinstance(item, Mapping):
        return False
    marker = item.get("pull_request")
    if marker is not None:
        return True
    url = str(item.get("pull_request_url", "") or item.get("html_url", ""))
    return "/pull/" in url


def extract_qualifying_capabilities(issues: Iterable[Mapping[str, Any]]) -> list[int]:
    """Return sorted qualifying capability numbers, excluding PR objects.

    Each entry must be a mapping with an integer ``number`` and a string
    ``body``. Entries that are pull requests are never ingested. Entries
    with a missing/invalid number are skipped; a ``None`` page entry is a
    caller error reported fail-closed by
    :func:`collect_qualifying_capabilities_from_pages`.
    """
    wanted: set[int] = set()
    for item in issues:
        if item is None:
            raise MandatoryLifecycleError("inventory page is incomplete (None entry)")
        if not isinstance(item, Mapping):
            raise MandatoryLifecycleError("inventory entry must be a JSON object")
        if is_pull_request_record(item):
            continue
        try:
            number = int(item.get("number", 0))
        except (TypeError, ValueError):
            continue
        if number <= 0:
            continue
        body = item.get("body", "")
        if not isinstance(body, str):
            continue
        if parse_automation_qualification(body) == QUALIFICATION_ISSUE_NUMBER:
            wanted.add(number)
    return sorted(wanted)


def collect_qualifying_capabilities_from_pages(
    pages: Sequence[Sequence[Mapping[str, Any]] | None],
) -> list[int]:
    """Merge every paginated inventory page without any enumeration cap.

    ``pages`` holds one list per REST page in order; a ``None`` page means
    a partial fetch failure and fails closed (no silent truncation).
    """
    wanted: set[int] = set()
    for index, page in enumerate(pages):
        if page is None:
            raise MandatoryLifecycleError(
                f"{BLOCKED_INVENTORY_INCOMPLETE}: page {index} unavailable"
            )
        for capability in extract_qualifying_capabilities(page):
            wanted.add(capability)
    return sorted(wanted)


def load_capability_inventory_file(path: str | Path) -> list[int]:
    """Load a chmod-600 JSON/JSONL inventory file with bounded streaming.

    Accepts either a JSON list of issue objects or JSONL (one object per
    line, blank lines ignored). Pull-request objects are excluded and only
    issues declaring ``automation-qualification: #7`` are returned sorted.
    Corrupt or missing files fail closed with :class:`MandatoryLifecycleError`.
    Only counts and issue numbers are returned: bodies never leave this call.
    """
    raw_path = Path(path)
    try:
        text = raw_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MandatoryLifecycleError(
            f"{BLOCKED_INVENTORY_INCOMPLETE}: cannot read inventory file: {exc}"
        ) from exc
    stripped = text.strip()
    if not stripped:
        # An empty JSONL inventory means zero open issues (idle state):
        # no waiting capabilities, no dispatch, lease classified only.
        return []
    entries: list[Mapping[str, Any]] = []
    if stripped.startswith("["):
        try:
            decoded: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MandatoryLifecycleError(
                f"{BLOCKED_INVENTORY_INCOMPLETE}: corrupt inventory JSON: {exc}"
            ) from exc
        if not isinstance(decoded, list):
            raise MandatoryLifecycleError(
                f"{BLOCKED_INVENTORY_INCOMPLETE}: inventory JSON must be a list"
            )
        for entry in decoded:
            if entry is None or not isinstance(entry, dict):
                raise MandatoryLifecycleError(
                    f"{BLOCKED_INVENTORY_INCOMPLETE}: corrupt inventory entry"
                )
            entries.append(entry)
    else:
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry_obj: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MandatoryLifecycleError(
                    f"{BLOCKED_INVENTORY_INCOMPLETE}: corrupt JSONL line {line_no}: {exc}"
                ) from exc
            if not isinstance(entry_obj, dict):
                raise MandatoryLifecycleError(
                    f"{BLOCKED_INVENTORY_INCOMPLETE}: corrupt JSONL entry at line {line_no}"
                )
            entries.append(entry_obj)
    return extract_qualifying_capabilities(entries)


@dataclass(frozen=True)
class TrackerComment:
    """One slimmed tracker comment (no raw Underlying bodies retained)."""

    body: str
    association: str
    login: str


def load_tracker_comments_file(path: str | Path) -> list[TrackerComment]:
    """Load slim tracker comments from a file, failing closed on corruption.

    The file must hold a JSON list of ``{body, association, login}`` objects.
    A corrupt, missing, or mistyped file raises :class:`MandatoryLifecycleError`
    so callers record typed BLOCKED instead of dispatching on empty history.
    """
    raw_path = Path(path)
    try:
        decoded: Any = json.loads(raw_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MandatoryLifecycleError(
            f"{BLOCKED_HISTORY_INCOMPLETE}: tracker comments file is missing"
        ) from exc
    except OSError as exc:
        raise MandatoryLifecycleError(
            f"{BLOCKED_HISTORY_INCOMPLETE}: cannot read tracker comments file: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise MandatoryLifecycleError(
            f"{BLOCKED_HISTORY_INCOMPLETE}: corrupt tracker comments JSON: {exc}"
        ) from exc
    if not isinstance(decoded, list):
        raise MandatoryLifecycleError(
            f"{BLOCKED_HISTORY_INCOMPLETE}: tracker comments file must hold a JSON list"
        )
    out: list[TrackerComment] = []
    for index, entry in enumerate(decoded):
        if not isinstance(entry, dict):
            raise MandatoryLifecycleError(
                f"{BLOCKED_HISTORY_INCOMPLETE}: corrupt comment entry {index}"
            )
        body = entry.get("body", "")
        association = entry.get("association", "")
        login = entry.get("login", "")
        if not isinstance(body, str) or not isinstance(association, str):
            raise MandatoryLifecycleError(
                f"{BLOCKED_HISTORY_INCOMPLETE}: corrupt comment entry {index}"
            )
        if not isinstance(login, str):
            raise MandatoryLifecycleError(
                f"{BLOCKED_HISTORY_INCOMPLETE}: corrupt comment entry {index}"
            )
        out.append(TrackerComment(body=body, association=association, login=login))
    return out


def assemble_comment_history_from_pages(
    pages: Sequence[Sequence[Mapping[str, Any]] | None],
) -> list[TrackerComment]:
    """Assemble the full paginated tracker history, failing closed on gaps.

    Every page must be present; a ``None`` page (partial fetch failure) or a
    mistyped entry raises :class:`MandatoryLifecycleError` with a typed
    BLOCKED code. Callers must never treat a truncated list as empty history.
    """
    out: list[TrackerComment] = []
    for index, page in enumerate(pages):
        if page is None:
            raise MandatoryLifecycleError(
                f"{BLOCKED_HISTORY_INCOMPLETE}: comment page {index} unavailable"
            )
        for entry_index, entry in enumerate(page):
            if not isinstance(entry, Mapping):
                raise MandatoryLifecycleError(
                    f"{BLOCKED_HISTORY_INCOMPLETE}: corrupt comment "
                    f"page {index} entry {entry_index}"
                )
            body = entry.get("body", "")
            association = entry.get(
                "authorAssociation", entry.get("author_association", entry.get("association", ""))
            )
            author = entry.get("author", entry.get("user", {}))
            login = ""
            if isinstance(author, Mapping):
                login_value = author.get("login", entry.get("login", ""))
                login = str(login_value or "")
            if not isinstance(body, str) or not isinstance(association, str):
                raise MandatoryLifecycleError(
                    f"{BLOCKED_HISTORY_INCOMPLETE}: corrupt comment "
                    f"page {index} entry {entry_index}"
                )
            out.append(TrackerComment(body=body, association=association, login=login))
    return out


def tracker_comments_to_triples(comments: Iterable[TrackerComment]) -> list[tuple[str, str, str]]:
    """Convert slim comments to ``(body, association, login)`` triples."""
    return [(item.body, item.association, item.login) for item in comments]


def shared_run_identity(sha: str, *, contract: str = CONTRACT_VERSION) -> str:
    """Return the atomic durable dispatch identity for one SHA/config."""
    normalized = normalize_sha(sha)
    slug = (contract or CONTRACT_VERSION).strip().replace("/", "-")
    return f"mandatory-qualification-{normalized[:12]}-{slug}"


def format_shared_dispatch_marker(
    capabilities: Sequence[int], sha: str, *, contract: str = CONTRACT_VERSION
) -> str:
    """Format one shared marker covering every waiting capability for a SHA."""
    normalized = normalize_sha(sha)
    waiting = sorted({int(cap) for cap in capabilities if int(cap) > 0})
    if not waiting:
        raise MandatoryLifecycleError("shared dispatch requires at least one capability")
    joined = ",".join(str(cap) for cap in waiting)
    resolved_contract = (contract or "").strip() or CONTRACT_VERSION
    return (
        f"<!-- continuum-qualification-required-shared "
        f"qualification={QUALIFICATION_ISSUE_NUMBER} sha={normalized} "
        f"contract={resolved_contract} capabilities={joined} -->"
    )


def parse_shared_dispatch_markers(bodies: Iterable[str]) -> list[tuple[str, list[int]]]:
    """Parse shared markers as ``(sha, capabilities)`` pairs in order."""
    out: list[tuple[str, list[int]]] = []
    for body in bodies:
        for match in _SHARED_DISPATCH_MARKER_RE.finditer(body or ""):
            if int(match.group("qualification")) != QUALIFICATION_ISSUE_NUMBER:
                continue
            caps = [
                int(token)
                for token in re.split(r"[,\s]+", match.group("capabilities").strip())
                if token.strip().isdigit() and int(token) > 0
            ]
            out.append((match.group("sha").lower(), sorted(set(caps))))
    return out


@dataclass(frozen=True)
class SharedDispatchPlan:
    """Authoritative idempotent plan for one SHA across capabilities."""

    should_dispatch: bool
    reason: str
    owner_capability: int | None
    waiting_capabilities: tuple[int, ...]
    reuse_sha: str | None


def plan_shared_qualification_dispatch(
    *,
    capabilities: Sequence[int],
    current_main_sha: str,
    dispatches: Sequence[MandatoryQualification],
    results: Sequence[TrustedResult],
    lease: TrackerLease,
    shared_markers: Sequence[tuple[str, list[int]]] | None = None,
) -> SharedDispatchPlan:
    """Plan exactly one authoritative A-F run per SHA/config generation.

    Multiple open capabilities may wait on the same validator issue #7 and
    the same current main commit. Only the deterministic owner (smallest
    capability number) mints the single ``workflow_dispatch`` event while
    every waiting capability is recorded separately. A second capability,
    duplicate push/PR-close/tracker-comment events, or an already dispatched
    SHA must not create another expensive run. A new code SHA restarts with
    new evidence; stale results never qualify new code.
    """
    current = normalize_sha(current_main_sha)
    waiting = tuple(sorted({int(cap) for cap in capabilities if int(cap) > 0}))
    if not waiting:
        return SharedDispatchPlan(
            should_dispatch=False,
            reason=f"{BLOCKED_NO_CAPABILITIES}: lease classified only, no dispatch decision",
            owner_capability=None,
            waiting_capabilities=(),
            reuse_sha=None,
        )
    current_latest: TrustedResult | None = None
    for result in results:
        if result.sha == current and result.result in ("pass", "fail"):
            current_latest = result
    if current_latest is not None:
        return SharedDispatchPlan(
            should_dispatch=False,
            reason=(
                f"no duplicate: SHA {current[:12]} already has trusted {current_latest.result}"
            ),
            owner_capability=None,
            waiting_capabilities=waiting,
            reuse_sha=current,
        )
    for item in dispatches:
        if item.sha == current:
            return SharedDispatchPlan(
                should_dispatch=False,
                reason=(
                    f"no duplicate: SHA {current[:12]} already dispatched "
                    f"(capability #{item.capability} owns the shared run); "
                    f"{len(waiting)} waiting"
                ),
                owner_capability=None,
                waiting_capabilities=waiting,
                reuse_sha=current,
            )
    for sha, _caps in shared_markers or []:
        if sha == current:
            return SharedDispatchPlan(
                should_dispatch=False,
                reason=(
                    f"no duplicate: shared marker for {current[:12]} already covers "
                    f"{len(waiting)} waiting"
                ),
                owner_capability=None,
                waiting_capabilities=waiting,
                reuse_sha=current,
            )
    if lease.state == "active":
        return SharedDispatchPlan(
            should_dispatch=False,
            reason=(
                f"{BLOCKED_LEASE_ACTIVE}: live run owns the reservation; "
                f"{len(waiting)} waiting recorded separately"
            ),
            owner_capability=None,
            waiting_capabilities=waiting,
            reuse_sha=current,
        )
    owner = waiting[0]
    if lease.state == "orphaned":
        return SharedDispatchPlan(
            should_dispatch=True,
            reason=(
                f"orphaned lease recovery: dispatch one shared run for {current[:12]} "
                f"covering {len(waiting)} capabilities (owner #{owner}), keep label"
            ),
            owner_capability=owner,
            waiting_capabilities=waiting,
            reuse_sha=None,
        )
    return SharedDispatchPlan(
        should_dispatch=True,
        reason=(
            f"dispatch shared run for exact main {current[:12]} "
            f"covering {len(waiting)} capabilities (owner #{owner})"
        ),
        owner_capability=owner,
        waiting_capabilities=waiting,
        reuse_sha=None,
    )


@dataclass(frozen=True)
class RunnerDecision:
    """Bounded idempotent outcome for one observed runner/API state."""

    action: str
    reason: str
    retry_allowed: bool


def classify_runner_state(
    *,
    status: str,
    conclusion: str | None = None,
    http_status: int | None = None,
    sha_matches_head: bool = True,
    attempt: int = 0,
    max_attempts: int = MAX_INFRA_DISPATCH_ATTEMPTS,
) -> RunnerDecision:
    """Classify success/failure/429/stale/canceled/403/5xx runner states.

    Returns a bounded idempotent outcome: at most one claimant retries an
    infrastructure failure, terminal evidence never retries, and a canceled
    stale HEAD re-dispatches the current SHA instead of accepting old output.
    """
    normalized_status = (status or "").strip().lower()
    normalized_conclusion = (conclusion or "").strip().lower()
    code = http_status or 0

    if normalized_status in ("in_progress", "queued", "waiting", "requested"):
        return RunnerDecision(
            action="retain",
            reason="live run owns the slot; duplicate dispatch refused",
            retry_allowed=False,
        )
    if normalized_status == "postponed" or normalized_conclusion == "postponed":
        allowed = should_retry_infra_failure(
            failure_kind="postponed", attempt=attempt, max_attempts=max_attempts
        )
        return RunnerDecision(
            action="retry-bounded" if allowed else "blocked-terminal",
            reason="postponed job: bounded single-claimant retry"
            if allowed
            else ("postponed job: retry budget exhausted, typed BLOCKED"),
            retry_allowed=allowed,
        )
    if code == 429 or normalized_status == "rate_limited_429":
        allowed = should_retry_infra_failure(
            failure_kind="rate_limited_429", attempt=attempt, max_attempts=max_attempts
        )
        return RunnerDecision(
            action="retry-bounded" if allowed else "blocked-terminal",
            reason="429 restart: bounded single-claimant retry"
            if allowed
            else ("429 restart: retry budget exhausted, typed BLOCKED"),
            retry_allowed=allowed,
        )
    if code in (403, 500, 502, 503, 504) or normalized_status in (
        "transient_403",
        "transient_5xx",
        "runner_error",
        "api_unavailable",
    ):
        kind = "transient_403" if code == 403 else "transient_5xx"
        allowed = should_retry_infra_failure(
            failure_kind=kind, attempt=attempt, max_attempts=max_attempts
        )
        return RunnerDecision(
            action="retry-bounded" if allowed else "blocked-terminal",
            reason=f"transient {code or kind}: bounded single-claimant retry"
            if allowed
            else (f"transient {code or kind}: retry budget exhausted, typed BLOCKED"),
            retry_allowed=allowed,
        )
    if normalized_status in ("canceled", "cancelled") or normalized_conclusion in (
        "cancelled",
        "canceled",
    ):
        if not sha_matches_head:
            return RunnerDecision(
                action="re-dispatch-exact-main",
                reason="canceled stale HEAD: re-dispatch current SHA, stale output never qualifies",
                retry_allowed=True,
            )
        allowed = should_retry_infra_failure(
            failure_kind="canceled_same_sha", attempt=attempt, max_attempts=max_attempts
        )
        return RunnerDecision(
            action="retry-bounded" if allowed else "blocked-terminal",
            reason="canceled same-SHA run: bounded single-claimant retry"
            if allowed
            else ("canceled same-SHA run: retry budget exhausted, typed BLOCKED"),
            retry_allowed=allowed,
        )
    if normalized_status == "completed" and normalized_conclusion in ("success", "failure"):
        return RunnerDecision(
            action="none",
            reason=f"terminal {normalized_conclusion}: evidence decides, no worker retry",
            retry_allowed=False,
        )
    if normalized_status == "wrong_attempt":
        return RunnerDecision(
            action="retain",
            reason="wrong run attempt ignored: live attempt owns the slot, no ghost worker",
            retry_allowed=False,
        )
    return RunnerDecision(
        action="blocked-terminal",
        reason=f"{BLOCKED_RUNNER_TERMINAL}: unknown runner state {status!r}, fail closed",
        retry_allowed=False,
    )


def should_retry_infra_failure(
    *, failure_kind: str, attempt: int, max_attempts: int = MAX_INFRA_DISPATCH_ATTEMPTS
) -> bool:
    """Return whether one bounded infrastructure retry remains for a kind."""
    if failure_kind not in INFRA_RETRYABLE_KINDS:
        return False
    if attempt < 0 or max_attempts <= 0:
        return False
    return attempt < max_attempts


def is_genuinely_active_run(
    *,
    has_active_run: bool,
    run_sha: str = "",
    current_sha: str = "",
    run_conclusion: str | None = None,
) -> bool:
    """Distinguish a genuinely active self-proving run from a ghost slot.

    A run counts as active only when the API reports a live run AND, when both
    SHAs are known, the run tests the current HEAD. A wrong-SHA, terminal, or
    canceled run never holds the lease, so no ghost slot blocks recovery.
    """
    if not has_active_run:
        return False
    conclusion = (run_conclusion or "").strip().lower()
    if conclusion in ("success", "failure", "cancelled", "canceled"):
        return False
    if run_sha and current_sha:
        try:
            if normalize_sha(run_sha) != normalize_sha(current_sha):
                return False
        except MandatoryLifecycleError:
            return False
    return True


def format_blocked_reason(code: str, *, detail: str = "") -> str:
    """Format a typed machine-reconstructable blocked reason (never PASS)."""
    normalized = (code or "").strip() or BLOCKED_RUNNER_TERMINAL
    suffix = f": {detail.strip()}" if detail.strip() else ""
    return f"blocked: {normalized}{suffix}; qualification stays FAIL/BLOCKED, never PASS"


def build_dispatch_record(
    *,
    capabilities: Sequence[int],
    sha: str,
    contract: str = CONTRACT_VERSION,
    actor: str = "",
    run_id: str = "",
    run_url: str = "",
) -> dict[str, Any]:
    """Build a privacy-safe machine-reconstructable dispatch record.

    Carries only capability refs, tracker #7, tested SHA, contract version,
    trusted actor, and durable workflow run URL/ID. Never includes issue
    bodies, books, secrets, chat IDs, or full comments.
    """
    normalized = normalize_sha(sha)
    waiting = sorted({int(cap) for cap in capabilities if int(cap) > 0})
    if not waiting:
        raise MandatoryLifecycleError("dispatch record requires at least one capability")
    return {
        "kind": "mandatory-qualification-dispatch",
        "qualification": QUALIFICATION_ISSUE_NUMBER,
        "capabilities": waiting,
        "sha": normalized,
        "contract": (contract or "").strip() or CONTRACT_VERSION,
        "identity": shared_run_identity(normalized, contract=contract),
        "actor": (actor or "").strip()[:120],
        "run_id": str(run_id or "")[:64],
        "run_url": str(run_url or "")[:512],
    }


def build_blocked_record(
    *,
    capabilities: Sequence[int],
    sha: str,
    contract: str = CONTRACT_VERSION,
    blocked_code: str,
    detail: str = "",
) -> dict[str, Any]:
    """Build a privacy-safe typed BLOCKED record for the same identity."""
    normalized = normalize_sha(sha)
    waiting = sorted({int(cap) for cap in capabilities if int(cap) > 0})
    return {
        "kind": "mandatory-qualification-blocked",
        "qualification": QUALIFICATION_ISSUE_NUMBER,
        "capabilities": waiting,
        "sha": normalized,
        "contract": (contract or "").strip() or CONTRACT_VERSION,
        "identity": shared_run_identity(normalized, contract=contract),
        "blocked_reason": format_blocked_reason(blocked_code, detail=detail),
    }


def serialize_record(record: Mapping[str, Any]) -> str:
    """Serialize a dispatch/blocked record to canonical JSON."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def parse_record_text(text: str) -> dict[str, Any]:
    """Parse and validate a serialized record, failing closed on corruption."""
    try:
        decoded: Any = json.loads(text or "")
    except json.JSONDecodeError as exc:
        raise MandatoryLifecycleError(f"corrupt dispatch record: {exc}") from exc
    if not isinstance(decoded, dict):
        raise MandatoryLifecycleError("dispatch record must be a JSON object")
    for key in ("capabilities", "sha", "contract", "qualification"):
        if key not in decoded:
            raise MandatoryLifecycleError(f"dispatch record is missing {key!r}")
    normalize_sha(str(decoded["sha"]))
    return dict(decoded)


def needs_scheduled_recovery(
    *,
    failure_kind: str,
    attempt: int,
    fingerprint: str = "",
    scope: str = "",
    max_attempts: int = MAX_INFRA_DISPATCH_ATTEMPTS,
) -> bool:
    """Return whether cron/watchdog may retry an infra failure on the same SHA.

    Recovery is deterministic and bounded: only retryable infrastructure kinds
    within budget, and never for fingerprints whose #290 repair budget is
    exhausted (no infinite same-fingerprint mutation loop). Terminal
    qualification FAIL stays BLOCKED without a new run.
    """
    if failure_kind not in INFRA_RETRYABLE_KINDS:
        return False
    if not should_retry_infra_failure(
        failure_kind=failure_kind, attempt=attempt, max_attempts=max_attempts
    ):
        return False
    if fingerprint and not is_repair_allowed(fingerprint, scope=scope):
        return False
    return True


__all__ = [
    "BLOCKED_HISTORY_INCOMPLETE",
    "BLOCKED_INVENTORY_INCOMPLETE",
    "BLOCKED_LEASE_ACTIVE",
    "BLOCKED_NO_CAPABILITIES",
    "BLOCKED_RUNNER_INFRA",
    "BLOCKED_RUNNER_TERMINAL",
    "BLOCKED_LABEL",
    "BLOCKER_BOOK_SOURCE",
    "BLOCKER_MODEL_PROVIDER",
    "BLOCKER_SERVICE",
    "BLOCKER_TELEGRAM_PEER",
    "BLOCKER_TELEGRAM_TOKEN",
    "BLOCKER_TELEGRAM_TRANSPORT",
    "CONTRACT_VERSION",
    "EXHAUSTED_BREAKER_FINGERPRINTS",
    "EXHAUSTED_BREAKER_SCOPES",
    "IN_PROGRESS_LABEL",
    "QUALIFICATION_ISSUE_NUMBER",
    "QUALIFYING_LABEL",
    "TYPED_BLOCKERS",
    "CompletionDecision",
    "DispatchDecision",
    "INFRA_RETRYABLE_KINDS",
    "LeaseRecovery",
    "MAX_INFRA_DISPATCH_ATTEMPTS",
    "MandatoryLifecycleError",
    "MandatoryQualification",
    "RunnerDecision",
    "SharedDispatchPlan",
    "TrackerComment",
    "TrackerLease",
    "TrustedResult",
    "assemble_comment_history_from_pages",
    "build_blocked_record",
    "build_dispatch_record",
    "build_dispatch_tuple",
    "classify_runner_state",
    "classify_tracker_lease",
    "collect_qualifying_capabilities_from_pages",
    "extract_qualifying_capabilities",
    "format_blocked_reason",
    "format_dispatch_marker",
    "format_shared_dispatch_marker",
    "gate_capability_completion",
    "is_genuinely_active_run",
    "is_pull_request_record",
    "is_repair_allowed",
    "is_trusted_author",
    "load_capability_inventory_file",
    "load_tracker_comments_file",
    "needs_scheduled_recovery",
    "normalize_sha",
    "orphaned_lease_recovery",
    "parse_automation_qualification",
    "parse_dispatch_markers",
    "parse_record_text",
    "parse_shared_dispatch_markers",
    "parse_trusted_results",
    "post_merge_state",
    "pr_body_closes_declared_capability",
    "pr_reference_kind",
    "requires_exact_main_qualification",
    "serialize_record",
    "shared_run_identity",
    "should_dispatch_qualification",
    "should_retry_infra_failure",
    "tracker_comments_to_triples",
    "typed_blocker_reason",
    "validate_pr_body_for_capability",
    "validate_task_admission",
]
