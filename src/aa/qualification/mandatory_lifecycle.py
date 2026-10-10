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

import re
from dataclasses import dataclass

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


__all__ = [
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
    "LeaseRecovery",
    "MandatoryLifecycleError",
    "MandatoryQualification",
    "TrackerLease",
    "TrustedResult",
    "build_dispatch_tuple",
    "classify_tracker_lease",
    "format_dispatch_marker",
    "gate_capability_completion",
    "is_repair_allowed",
    "is_trusted_author",
    "normalize_sha",
    "orphaned_lease_recovery",
    "parse_automation_qualification",
    "parse_dispatch_markers",
    "parse_trusted_results",
    "post_merge_state",
    "pr_body_closes_declared_capability",
    "pr_reference_kind",
    "requires_exact_main_qualification",
    "should_dispatch_qualification",
    "typed_blocker_reason",
    "validate_pr_body_for_capability",
    "validate_task_admission",
]
