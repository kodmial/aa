"""Issue-driven bounded manual-testing campaign (issue #6).

The complete bot and OpenCode run together in one bounded GitHub Actions
job that is started, stopped and inspected from the dedicated repository
control issue. A single owner ``/run`` comment starts one manual-testing
campaign with deterministic hard bounds:

- maximum 4 runtime dispatches;
- each runtime runs a fixed 5-hour application window after readiness;
- maximum 20 hours aggregate requested runtime;
- unconditional campaign expiry 24 hours after the accepted ``/run``;
- no automatic renewal, rollover, reset or extension;
- a repeated ``/run`` while a campaign is active is an idempotent no-op;
- a completed/expired/stopped campaign restarts only via a new owner
  ``/run`` comment.

Campaign state is derived from durable GitHub state only (owner comments
on the control issue plus the authoritative ``aa-runtime.yml`` workflow
run history), so a fresh reconciler wake-up re-reads the same truth and
duplicate cron/events cannot create two pollers. A successfully committed
runtime dispatch consumes one campaign start even when the run later
fails or ends early, which prevents failure loops.

All helpers here are pure and privacy-safe: they handle only command
tokens, timestamps, run ids and counters, never Telegram message bodies,
prompts, corpus text or credentials.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

CONTROL_ISSUE_NUMBER = 31

RUN_COMMAND = "/run"
STATUS_COMMAND = "/bot status"
STOP_COMMAND = "/bot stop"

# Campaign hard bounds (issue #6).
MAX_STARTS = 4
RUNTIME_SECONDS = 5 * 3600
AGGREGATE_REQUESTED_SECONDS = 20 * 3600
CAMPAIGN_LIFETIME_SECONDS = 24 * 3600

# Operational bounds: the reconciler wakes every 30 minutes, and the
# runner/job timeout stays below the 6-hour hosted-runner ceiling so
# Telegram/OpenCode cleanup always has headroom.
RECONCILE_MINUTES = 30
JOB_TIMEOUT_MINUTES = 330
RUNNER_CEILING_MINUTES = 360

ACTIVE_RUN_STATUSES: tuple[str, ...] = (
    "queued",
    "in_progress",
    "waiting",
    "requested",
    "pending",
)


def parse_control_command(text: str) -> str | None:
    """Return the recognized control command for a comment body.

    Recognizes exactly ``/run``, ``/bot status`` and ``/bot stop``
    (surrounding whitespace is ignored). Anything else returns ``None``
    so callers can reject it without echoing untrusted text.
    """
    normalized = text.strip()
    if normalized == RUN_COMMAND:
        return "run"
    if normalized == STATUS_COMMAND:
        return "status"
    if normalized == STOP_COMMAND:
        return "stop"
    return None


def is_run_command(text: str) -> bool:
    """Whether ``text`` is exactly the owner ``/run`` campaign command."""
    return parse_control_command(text) == "run"


def campaign_expires_at(accepted_at: float) -> float:
    """Return the unconditional campaign expiry epoch for ``accepted_at``."""
    return accepted_at + float(CAMPAIGN_LIFETIME_SECONDS)


def campaign_is_expired(now: float, accepted_at: float) -> bool:
    """Whether the campaign started at ``accepted_at`` has expired."""
    return now - accepted_at >= float(CAMPAIGN_LIFETIME_SECONDS)


def starts_remaining(starts_used: int) -> int:
    """Return how many of the 4 campaign starts are still available."""
    return max(0, MAX_STARTS - starts_used)


def aggregate_requested_seconds(starts_used: int) -> int:
    """Return the aggregate requested runtime for ``starts_used`` starts."""
    return starts_used * RUNTIME_SECONDS


def validate_requested_duration(seconds: float) -> None:
    """Reject any campaign runtime duration other than exactly 5 hours."""
    if seconds != float(RUNTIME_SECONDS):
        raise ValueError(f"campaign runtime must be exactly {RUNTIME_SECONDS}s (5h)")


def validate_job_timeout(timeout_minutes: int) -> None:
    """Reject job timeouts at or above the 6-hour runner ceiling."""
    if timeout_minutes <= 0:
        raise ValueError("job timeout must be positive")
    if timeout_minutes >= RUNNER_CEILING_MINUTES:
        raise ValueError(
            f"job timeout ({timeout_minutes}m) must stay below "
            f"the {RUNNER_CEILING_MINUTES}m runner ceiling"
        )


def count_starts_after(run_created_at: list[float], accepted_at: float) -> int:
    """Count committed runtime dispatches created after ``accepted_at``.

    Every successfully committed dispatch consumes one campaign start,
    including runs that later fail, end early or are cancelled, which
    prevents failure loops from minting fresh starts.
    """
    return sum(1 for created in run_created_at if created > accepted_at)


def campaign_stopped_after(stop_times: list[float], accepted_at: float) -> bool:
    """Whether an owner stop terminated the campaign from ``accepted_at``."""
    return any(stop > accepted_at for stop in stop_times)


def resolve_active_start(
    run_times: list[float],
    run_created_at: list[float],
    stop_times: list[float],
    now: float,
) -> float | None:
    """Return the accepted ``/run`` epoch of the active campaign, if any.

    ``run_times`` are owner ``/run`` comment epochs, ``run_created_at``
    are ``aa-runtime.yml`` run creation epochs, and ``stop_times`` are
    owner ``/bot stop`` epochs. The earliest candidate that is unexpired,
    unterminated and still has starts left wins; duplicates posted while
    that campaign is active are idempotent no-ops. Any stop posted after
    the accepted ``/run`` terminates that campaign, so later scheduled
    wake-ups stay permanent no-ops until a fresh owner ``/run`` posted
    after the stop.
    """
    ordered = sorted(run_times)
    for candidate in ordered:
        relevant_stops = [stop for stop in stop_times if stop > candidate]
        if relevant_stops:
            continue
        if campaign_is_expired(now, candidate):
            continue
        if count_starts_after(run_created_at, candidate) >= MAX_STARTS:
            continue
        exhausted_before = False
        for prev in ordered:
            if prev >= candidate:
                break
            prev_runs = sorted(t for t in run_created_at if t > prev)
            if len(prev_runs) >= MAX_STARTS and candidate <= prev_runs[MAX_STARTS - 1]:
                exhausted_before = True
                break
        if exhausted_before:
            continue
        return candidate
    return None


def campaign_is_active(
    now: float,
    accepted_at: float,
    starts_used: int,
    *,
    stopped: bool,
    runtime_active: bool = False,
) -> bool:
    """Whether the campaign from ``accepted_at`` is still active.

    ``runtime_active`` does not change the answer: an active poller
    means the reconciler must no-op, but the campaign itself stays
    active until it expires, is stopped or exhausts its starts.
    """
    _ = runtime_active
    if stopped:
        return False
    if campaign_is_expired(now, accepted_at):
        return False
    return starts_used < MAX_STARTS


def should_dispatch(
    now: float,
    accepted_at: float,
    starts_used: int,
    *,
    stopped: bool,
    runtime_active: bool,
) -> bool:
    """Whether the reconciler may dispatch exactly one 5-hour runtime."""
    if runtime_active:
        return False
    return campaign_is_active(
        now, accepted_at, starts_used, stopped=stopped, runtime_active=runtime_active
    )


def campaign_summary(
    accepted_at: float,
    starts_used: int,
    now: float,
    *,
    stopped: bool,
    runtime_active: bool,
) -> dict[str, object]:
    """Return a privacy-safe campaign snapshot (ids/counts/durations only)."""
    expires_at = campaign_expires_at(accepted_at)
    return {
        "control_issue": CONTROL_ISSUE_NUMBER,
        "starts_used": starts_used,
        "starts_max": MAX_STARTS,
        "starts_remaining": starts_remaining(starts_used),
        "runtime_seconds": RUNTIME_SECONDS,
        "aggregate_requested_seconds": aggregate_requested_seconds(starts_used),
        "aggregate_max_seconds": AGGREGATE_REQUESTED_SECONDS,
        "campaign_lifetime_seconds": CAMPAIGN_LIFETIME_SECONDS,
        "expires_in_seconds": max(0, int(expires_at - now)),
        "expired": campaign_is_expired(now, accepted_at),
        "stopped": stopped,
        "runtime_active": runtime_active,
        "active": campaign_is_active(
            now, accepted_at, starts_used, stopped=stopped, runtime_active=runtime_active
        ),
    }


# ---------------------------------------------------------------------------
# Authoritative idempotent dispatch (issue #119).
#
# Single shared state-transition implementation for the owner `/run`
# command handler and the scheduled reconciler. Both callers must use
# these helpers (or mirror them exactly in workflow JS) instead of
# maintaining divergent campaign semantics.
#
# Durable state model (existing representation only):
# - owner `/run` / `/bot stop` comments on the control issue, plus
# - committed `aa-runtime.yml` workflow runs, plus
# - machine-readable dispatch markers posted as control-issue comments
#   after a dispatch is accepted/committed.
#
# Rules:
# - the command is parsed from the immutable event payload, never from a
#   re-read/edited comment body;
# - pre-existing campaign resolution excludes the current event comment ID,
#   so the initiating comment can never self-match as "already active";
# - re-delivery of the same event (same comment ID) is idempotent via the
#   dispatch marker for that source comment ID;
# - a marker is posted only after the dispatch API accepts the run
#   (failure-before-commit consumes no start);
# - once committed, the start is consumed even if the runtime later fails
#   (failed runs still count via the run history);
# - effective starts = max(committed runs after accepted_at,
#   committed markers for the generation), which bridges the
#   eventually-consistent run listing without ever double-dispatching to
#   "repair" an uncertain dispatch (exactly one dispatch attempt per
#   serialized handler execution);
# - control and reconciler handlers serialize on one concurrency group and
#   always re-read authoritative state immediately before deciding, so the
#   second contender sees the first contender's marker/run and no-ops.
# ---------------------------------------------------------------------------

DISPATCH_MARKER_KIND = "aa-campaign-dispatch"

_MARKER_RE = re.compile(
    r"<!--\s*aa-campaign-dispatch\s+"
    r"generation=(?P<generation>\S+)\s+"
    r"seq=(?P<seq>\d+)"
    r"(?:\s+source-comment=(?P<source>\S+))?"
    r"\s*-->",
)

InitialDispatchAction = Literal[
    "dispatch_new", "noop_duplicate", "noop_active", "noop_runtime_active", "noop_invalid"
]

ReconcileAction = Literal[
    "dispatch",
    "noop_no_campaign",
    "noop_runtime_active",
    "noop_pending",
    "noop_exhausted",
    "noop_inactive",
]


@dataclass(frozen=True)
class ControlComment:
    """Durable control-issue comment (ids/counters only, never message text)."""

    comment_id: int
    body: str
    created_at: float
    is_owner: bool


@dataclass(frozen=True)
class RuntimeRun:
    """Authoritative runtime run reference (ids/counters only)."""

    run_id: int
    created_at: float
    status: str
    conclusion: str | None


@dataclass(frozen=True)
class CampaignGeneration:
    """Explicit durable campaign identity for one accepted `/run`."""

    accepted_at: float
    source_comment_id: int
    generation_id: str


@dataclass(frozen=True)
class InitialDispatchDecision:
    """Shared outcome for the owner `/run` event handler."""

    action: InitialDispatchAction
    generation: CampaignGeneration | None
    starts_used: int
    next_seq: int


@dataclass(frozen=True)
class ReconcileDecision:
    """Shared outcome for the scheduled reconciler."""

    action: ReconcileAction
    generation: CampaignGeneration | None
    starts_used: int
    next_seq: int


def is_authorized_control_event(actor: str, owner: str, issue_number: int) -> bool:
    """Whether an event is an authenticated owner command on the control issue."""
    return bool(actor) and actor == owner and issue_number == CONTROL_ISSUE_NUMBER


def generation_id_for_comment(comment_id: int) -> str:
    """Return the explicit durable generation id for a `/run` comment id."""
    return f"c{int(comment_id)}"


def format_dispatch_marker(
    generation_id: str, seq: int, source_comment_id: int | None = None
) -> str:
    """Return the machine-readable commit marker for a successful dispatch."""
    suffix = f" source-comment={int(source_comment_id)}" if source_comment_id is not None else ""
    return f"<!-- {DISPATCH_MARKER_KIND} generation={generation_id} seq={int(seq)}{suffix} -->"


def parse_dispatch_marker(body: str) -> dict[str, object] | None:
    """Parse one dispatch marker from a comment body, if present."""
    match = _MARKER_RE.search(body)
    if match is None:
        return None
    return {
        "generation": match.group("generation"),
        "seq": int(match.group("seq")),
        "source_comment": match.group("source"),
    }


def markers_for_generation(bodies: list[str], generation_id: str) -> list[int]:
    """Return committed seq numbers for ``generation_id`` (durable markers).

    Deduplicated by seq: a retried marker post after an ambiguous success
    may persist the same seq twice, and counting both would wedge the
    reconciler in ``noop_pending`` (markers greater than runs forever).
    """
    seqs: set[int] = set()
    for body in bodies:
        parsed = parse_dispatch_marker(body)
        if parsed is not None and parsed.get("generation") == generation_id:
            value = parsed.get("seq")
            if isinstance(value, int):
                seqs.add(value)
    return sorted(seqs)


def has_marker_for_source(bodies: list[str], source_comment_id: int) -> bool:
    """Whether a committed dispatch marker already exists for a source comment."""
    wanted = str(int(source_comment_id))
    for body in bodies:
        parsed = parse_dispatch_marker(body)
        if parsed is not None and parsed.get("source_comment") == wanted:
            return True
    return False


def effective_starts_used(runs_after_accepted: int, marker_count: int) -> int:
    """Return the authoritative start counter bridging eventual consistency."""
    return max(int(runs_after_accepted), int(marker_count))


def resolve_active_generation(
    comments: list[ControlComment],
    run_created_at: list[float],
    now: float,
    *,
    exclude_comment_id: int | None = None,
) -> CampaignGeneration | None:
    """Resolve the pre-existing active campaign generation, if any.

    ``exclude_comment_id`` is the current event comment id and is always
    excluded, so the initiating `/run` can never self-match as already
    active. All other semantics match :func:`resolve_active_start`.
    """
    candidates: list[tuple[float, int]] = sorted(
        (c.created_at, c.comment_id)
        for c in comments
        if c.is_owner
        and (exclude_comment_id is None or c.comment_id != exclude_comment_id)
        and parse_control_command(c.body) == "run"
    )
    stop_times = [
        c.created_at for c in comments if c.is_owner and parse_control_command(c.body) == "stop"
    ]
    ordered = sorted(run_created_at)
    ordered_pairs = sorted(candidates)
    for index, (candidate, source_id) in enumerate(ordered_pairs):
        if any(stop > candidate for stop in stop_times):
            continue
        if campaign_is_expired(now, candidate):
            continue
        if count_starts_after(ordered, candidate) >= MAX_STARTS:
            continue
        exhausted_before = False
        for prev, _prev_id in ordered_pairs[:index]:
            prev_runs = sorted(t for t in ordered if t > prev)
            if len(prev_runs) >= MAX_STARTS and candidate <= prev_runs[MAX_STARTS - 1]:
                exhausted_before = True
                break
        if exhausted_before:
            continue
        return CampaignGeneration(
            accepted_at=candidate,
            source_comment_id=source_id,
            generation_id=generation_id_for_comment(source_id),
        )
    return None


def effective_starts_for_generation(
    generation: CampaignGeneration,
    comment_bodies: list[str],
    run_created_at: list[float],
) -> int:
    """Return the authoritative starts-used counter for a generation."""
    runs_after = count_starts_after(sorted(run_created_at), generation.accepted_at)
    markers = len(markers_for_generation(comment_bodies, generation.generation_id))
    return effective_starts_used(runs_after, markers)


def decide_initial_dispatch(
    *,
    event_body: str,
    event_comment_id: int,
    event_created_at: float,
    comments: list[ControlComment],
    comment_bodies: list[str],
    run_created_at: list[float],
    runtime_active: bool,
    now: float,
) -> InitialDispatchDecision:
    """Decide one owner `/run` event using the shared transition.

    ``event_body`` is the immutable payload body (never a re-read body).
    ``comments`` is the re-read history including all markers. Exactly one
    dispatch attempt follows only from ``dispatch_new``; every ``noop_*``
    must not dispatch.
    """
    if parse_control_command(event_body) != "run":
        return InitialDispatchDecision(
            action="noop_invalid", generation=None, starts_used=0, next_seq=0
        )
    if has_marker_for_source(comment_bodies, event_comment_id):
        preexisting = resolve_active_generation(
            comments, run_created_at, now, exclude_comment_id=event_comment_id
        )
        if preexisting is not None:
            starts = effective_starts_for_generation(preexisting, comment_bodies, run_created_at)
            return InitialDispatchDecision(
                action="noop_duplicate",
                generation=preexisting,
                starts_used=starts,
                next_seq=starts + 1,
            )
        # Marker exists but the generation itself expired/stopped: the event
        # was already committed and must never consume another start.
        return InitialDispatchDecision(
            action="noop_duplicate", generation=None, starts_used=0, next_seq=0
        )
    preexisting = resolve_active_generation(
        comments, run_created_at, now, exclude_comment_id=event_comment_id
    )
    if preexisting is not None:
        starts = effective_starts_for_generation(preexisting, comment_bodies, run_created_at)
        return InitialDispatchDecision(
            action="noop_active",
            generation=preexisting,
            starts_used=starts,
            next_seq=starts + 1,
        )
    if runtime_active:
        return InitialDispatchDecision(
            action="noop_runtime_active", generation=None, starts_used=0, next_seq=0
        )
    new_generation = CampaignGeneration(
        accepted_at=float(event_created_at),
        source_comment_id=int(event_comment_id),
        generation_id=generation_id_for_comment(event_comment_id),
    )
    return InitialDispatchDecision(
        action="dispatch_new", generation=new_generation, starts_used=0, next_seq=1
    )


def decide_reconcile(
    *,
    comments: list[ControlComment],
    comment_bodies: list[str],
    run_created_at: list[float],
    runtime_active: bool,
    now: float,
) -> ReconcileDecision:
    """Decide one scheduled reconciler tick using the shared transition."""
    generation = resolve_active_generation(comments, run_created_at, now)
    if generation is None:
        return ReconcileDecision(
            action="noop_no_campaign", generation=None, starts_used=0, next_seq=0
        )
    if runtime_active:
        starts = effective_starts_for_generation(generation, comment_bodies, run_created_at)
        return ReconcileDecision(
            action="noop_runtime_active",
            generation=generation,
            starts_used=starts,
            next_seq=starts + 1,
        )
    runs_after = count_starts_after(sorted(run_created_at), generation.accepted_at)
    markers = len(markers_for_generation(comment_bodies, generation.generation_id))
    starts = effective_starts_used(runs_after, markers)
    if starts >= MAX_STARTS:
        return ReconcileDecision(
            action="noop_exhausted",
            generation=generation,
            starts_used=starts,
            next_seq=starts + 1,
        )
    if markers > runs_after:
        # A dispatch was committed but its run is not yet listed (eventual
        # consistency). Wait for the runtime to appear; never dispatch a
        # second poller to "repair" the uncertain window.
        return ReconcileDecision(
            action="noop_pending",
            generation=generation,
            starts_used=starts,
            next_seq=starts + 1,
        )
    stopped = any(
        c.is_owner
        and parse_control_command(c.body) == "stop"
        and c.created_at > generation.accepted_at
        for c in comments
    )
    if not campaign_is_active(
        now, generation.accepted_at, starts, stopped=stopped, runtime_active=False
    ):
        if starts >= MAX_STARTS:
            return ReconcileDecision(
                action="noop_exhausted",
                generation=generation,
                starts_used=starts,
                next_seq=starts + 1,
            )
        return ReconcileDecision(
            action="noop_inactive",
            generation=generation,
            starts_used=starts,
            next_seq=starts + 1,
        )
    return ReconcileDecision(
        action="dispatch", generation=generation, starts_used=starts, next_seq=starts + 1
    )


def status_snapshot(
    *,
    comments: list[ControlComment],
    comment_bodies: list[str],
    run_created_at: list[float],
    runtime_active: bool,
    now: float,
) -> dict[str, object]:
    """Return a privacy-safe status snapshot with pending-dispatch awareness."""
    generation = resolve_active_generation(comments, run_created_at, now)
    if generation is None:
        return {"active": False, "generation": None, "starts_used": 0}
    starts = effective_starts_for_generation(generation, comment_bodies, run_created_at)
    runs_after = count_starts_after(sorted(run_created_at), generation.accepted_at)
    markers = len(markers_for_generation(comment_bodies, generation.generation_id))
    stopped = any(
        c.is_owner
        and parse_control_command(c.body) == "stop"
        and c.created_at > generation.accepted_at
        for c in comments
    )
    return {
        "active": campaign_is_active(
            now,
            generation.accepted_at,
            starts,
            stopped=stopped,
            runtime_active=runtime_active,
        ),
        "generation": generation.generation_id,
        "source_comment": generation.source_comment_id,
        "accepted_at": generation.accepted_at,
        "starts_used": starts,
        "starts_max": MAX_STARTS,
        "pending_dispatch": markers > runs_after,
        "stopped": stopped,
        "expired": campaign_is_expired(now, generation.accepted_at),
        "runtime_active": runtime_active,
    }
