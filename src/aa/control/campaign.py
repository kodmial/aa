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
    that campaign is active are idempotent no-ops. A stop posted after
    the accepted ``/run`` (and before any newer ``/run``) terminates
    that campaign, so later scheduled wake-ups stay permanent no-ops
    until a fresh owner ``/run``.
    """
    ordered = sorted(run_times)
    for index, candidate in enumerate(ordered):
        horizon = ordered[index + 1] if index + 1 < len(ordered) else None
        relevant_stops = [
            stop for stop in stop_times if stop > candidate and (horizon is None or stop < horizon)
        ]
        if relevant_stops:
            continue
        if campaign_is_expired(now, candidate):
            continue
        if count_starts_after(run_created_at, candidate) >= MAX_STARTS:
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
