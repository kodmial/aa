"""Unit and workflow-contract tests for the bounded campaign (issue #6).

The single owner ``/run`` comment on control issue #31 starts at most
4 x 5h runtimes inside a 24h wall-clock campaign. A 30-minute scheduled
reconciler continues only an explicitly activated campaign and stays a
permanent no-op otherwise. All assertions use ids, counts, durations and
timestamps only; no test or summary may carry message text or credentials.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aa.config import Settings
from aa.control.campaign import (
    AGGREGATE_REQUESTED_SECONDS,
    CAMPAIGN_LIFETIME_SECONDS,
    CONTROL_ISSUE_NUMBER,
    JOB_TIMEOUT_MINUTES,
    MAX_STARTS,
    RECONCILE_MINUTES,
    RUNNER_CEILING_MINUTES,
    RUNTIME_SECONDS,
    aggregate_requested_seconds,
    campaign_is_active,
    campaign_is_expired,
    campaign_summary,
    count_starts_after,
    is_run_command,
    parse_control_command,
    resolve_active_start,
    should_dispatch,
    starts_remaining,
    validate_job_timeout,
    validate_requested_duration,
)
from aa.control.runtime_control import RuntimeController

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _read(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_campaign_hard_bounds() -> None:
    assert CONTROL_ISSUE_NUMBER == 31
    assert MAX_STARTS == 4
    assert RUNTIME_SECONDS == 5 * 3600
    assert AGGREGATE_REQUESTED_SECONDS == 20 * 3600
    assert CAMPAIGN_LIFETIME_SECONDS == 24 * 3600
    assert RECONCILE_MINUTES == 30
    assert JOB_TIMEOUT_MINUTES < RUNNER_CEILING_MINUTES
    assert JOB_TIMEOUT_MINUTES == 330
    assert RUNNER_CEILING_MINUTES == 360


def test_parse_control_commands() -> None:
    assert parse_control_command("/run") == "run"
    assert parse_control_command("  /run  \n") == "run"
    assert parse_control_command("/bot status") == "status"
    assert parse_control_command("/bot stop") == "stop"
    assert parse_control_command("/bot start 1h") is None
    assert parse_control_command("/run now") is None
    assert parse_control_command("hello") is None
    assert parse_control_command("") is None
    assert is_run_command("/run")
    assert not is_run_command("/bot status")


def test_duplicate_run_is_idempotent_noop() -> None:
    now = 1_000_000.0
    accepted = now - 3600.0
    assert campaign_is_active(now, accepted, 1, stopped=False, runtime_active=False)
    # A repeated /run resolves to the same accepted start, not a new campaign.
    resolved = resolve_active_start([accepted, now - 60.0], [accepted + 10.0], [], now)
    assert resolved == accepted


def test_campaign_expires_within_24h() -> None:
    accepted = 1_000_000.0
    assert not campaign_is_expired(accepted + 23 * 3600, accepted)
    assert campaign_is_expired(accepted + 24 * 3600, accepted)
    assert not campaign_is_active(
        accepted + 25 * 3600, accepted, 1, stopped=False, runtime_active=False
    )
    # After expiry the same history resolves to no active campaign, so only
    # a fresh owner /run can start again.
    assert resolve_active_start([accepted], [accepted + 10.0], [], accepted + 25 * 3600) is None


def test_four_starts_max_and_20h_aggregate() -> None:
    accepted = 2_000_000.0
    runs = [accepted + 10.0 * (i + 1) for i in range(4)]
    assert count_starts_after(runs, accepted) == 4
    assert aggregate_requested_seconds(4) == 20 * 3600
    assert starts_remaining(4) == 0
    assert not campaign_is_active(
        accepted + 3600.0, accepted, 4, stopped=False, runtime_active=False
    )
    assert resolve_active_start([accepted], runs, [], accepted + 3600.0) is None


def test_failed_or_short_run_still_consumes_start() -> None:
    accepted = 3_000_000.0
    # Dispatches committed to the run history count even when the run
    # itself failed or ended early; there is no reset on failure.
    assert count_starts_after([accepted + 5.0], accepted) == 1
    assert not should_dispatch(
        accepted + 600.0,
        accepted,
        4,
        stopped=False,
        runtime_active=False,
    )


def test_stop_terminates_campaign() -> None:
    accepted = 4_000_000.0
    now = accepted + 3600.0
    assert not campaign_is_active(now, accepted, 1, stopped=True, runtime_active=False)
    assert not should_dispatch(now, accepted, 1, stopped=True, runtime_active=False)
    assert resolve_active_start([accepted], [accepted + 10.0], [accepted + 20.0], now) is None
    # A new owner /run after the stop starts a fresh campaign.
    fresh = now + 60.0
    assert (
        resolve_active_start([accepted, fresh], [accepted + 10.0], [accepted + 20.0], fresh + 1.0)
        == fresh
    )


def test_reconciler_dispatch_rules() -> None:
    accepted = 5_000_000.0
    now = accepted + 3600.0
    # No-op with no active campaign is handled by resolve_active_start -> None.
    assert resolve_active_start([], [], [], now) is None
    # No-op while a runtime is already active (never queue a second poller).
    assert not should_dispatch(now, accepted, 1, stopped=False, runtime_active=True)
    # Dispatch exactly one 5h runtime only when active, unexpired, starts left.
    assert should_dispatch(now, accepted, 1, stopped=False, runtime_active=False)
    assert not should_dispatch(now, accepted, 0, stopped=True, runtime_active=False)


def test_requested_duration_and_timeout_bounds() -> None:
    validate_requested_duration(float(5 * 3600))
    with pytest.raises(ValueError):
        validate_requested_duration(3600.0)
    with pytest.raises(ValueError):
        validate_requested_duration(0.0)
    validate_job_timeout(330)
    with pytest.raises(ValueError):
        validate_job_timeout(360)
    with pytest.raises(ValueError):
        validate_job_timeout(400)


def test_runtime_controller_and_settings_enforce_5h_max() -> None:
    RuntimeController(session_duration_seconds=float(5 * 3600))
    with pytest.raises(ValueError):
        RuntimeController(session_duration_seconds=float(5 * 3600 + 1))
    settings = Settings.from_env({"BOT_SESSION_DURATION_SECONDS": "18000"})
    settings.validate()
    bad = Settings.from_env({"BOT_SESSION_DURATION_SECONDS": "18001"})
    with pytest.raises(ValueError):
        bad.validate()


def test_campaign_summary_is_privacy_safe() -> None:
    accepted = 6_000_000.0
    summary = campaign_summary(accepted, 2, accepted + 3600.0, stopped=False, runtime_active=True)
    assert summary["starts_used"] == 2
    assert summary["starts_max"] == 4
    assert summary["aggregate_requested_seconds"] == 2 * 5 * 3600
    raw = str(summary)
    for probe in ("TELEGRAM", "AGE-SECRET", "prompt", "corpus"):
        assert probe not in raw


def test_runtime_workflow_is_bounded_single_poller() -> None:
    text = _read("aa-runtime.yml")
    assert "concurrency:" in text
    assert "aa-bot-runtime" in text
    assert "cancel-in-progress: false" in text
    assert "5h" in text
    assert "18000" in text
    assert "timeout-minutes: 330" in text
    assert "210" not in text
    assert "15m" not in text
    assert "verify_runtime_qualification.py" in text
    assert "restore_canonical.py --no-network-fallback" in text
    assert "restore_canonical.py --lang ru --no-network-fallback" in text
    assert "issues: write" in text
    # The runtime never runs unbounded or queues behind itself silently.
    assert "Reject duplicate runtime" in text
    for forbidden in ("sleep infinity", "tail -f /dev/null", "nohup", "tmux"):
        assert forbidden not in text


def test_control_workflow_handles_run_status_stop() -> None:
    text = _read("aa-runtime-control.yml")
    assert "issue.number == 31" in text or "CONTROL_ISSUE" in text
    assert "/run" in text
    assert "/bot status" in text
    assert "/bot stop" in text
    assert "repository_owner" in text
    assert "already active" in text
    assert "idempotent" in text or "no-op" in text
    # Duplicate starts are rejected, never replacing the current runtime.
    assert "cancel-in-progress: false" in text
    assert "duration" in text and "5h" in text


def test_reconciler_is_bounded_noop_by_default() -> None:
    text = _read("aa-runtime-reconciler.yml")
    assert "*/30 * * * *" in text
    assert "workflow_dispatch" in text
    assert "aa-runtime.yml" in text
    assert "starts_used" in text or "starts" in text
    assert "24" in text
    assert "no-op" in text
    assert "cancel-in-progress: false" in text
    # The reconciler continues a campaign; it never auto-renews one.
    assert "renew" in text
    assert "pull_request" not in text
    assert "push:" not in text or "branches" not in text or "main" in text
