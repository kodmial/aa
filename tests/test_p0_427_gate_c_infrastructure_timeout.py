"""Regression coverage for the infrastructure-timeout diagnostic harness (kodmial/aa#427).

Confirmed incident (runs 39781653706 and 39824142255, Gate C
``live-scenario-watchdog-timeout-stuck-provider``): the child's bounded
stalled-turn recovery raced the parent's campaign-budget preempt, so the
published typed stall evidence rendered fabricated ``0`` values for stage
and delivery timing, the timeout reason degraded to ``unknown``, and the
persisted marker still described the previous scenario instead of the
preempted one. This module owns the behavioral contract:

- honesty: the parent watchdog publisher names the bounded condition it
  actually observed as its ``timeout_reason``, and absent measurements are
  rendered ``unavailable`` by the diagnostic issue template, never as a
  fabricated ``0 ms``;
- marker consistency: a preempted child abort must leave the marker
  describing the preempted scenario's own staged attribution and pending
  cleanup, never the previous scenario's completed cleanup;
- preserved invariants: no fixed wall clock may kill a child that keeps
  making observable progress, and the bounded recovery lifecycle still
  holds inside its declared window and still preempts honestly after it.

Every test uses real event-loop/process timing with synthetic privacy-safe
progress markers only; no qualification prompt text is encoded here.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from pathlib import Path
from typing import Any

import pytest

from aa.opencode.errors import OpenCodeRateLimitError
from aa.qualification.live_turn_repair import (
    build_timeout_attribution,
    fatal_provider_failure,
    request_state_from_audit,
)
from aa.qualification.live_watchdog import (
    ENV_RECOVERY_ALLOWANCE_S,
    ENV_SCENARIO_INACTIVITY_S,
    ENV_TOTAL_DEADLINE_S,
    _timeout_lane,
    read_live_progress,
    record_live_progress,
    supervise_live_gate_c,
)
from aa.qualification.product_contract_live import LaneResult

ROOT = Path(__file__).resolve().parents[1]
SHA = "b" * 40
WORKFLOW = ROOT / ".github" / "workflows" / "aa-self-proving-qualification.yml"


# --- Helpers ------------------------------------------------------------


def _watchdog_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    progress_path = tmp_path / "p0427-progress.json"
    monkeypatch.setenv("AA_QUALIFICATION_LIVE_PROGRESS_PATH", str(progress_path))
    monkeypatch.delenv("AA_QUALIFICATION_JOB_STARTED_AT", raising=False)
    monkeypatch.delenv("AA_QUALIFICATION_JOB_STARTED_MONOTONIC", raising=False)
    return progress_path


def _watchdog_entry(metrics: dict[str, Any]) -> dict[str, Any]:
    entries = [
        dict(item)
        for item in metrics.get("scenario_diagnostics", [])
        if isinstance(item, dict) and item.get("watchdog_timeout")
    ]
    assert entries, metrics
    return entries[-1]


def _lane_source() -> str:
    from aa.qualification import product_contract_live as live_mod

    lines, _ = inspect.getsourcelines(live_mod.run_live_telegram_evidence_lane)
    return "".join(lines)


def _attempt_row(
    operation: str,
    causality_id: str,
    *,
    state: str,
    started: float,
    error_kind: str = "",
    failure_category: str = "",
) -> dict[str, Any]:
    """Build one realistic privacy-safe OpenCode attempt-ledger row."""
    row: dict[str, Any] = {
        "attempt_id": f"att-{operation}-{started}",
        "operation": operation,
        "causality_id": causality_id,
        "state": state,
        "attempt_started_monotonic": started,
        "latency_ms": 12.5,
    }
    if error_kind:
        row["error_kind"] = error_kind
    if failure_category:
        row["failure_category"] = failure_category
    return row


# --- Honest parent attribution ------------------------------------------


def test_watchdog_preempt_names_its_honest_timeout_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A no-real-progress preempt must carry the actual trigger, not unknown.

    The child reaches a provider stage, then goes silent forever; the local
    request evidence stays ``progress-unobservable`` because the pinned
    local request audit cannot see server-side streaming behavior. The
    parent must report the bounded condition it observed and must not have
    to guess at any server-side cause.
    """
    progress_path = _watchdog_env(monkeypatch, tmp_path)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "30")
    monkeypatch.setenv(ENV_SCENARIO_INACTIVITY_S, "0.4")
    monkeypatch.setenv(ENV_RECOVERY_ALLOWANCE_S, "0.8")

    async def lane(run_id: str) -> LaneResult:
        _ = run_id
        record_live_progress(
            "scenario-start",
            scenario_index=1,
            family="ordinary-text",
            scenario_total=2,
            completed_scenario_count=0,
            active_stage="planner",
            real_progress=True,
        )
        record_live_progress("scenario-stage", active_stage="opencode", real_progress=True)
        # Liveness heartbeat only: it never resets the inactivity bound.
        record_live_progress(
            "scenario-heartbeat",
            scenario_index=1,
            request_state="request-completed",
            request_started=True,
            request_progress_observable="progress-unobservable",
        )
        await asyncio.sleep(3.0)
        return LaneResult(lane="live-telegram-evidence", status="FAIL")

    started = time.monotonic()
    result = supervise_live_gate_c(
        lane, run_id="p0427-reason", main_sha=SHA, progress_path=progress_path
    )
    elapsed = time.monotonic() - started
    assert result.status == "FAIL"
    assert result.failed == ("live-scenario-watchdog-timeout-stuck-provider",)
    assert result.metrics["watchdog_trigger"] == "no-real-progress"
    entry = _watchdog_entry(result.metrics)
    assert entry["timeout_reason"] == "no-real-progress"
    assert entry["timeout_category"] == "stuck-provider"
    assert entry["request_state"] == "request-completed"
    assert result.metrics["watchdog_request_progress_observable"] == "progress-unobservable"
    assert entry["request_progress_observable"] == "progress-unobservable"
    # Nothing invented: with no child abort ever observed there is no
    # cleanup evidence to publish.
    assert result.metrics["watchdog_stall_cleanup"] == {}
    assert entry.get("cleanup") in (None, {})
    # The preempt is bounded, but is never framed as an exact fixed wall.
    assert 0.4 <= elapsed < 10.0, elapsed


def test_watchdog_total_deadline_preempt_is_honestly_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A campaign-budget preempt through real observable progress."""
    progress_path = _watchdog_env(monkeypatch, tmp_path)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "1")
    monkeypatch.setenv(ENV_SCENARIO_INACTIVITY_S, "0.25")

    async def lane(run_id: str) -> LaneResult:
        _ = run_id
        for _ in range(20):
            record_live_progress("scenario-stage", active_stage="opencode", real_progress=True)
            await asyncio.sleep(0.05)
        return LaneResult(lane="live-telegram-evidence", status="FAIL")

    started = time.monotonic()
    result = supervise_live_gate_c(
        lane, run_id="p0427-total", main_sha=SHA, progress_path=progress_path
    )
    elapsed = time.monotonic() - started
    assert result.metrics["watchdog_trigger"] == "total-deadline"
    entry = _watchdog_entry(result.metrics)
    assert entry["timeout_reason"] == "total-deadline"
    assert elapsed >= 1.0
    assert elapsed < 10.0


def test_progressing_child_is_not_killed_by_any_fixed_wall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A responsive child outlives its inactivity bound without preemption.

    Real timing regression for the no-fixed-wall-clock contract: the
    scenario runs many times longer than its inactivity bound while
    progress keeps arriving, and the parent must hand through the child's
    own lane result unmodified.
    """
    progress_path = _watchdog_env(monkeypatch, tmp_path)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "30")
    monkeypatch.setenv(ENV_SCENARIO_INACTIVITY_S, "0.25")

    async def lane(run_id: str) -> LaneResult:
        _ = run_id
        record_live_progress(
            "scenario-start",
            scenario_index=1,
            family="ordinary-text",
            scenario_total=1,
            completed_scenario_count=0,
            real_progress=True,
        )
        for _ in range(6):
            await asyncio.sleep(0.12)
            record_live_progress("scenario-stage", active_stage="opencode", real_progress=True)
        await asyncio.sleep(0.12)
        return LaneResult(lane="live-telegram-evidence", status="FAIL")

    started = time.monotonic()
    result = supervise_live_gate_c(
        lane, run_id="p0427-progressing", main_sha=SHA, progress_path=progress_path
    )
    elapsed = time.monotonic() - started
    assert result.status == "FAIL"
    assert result.metrics.get("watchdog_triggered") is not True
    assert "watchdog_trigger" not in result.metrics
    # The child ran about 0.9 s, several times its 0.25 s inactivity bound,
    # and was never preempted by any wall clock.
    assert elapsed >= 0.85, elapsed
    assert elapsed < 10.0


# --- Preempted recovery inherits the right stall's evidence --------------


def test_preempted_cleanup_publishes_pending_state_not_stale_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A preempted pending abort never inherits the previous stall's record."""
    progress_path = _watchdog_env(monkeypatch, tmp_path)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "30")
    monkeypatch.setenv(ENV_SCENARIO_INACTIVITY_S, "0.4")
    monkeypatch.setenv(ENV_RECOVERY_ALLOWANCE_S, "0.6")

    async def lane(run_id: str) -> LaneResult:
        _ = run_id
        # Scenario 1 stalls and its bounded cleanup completes.
        record_live_progress(
            "scenario-start",
            scenario_index=1,
            family="first-stall",
            scenario_total=2,
            completed_scenario_count=0,
            real_progress=True,
        )
        record_live_progress(
            "scenario-timeout",
            scenario_index=1,
            family="first-stall",
            timeout_category="stuck-provider",
            timeout_reason="no-real-progress",
            last_active_stage="opencode",
            request_state="request-completed",
            request_started=True,
            request_progress_observable="progress-unobservable",
            cleanup={"quiesced": True, "non_cooperative": False, "reason": "no-real-progress"},
            completed_scenario_count=1,
            scenario_total=2,
            scenario_diagnostics=[
                {
                    "scenario_index": 1,
                    "family": "first-stall",
                    "delivery_timeout": True,
                    "delivery_latency_ms": 1801000.0,
                    "stage_elapsed_ms": 1800000.0,
                    "timeout_category": "stuck-provider",
                    "cleanup": {"quiesced": True, "reason": "no-real-progress"},
                }
            ],
            real_progress=False,
        )
        # Scenario 2 progresses once, then declares recovery with the same
        # markers the production lane now writes: this scenario's staged
        # attribution, its own pending cleanup and its diagnostics.
        record_live_progress(
            "scenario-start",
            scenario_index=2,
            family="second-stall",
            scenario_total=2,
            completed_scenario_count=1,
            real_progress=True,
        )
        record_live_progress(
            "scenario-abort-start",
            scenario_index=2,
            family="second-stall",
            scenario_total=2,
            timeout_category="stuck-provider",
            timeout_reason="campaign-budget",
            last_active_stage="opencode",
            request_state="unconfirmed-in-flight",
            request_started=True,
            request_progress_observable="progress-unobservable",
            cleanup={"skipped": "campaign-budget"},
            completed_scenario_count=2,
            scenario_diagnostics=[
                {
                    "scenario_index": 2,
                    "family": "second-stall",
                    "delivery_timeout": True,
                    "delivery_latency_ms": 1320000.0,
                    "stage_elapsed_ms": 1320000.0,
                    "timeout_category": "stuck-provider",
                    "timeout_reason": "campaign-budget",
                }
            ],
            latency_samples=[
                {
                    "causality_id": "sim-2",
                    "family": "second-stall",
                    "mode": "text",
                    "outcome": "timeout",
                    "latency_ms": 1320000.0,
                    "censored": True,
                }
            ],
            stall_recovery_in_progress=True,
            stall_recovery_started_monotonic=time.monotonic(),
            real_progress=False,
        )
        await asyncio.sleep(3.0)
        return LaneResult(lane="live-telegram-evidence", status="FAIL")

    result = supervise_live_gate_c(
        lane, run_id="p0427-stale", main_sha=SHA, progress_path=progress_path
    )
    assert result.status == "FAIL"
    assert result.metrics["watchdog_trigger"] == "no-real-progress"
    # The evidence describes the preempted second stall: its request state
    # was honestly still unconfirmed, never the first stall's completed one.
    assert result.metrics["last_scenario_family"] == "second-stall"
    assert result.metrics["watchdog_request_state"] == "unconfirmed-in-flight"
    entry = _watchdog_entry(result.metrics)
    entry = _watchdog_entry(result.metrics)
    assert entry["timeout_reason"] == "no-real-progress"
    assert entry["timeout_category"] == "stuck-provider"
    # Preemption is published as the pending pre-cleanup state, not as a
    # completed cleanup and not as the previous scenario's success.
    assert result.metrics["watchdog_stall_cleanup"] == {"skipped": "campaign-budget"}
    assert entry["request_state"] == "unconfirmed-in-flight"
    # The preempted scenario's own measured entry stays in the marker.
    persisted = read_live_progress(progress_path) or {}
    diagnostics = [dict(item) for item in persisted.get("scenario_diagnostics", [])]
    assert any(
        item.get("scenario_index") == 2 and item.get("delivery_latency_ms") == 1320000.0
        for item in diagnostics
    ), diagnostics


def test_preempt_without_child_evidence_publishes_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Killed before any typed marker: no invented category or cleanup."""
    progress_path = _watchdog_env(monkeypatch, tmp_path)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "0.8")

    async def lane(run_id: str) -> LaneResult:
        _ = run_id
        record_live_progress("scenarios-start", scenario_total=1, real_progress=True)
        time.sleep(5)
        return LaneResult(lane="live-telegram-evidence", status="FAIL")

    result = supervise_live_gate_c(
        lane, run_id="p0427-no-evidence", main_sha=SHA, progress_path=progress_path
    )
    assert result.status == "FAIL"
    assert result.failed == ("live-scenario-watchdog-timeout-unknown",)
    entry = _watchdog_entry(result.metrics)
    assert entry["timeout_reason"] == "total-deadline"
    assert result.metrics["watchdog_stall_cleanup"] == {}


# --- Lane wiring invariants (source-level, mirrors other P0 tests) ------


def test_lane_persists_preempted_scenario_attribution_on_abort_start() -> None:
    """The production abort marker embeds this scenario's staged evidence."""
    source = _lane_source()
    abort_index = source.index('"scenario-abort-start"')
    quiesce_index = source.index("quiesce_and_abort_stalled_turn")
    record = source[abort_index:quiesce_index]
    assert "family=family," in record
    assert 'timeout_category=stalled_attribution["timeout_category"]' in record
    assert 'request_state=stalled_attribution["request_state"]' in record
    assert "request_progress_observable=stalled_attribution[" in record
    assert "cleanup=stall_cleanup" in record
    assert "scenario_diagnostics=scenario_diagnostics" in record
    assert "latency_samples=latency_samples" in record
    assert "stall_recovery_in_progress=True" in record


def test_watchdog_source_keeps_bounded_recovery_and_real_progress_bounds() -> None:
    """Preserved lifecycle: declared-window hold plus real-progress bound."""
    from aa.qualification.live_watchdog import supervise_live_gate_c as supervise_fn

    source = "".join(inspect.getsourcelines(supervise_fn)[0])
    assert "stall_recovery_in_progress" in source
    assert "recovery_allowance_s()" in source
    assert "inactivity_s" in source
    assert "total-deadline"


# --- Publication template honesty ---------------------------------------


def test_diagnostic_issue_renders_absent_measurements_as_unavailable() -> None:
    """Absent stage/delivery measurements must render ``unavailable``, not 0."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "stallMeasurementLine('Stage elapsed ms', stall.stage_elapsed_ms)" in workflow
    assert "stallMeasurementLine('Delivery latency ms', stall.delivery_latency_ms)" in workflow
    # The fabricating coercions are gone.
    assert "Math.round(Number(stall.stage_elapsed_ms" not in workflow
    assert "Math.round(Number(stall.delivery_latency_ms" not in workflow
    helper_source = workflow.split("function stallMeasurementLine", 1)[1]
    assert "unavailable" in helper_source


# --- Typed failing-lane marker integrity (unit level) --------------------


def test_timeout_lane_entry_carries_trigger_and_preserved_diagnostics() -> None:
    child_entry = {
        "scenario_index": 2,
        "family": "ordinary-text",
        "delivery_timeout": True,
        "timeout_category": "stuck-provider",
        "timeout_reason": "campaign-budget",
        "delivery_latency_ms": 123.0,
        "cleanup": {"skipped": "campaign-budget"},
    }
    marker: dict[str, Any] = {
        "timeout_category": "stuck-provider",
        "scenario_diagnostics": [child_entry],
        "scenario_index": 2,
        "family": "ordinary-text",
        "request_state": "unconfirmed-in-flight",
        "request_started": True,
        "request_progress_observable": "progress-unobservable",
        "cleanup": {"skipped": "campaign-budget"},
    }
    result = _timeout_lane(
        trigger="total-deadline", progress=marker, main_sha=SHA, elapsed_s=1.0, deadline_s=1.0
    )
    assert result.status == "FAIL"
    diagnostics = [dict(item) for item in result.metrics["scenario_diagnostics"]]
    parent_entries = [item for item in diagnostics if item.get("watchdog_timeout")]
    assert len(parent_entries) == 1
    parent_entry = parent_entries[0]
    # The parent names its real trigger as the timeout reason and keeps the
    # child's measured evidence preserved, never erased.
    assert parent_entry["timeout_reason"] == "total-deadline"
    assert parent_entry["timeout_category"] == "stuck-provider"
    assert parent_entry["watchdog_trigger"] == "total-deadline"
    assert any(item.get("delivery_latency_ms") == 123.0 for item in diagnostics)
    assert result.metrics["watchdog_stall_cleanup"] == {"skipped": "campaign-budget"}
    assert result.metrics["watchdog_request_state"] == "unconfirmed-in-flight"


# --- Terminal provider failure is observable, never masked ----------------


def test_cleanup_completion_never_masks_a_terminal_turn_failure() -> None:
    """A later session-delete success must not rewrite a failed turn as done.

    Confirmed incident shape (run 44611952510): the verifier's
    ``message-structured`` request failed with a provider 429, then the
    turn's ``session-delete`` cleanup completed. The newest-attempt rule
    picked the cleanup and published ``request-completed``, so the lane
    waited out the full inactivity bound and fabricated an unobservable
    infrastructure timeout.
    """
    causality = "turn-429"
    audit = [
        _attempt_row(
            "message-structured",
            causality,
            state="failed",
            started=1.0,
            error_kind="rate-limit",
            failure_category="OpenCodeRateLimitError",
        ),
        _attempt_row("session-delete", causality, state="completed", started=2.0),
    ]
    report = request_state_from_audit(audit, causality)
    assert report["request_state"] == "request-failed"
    assert report["last_request_state"] == "failed"
    assert report["last_confirmed_request"] == "message-structured"
    assert report["last_request_error_kind"] == "rate-limit"
    assert report["last_request_failure_category"] == "OpenCodeRateLimitError"
    # The timeout attribution therefore names a confirmed provider error,
    # never "server alive but session unobserved".
    attribution = build_timeout_attribution(
        causality_id=causality,
        family="ordinary-text",
        last_active_stage="opencode",
        stage_changed_at=time.monotonic() - 1.0,
        started_monotonic=time.monotonic() - 2.0,
        request_audit=audit,
        timeout_reason="no-real-progress",
    )
    assert attribution["stall_kind"] == "provider-error-confirmed"
    assert attribution["request_state"] == "request-failed"
    fatal = fatal_provider_failure(audit, causality)
    assert fatal is not None
    assert fatal["error_kind"] == "rate-limit"
    assert fatal["operation"] == "message-structured"


def test_newer_turn_attempt_supersedes_an_earlier_terminal_failure() -> None:
    """A retry still running or completed stays authoritative for the turn."""
    causality = "turn-retry"
    failed = _attempt_row(
        "message-structured",
        causality,
        state="failed",
        started=1.0,
        error_kind="rate-limit",
    )
    recovered = _attempt_row("message-structured", causality, state="completed", started=2.0)
    completed = request_state_from_audit([failed, recovered], causality)
    assert completed["request_state"] == "request-completed"
    assert fatal_provider_failure([failed, recovered], causality) is None

    in_flight = _attempt_row("message-structured", causality, state="in-flight", started=2.0)
    running = request_state_from_audit([failed, in_flight], causality)
    assert running["request_state"] == "http-in-flight"
    assert fatal_provider_failure([failed, in_flight], causality) is None


def test_non_fatal_provider_errors_are_not_surfaced_as_fatal() -> None:
    """Only a 429 is non-deliverable; other errors still produce a reply."""
    causality = "turn-soft-error"
    for kind in ("transient", "deterministic", "provider-access", "session-not-found"):
        audit = [
            _attempt_row(
                "message-structured",
                causality,
                state="failed",
                started=1.0,
                error_kind=kind,
                failure_category="OpenCodeError",
            )
        ]
        assert request_state_from_audit(audit, causality)["request_state"] == "request-failed"
        assert fatal_provider_failure(audit, causality) is None


def test_confirmed_429_reaches_the_runner_lifecycle_without_any_wall_clock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Real timing: a terminal 429 exits promptly, not after the silence bound.

    The synthetic lane owns a realistic attempt ledger: the turn is
    in-flight first, then the verifier's message request terminally fails
    with a 429 while a later session-delete completes. The same pure
    detector the production await loop uses must surface the failure in
    about a probe period, long before the deliberately huge inactivity
    bound, and the existing parent lifecycle must re-raise the provider
    error unchanged.
    """
    progress_path = _watchdog_env(monkeypatch, tmp_path)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "30")
    monkeypatch.setenv(ENV_SCENARIO_INACTIVITY_S, "30")
    causality = "turn-live-429"
    audit: list[dict[str, Any]] = [
        _attempt_row("message-structured", causality, state="in-flight", started=1.0)
    ]

    async def lane(run_id: str) -> LaneResult:
        _ = run_id
        record_live_progress(
            "scenario-start",
            scenario_index=1,
            family="ordinary-text",
            scenario_total=2,
            completed_scenario_count=0,
            active_stage="opencode",
            real_progress=True,
        )
        started = time.monotonic()
        while time.monotonic() - started < 20.0:
            await asyncio.sleep(0.02)
            if time.monotonic() - started >= 0.3:
                audit[0] = _attempt_row(
                    "message-structured",
                    causality,
                    state="failed",
                    started=1.0,
                    error_kind="rate-limit",
                    failure_category="OpenCodeRateLimitError",
                )
                audit.append(
                    _attempt_row("session-delete", causality, state="completed", started=2.0)
                )
            if fatal_provider_failure(audit, causality) is not None:
                raise OpenCodeRateLimitError("opencode provider rate-limited")
        return LaneResult(lane="live-telegram-evidence", status="FAIL")

    started = time.monotonic()
    with pytest.raises(OpenCodeRateLimitError):
        supervise_live_gate_c(lane, run_id="p0427-429", main_sha=SHA, progress_path=progress_path)
    elapsed = time.monotonic() - started
    # Bounded by the event, not by the 30 s inactivity/campaign walls.
    assert 0.2 <= elapsed < 8.0, elapsed


def _first_scenario_await_loop(source: str) -> str:
    """Extract only the main scenario await loop from the lane source.

    A slice that runs to the end of the function also contains the later
    multi-turn and voice await loops, so a probe moved out of the main
    scenario loop would still satisfy the assertions. Bound the extraction
    by indentation so only the first loop's body is inspected.
    """
    lines = source.splitlines(keepends=True)
    start = next(
        index for index, line in enumerate(lines) if "while len(api.sent_texts) <= before:" in line
    )
    loop_indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines):
        stripped = lines[end].strip()
        if stripped:
            line_indent = len(lines[end]) - len(lines[end].lstrip())
            if line_indent <= loop_indent:
                break
        end += 1
    return "".join(lines[start:end])


def test_lane_probes_the_request_audit_inside_the_await_loop() -> None:
    """Production wiring: the await loop raises the typed 429 promptly."""
    source = _lane_source()
    loop = _first_scenario_await_loop(source)
    # The bounded slice must be the main scenario loop, never a later
    # multi-turn/voice loop that happens to contain the same wait shape.
    assert "while len(api.sent_texts) <= before:" in loop
    assert loop.count("while len(api.sent_texts) <= before:") == 1
    assert "_send_raw_text" not in loop
    # The await loop samples the already-published request audit through the
    # exception-safe helper while a scenario is still waiting for delivery.
    assert "_fatal_request_failure(" in loop
    assert "raise OpenCodeRateLimitError(" in loop
    assert "scenario-provider-error" in loop
    # The detector itself reads the same request evidence used for stall
    # attribution, so a later cleanup attempt cannot hide the failed turn.
    helper = source[source.index("def _fatal_request_failure") :]
    helper = helper[: helper.index("def _collect_partial_evidence")]
    assert "fatal_provider_failure(" in helper
    assert "request_latency_audit" in helper
    # No fixed per-turn cutoff is introduced by the probe: only the existing
    # inactivity bound and an event-driven sampling interval remain.
    assert "_SCENARIO_PROVIDER_PROBE_INTERVAL_S" in loop
    assert "scenario_inactivity_limit_s" in loop
    assert "await asyncio.sleep(0.02)" in loop
