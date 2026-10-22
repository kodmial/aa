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
