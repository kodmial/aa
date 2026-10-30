"""kodmial/aa#453: end-to-end watchdog kill preserves the measured diagnostics.

A lane runner writes synthetic privacy-safe progress markers with the
incremental ``partial_evidence`` snapshot (exactly the contract the real
child lane follows), then hangs forever; the parent watchdog preempts on
``no-real-progress`` and the resulting FAIL lane must carry the complete
already-measured partial diagnostics with explicit coverage, honest
latency aggregates and distinct event identity — instead of the empty
metrics the incident run lost.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from aa.qualification.live_partial_evidence import (
    EVENT_KIND_CHILD_TIMEOUT,
    EVENT_KIND_WATCHDOG_TERMINATION,
    build_partial_evidence_snapshot,
)
from aa.qualification.live_watchdog import (
    ENV_SCENARIO_INACTIVITY_S,
    ENV_TOTAL_DEADLINE_S,
    record_live_progress,
    supervise_live_gate_c,
)
from aa.qualification.product_contract_live import LaneResult

SHA = "c" * 40
CAUSALITY = "da" * 8


def _partial(phase: str, *, started: int, interrupted: int) -> dict[str, Any]:
    return build_partial_evidence_snapshot(
        phase=phase,
        scenario_index=2,
        family="ordinary-text",
        causality_id=CAUSALITY,
        current_request_state="in-flight",
        scenario_total=2,
        completed_scenario_count=1,
        scenarios_started=started,
        scenarios_interrupted=interrupted,
        product_outcome_counts={"served": 1},
        delivered_service_error_count=0,
        stage_snapshots=[
            {
                "planner_latency_ms": 480.0,
                "planner_outcome": "llm-primary",
                "answer_latency_ms": 0,
                "answer_outcome": "",
            }
        ],
        attempt_entries=(
            {
                "causality_id": CAUSALITY,
                "operation": "message",
                "state": "in-flight",
                "started_at_s": 9000.0,
            },
        ),
        served_audit=({"agent": "planner", "served": "model/alpha"},),
        token_audit=({"agent": "planner", "input": 7, "output": 2},),
        request_audit=(
            {
                "causality_id": CAUSALITY,
                "operation": "message",
                "state": "completed",
                "latency_ms": 640.0,
            },
        ),
        process_state={"owned_by_runtime": True, "alive": True},
        server_evidence={
            "schema": "aa-opencode-server-evidence/1",
            "state": "supported",
            "last_session_state": "busy",
        },
    )


def test_watchdog_preempt_preserves_full_partial_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    progress_path = tmp_path / "p0453-progress.json"
    monkeypatch.setenv("AA_QUALIFICATION_LIVE_PROGRESS_PATH", str(progress_path))
    monkeypatch.delenv("AA_QUALIFICATION_JOB_STARTED_AT", raising=False)
    monkeypatch.delenv("AA_QUALIFICATION_JOB_STARTED_MONOTONIC", raising=False)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "30")
    monkeypatch.setenv(ENV_SCENARIO_INACTIVITY_S, "0.3")

    async def lane(run_id: str) -> LaneResult:
        _ = run_id
        # Scenario 1 completes with a measured sample.
        record_live_progress(
            "scenario-start",
            scenario_index=1,
            family="ordinary-text",
            scenario_total=2,
            completed_scenario_count=0,
            real_progress=True,
            partial_evidence=_partial("scenario-start", started=1, interrupted=0),
        )
        record_live_progress(
            "scenario-complete",
            scenario_index=1,
            family="ordinary-text",
            completed_scenario_count=1,
            scenario_diagnostics=[
                {
                    "scenario_index": 1,
                    "family": "ordinary-text",
                    "delivery_timeout": False,
                    "production_outcome": "served",
                }
            ],
            latency_samples=[
                {
                    "causality_id": CAUSALITY,
                    "family": "ordinary-text",
                    "outcome": "served",
                    "latency_ms": 1500.0,
                    "censored": False,
                    "product_outcome": "served",
                }
            ],
            real_progress=True,
            partial_evidence=_partial("scenario-complete", started=1, interrupted=0),
        )
        # Scenario 2 stalls; the child records its own scenario timeout.
        record_live_progress(
            "scenario-start",
            scenario_index=2,
            family="ordinary-text",
            scenario_total=2,
            completed_scenario_count=1,
            real_progress=True,
            partial_evidence=_partial("scenario-start", started=2, interrupted=0),
        )
        record_live_progress(
            "scenario-timeout",
            scenario_index=2,
            family="ordinary-text",
            timeout_category="stuck-provider",
            timeout_reason="no-real-progress",
            scenario_diagnostics=[
                {
                    "scenario_index": 2,
                    "family": "ordinary-text",
                    "delivery_timeout": True,
                    "event_kind": EVENT_KIND_CHILD_TIMEOUT,
                    "event_id": "child-e2",
                    "turn_causality_id": CAUSALITY,
                    "timeout_category": "stuck-provider",
                }
            ],
            completed_scenario_count=1,
            partial_evidence=_partial("scenario-timeout", started=2, interrupted=1),
        )
        # ... then hangs forever (wedged turn).
        await asyncio.sleep(10.0)
        return LaneResult(lane="live-telegram-evidence", status="FAIL")

    started = time.monotonic()
    result = supervise_live_gate_c(
        lane, run_id="p0453-preserve", main_sha=SHA, progress_path=progress_path
    )
    elapsed = time.monotonic() - started
    assert result.status == "FAIL"
    assert result.metrics["watchdog_trigger"] == "no-real-progress"
    assert elapsed < 10.0
    metrics = result.metrics

    # --- Measured child diagnostics survive the kill ----------------------
    assert metrics["watchdog_diagnostics_status"] == "observed-partial"
    assert metrics["partial_evidence_reported"] is True
    attempts = metrics["opencode_attempt_audit"]
    assert attempts and attempts[-1]["state"] == "in-flight"
    assert metrics["served_models_by_agent"] == {"planner": ["model/alpha"]}
    assert metrics["opencode_token_usage_by_agent"] == {
        "planner": {
            "requests": 1,
            "input_total": 7,
            "output_total": 2,
            "reasoning_total": 0,
            "cache_read_total": 0,
            "cache_write_total": 0,
        }
    }
    assert metrics["opencode_request_latency_ms"]["message"]["p95"] == 640.0
    assert metrics["opencode_runtime_process"] == {"owned_by_runtime": True, "alive": True}
    assert metrics["opencode_server_events"]["state"] == "supported"
    assert metrics["opencode_server_events"]["last_session_state"] == "busy"
    assert metrics["stage_outcome_counts"]["planner"] == {"llm-primary": 1}
    assert metrics["stage_latency_ms"]["planner"]["p95"] == 480.0
    assert metrics["stage_latency_coverage"]["stages"]["planner"] == {"measured": 1}
    assert metrics["stage_latency_coverage"]["stages"]["answer"] == {"unobserved": 1}
    assert metrics["scenarios_started"] == 2
    assert metrics["scenarios_interrupted"] == 1
    assert metrics["product_outcome_counts"] == {"served": 1}

    # --- Honest latency aggregates over persisted samples -----------------
    assert metrics["turn_latency_sample_count"] == 1
    assert metrics["turn_latency_p50_ms"] == 1500.0
    assert metrics["turn_latency_p95_ms"] == 1500.0
    assert metrics["turn_latency_max_ms"] == 1500.0

    # --- No double counting of the preempted scenario ----------------------
    assert metrics["interrupted_scenarios_distinct"] == 1
    kinds = [
        str(item.get("event_kind"))
        for item in metrics["scenario_diagnostics"]
        if item.get("delivery_timeout")
    ]
    assert kinds.count(EVENT_KIND_CHILD_TIMEOUT) == 1
    assert kinds.count(EVENT_KIND_WATCHDOG_TERMINATION) == 1
    watchdog_rows = [
        item
        for item in metrics["scenario_diagnostics"]
        if item.get("event_kind") == EVENT_KIND_WATCHDOG_TERMINATION
    ]
    assert watchdog_rows[0]["turn_causality_id"] == CAUSALITY
    assert watchdog_rows[0]["event_id"]
