"""kodmial/aa#453: watchdog kills preserve complete partial diagnostics."""

from typing import Any, cast

import pytest

from aa.qualification.c_focus import build_c_focus_result
from aa.qualification.live_partial_evidence import (
    DIAGNOSTICS_UNAVAILABLE,
    EVENT_KIND_CHILD_TIMEOUT,
    EVENT_KIND_WATCHDOG_TERMINATION,
    SCHEMA_VERSION,
    build_partial_evidence_snapshot,
    merge_partial_into_metrics,
    stage_latency_coverage,
    stage_latency_view,
    validated_partial_evidence,
)
from aa.qualification.live_watchdog import _failed_lane, _timeout_lane

pytestmark = pytest.mark.usefixtures("monkeypatch")


def _make_partial(**overrides: object) -> dict[str, object]:
    base = build_partial_evidence_snapshot(
        phase="scenario-heartbeat",
        scenario_index=11,
        family="actionable-question",
        causality_id="caus" * 8,
        current_request_state="in-flight",
        scenario_total=18,
        completed_scenario_count=10,
        scenarios_started=11,
        scenarios_interrupted=1,
        product_outcome_counts={"served": 9},
        delivered_service_error_count=1,
        stage_snapshots=[
            {
                "planner_latency_ms": 101.0,
                "planner_outcome": "llm-primary",
                "retrieval_latency_ms": 0,
                "retrieval_outcome": "",
                "answer_latency_ms": 25.5,
                "answer_outcome": "served",
                "verifier_latency_ms": 0,
                "verifier_outcome": "verifier-stamped",
            }
        ],
        attempt_entries=(
            {
                "causality_id": "caus" * 8,
                "operation": "message",
                "state": "in-flight",
                "started_at_s": 8123.5,
            },
            {
                "causality_id": "caus" * 8,
                "operation": "session.create",
                "state": "completed",
                "late_outcome": "served",
                "started_at_s": 8100.0,
            },
        ),
        served_audit=(
            {"agent": "planner", "served": "gpt/x"},
            {"agent": "answer", "served": "gpt/y"},
        ),
        token_audit=({"agent": "planner", "input": 10, "output": 3},),
        request_audit=(
            {
                "operation": "message",
                "state": "completed",
                "latency_ms": 120.0,
                "causality_id": "caus" * 8,
            },
            {"operation": "message", "state": "in-flight", "latency_ms": 0.0},
        ),
        process_state={"owned_by_runtime": True, "alive": True},
        server_evidence={
            "schema": "aa-opencode-server-evidence/1",
            "state": "supported",
            "last_session_state": "busy",
        },
    )
    base.update(overrides)
    return base


def _latency_samples() -> list[dict[str, object]]:
    return [
        {
            "causality_id": "c1",
            "family": "actionable-question",
            "outcome": "served",
            "latency_ms": 1000.0,
            "censored": False,
            "product_outcome": "served",
        },
        {
            "causality_id": "c2",
            "family": "actionable-question",
            "outcome": "served",
            "latency_ms": 3000.0,
            "censored": False,
            "product_outcome": "served",
        },
        {
            "causality_id": "c3",
            "family": "actionable-question",
            "outcome": "timeout",
            "latency_ms": 5000.0,
            "censored": True,
            "product_outcome": "timeout",
        },
    ]


def _diagnostics_with_child_timeout() -> list[dict[str, object]]:
    return [
        {
            "scenario_index": 5,
            "family": "actionable-question",
            "delivery_timeout": False,
            "production_outcome": "served",
        },
        {
            "scenario_index": 11,
            "family": "actionable-question",
            "delivery_timeout": True,
            "event_kind": EVENT_KIND_CHILD_TIMEOUT,
            "event_id": "child-e1",
            "turn_causality_id": "caus" * 8,
            "timeout_category": "stuck-provider",
        },
    ]


def _timeout_progress(partial: dict[str, object] | None) -> dict[str, object]:
    progress: dict[str, object] = {
        "phase": "scenario-heartbeat",
        "scenario_index": 11,
        "family": "actionable-question",
        "active_stage": "opencode",
        "last_active_stage": "answer",
        "request_state": "in-flight",
        "request_started": True,
        "request_progress_observable": "progress-unobservable",
        "timeout_category": "stuck-provider",
        "completed_scenario_count": 10,
        "scenario_total": 18,
        "cleanup": {"stall_recovery_state": "restart-required"},
        "scenario_diagnostics": _diagnostics_with_child_timeout(),
        "latency_samples": _latency_samples(),
    }
    if partial is not None:
        progress["partial_evidence"] = partial
    return progress


def test_timeout_lane_merges_validated_partial_evidence() -> None:
    """The parent's FAIL metrics carry the child's measured diagnostics."""
    partial = _make_partial()
    result = _timeout_lane(
        trigger="total-deadline",
        progress=_timeout_progress(partial),
        main_sha="abc123",
        elapsed_s=110.0,
        deadline_s=7200.0,
    )
    assert result.status == "FAIL"
    metrics = result.metrics
    assert metrics["watchdog_diagnostics_status"] == "observed-partial"
    assert metrics["partial_evidence_reported"] is True
    # Client-ledger sources land under the standard metric names.
    attempts = metrics["opencode_attempt_audit"]
    assert isinstance(attempts, list) and len(attempts) == 2
    assert all(isinstance(key, str) for entry in attempts for key in entry), (
        "attempt projection must stay allow-listed"
    )
    assert metrics["opencode_attempt_in_flight_count"] == 1
    assert metrics["opencode_attempt_late_count"] == 1
    assert metrics["served_models_by_agent"] == {"planner": ["gpt/x"], "answer": ["gpt/y"]}
    assert metrics["opencode_token_usage_by_agent"] == {
        "planner": {
            "requests": 1,
            "input_total": 10,
            "output_total": 3,
            "reasoning_total": 0,
            "cache_read_total": 0,
            "cache_write_total": 0,
        }
    }
    latency_view = metrics["opencode_request_latency_ms"]
    assert latency_view["message"] == {"count": 1, "p50": 120.0, "p95": 120.0, "max": 120.0}
    assert metrics["opencode_runtime_process"] == {"owned_by_runtime": True, "alive": True}
    assert metrics["stage_outcome_counts"] == {
        "planner": {"llm-primary": 1},
        "answer": {"served": 1},
        "verifier": {"verifier-stamped": 1},
    }
    assert metrics["stage_latency_ms"]["answer"] == {
        "count": 1,
        "p50": 25.5,
        "p95": 25.5,
        "max": 25.5,
    }
    assert metrics["stage_latency_coverage"]["stages"]["answer"] == {"measured": 1}
    # Server-side evidence travels with the provider-confirmed source type.
    assert metrics["opencode_server_events"]["state"] == "supported"
    assert metrics["opencode_server_events"]["last_session_state"] == "busy"
    coverage = metrics["partial_evidence_coverage"]
    assert coverage["server_events"] == "observed"
    assert coverage["stage_telemetry"] == "observed"
    assert coverage["client_attempt_ledger"] == "observed"
    assert metrics["scenarios_started"] == 11
    assert metrics["scenarios_interrupted"] == 1
    assert metrics["product_outcome_counts"] == {"served": 9}
    assert metrics["delivered_service_error_count"] == 1


def test_timeout_lane_dedups_events_and_counts_honest_latencies() -> None:
    """Child scenario-timeout and watchdog termination are distinct typed events."""
    result = _timeout_lane(
        trigger="no-real-progress",
        progress=_timeout_progress(_make_partial()),
        main_sha="abc123",
        elapsed_s=110.0,
        deadline_s=7200.0,
    )
    metrics = result.metrics
    diagnostics = metrics["scenario_diagnostics"]
    child_rows = [
        item for item in diagnostics if item.get("event_kind") == EVENT_KIND_CHILD_TIMEOUT
    ]
    watchdog_rows = [
        item for item in diagnostics if item.get("event_kind") == EVENT_KIND_WATCHDOG_TERMINATION
    ]
    assert len(child_rows) == 1 and len(watchdog_rows) == 1
    # Distinct event identity: no double counting of one scenario as one row.
    assert child_rows[0]["event_id"] != watchdog_rows[0]["event_id"]
    assert watchdog_rows[0]["event_id"]
    assert watchdog_rows[0]["turn_causality_id"] == "caus" * 8
    assert metrics["interrupted_scenarios_distinct"] == 1
    # Honest latency aggregates over the persisted samples.
    assert metrics["turn_latency_sample_count"] == 3
    assert metrics["turn_latency_p50_ms"] == 3000.0
    assert metrics["turn_latency_p95_ms"] == 5000.0
    assert metrics["turn_latency_max_ms"] == 5000.0


def test_timeout_lane_without_partial_marks_diagnostics_unavailable() -> None:
    """No partial marker: explicit typed marker, never fabricated metrics."""
    result = _timeout_lane(
        trigger="total-deadline",
        progress=_timeout_progress(None),
        main_sha="abc123",
        elapsed_s=110.0,
        deadline_s=7200.0,
    )
    metrics = result.metrics
    assert metrics["watchdog_diagnostics_status"] == DIAGNOSTICS_UNAVAILABLE
    assert metrics["partial_evidence_reported"] is False
    assert metrics["diagnostics_unavailable_sources"] == [
        "client_attempt_ledger",
        "request_latency_audit",
        "runtime_process_state",
        "scenario_accounting",
        "served_model_audit",
        "server_events",
        "stage_telemetry",
        "token_usage_audit",
    ]
    for forbidden in (
        "opencode_attempt_audit",
        "served_models_by_agent",
        "opencode_token_usage_by_agent",
        "opencode_server_events",
    ):
        assert forbidden not in metrics
    # The persisted samples still produce the parent's own honest
    # aggregates; a fabricated zero turn is never inserted.
    assert metrics["turn_latency_p50_ms"] == 3000.0
    assert metrics["turn_latency_p95_ms"] == 5000.0
    assert metrics["watchdog_trigger"] == "total-deadline"
    # Without any samples the aggregates stay explicitly absent.
    empty_progress = dict(_timeout_progress(None))
    empty_progress["latency_samples"] = []
    empty_progress["scenario_diagnostics"] = []
    result2 = _timeout_lane(
        trigger="total-deadline",
        progress=empty_progress,
        main_sha="abc123",
        elapsed_s=110.0,
        deadline_s=7200.0,
    )
    assert result2.metrics["turn_latency_p50_ms"] is None
    assert result2.metrics["turn_latency_p95_ms"] is None
    assert result2.metrics["turn_latency_max_ms"] is None


def test_invalid_partial_evidence_fails_closed() -> None:
    """Wrong schema or shape must not be merged (fail-closed)."""
    for bad in (
        {"schema_version": "aa-live-partial-evidence/0", "partial": True},
        {"schema_version": SCHEMA_VERSION, "partial": False},
        "self-referential-string",
        17,
    ):
        progress = _timeout_progress(None)
        progress["partial_evidence"] = bad
        assert validated_partial_evidence(progress) is None
        result = _timeout_lane(
            trigger="total-deadline",
            progress=progress,
            main_sha="abc123",
            elapsed_s=110.0,
            deadline_s=7200.0,
        )
        assert result.metrics["watchdog_diagnostics_status"] == DIAGNOSTICS_UNAVAILABLE


def test_failed_lane_crash_keeps_partial_or_types_unavailable() -> None:
    """A child crash before the first scenario stays honestly typed."""
    result = _failed_lane("runtime-not-ready", main_sha="sha", progress=_timeout_progress(None))
    assert result.status == "FAIL"
    assert result.metrics["watchdog_diagnostics_status"] == DIAGNOSTICS_UNAVAILABLE

    result2 = _failed_lane("runtime-not-ready", main_sha="sha", progress=_timeout_progress({}))
    assert result2.metrics["watchdog_diagnostics_status"] == DIAGNOSTICS_UNAVAILABLE


def test_merge_partial_without_any_observed_source_is_unavailable() -> None:
    """An all-empty partial must not fabricate success or fake zeros."""
    empty_partial = build_partial_evidence_snapshot(phase="observer-start")
    result = _timeout_lane(
        trigger="no-real-progress",
        progress=_timeout_progress(empty_partial),
        main_sha="sha",
        elapsed_s=1.0,
        deadline_s=7200.0,
    )
    metrics = result.metrics
    assert metrics["watchdog_diagnostics_status"] == DIAGNOSTICS_UNAVAILABLE
    assert "opencode_attempt_audit" not in metrics


def test_stage_latency_coverage_vocabulary() -> None:
    """Per-stage coverage stays closed-vocabulary and honest."""
    verdicts = stage_latency_coverage(
        {
            "planner_latency_ms": 512.5,
            "planner_outcome": "llm-primary",
            "retrieval_latency_ms": 0,
            "retrieval_outcome": "index-loaded",
            "answer_latency_ms": 0,
            "answer_outcome": "",
            "verifier_outcome": "stamped",
        }
    )
    assert verdicts == {
        "planner": "measured",
        "retrieval": "zero-unproven",
        "answer": "unobserved",
        "verifier": "not-measured",
    }
    aggregates, coverage = stage_latency_view(
        [
            {"planner_latency_ms": 100.0, "planner_outcome": "ok"},
            {"retrieval_latency_ms": 0, "retrieval_outcome": "loaded"},
            {"verifier_outcome": "stamped"},
        ]
    )
    assert aggregates == {"planner": {"count": 1, "max": 100.0, "p50": 100.0, "p95": 100.0}}
    assert coverage["stages"]["retrieval"] == {"zero-unproven": 1, "not-measured": 2}
    assert coverage["stages"]["verifier"] == {"not-measured": 3}


def test_c_focus_result_carries_partial_coverage() -> None:
    """The c-focus artifact carries the explicit partial-evidence coverage."""
    metrics = {
        "partial_evidence_reported": True,
        "watchdog_diagnostics_status": "observed-partial",
        "partial_evidence_coverage": {
            "client_attempt_ledger": "observed",
            "server_events": "disconnected",
        },
    }
    live_summary = {
        "lanes": [
            {
                "lane": "live-telegram-evidence",
                "status": "FAIL",
                "metrics": metrics,
            }
        ]
    }
    result = build_c_focus_result(
        main_sha="sha",
        run_id="run",
        gate_c_evidence={"sha": "sha", "status": "FAIL"},
        live_summary=live_summary,
    )
    diagnostics = result["privacy_safe_diagnostics"]
    assert diagnostics["partial_evidence_reported"] is True
    assert diagnostics["watchdog_diagnostics_status"] == "observed-partial"
    assert diagnostics["partial_evidence_coverage"] == {
        "client_attempt_ledger": "observed",
        "server_events": "disconnected",
    }
    # A real FAIL stays C_FAIL (verdict vocabulary unchanged).
    assert result["verdict"] == "C_FAIL"


def test_merge_rejects_fabricated_scenario_totals() -> None:
    """Non-numeric scenario totals are dropped, never coerced into counts."""
    partial = _make_partial()
    partial["scenario_totals"] = {
        "started": "many",
        "interrupted": "1",
        "completed_scenario_count": 10,
        "total": 18,
    }
    metrics = merge_partial_into_metrics({}, partial)
    assert "scenarios_started" not in metrics
    assert metrics["completed_scenario_count"] == 10
    assert "scenarios_interrupted" not in metrics


def test_child_partial_snapshot_shape_and_unobserved_defaults() -> None:
    """build_partial_evidence_snapshot validates without network or secrets."""
    partial = cast(dict[str, Any], _make_partial())
    current = cast(dict[str, Any], partial["current_scenario"])
    assert partial["schema_version"] == SCHEMA_VERSION
    assert partial["partial"] is True
    assert current["causality_id"] == "caus" * 8
    assert current["request_state"] == "in-flight"
    # The in-flight 0.0 placeholder is never a measured request.
    request_view = cast(dict[str, Any], partial["opencode_request_latency_ms"])
    assert request_view["message"]["count"] == 1
    assert partial["server_evidence_state"] == "supported"

    absent = build_partial_evidence_snapshot(phase="observer-start")
    assert absent["server_evidence_state"] == "unobserved"
    assert absent["coverage"]["server_events"] == "unobserved"
    assert absent["server_events"]["progress_meaning"] == "none"
    assert validated_partial_evidence({"partial_evidence": absent}) == absent
