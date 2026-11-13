"""kodmial/aa#454: bounded Gate C canary preflight regressions."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from pathlib import Path
from typing import Any

import pytest

from aa.opencode.client import HttpOpenCodeClient
from aa.qualification.gate_c_canary import (
    CATEGORY_DIAGNOSTIC_MISSING,
    CATEGORY_EXTERNAL_PROGRESS_UNOBSERVED,
    CATEGORY_ORPHAN_WORK,
    CATEGORY_PRIVACY_VIOLATION,
    CATEGORY_READY,
    CATEGORY_REAL_PATH_TYPED_FAILURE,
    CATEGORY_UNSUPPORTED_SERVER,
    SELF_TEST_PROMPT_SENTINEL,
    STATUS_BLOCKED,
    STATUS_FAIL,
    STATUS_INCONCLUSIVE,
    STATUS_READY,
    build_canary_result,
    build_real_path_report,
    evaluate_canary,
    find_privacy_marker,
    privacy_markers,
    run_bounded_real_path,
    run_observer_self_test,
    server_endpoints_compatible,
    server_sse_route_observed,
)
from aa.qualification.live_partial_evidence import build_partial_evidence_snapshot
from aa.qualification.live_watchdog import ENV_PROGRESS_PATH
from aa.qualification.product_contract_live import LaneResult

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "aa-self-proving-qualification.yml"


class _HangingHttpServer:
    """A real loopback TCP server that accepts and never answers."""

    def __init__(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen(8)
        self.host, self.port = self._socket.getsockname()[:2]
        self._stop = threading.Event()
        self._connections: list[socket.socket] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        self._socket.settimeout(0.2)
        while not self._stop.is_set():
            try:
                connection, _ = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            self._connections.append(connection)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        self._stop.set()
        for connection in self._connections:
            try:
                connection.close()
            except OSError:
                pass
        try:
            self._socket.close()
        except OSError:
            pass


def _real_path(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "PASS",
        "real_turns": 1,
        "delivered_turns": 1,
        "families": ["meta-capability"],
        "typed_cause": "",
        "budget_expiration_category": "none",
        "attempt_coverage": "observed",
        "stage_coverage": "observed",
        "server_coverage": "observed",
        "server_evidence_state": "supported",
        "server_endpoints_compatible": True,
        "server_state_observed": True,
        "attempt_evidence": {"rows": 1, "model_requests": 1},
        "in_flight_attempt_preserved": False,
        "scenario_diagnostics": [
            {
                "scenario_index": 1,
                "family": "meta-capability",
                "delivery_status": "delivered",
                "product_outcome": "answered",
                "failure_reason": "",
            }
        ],
    }
    payload.update(overrides)
    return payload


def _decision(self_test: dict[str, Any], real_path: dict[str, Any] | None) -> Any:
    return evaluate_canary(
        self_test=self_test,
        real_path=real_path,
        preflight_budget_s=600.0,
        live_budget_s=300.0,
        elapsed_s=10.0,
    )


@pytest.mark.asyncio
async def test_observable_self_test_and_approved_real_turn_are_ready(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = await run_observer_self_test(tmp_path)
    stdout = capsys.readouterr().out

    assert "Gate C canary opencode-attempt" in stdout
    assert "stage=answer" in stdout
    assert "scenario_index=1" in stdout
    assert "GATECCANARYPROMPTSENTINEL454" not in stdout
    assert "GATECCANARYSSESENTINEL454" not in stdout

    assert report["schema_version"] == "aa-gate-c-canary-observer-self-test/1"
    assert report["status"] == "PASS", report["failed_checks"]
    assert report["llm_calls"] == 0
    assert report["privacy_safe"] is True
    assert report["no_orphan"] is True
    assert report["server_telemetry"]["state"] == "supported"
    assert report["server_telemetry"]["server_error_kind"] == "APIError"
    assert report["persisted_in_flight_snapshots"] >= 1
    assert report["watchdog_marker_written"] is True
    assert report["partial_evidence_coverage"]["client_attempt_ledger"] == "observed"
    assert report["partial_evidence_coverage"]["server_events"] == "observed"
    assert report["partial_evidence_coverage"]["stage_telemetry"] == "observed"
    assert report["checks"]["stdout-attempt-stage-scenario"] is True
    assert report["checks"]["raw-sse-payload-not-retained"] is True

    decision = _decision(report, _real_path())
    assert decision.status == STATUS_READY
    assert decision.category == CATEGORY_READY
    assert decision.ready is True

    # The self-test crosses the real HTTP boundary with a sentinel prompt;
    # no persisted artifact may contain it or an auth header.
    for path in tmp_path.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "GATECCANARYPROMPTSENTINEL454" not in text
        assert "GATECCANARYSSESENTINEL454" not in text
        assert "Authorization" not in text
        assert "Bearer " not in text


def test_missing_attempt_or_empty_cause_fails_closed() -> None:
    # A synthetic self-test pass is enough for the pure decision: the real
    # self-test itself is covered above.
    self_test = {
        "schema_version": "aa-gate-c-canary-observer-self-test/1",
        "status": "PASS",
        "checks": {"server-telemetry-supported": True},
        "failed_checks": [],
        "server_telemetry": {"state": "supported"},
        "privacy_safe": True,
        "no_orphan": True,
        "partial_evidence_coverage": {"runtime_process_state": "observed"},
    }

    missing_attempt = _decision(self_test, _real_path(attempt_coverage="missing"))
    assert missing_attempt.status == STATUS_BLOCKED
    assert missing_attempt.category == CATEGORY_DIAGNOSTIC_MISSING
    assert missing_attempt.missing_field == "real_path.attempt_coverage"

    # A lane FAIL with no typed cause must never be upgraded to READY.
    empty_cause = _decision(self_test, _real_path(status="FAIL"))
    assert empty_cause.status == STATUS_FAIL
    assert empty_cause.category == CATEGORY_REAL_PATH_TYPED_FAILURE

    typed_cause = _decision(
        self_test,
        _real_path(typed_cause="generation-empty-candidate"),
    )
    assert typed_cause.status == STATUS_FAIL
    assert typed_cause.category == CATEGORY_REAL_PATH_TYPED_FAILURE

    # No real turn is BLOCKED, not a fabricated success.
    no_turn = _decision(self_test, _real_path(real_turns=0))
    assert no_turn.status == STATUS_BLOCKED

    # A scenario can complete without a real model-message attempt (for
    # example a deterministic fast path). That is not proof of the
    # diagnostic chain and must never authorize the 18-case campaign.
    no_model = _decision(
        self_test,
        _real_path(attempt_evidence={"rows": 2, "model_requests": 0}),
    )
    assert no_model.status == STATUS_BLOCKED
    assert no_model.category == CATEGORY_DIAGNOSTIC_MISSING
    assert no_model.missing_field == "real_path.attempt_evidence.model_requests"

    # Endpoint availability without an actual session status/event fact is
    # not server-side evidence of the real turn.
    no_server_fact = _decision(self_test, _real_path(server_state_observed=False))
    assert no_server_fact.status == STATUS_BLOCKED
    assert no_server_fact.category == CATEGORY_UNSUPPORTED_SERVER
    assert no_server_fact.missing_field == "real_path.server_state_observed"


@pytest.mark.asyncio
async def test_unresponsive_http_is_bounded_inconclusive() -> None:
    server = _HangingHttpServer()
    client = HttpOpenCodeClient(server.base_url, request_timeout=30)

    async def hung_request() -> Any:
        # A real loopback HTTP call that never receives response headers.
        return await client._call("GET", "/session/status")

    try:
        lane_result = await run_bounded_real_path(hung_request, budget_s=0.05)
    finally:
        server.stop()
        # Let the cancelled executor worker observe the closed socket and
        # record its terminal outcome; no orphan thread survives the test.
        await asyncio.sleep(0.5)
    assert lane_result.status == "INCOMPLETE"
    assert lane_result.metrics["budget_expiration_category"] == "campaign-budget"
    assert client.quiescence_proven() is True

    real_path = build_real_path_report(lane_result)
    decision = _decision(
        {
            "schema_version": "aa-gate-c-canary-observer-self-test/1",
            "status": "PASS",
            "checks": {"server-telemetry-supported": True},
            "failed_checks": [],
            "server_telemetry": {"state": "supported"},
            "privacy_safe": True,
            "no_orphan": True,
        },
        real_path,
    )
    assert decision.status == STATUS_INCONCLUSIVE
    assert decision.category == CATEGORY_EXTERNAL_PROGRESS_UNOBSERVED

    # Budget expiry stays INCONCLUSIVE even when no attempt/stage coverage
    # could be recorded: external non-completion is not a product failure.
    budget_only = _decision(
        {
            "schema_version": "aa-gate-c-canary-observer-self-test/1",
            "status": "PASS",
            "checks": {"server-telemetry-supported": True},
            "failed_checks": [],
            "server_telemetry": {"state": "supported"},
            "privacy_safe": True,
            "no_orphan": True,
        },
        _real_path(
            status="INCOMPLETE",
            real_turns=0,
            attempt_coverage="missing",
            stage_coverage="missing",
            budget_expiration_category="no-real-progress",
        ),
    )
    assert budget_only.status == STATUS_INCONCLUSIVE
    assert budget_only.category == CATEGORY_EXTERNAL_PROGRESS_UNOBSERVED


def test_canary_reuses_frozen_existing_core_scenarios() -> None:
    from aa.qualification.product_contract_live import (
        GATE_C_CORE_SCENARIOS,
        GATE_C_HELD_OUT_SCENARIOS,
        GATE_C_REGRESSION_SCENARIO_COUNT,
        gate_c_canary_scenarios,
    )

    assert (
        len(GATE_C_CORE_SCENARIOS) + len(GATE_C_HELD_OUT_SCENARIOS)
        == GATE_C_REGRESSION_SCENARIO_COUNT
        == 18
    )
    canary = gate_c_canary_scenarios()
    assert 1 <= len(canary) <= 2
    assert all(scenario in GATE_C_CORE_SCENARIOS for scenario in canary)
    assert [family for family, _, _ in canary] == [
        "meta-capability",
        "substantive-drinking",
    ]


def test_workflow_runs_canary_before_gate_c_and_gates_on_ready() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    canary = text.index("Gate C canary - bounded diagnostic preflight")
    gate_c = text.index("Gate C - LIVE exact production conversation path")
    assert canary < gate_c
    assert "id: gate-c-canary" in text
    assert "timeout-minutes: 10" in text
    assert "scripts/run_gate_c_canary.py" in text
    assert "--preflight-budget-s 600" in text
    assert "--live-budget-s 300" in text
    assert "steps.gate-c-canary.outcome == 'success'" in text
    assert "self-proving-out/gate-c-canary-result.json" in text
    assert "self-proving-out/observer-self-test.json" in text
    assert "self-proving-out/attempt-observer/" in text
    assert "self-proving-out/canary-progress.json" in text
    assert "self-proving-out/canary-attempt-observer/" in text
    assert "Gate C validated checkpoint present; canary not required (resume)." in text
    # The short canary shares the prepared image, corpus and L2 index with
    # the campaign: no per-job dependency install or E5 rebuild.
    canary_block = text[canary:gate_c]
    assert "pip install" not in canary_block
    assert "build_retrieval_index.py" not in canary_block
    assert "manage_derived_cache.py" not in canary_block


def test_canary_result_artifact_is_not_qualification_evidence() -> None:
    from aa.qualification.gate_c_canary import EXIT_BY_STATUS

    assert EXIT_BY_STATUS == {
        STATUS_READY: 0,
        STATUS_FAIL: 1,
        STATUS_BLOCKED: 2,
        STATUS_INCONCLUSIVE: 3,
    }
    # The persisted payload shape carries an explicit non-qualification
    # marker and no prompt/reply fields.
    decision = evaluate_canary(
        self_test={
            "schema_version": "aa-gate-c-canary-observer-self-test/1",
            "status": "PASS",
            "checks": {"server-telemetry-supported": True},
            "failed_checks": [],
            "server_telemetry": {"state": "supported"},
            "privacy_safe": True,
            "no_orphan": True,
        },
        real_path=_real_path(),
        preflight_budget_s=600.0,
        live_budget_s=300.0,
        elapsed_s=12.5,
    )
    result = build_canary_result(
        main_sha="a" * 40,
        run_id="123",
        self_test={
            "schema_version": "aa-gate-c-canary-observer-self-test/1",
            "status": "PASS",
            "server_telemetry": {"state": "supported"},
        },
        real_path=_real_path(),
        decision=decision,
        preflight_budget_s=600.0,
        live_budget_s=300.0,
        elapsed_s=12.5,
        privacy_safe_paths=["gate-c-canary-result.json"],
    )
    assert result["schema_version"] == "aa-gate-c-canary-result/1"
    assert result["ready"] is True
    assert result["run_sha"] == "a" * 40
    assert result["main_sha"] == "a" * 40
    assert result["stage_attempt_evidence_coverage"]["client_attempt_ledger"] == "observed"
    assert result["stage_attempt_evidence_coverage"]["stage_telemetry"] == "observed"
    assert result["budget_expiration_category"] == "none"
    assert result["privacy_safe_paths"] == ["gate-c-canary-result.json"]
    assert result["product_qualified"] is False
    assert result["qualification_evidence"] is False
    assert result["gate_scenarios_executed"] == 0
    assert result["real_turns"] == 1
    assert result["delivered_turns"] == 1
    assert result["total_time_s"] == 12.5
    serialized = json.dumps(result, ensure_ascii=False)
    assert "GATECCANARYPROMPTSENTINEL454" not in serialized


def test_preflight_budget_expiry_is_persisted_as_typed_inconclusive() -> None:
    from aa.qualification.gate_c_canary import (
        CATEGORY_EXTERNAL_PROGRESS_UNOBSERVED,
        CanaryDecision,
    )

    decision = CanaryDecision(
        STATUS_INCONCLUSIVE,
        CATEGORY_EXTERNAL_PROGRESS_UNOBSERVED,
        "budget.preflight_budget_s",
        "preflight budget expired before the real-path canary",
    )
    result = build_canary_result(
        main_sha="a" * 40,
        run_id="123",
        self_test={
            "schema_version": "aa-gate-c-canary-observer-self-test/1",
            "status": "PASS",
            "server_telemetry": {"state": "supported"},
        },
        real_path=None,
        decision=decision,
        preflight_budget_s=600.0,
        live_budget_s=300.0,
        elapsed_s=601.0,
        privacy_safe_paths=["gate-c-canary-result.json"],
    )
    assert result["status"] == STATUS_INCONCLUSIVE
    assert result["ready"] is False
    assert result["budget_expiration_category"] == "preflight-budget"
    assert result["budget"]["category"] == "preflight-budget"
    assert result["budget"]["expired"] is True
    assert result["product_qualified"] is False


def test_unsupported_server_privacy_and_orphan_fail_closed() -> None:
    base_self_test = {
        "schema_version": "aa-gate-c-canary-observer-self-test/1",
        "status": "PASS",
        "checks": {"server-telemetry-supported": True},
        "failed_checks": [],
        "server_telemetry": {"state": "supported"},
        "privacy_safe": True,
        "no_orphan": True,
    }

    unsupported = _decision(
        {**base_self_test, "server_telemetry": {"state": "unsupported"}},
        _real_path(),
    )
    assert unsupported.status == STATUS_BLOCKED
    assert unsupported.category == CATEGORY_UNSUPPORTED_SERVER

    privacy = _decision({**base_self_test, "privacy_safe": False}, _real_path())
    assert privacy.status == STATUS_BLOCKED
    assert privacy.category == CATEGORY_PRIVACY_VIOLATION

    orphan = _decision({**base_self_test, "no_orphan": False}, _real_path())
    assert orphan.status == STATUS_BLOCKED
    assert orphan.category == CATEGORY_ORPHAN_WORK

    # The actual pinned server must expose the documented surface too.
    real_unsupported = _decision(
        base_self_test,
        _real_path(server_evidence_state="unsupported"),
    )
    assert real_unsupported.status == STATUS_BLOCKED
    assert real_unsupported.category == CATEGORY_UNSUPPORTED_SERVER

    no_endpoints = _decision(
        base_self_test,
        _real_path(server_endpoints_compatible=False),
    )
    assert no_endpoints.status == STATUS_BLOCKED
    assert no_endpoints.category == CATEGORY_UNSUPPORTED_SERVER
    assert no_endpoints.missing_field == "real_path.server_endpoints_compatible"


def test_doc_alone_does_not_prove_server_telemetry_compatibility() -> None:
    doc_only = {
        "state": "supported",
        "endpoints": {"/doc": {"state": "supported", "http_status": 200}},
    }
    assert server_endpoints_compatible(doc_only) is False

    missing_sse = {
        "state": "supported",
        "endpoints": {
            "/doc": {"state": "supported", "http_status": 200},
            "/session/status": {"session-status": 0},
        },
    }
    assert server_endpoints_compatible(missing_sse) is False

    compatible: dict[str, Any] = {
        "state": "supported",
        "endpoints": {
            "/doc": {"state": "supported", "http_status": 200},
            "/session/status": {"session-status": 0, "unknown-state": 0},
            "/event": {
                "event-stream-status": 200,
                "event-stream-streamed": 1,
                "event-stream-quiet": 1,
            },
        },
    }
    assert server_endpoints_compatible(compatible) is True

    broken_sse: dict[str, Any] = {
        **compatible,
        "endpoints": {
            **compatible["endpoints"],
            "/event": {"event-stream-status": 404, "event-stream-http-error": 404},
        },
    }
    assert server_endpoints_compatible(broken_sse) is False

    unknown_status_shape: dict[str, Any] = {
        **compatible,
        "endpoints": {
            **compatible["endpoints"],
            "/session/status": {"session-status": 1, "unknown-state": 1},
        },
    }
    assert server_endpoints_compatible(unknown_status_shape) is False


def test_real_path_lane_evidence_with_observed_model_request_is_ready() -> None:
    lane_result = LaneResult(
        lane="gate-c-canary-real-path",
        status="PASS",
        metrics={
            "real_turns": 1,
            "scenario_diagnostics": [
                {
                    "scenario_index": 1,
                    "family": "meta-capability",
                    "delivery_timeout": False,
                    "delivery_status": "delivered",
                    "product_outcome": "answer_success",
                    "failure_reason": "",
                }
            ],
            "attempt_audit": [
                {
                    "attempt_id": "a" * 32,
                    "operation": "message-text",
                    "state": "completed",
                    "stage": "answer",
                    "agent": "aa",
                    "model": "provider/model-primary",
                    "causality_id": "c" * 32,
                }
            ],
            "partial_evidence": {
                "coverage": {
                    "client_attempt_ledger": "observed",
                    "stage_telemetry": "observed",
                    "server_events": "observed",
                    "served_model_audit": "observed",
                    "runtime_process_state": "observed",
                },
                "served_models_by_agent": {"aa": ["provider/model-primary"]},
                "stage_outcome_counts": {"answer": {"answered": 1}},
            },
            "server_evidence": {
                "state": "supported",
                "endpoints": {
                    "/doc": {"state": "supported", "http_status": 200},
                    "/session/status": {"session-status": 1, "unknown-state": 0},
                    "/event": {
                        "event-stream-status": 200,
                        "event-stream-streamed": 1,
                        "event-stream-quiet": 1,
                    },
                },
                "last_session_state": "idle",
                "session_state_counts": {"idle": 1, "busy": 0, "retry": 0},
            },
        },
    )

    report = build_real_path_report(lane_result)
    assert report["attempt_evidence"]["model_requests"] == 1
    assert report["server_state_observed"] is True
    decision = _decision(
        {
            "schema_version": "aa-gate-c-canary-observer-self-test/1",
            "status": "PASS",
            "checks": {"server-telemetry-supported": True},
            "failed_checks": [],
            "server_telemetry": {"state": "supported"},
            "privacy_safe": True,
            "no_orphan": True,
        },
        report,
    )
    assert decision.status == STATUS_READY
    assert decision.category == CATEGORY_READY


def test_attempt_ledger_supplies_typed_cause_and_bounded_model_evidence() -> None:
    lane_result = LaneResult(
        lane="gate-c-canary-real-path",
        status="PASS",
        metrics={
            "real_turns": 1,
            "scenario_diagnostics": [
                {
                    "scenario_index": 1,
                    "family": "substantive-drinking",
                    "delivery_timeout": False,
                    "delivery_status": "delivered",
                    "product_outcome": "answer_success",
                    "failure_reason": "",
                }
            ],
            "attempt_audit": [
                {
                    "attempt_id": "a" * 32,
                    "operation": "message-text",
                    "state": "failed",
                    "stage": "answer",
                    "agent": "aa",
                    "model": "provider/model-primary",
                    "fallback_model": "provider/model-fallback",
                    "failure_category": "conversational-generation-failed",
                    "causality_id": "c" * 32,
                }
            ],
            "partial_evidence": {
                "coverage": {
                    "client_attempt_ledger": "observed",
                    "stage_telemetry": "observed",
                    "server_events": "observed",
                    "served_model_audit": "observed",
                    "runtime_process_state": "observed",
                },
                "served_models_by_agent": {"aa": ["provider/model-primary"]},
                "stage_outcome_counts": {"answer": {"answered": 1}},
            },
            "server_evidence": {
                "state": "supported",
                "endpoints": {
                    "/doc": {"state": "supported", "http_status": 200},
                    "/session/status": {"session-status": 0, "unknown-state": 0},
                    "/event": {
                        "event-stream-status": 200,
                        "event-stream-streamed": 1,
                        "event-stream-quiet": 1,
                    },
                },
                "server_error_kind": "APIError",
                "server_error_status_code": 503,
                "last_session_state": "busy",
                "session_state_counts": {"idle": 0, "busy": 1, "retry": 0},
            },
        },
    )

    report = build_real_path_report(lane_result)
    assert report["typed_cause"] == "conversational-generation-failed"
    assert report["server_state_observed"] is True
    assert report["attempt_evidence"]["model_requests"] == 1
    assert report["attempt_coverage"] == "observed"
    assert report["served_model_coverage"] == "observed"
    assert report["attempt_evidence"]["state_counts"] == {"failed": 1}
    assert report["attempt_evidence"]["requested_models"] == ["provider/model-primary"]
    assert report["served_models_by_agent"] == {"aa": ["provider/model-primary"]}
    assert report["server_evidence"]["server_error_kind"] == "APIError"
    assert report["server_evidence"]["server_error_status_code"] == 503

    decision = _decision(
        {
            "schema_version": "aa-gate-c-canary-observer-self-test/1",
            "status": "PASS",
            "checks": {"server-telemetry-supported": True},
            "failed_checks": [],
            "server_telemetry": {"state": "supported"},
            "privacy_safe": True,
            "no_orphan": True,
        },
        report,
    )
    assert decision.status == STATUS_FAIL
    assert decision.category == CATEGORY_REAL_PATH_TYPED_FAILURE

    result = build_canary_result(
        main_sha="a" * 40,
        run_id="123",
        self_test={"schema_version": "aa-gate-c-canary-observer-self-test/1"},
        real_path=report,
        decision=decision,
        preflight_budget_s=600.0,
        live_budget_s=300.0,
        elapsed_s=3.0,
        privacy_safe_paths=["gate-c-canary-result.json"],
    )
    assert result["real_path"]["attempt_evidence"]["state_counts"] == {"failed": 1}
    assert result["real_path"]["served_models_by_agent"] == {"aa": ["provider/model-primary"]}
    assert result["real_path"]["server_evidence"]["server_error_kind"] == "APIError"
    assert result["real_path"]["server_evidence"]["server_error_status_code"] == 503
    assert result["stage_attempt_evidence_coverage"]["served_model_audit"] == "observed"
    assert "conversational-generation-failed" in result["real_path"]["typed_cause"]

    # Quality checks record diagnosis but never turn into a canary failure.
    from aa.qualification.product_contract_live import _canary_infrastructure_failure

    assert _canary_infrastructure_failure("live-real-opencode-runtime-ready") is True
    assert _canary_infrastructure_failure("live-transport-meta-capability-accepted") is True
    assert _canary_infrastructure_failure("live-delivery-meta-capability-timeout") is True
    assert _canary_infrastructure_failure("live-book-grounding-meta-capability-1") is False
    assert _canary_infrastructure_failure("live-answer-relevance-meta-capability-1") is False
    assert _canary_infrastructure_failure("live-meta-direct-1") is False


def test_elapsed_preflight_budget_is_typed_inconclusive() -> None:
    decision = evaluate_canary(
        self_test={
            "schema_version": "aa-gate-c-canary-observer-self-test/1",
            "status": "PASS",
            "checks": {"server-telemetry-supported": True},
            "failed_checks": [],
            "server_telemetry": {"state": "supported"},
            "privacy_safe": True,
            "no_orphan": True,
        },
        real_path=_real_path(),
        preflight_budget_s=600.0,
        live_budget_s=300.0,
        elapsed_s=601.0,
    )
    assert decision.status == STATUS_INCONCLUSIVE
    assert decision.category == CATEGORY_EXTERNAL_PROGRESS_UNOBSERVED
    assert decision.missing_field == "budget.preflight_budget_s"
    assert decision.detail


def test_global_event_route_satisfies_canary_server_compatibility() -> None:
    evidence = {
        "state": "supported",
        "endpoints": {
            "/doc": {"state": "supported", "http_status": 200},
            "/session/status": {"session-status": 0, "unknown-state": 0},
            "/global/event": {
                "event-stream-status": 200,
                "event-stream-streamed": 1,
                "event-stream-quiet": 1,
            },
        },
    }
    assert server_sse_route_observed(evidence) is True
    assert server_endpoints_compatible(evidence) is True


def test_real_path_prompt_or_secret_marker_blocks_and_is_redacted() -> None:
    self_test = {
        "schema_version": "aa-gate-c-canary-observer-self-test/1",
        "status": "PASS",
        "checks": {"server-telemetry-supported": True},
        "failed_checks": [],
        "server_telemetry": {"state": "supported"},
        "privacy_safe": True,
        "no_orphan": True,
    }
    leaked = _real_path(
        scenario_diagnostics=[
            {
                "scenario_index": 1,
                "family": "meta-capability",
                "delivery_status": "delivered",
                "product_outcome": "answered",
                "failure_reason": SELF_TEST_PROMPT_SENTINEL,
            }
        ]
    )
    assert find_privacy_marker(leaked, privacy_markers()) == SELF_TEST_PROMPT_SENTINEL
    decision = _decision(self_test, leaked)
    assert decision.status == STATUS_BLOCKED
    assert decision.category == CATEGORY_PRIVACY_VIOLATION
    assert decision.missing_field == "real_path.privacy"

    result = build_canary_result(
        main_sha="a" * 40,
        run_id="123",
        self_test=self_test,
        real_path=leaked,
        decision=decision,
        preflight_budget_s=600.0,
        live_budget_s=300.0,
        elapsed_s=3.0,
        privacy_safe_paths=["gate-c-canary-result.json"],
    )
    serialized = json.dumps(result, ensure_ascii=False)
    assert SELF_TEST_PROMPT_SENTINEL not in serialized
    assert "redacted" in serialized
    assert result["status"] == STATUS_BLOCKED


def _load_canary_script() -> Any:
    import importlib.util
    import sys

    path = ROOT / "scripts" / "run_gate_c_canary.py"
    name = "aa_run_gate_c_canary_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _self_test_pass() -> dict[str, Any]:
    return {
        "schema_version": "aa-gate-c-canary-observer-self-test/1",
        "status": "PASS",
        "checks": {"server-telemetry-supported": True},
        "failed_checks": [],
        "server_telemetry": {"state": "supported"},
        "privacy_safe": True,
        "no_orphan": True,
        "partial_evidence_coverage": {"runtime_process_state": "observed"},
    }


def _lane_ready() -> LaneResult:
    return LaneResult(
        lane="gate-c-canary-real-path",
        status="PASS",
        metrics={
            "real_turns": 1,
            "scenario_diagnostics": [
                {
                    "scenario_index": 1,
                    "family": "meta-capability",
                    "delivery_timeout": False,
                    "delivery_status": "delivered",
                    "product_outcome": "answer_success",
                    "failure_reason": "",
                }
            ],
            "attempt_audit": [
                {
                    "attempt_id": "a" * 32,
                    "operation": "message-text",
                    "state": "completed",
                    "stage": "answer",
                }
            ],
            "partial_evidence": {
                "coverage": {
                    "client_attempt_ledger": "observed",
                    "stage_telemetry": "observed",
                    "server_events": "observed",
                    "served_model_audit": "observed",
                    "runtime_process_state": "observed",
                }
            },
            "server_evidence": {
                "state": "supported",
                "endpoints": {
                    "/doc": {"state": "supported", "http_status": 200},
                    "/session/status": {"session-status": 1, "unknown-state": 0},
                    "/global/event": {
                        "event-stream-status": 200,
                        "event-stream-streamed": 1,
                        "event-stream-quiet": 1,
                    },
                },
                "last_session_state": "busy",
            },
        },
    )


def _patch_canary_preconditions(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "checked_out_sha", lambda root: "a" * 40)
    monkeypatch.setattr(module, "working_tree_clean", lambda root: True)
    monkeypatch.setattr(module, "_pinned_opencode_version", lambda: "1.2.3")
    monkeypatch.setattr(module, "_installed_opencode_version", lambda: "opencode 1.2.3")


def test_canary_cli_ready_is_not_qualification_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_canary_script()
    _patch_canary_preconditions(module, monkeypatch)

    async def _self_test(out_dir: Path | str) -> dict[str, Any]:
        return _self_test_pass()

    async def _lane(**kwargs: Any) -> LaneResult:
        return _lane_ready()

    monkeypatch.setattr(module, "run_observer_self_test", _self_test)
    monkeypatch.setattr(module, "run_live_telegram_evidence_lane", _lane)

    exit_code = module.main(["--main-sha", "a" * 40, "--out-dir", str(tmp_path), "--run-id", "42"])
    assert exit_code == 0
    result = json.loads((tmp_path / "gate-c-canary-result.json").read_text())
    assert result["status"] == STATUS_READY
    assert result["ready"] is True
    assert result["product_qualified"] is False
    assert result["qualification_evidence"] is False
    assert result["gate_scenarios_executed"] == 0
    assert result["stage_attempt_evidence_coverage"]["client_attempt_ledger"] == "observed"


def test_canary_cli_blocks_before_real_path_when_self_test_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_canary_script()
    _patch_canary_preconditions(module, monkeypatch)
    lane_called = False

    async def _self_test(out_dir: Path | str) -> dict[str, Any]:
        return {
            **_self_test_pass(),
            "status": "FAIL",
            "failed_checks": ["attempt-start-observed"],
        }

    async def _lane(**kwargs: Any) -> LaneResult:
        nonlocal lane_called
        lane_called = True
        return _lane_ready()

    monkeypatch.setattr(module, "run_observer_self_test", _self_test)
    monkeypatch.setattr(module, "run_live_telegram_evidence_lane", _lane)

    exit_code = module.main(["--main-sha", "a" * 40, "--out-dir", str(tmp_path), "--run-id", "42"])
    assert exit_code == 2
    assert lane_called is False
    result = json.loads((tmp_path / "gate-c-canary-result.json").read_text())
    assert result["status"] == STATUS_BLOCKED
    assert result["real_path"]["status"] == "not-executed"
    assert result["product_qualified"] is False


def test_canary_cli_bounded_timeout_is_inconclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_canary_script()
    _patch_canary_preconditions(module, monkeypatch)

    async def _self_test(out_dir: Path | str) -> dict[str, Any]:
        return _self_test_pass()

    async def _lane(**kwargs: Any) -> LaneResult:
        return LaneResult(
            lane="gate-c-canary-real-path",
            status="INCOMPLETE",
            incomplete=("canary-budget-expired",),
            metrics={
                "canary": True,
                "real_turns": 0,
                "scenario_diagnostics": [],
                "budget_expiration_category": "campaign-budget",
                "timeout_reason": "campaign-budget",
            },
        )

    monkeypatch.setattr(module, "run_observer_self_test", _self_test)
    monkeypatch.setattr(module, "run_live_telegram_evidence_lane", _lane)

    exit_code = module.main(["--main-sha", "a" * 40, "--out-dir", str(tmp_path), "--run-id", "42"])
    assert exit_code == 3
    result = json.loads((tmp_path / "gate-c-canary-result.json").read_text())
    assert result["status"] == STATUS_INCONCLUSIVE
    assert result["budget"]["category"] == "campaign-budget"
    assert result["budget"]["expired"] is True
    assert result["product_qualified"] is False


def test_canary_cli_provider_429_requires_fresh_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aa.opencode.errors import OpenCodeRateLimitError

    module = _load_canary_script()
    _patch_canary_preconditions(module, monkeypatch)

    async def _self_test(out_dir: Path | str) -> dict[str, Any]:
        return _self_test_pass()

    async def _lane(**kwargs: Any) -> LaneResult:
        raise OpenCodeRateLimitError("provider rate limit")

    monkeypatch.setattr(module, "run_observer_self_test", _self_test)
    monkeypatch.setattr(module, "run_live_telegram_evidence_lane", _lane)

    exit_code = module.main(["--main-sha", "a" * 40, "--out-dir", str(tmp_path), "--run-id", "42"])
    assert exit_code == 75
    assert (tmp_path / "restart-required.json").is_file()
    result = json.loads((tmp_path / "gate-c-canary-result.json").read_text())
    assert result["status"] == STATUS_INCONCLUSIVE
    assert result["product_qualified"] is False


def test_canary_cli_final_artifact_privacy_scan_blocks_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_canary_script()
    _patch_canary_preconditions(module, monkeypatch)

    async def _self_test(out_dir: Path | str) -> dict[str, Any]:
        leak_dir = Path(out_dir) / "attempt-observer"
        leak_dir.mkdir(parents=True, exist_ok=True)
        (leak_dir / "leak.txt").write_text(SELF_TEST_PROMPT_SENTINEL, encoding="utf-8")
        return _self_test_pass()

    async def _lane(**kwargs: Any) -> LaneResult:
        return _lane_ready()

    monkeypatch.setattr(module, "run_observer_self_test", _self_test)
    monkeypatch.setattr(module, "run_live_telegram_evidence_lane", _lane)

    exit_code = module.main(["--main-sha", "a" * 40, "--out-dir", str(tmp_path), "--run-id", "42"])
    assert exit_code == 2
    result = json.loads((tmp_path / "gate-c-canary-result.json").read_text())
    assert result["status"] == STATUS_BLOCKED
    assert result["category"] == CATEGORY_PRIVACY_VIOLATION
    assert result["missing_field"] == "artifacts"
    assert result["observer_self_test"]["privacy_safe"] is False
    assert SELF_TEST_PROMPT_SENTINEL not in json.dumps(result)


@pytest.mark.asyncio
async def test_bounded_timeout_preserves_persisted_in_flight_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = build_partial_evidence_snapshot(
        phase="canary-first-turn",
        causality_id="c" * 32,
        attempt_entries=[
            {
                "attempt_id": "a" * 32,
                "operation": "message-text",
                "state": "in-flight",
                "stage": "answer",
            }
        ],
    )
    progress_path = tmp_path / "canary-progress.json"
    progress_path.write_text(json.dumps({"partial_evidence": partial}), encoding="utf-8")
    monkeypatch.setenv(ENV_PROGRESS_PATH, str(progress_path))

    server = _HangingHttpServer()
    client = HttpOpenCodeClient(server.base_url, request_timeout=30)

    async def hung_request() -> Any:
        return await client._call("GET", "/session/status")

    try:
        lane_result = await run_bounded_real_path(hung_request, budget_s=0.05)
    finally:
        server.stop()
        await asyncio.sleep(0.5)

    assert lane_result.status == "INCOMPLETE"
    report = build_real_path_report(lane_result)
    assert report["in_flight_attempt_preserved"] is True
    assert report["attempt_in_flight_count"] == 1
    assert report["budget_expiration_category"] == "campaign-budget"
    decision = _decision(
        {
            "schema_version": "aa-gate-c-canary-observer-self-test/1",
            "status": "PASS",
            "checks": {"server-telemetry-supported": True},
            "failed_checks": [],
            "server_telemetry": {"state": "supported"},
            "privacy_safe": True,
            "no_orphan": True,
        },
        report,
    )
    assert decision.status == STATUS_INCONCLUSIVE
    assert decision.category == CATEGORY_EXTERNAL_PROGRESS_UNOBSERVED
