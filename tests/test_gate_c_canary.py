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
    CATEGORY_READY,
    CATEGORY_REAL_PATH_TYPED_FAILURE,
    STATUS_BLOCKED,
    STATUS_FAIL,
    STATUS_INCONCLUSIVE,
    STATUS_READY,
    build_canary_result,
    build_real_path_report,
    evaluate_canary,
    run_bounded_real_path,
    run_observer_self_test,
)

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
        "families": ["meta-capability"],
        "typed_cause": "",
        "budget_expiration_category": "none",
        "attempt_coverage": "observed",
        "stage_coverage": "observed",
        "server_coverage": "observed",
        "server_evidence_state": "supported",
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
) -> None:
    report = await run_observer_self_test(tmp_path)

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
    assert "scripts/run_gate_c_canary.py" in text
    assert "--preflight-budget-s 600" in text
    assert "--live-budget-s 300" in text
    assert "steps.gate-c-canary.outcome == 'success'" in text
    assert "self-proving-out/gate-c-canary-result.json" in text
    assert "self-proving-out/observer-self-test.json" in text
    assert "self-proving-out/canary-progress.json" in text
    assert "self-proving-out/canary-attempt-observer/" in text
    assert "Gate C validated checkpoint present; canary not required (resume)." in text


def test_canary_result_artifact_is_not_qualification_evidence() -> None:
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
    assert result["product_qualified"] is False
    assert result["qualification_evidence"] is False
    assert result["gate_scenarios_executed"] == 0
    assert result["total_time_s"] == 12.5
    serialized = json.dumps(result, ensure_ascii=False)
    assert "GATECCANARYPROMPTSENTINEL454" not in serialized
