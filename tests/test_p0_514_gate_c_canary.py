"""kodmial/aa#514: bounded Gate C diagnostic canary regressions.

The canary must prove the production diagnostic chain *before* the unchanged
18-scenario Gate C campaign:

- deterministic provider-free observer/server self-test against a controlled
  local OpenCode surface;
- one or two existing core scenarios through the production path;
- a genuinely bounded INCONCLUSIVE on external silence (never provider-failed);
- fail-closed BLOCKED when the observer or server telemetry is missing;
- no prompt/reply/session/secret text in artifacts or stdout.
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from aa.opencode.client import HttpOpenCodeClient
from aa.opencode.observer import StallSnapshotObserver
from aa.qualification.gate_c_canary import (
    BUDGET_EXPIRED,
    BUDGET_WITHIN,
    SCHEMA_VERSION,
    SELF_TEST_PROMPT_SENTINEL,
    SELF_TEST_SESSION_SENTINEL,
    SELF_TEST_SSE_SECRET_SENTINEL,
    STATUS_BLOCKED,
    STATUS_INCONCLUSIVE,
    STATUS_READY,
    GateCCanaryError,
    RealPathEvidence,
    ScenarioEvidence,
    SelfTestEvidence,
    _scan_for_sentinels,
    _server_observation_checks,
    assert_no_text_leak,
    evaluate_canary,
    observer_self_test,
    run_canary,
    write_canary_result,
)


def _self_test_ready() -> SelfTestEvidence:
    return SelfTestEvidence(
        passed=True,
        snapshot_count=2,
        attempt_start_observed=True,
        trace_observed=True,
        flush_observed=True,
        process_alive_observed=True,
        scenario_correlation_observed=True,
        stage_attempt_stdout_observed=True,
        watchdog_marker_observed=True,
        server_evidence={"state": "supported", "progress_meaning": "none"},
        server_unobserved_verified=True,
        privacy_validated=True,
        orphan_absent=True,
    )


def _scenario(family: str = "meta-capability", **overrides: Any) -> ScenarioEvidence:
    values: dict[str, Any] = {
        "family": family,
        "delivered": True,
        "answer_outcome": "served",
        "product_outcome": "answer_success",
        "attempt_observed": True,
        "stage_observed": True,
        "causality_observed": True,
        "server_evidence_state": "supported",
        "model": "mock/mock-model",
        "latency_ms": 12.0,
    }
    values.update(overrides)
    return ScenarioEvidence(**values)


def _server_evidence_ready() -> dict[str, Any]:
    """Supported native v2 surface: OpenAPI, active sessions and SSE."""
    return {
        "state": "supported",
        "progress_meaning": "none",
        "doc_endpoint": "/openapi.json",
        "session_status_endpoint": "/api/session/active",
        "event_endpoint": "/api/event",
        "endpoints": {
            "/doc": {"http_status": 200, "state": "supported"},
            "/session/status": {"session-status": 1, "unknown-state": 0, "busy": 1},
            "/api/event": {
                "event-stream-status": 200,
                "event-stream-streamed": 1,
                "event-stream-quiet": 1,
            },
        },
    }


def _real_path_ready(turns: int = 2) -> RealPathEvidence:
    families = ("meta-capability", "substantive-drinking")
    return RealPathEvidence(
        scenarios=tuple(_scenario(families[index]) for index in range(turns)),
        server_evidence=_server_evidence_ready(),
    )


# --- Deterministic self-test on the real interfaces -----------------------


@pytest.mark.asyncio
async def test_observer_self_test_proves_the_wired_diagnostic_chain(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence = await observer_self_test(tmp_path / "self-test")
    assert evidence.passed, evidence.missing_fields
    assert evidence.snapshot_count >= 2
    assert evidence.attempt_start_observed
    assert evidence.scenario_correlation_observed
    assert evidence.trace_observed
    assert evidence.flush_observed
    assert evidence.process_alive_observed
    assert evidence.watchdog_marker_observed
    assert evidence.server_unobserved_verified
    assert evidence.server_evidence["state"] == "supported"
    assert evidence.server_evidence["server_error_status_code"] == 503
    assert evidence.server_evidence["server_error_kind"] == "APIError"
    assert evidence.server_evidence["session_state_counts"]["busy"] >= 1
    assert evidence.server_evidence["event_type_counts"]["session.status"] >= 1
    assert evidence.server_evidence["event_type_counts"]["session.execution.failed"] >= 1
    assert evidence.privacy_validated

    # The live stdout lines carry the bounded stage/attempt projection only.
    stdout = capsys.readouterr().out
    assert "attempt_state=in-flight" in stdout
    assert "attempt_id=" in stdout
    assert SELF_TEST_PROMPT_SENTINEL not in stdout
    assert SELF_TEST_SESSION_SENTINEL not in stdout
    assert SELF_TEST_SSE_SECRET_SENTINEL not in stdout

    # The controlled server deliberately embeds session ids and secret
    # markers; no persisted diagnostic file may retain them.
    persisted = list((tmp_path / "self-test").rglob("*.json"))
    assert persisted
    for path in persisted:
        payload = path.read_text(encoding="utf-8")
        assert SELF_TEST_PROMPT_SENTINEL not in payload
        assert SELF_TEST_SESSION_SENTINEL not in payload
        assert SELF_TEST_SSE_SECRET_SENTINEL not in payload


# --- DoD case (a): ready path ---------------------------------------------


@pytest.mark.asyncio
async def test_ready_path_runs_both_existing_scenarios_and_writes_privacy_safe_artifact(
    tmp_path: Path,
) -> None:
    # The real provider-free self-test runs against the controlled fake
    # OpenCode surface; only the provider/model turn is replaced by bounded
    # ready evidence so this CI test stays deterministic and offline.
    async def real_path_runner(_out: Path, _budget: float) -> RealPathEvidence:
        return _real_path_ready(2)

    result = await run_canary(
        main_sha="a" * 40,
        run_id="514",
        out_dir=tmp_path,
        preflight_budget_s=30.0,
        live_probe_budget_s=10.0,
        scenario_limit=2,
        real_path_runner=real_path_runner,
    )

    assert result.self_test.passed, result.self_test.missing_fields
    assert result.self_test.server_evidence["state"] == "supported"
    assert result.status == STATUS_READY
    assert result.reason == "canary-ready"
    assert result.exit_code == 0
    assert result.budget_category == BUDGET_WITHIN
    assert result.evidence_coverage() == {
        "stage": "observed",
        "attempt": "observed",
        "causality": "observed",
        "server_events": "observed",
    }

    path = tmp_path / "gate-c-canary-result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["main_sha"] == "a" * 40
    assert payload["run_id"] == "514"
    assert payload["status"] == STATUS_READY
    assert payload["product_qualified"] is False
    assert payload["gate_c_executed"] is False
    assert payload["real_turn_count"] == 2
    assert payload["budget"]["category"] == BUDGET_WITHIN
    assert payload["artifacts"]["result"] == "gate-c-canary-result.json"
    assert payload["privacy_validated"] is True
    assert "prompt" not in json.dumps(payload).casefold()


# --- DoD case (b): missing observer / typed cause fails closed -------------


@pytest.mark.asyncio
async def test_missing_observer_blocks_before_any_real_path_turn(
    tmp_path: Path,
) -> None:
    started = False

    async def self_test_runner(_out: Path, _timeout: float) -> SelfTestEvidence:
        return SelfTestEvidence(
            passed=False,
            missing_fields=("observer.snapshot_count",),
        )

    async def real_path_runner(_out: Path, _budget: float) -> RealPathEvidence:
        nonlocal started
        started = True
        return _real_path_ready(2)

    result = await run_canary(
        main_sha="b" * 40,
        run_id="514",
        out_dir=tmp_path,
        preflight_budget_s=10.0,
        live_probe_budget_s=5.0,
        self_test_runner=self_test_runner,
        real_path_runner=real_path_runner,
    )

    assert result.status == STATUS_BLOCKED
    assert result.reason == "diagnostic-self-test-failed"
    assert result.exit_code == 2
    assert "observer.snapshot_count" in result.missing_fields
    assert not started
    payload = json.loads((tmp_path / "gate-c-canary-result.json").read_text(encoding="utf-8"))
    assert payload["real_path"]["error"] == "skipped-self-test-blocked"
    assert payload["gate_c_executed"] is False


@pytest.mark.asyncio
async def test_self_declared_passed_without_persisted_evidence_blocks_before_real_path(
    tmp_path: Path,
) -> None:
    """A runner cannot claim READY while omitting the evidence contract."""
    started = False

    async def self_test_runner(_out: Path, _timeout: float) -> SelfTestEvidence:
        # Deliberately contradictory: passed with no snapshots, attempt,
        # watchdog marker, server facts or orphan check.
        return SelfTestEvidence(passed=True, privacy_validated=True)

    async def real_path_runner(_out: Path, _budget: float) -> RealPathEvidence:
        nonlocal started
        started = True
        return _real_path_ready(2)

    result = await run_canary(
        main_sha="b" * 40,
        run_id="514",
        out_dir=tmp_path,
        preflight_budget_s=10.0,
        live_probe_budget_s=5.0,
        self_test_runner=self_test_runner,
        real_path_runner=real_path_runner,
    )

    assert result.status == STATUS_BLOCKED
    assert result.reason == "diagnostic-self-test-failed"
    assert "observer.snapshot_count" in result.missing_fields
    assert "server_events.state" in result.missing_fields
    assert not started
    payload = json.loads((tmp_path / "gate-c-canary-result.json").read_text(encoding="utf-8"))
    assert payload["real_path"]["error"] == "skipped-self-test-blocked"
    assert payload["product_qualified"] is False
    assert payload["gate_c_executed"] is False


def test_unsupported_real_server_telemetry_blocks_as_infrastructure() -> None:
    result = evaluate_canary(
        main_sha="c" * 40,
        run_id="514",
        self_test=_self_test_ready(),
        real_path=RealPathEvidence(error="unsupported-server-telemetry"),
        preflight_budget_s=600.0,
        live_probe_budget_s=300.0,
        elapsed_s=10.0,
    )
    assert result.status == STATUS_BLOCKED
    assert result.reason == "unsupported-server-telemetry"
    assert result.exit_code == 2


def test_concrete_typed_cause_fails_without_claiming_a_product_verdict() -> None:
    result = evaluate_canary(
        main_sha="c" * 40,
        run_id="514",
        self_test=_self_test_ready(),
        real_path=RealPathEvidence(
            scenarios=(
                _scenario(
                    failure_cause="conversational-generation-failed",
                    answer_outcome="conversational-generation-failed",
                    product_outcome="rejected",
                ),
            ),
            server_evidence=_server_evidence_ready(),
        ),
        preflight_budget_s=600.0,
        live_probe_budget_s=300.0,
        elapsed_s=41.0,
        scenario_limit=1,
    )
    assert result.status == "FAIL"
    assert result.reason == "conversational-generation-failed"
    assert result.exit_code == 4
    assert result.to_dict()["product_qualified"] is False
    assert result.to_dict()["gate_c_executed"] is False


def test_missing_real_path_attempt_coverage_blocks() -> None:
    result = evaluate_canary(
        main_sha="c" * 40,
        run_id="514",
        self_test=_self_test_ready(),
        real_path=RealPathEvidence(
            scenarios=(_scenario(attempt_observed=False),),
            server_evidence=_server_evidence_ready(),
        ),
        preflight_budget_s=600.0,
        live_probe_budget_s=300.0,
        elapsed_s=10.0,
        scenario_limit=1,
    )
    assert result.status == STATUS_BLOCKED
    assert result.reason == "missing-diagnostic-chain"
    assert "real_path.attempt_observed" in result.missing_fields


def test_unsupported_server_telemetry_is_a_typed_infrastructure_block() -> None:
    result = evaluate_canary(
        main_sha="c" * 40,
        run_id="514",
        self_test=_self_test_ready(),
        real_path=RealPathEvidence(
            scenarios=(_scenario(server_evidence_state="unsupported"),),
            server_evidence={"state": "unsupported", "progress_meaning": "none"},
        ),
        preflight_budget_s=600.0,
        live_probe_budget_s=300.0,
        elapsed_s=10.0,
        scenario_limit=1,
    )
    assert result.status == STATUS_BLOCKED
    assert result.reason == "unsupported-server-telemetry"
    assert result.exit_code == 2


def test_absent_server_endpoint_is_unsupported_server_telemetry() -> None:
    result = evaluate_canary(
        main_sha="c" * 40,
        run_id="514",
        self_test=_self_test_ready(),
        real_path=RealPathEvidence(error="server-events-unavailable"),
        preflight_budget_s=600.0,
        live_probe_budget_s=300.0,
        elapsed_s=10.0,
        scenario_limit=1,
    )
    assert result.status == STATUS_BLOCKED
    assert result.reason == "unsupported-server-telemetry"


def test_unknown_product_outcome_is_not_promoted_to_a_canary_failure() -> None:
    """The canary proves wiring; unattributed quality stays Gate C's call."""
    result = evaluate_canary(
        main_sha="c" * 40,
        run_id="514",
        self_test=_self_test_ready(),
        real_path=RealPathEvidence(
            scenarios=(
                _scenario(
                    answer_outcome="",
                    product_outcome="unknown",
                    failure_cause="unknown",
                ),
            ),
            server_evidence=_server_evidence_ready(),
        ),
        preflight_budget_s=600.0,
        live_probe_budget_s=300.0,
        elapsed_s=10.0,
        scenario_limit=1,
    )
    assert result.status == STATUS_READY
    assert result.reason == "canary-ready"


# --- DoD case (c): unresponsive HTTP is bounded INCONCLUSIVE ---------------


class _HangingServer:
    """A real TCP server that accepts and never answers."""

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self._sock.settimeout(0.2)
        self._stop = threading.Event()
        self._accepted: list[socket.socket] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            self._accepted.append(conn)

    @property
    def base_url(self) -> str:
        host, port = self._sock.getsockname()[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        self._stop.set()
        for conn in self._accepted:
            try:
                conn.close()
            except OSError:
                pass
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)


@pytest.mark.asyncio
async def test_unresponsive_http_produces_inconclusive_with_snapshots(
    tmp_path: Path,
) -> None:
    server = _HangingServer()
    client = HttpOpenCodeClient(server.base_url, request_timeout=30.0)
    observer = StallSnapshotObserver(
        attempt_source=lambda: client.attempt_audit,
        out_dir=tmp_path / "attempt-observer",
        delays=(0.05, 0.10),
        interval_s=0.01,
        process_state_source=lambda: {
            "owned_by_runtime": True,
            "alive": True,
            "spawned": True,
            "started": True,
        },
    )
    task = asyncio.create_task(
        client._call("POST", "/session/ses_canary_514_private/message", {"text": "x"})
    )
    try:
        await observer.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and observer.snapshot_file_count < 2:
            await asyncio.sleep(0.01)
        assert observer.snapshot_file_count >= 2
        assert any(str(entry.get("state", "")) == "in-flight" for entry in client.attempt_audit)
        await observer.stop(flush_reason="test-budget-expired")
        evidence = RealPathEvidence(
            scenarios=(
                _scenario(
                    delivered=False,
                    answer_outcome="",
                    product_outcome="",
                    latency_ms=50.0,
                ),
            ),
            server_evidence=_server_evidence_ready(),
            budget_expired=True,
            in_flight_preserved=True,
        )
        result = evaluate_canary(
            main_sha="d" * 40,
            run_id="514",
            self_test=_self_test_ready(),
            real_path=evidence,
            preflight_budget_s=600.0,
            live_probe_budget_s=300.0,
            elapsed_s=50.0,
            scenario_limit=2,
        )
        assert result.status == STATUS_INCONCLUSIVE
        assert result.reason == "external-progress-unobserved"
        assert result.exit_code == 3
        assert result.budget_category == BUDGET_EXPIRED
        payload = result.to_dict()
        assert payload["real_path"]["budget_expired"] is True
        assert payload["real_path"]["in_flight_preserved"] is True
        assert payload["product_qualified"] is False
        assert not _scan_for_sentinels(list((tmp_path / "attempt-observer").glob("*.json")))
    finally:
        server.stop()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - cleanup
            pass


@pytest.mark.asyncio
async def test_run_canary_bounds_an_unresponsive_real_path_runner(
    tmp_path: Path,
) -> None:
    """The orchestration itself returns INCONCLUSIVE, not a product failure."""

    async def ready_self_test(_out: Path, _timeout: float) -> SelfTestEvidence:
        return _self_test_ready()

    async def slow_real_path(_out: Path, _budget: float) -> RealPathEvidence:
        await asyncio.sleep(5.0)
        return _real_path_ready(2)

    result = await run_canary(
        main_sha="d" * 40,
        run_id="514",
        out_dir=tmp_path,
        preflight_budget_s=10.0,
        live_probe_budget_s=0.2,
        self_test_runner=ready_self_test,
        real_path_runner=slow_real_path,
    )

    assert result.status == STATUS_INCONCLUSIVE
    assert result.reason == "external-progress-unobserved"
    assert result.budget_category == BUDGET_EXPIRED
    payload = json.loads((tmp_path / "gate-c-canary-result.json").read_text(encoding="utf-8"))
    assert payload["real_path"]["budget_expired"] is True
    assert payload["product_qualified"] is False
    assert payload["gate_c_executed"] is False


# --- Server evidence: /doc, /session/status and SSE route fallback --------


@pytest.mark.asyncio
async def test_server_evidence_supports_global_event_fallback_and_unobserved_shape() -> None:
    fallback, first_unobserved = await _server_observation_checks(
        error_event=True, event_route="/global/event"
    )
    # The controlled fallback server reports an active drain, so real status
    # facts must make the probe observed rather than unobserved.
    assert first_unobserved is False
    assert fallback["state"] == "supported"
    assert fallback["event_endpoint"] == "/global/event"
    assert fallback["endpoints"]["/doc"]["state"] == "supported"
    assert fallback["server_error_status_code"] == 503

    absent, absent_unobserved = await _server_observation_checks(
        error_event=False,
        event_enabled=False,
        session_status=False,
    )
    assert absent_unobserved is True
    assert absent["state"] == "supported"
    assert not absent.get("session_state_counts", {}).get("busy")


# --- Privacy boundary ------------------------------------------------------


def test_artifact_rejects_text_keys_and_sentinel_leaks(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        assert_no_text_leak({"prompt": "hello"})

    leaking = evaluate_canary(
        main_sha="e" * 40,
        run_id="514",
        self_test=SelfTestEvidence(
            passed=False,
            missing_fields=(SELF_TEST_PROMPT_SENTINEL,),
        ),
        real_path=RealPathEvidence(error="skipped-self-test-blocked"),
        preflight_budget_s=600.0,
        live_probe_budget_s=300.0,
        elapsed_s=1.0,
    )
    with pytest.raises(GateCCanaryError):
        write_canary_result(tmp_path, leaking)
    assert not (tmp_path / "gate-c-canary-result.json").exists()


# --- Fail-closed orchestration --------------------------------------------


@pytest.mark.asyncio
async def test_injected_self_test_exception_fails_closed_without_real_turns(
    tmp_path: Path,
) -> None:
    started = False

    async def self_test_runner(_out: Path, _timeout: float) -> SelfTestEvidence:
        raise RuntimeError("observer construction failed")

    async def real_path_runner(_out: Path, _budget: float) -> RealPathEvidence:
        nonlocal started
        started = True
        return _real_path_ready(2)

    result = await run_canary(
        main_sha="f" * 40,
        run_id="514",
        out_dir=tmp_path,
        preflight_budget_s=10.0,
        live_probe_budget_s=5.0,
        self_test_runner=self_test_runner,
        real_path_runner=real_path_runner,
    )

    assert result.status == STATUS_BLOCKED
    assert result.reason == "diagnostic-self-test-failed"
    assert "self-test.RuntimeError" in result.missing_fields
    assert not started
    payload = json.loads((tmp_path / "gate-c-canary-result.json").read_text(encoding="utf-8"))
    assert payload["real_path"]["error"] == "skipped-self-test-blocked"
    assert payload["gate_c_executed"] is False
    assert payload["product_qualified"] is False


# --- Workflow and CLI integration -----------------------------------------

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "aa-self-proving-qualification.yml"


def test_workflow_gates_the_unchanged_campaign_behind_a_bounded_canary() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    canary_index = text.index("id: gate-c-canary")
    gate_index = text.index("id: gate-c\n")
    assert canary_index < gate_index
    canary_step = text[canary_index:gate_index]

    # The canary is bounded independently of the shared 135-minute job.
    assert "timeout-minutes: 12" in canary_step
    assert "--preflight-budget-s 600" in canary_step
    assert "--live-probe-budget-s 300" in canary_step
    assert "scripts/run_gate_c_canary.py" in canary_step
    assert "aa-gate-c-canary" in text

    # READY is the only authorization, and the artifact may never carry a
    # product/Gate C verdict or a different exact-main identity.
    assert 'payload.get("status") != "READY"' in text
    assert 'payload.get("product_qualified") is not False' in text
    assert 'payload.get("gate_c_executed") is not False' in text
    assert 'payload.get("main_sha"' in text

    # The diagnostic canary is not allowed to rebuild the prepared runtime,
    # index or model environment.
    assert "python scripts/build_retrieval_index.py" not in canary_step
    assert "prepare_opencode_verification_env.sh" not in canary_step


def test_cli_invalid_sha_publishes_blocked_artifact_and_never_stale_ready(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "gate-c-canary-result.json"
    stale.write_text('{"status": "READY"}\n', encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_gate_c_canary.py"),
            "--main-sha",
            "not-an-exact-sha",
            "--out-dir",
            str(tmp_path),
            "--run-id",
            "514",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 2
    payload = json.loads(stale.read_text(encoding="utf-8"))
    assert payload["status"] == STATUS_BLOCKED
    assert payload["reason"] == "invalid-main-sha"
    assert payload["gate_c_executed"] is False
    assert payload["product_qualified"] is False
    assert "prompt" not in json.dumps(payload).casefold()
    assert SELF_TEST_PROMPT_SENTINEL not in proc.stdout
