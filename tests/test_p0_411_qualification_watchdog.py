"""Regression coverage for the independent live Gate C watchdog (issue #411).

The watchdog must preempt a non-cooperative blocking provider/retrieval call
without waiting for the Actions job timeout, retain privacy-safe per-scenario
evidence, and preserve ordinary slow-but-responsive semantics.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest

from aa.opencode.errors import OpenCodeRateLimitError
from aa.qualification.live_watchdog import (
    ENV_FINALISATION_RESERVE_S,
    ENV_JOB_STARTED_AT,
    ENV_JOB_TIMEOUT_MINUTES,
    ENV_PROGRESS_PATH,
    ENV_SCENARIO_GRACE_S,
    ENV_TOTAL_DEADLINE_S,
    live_total_deadline_s,
    read_live_progress,
    record_live_progress,
    supervise_live_gate_c,
)
from aa.qualification.product_contract_live import LaneResult, LiveSummary

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_dead(pid: int, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _process_alive(pid):
            return True
        time.sleep(0.05)
    return not _process_alive(pid)


def test_blocking_scenario_8_is_preempted_with_typed_negative_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress_path = tmp_path / "live-watchdog-progress.json"
    monkeypatch.setenv(ENV_PROGRESS_PATH, str(progress_path))
    monkeypatch.delenv(ENV_JOB_STARTED_AT, raising=False)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "30")
    monkeypatch.setenv(ENV_SCENARIO_GRACE_S, "0.2")

    async def blocking_lane(run_id: str) -> LaneResult:
        assert run_id == "watchdog-run"
        for index in range(1, 8):
            record_live_progress(
                "scenario-start",
                scenario_index=index,
                family="long-conversation",
                scenario_deadline_at=time.time() + 5.0,
                completed_scenario_count=index - 1,
                scenario_total=18,
            )
            record_live_progress(
                "scenario-complete",
                scenario_index=index,
                family="long-conversation",
                clear_scenario_deadline=True,
                completed_scenario_count=index,
                scenario_total=18,
            )
        descendant = subprocess.Popen(["sleep", "60"])  # noqa: S607
        record_live_progress(
            "scenario-start",
            scenario_index=8,
            family="long-conversation",
            scenario_deadline_at=time.time() + 0.4,
            completed_scenario_count=7,
            scenario_total=18,
            descendant_pid=descendant.pid,
        )
        # Intentionally non-cooperative: no await point, no asyncio timeout
        # can fire in this process while it is blocked here.
        time.sleep(60)
        return LaneResult(lane="live-telegram-evidence", status="PASS")

    started = time.monotonic()
    result = supervise_live_gate_c(
        blocking_lane,
        run_id="watchdog-run",
        main_sha=SHA,
        progress_path=progress_path,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 15.0, f"watchdog waited too long: {elapsed:.1f}s"
    assert result.status == "FAIL"
    assert result.failed == ("live-scenario-watchdog-timeout-unknown",)
    metrics = result.metrics
    assert metrics["watchdog_triggered"] is True
    assert metrics["watchdog_trigger"] == "scenario-deadline"
    assert metrics["timeout_category"] == "unknown"
    assert metrics["scenario_timeout"] is True
    assert metrics["completed_scenario_count"] == 7
    assert metrics["last_scenario_index"] == 8
    assert metrics["scenario_total"] == 18
    assert metrics["exact_sha"] == SHA
    assert metrics["gate_boundary"]["telegram_egress_mode"] == "stubbed"
    assert metrics["scenario_diagnostics"][-1]["timeout_category"] == "unknown"
    persisted = read_live_progress(progress_path)
    descendant_pid = int(persisted["descendant_pid"])
    assert _wait_dead(descendant_pid)


def test_slow_but_responsive_lane_assesses_all_18_without_false_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress_path = tmp_path / "live-watchdog-progress.json"
    monkeypatch.setenv(ENV_PROGRESS_PATH, str(progress_path))
    monkeypatch.delenv(ENV_JOB_STARTED_AT, raising=False)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "30")
    monkeypatch.setenv(ENV_SCENARIO_GRACE_S, "5")

    async def responsive_lane(run_id: str) -> LaneResult:
        _ = run_id
        for index in range(1, 19):
            record_live_progress(
                "scenario-start",
                scenario_index=index,
                family="long-conversation",
                scenario_deadline_at=time.time() + 5.0,
                completed_scenario_count=index - 1,
                scenario_total=18,
            )
            await asyncio.sleep(0.01)
            record_live_progress(
                "scenario-complete",
                scenario_index=index,
                family="long-conversation",
                clear_scenario_deadline=True,
                completed_scenario_count=index,
                scenario_total=18,
            )
        return LaneResult(
            lane="live-telegram-evidence",
            status="PASS",
            passed=("live-egress-simulated-labeled",),
            metrics={"scenarios_executed": 18},
        )

    result = supervise_live_gate_c(
        responsive_lane,
        run_id="slow-responsive",
        main_sha=SHA,
        progress_path=progress_path,
    )
    assert result.status == "PASS"
    assert result.metrics["scenarios_executed"] == 18
    assert "watchdog_triggered" not in result.metrics


def test_total_deadline_kills_stalled_lane_without_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress_path = tmp_path / "live-watchdog-progress.json"
    monkeypatch.setenv(ENV_PROGRESS_PATH, str(progress_path))
    monkeypatch.delenv(ENV_JOB_STARTED_AT, raising=False)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "0.8")

    async def stalled_lane(run_id: str) -> LaneResult:
        _ = run_id
        time.sleep(60)
        return LaneResult(lane="live-telegram-evidence", status="PASS")

    started = time.monotonic()
    result = supervise_live_gate_c(
        stalled_lane,
        run_id="stalled",
        main_sha=SHA,
        progress_path=progress_path,
    )
    assert time.monotonic() - started < 10.0
    assert result.status == "FAIL"
    assert result.metrics["watchdog_trigger"] == "total-deadline"
    assert result.metrics["scenarios_executed"] == 0
    assert result.metrics["exact_sha"] == SHA


def test_total_deadline_reserves_finalisation_time_from_actions_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_000_000.0
    monkeypatch.setenv(ENV_JOB_STARTED_AT, str(now - 600.0))
    monkeypatch.setenv(ENV_JOB_TIMEOUT_MINUTES, "120")
    monkeypatch.setenv(ENV_FINALISATION_RESERVE_S, "900")
    monkeypatch.delenv(ENV_TOTAL_DEADLINE_S, raising=False)
    # 7200s job budget - 900s reserve - 600s already elapsed = 5700s.
    assert live_total_deadline_s(now=now) == 5700.0


def test_progress_evidence_is_privacy_safe_and_checkpointed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress_path = tmp_path / "progress.json"
    monkeypatch.setenv(ENV_PROGRESS_PATH, str(progress_path))
    record_live_progress(
        "scenario-complete",
        scenario_index=8,
        family="long-conversation",
        completed_scenario_count=7,
        scenario_total=18,
        scenario_diagnostics=[{"scenario_index": 7, "family": "topic-shift"}],
    )
    payload = read_live_progress(progress_path)
    rendered = str(payload).lower()
    assert payload["scenario_index"] == 8
    assert payload["completed_scenario_count"] == 7
    for forbidden in ("utterance", "answer", "passage", "secret", "token"):
        assert forbidden not in rendered


def test_watchdog_preserves_provider_429_runner_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress_path = tmp_path / "live-watchdog-progress.json"
    monkeypatch.setenv(ENV_PROGRESS_PATH, str(progress_path))
    monkeypatch.delenv(ENV_JOB_STARTED_AT, raising=False)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "30")

    async def rate_limited_lane(run_id: str) -> LaneResult:
        _ = run_id
        raise OpenCodeRateLimitError("opencode request rate-limited: http=429")

    with pytest.raises(OpenCodeRateLimitError, match="http=429"):
        supervise_live_gate_c(
            rate_limited_lane,
            run_id="rate-limited",
            main_sha=SHA,
            progress_path=progress_path,
        )


def test_watchdog_timeout_maps_to_authoritative_gate_c_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_self_proving_qualification as runner

    progress_path = tmp_path / "live-watchdog-progress.json"
    monkeypatch.setenv(ENV_PROGRESS_PATH, str(progress_path))
    monkeypatch.delenv(ENV_JOB_STARTED_AT, raising=False)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "0.8")

    async def stalled_lane(run_id: str) -> LaneResult:
        _ = run_id
        time.sleep(60)
        return LaneResult(lane="live-telegram-evidence", status="PASS")

    lane = supervise_live_gate_c(
        stalled_lane,
        run_id="gate-c-map",
        main_sha=SHA,
        progress_path=progress_path,
    )
    summary = LiveSummary(
        main_sha=SHA,
        status=lane.status,
        lanes=(lane,),
        static_gates={},
        run_id="gate-c-map",
    )
    live_out = tmp_path / "live-out"
    live_out.mkdir()
    (live_out / "product-contract-live-summary.json").write_text(
        __import__("json").dumps(summary.to_dict()), encoding="utf-8"
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    decided = runner._gate_c_live_evidence(SHA, "gate-c-map", "p" * 64, "r" * 64)
    assert decided is not None
    evidence, latencies, aggregate = decided
    assert evidence.status == "FAIL"
    assert evidence.sha == SHA
    assert evidence.failure_category == "live-scenario-watchdog-timeout-unknown"
    assert evidence.live_trusted is True
    assert latencies == []
    assert aggregate is None


def test_workflows_use_two_hour_qualification_limit_and_watchdog_markers() -> None:
    self_proving = (ROOT / ".github/workflows/aa-self-proving-qualification.yml").read_text(
        encoding="utf-8"
    )
    expected = "timeout-minutes: ${{ vars.AA_QUALIFICATION_JOB_TIMEOUT_MINUTES || 120 }}"
    assert self_proving.count(expected) == 2
    assert "timeout-minutes: 45" not in self_proving
    assert "AA_QUALIFICATION_JOB_STARTED_AT" in self_proving
    assert "AA_QUALIFICATION_FINALISATION_RESERVE_S" in self_proving
    assert "AA_QUALIFICATION_LIVE_PROGRESS_PATH" in self_proving
    assert "live-out/live-watchdog-progress.json" in self_proving

    real_book = (ROOT / ".github/workflows/aa-real-book-retrieval-qualification.yml").read_text(
        encoding="utf-8"
    )
    assert expected in real_book
    assert "timeout-minutes: 60" not in real_book
