"""Issue #412: no artificial qualification/stage/transport walls.

The owner requirement is behavioural: a healthy slow reply must survive,
while a genuinely non-cooperative hang is still terminated by the separate
process watchdog with typed negative evidence.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from aa.opencode.client import HttpOpenCodeClient
from aa.qualification.live_watchdog import (
    ENV_JOB_STARTED_MONOTONIC,
    ENV_PROGRESS_PATH,
    ENV_SCENARIO_INACTIVITY_S,
    ENV_TOTAL_DEADLINE_S,
    read_live_progress,
    record_live_progress,
    supervise_live_gate_c,
)
from aa.qualification.product_contract_live import LaneResult
from aa.time_policy import TimePolicy, current_time_policy

ROOT = Path(__file__).resolve().parents[1]
SHA = "b" * 40


def test_effective_policy_has_no_fixed_300s_or_900s_response_wall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "AA_TRANSPORT_TIMEOUT_S",
        "AA_QUALIFICATION_SCENARIO_WAIT_S",
        "AA_QUALIFICATION_SCENARIO_INACTIVITY_S",
        "AA_QUALIFICATION_CAMPAIGN_S",
        "AA_QUALIFICATION_FINALISATION_RESERVE_S",
    ):
        monkeypatch.delenv(name, raising=False)
    policy = current_time_policy()
    assert policy.transport_timeout_s > 900.0
    assert policy.qualification_scenario_wait_s != 300.0
    assert policy.qualification_campaign_s == 135 * 60.0
    assert policy.qualification_finalisation_reserve_s > 0.0
    assert policy.qualification_scenario_inactivity_s > 900.0
    projected = policy.to_safe_dict()
    assert projected["qualification_campaign_s"] == 8100.0


def test_a_repository_variable_cannot_shrink_the_effective_135m_workflow_cap() -> None:
    self_proving = (ROOT / ".github/workflows/aa-self-proving-qualification.yml").read_text(
        encoding="utf-8"
    )
    real_book = (ROOT / ".github/workflows/aa-real-book-retrieval-qualification.yml").read_text(
        encoding="utf-8"
    )
    floored = (
        "timeout-minutes: ${{ fromJSON(vars.AA_QUALIFICATION_JOB_TIMEOUT_MINUTES || '135') > 135"
    )
    assert self_proving.count(floored) == 2
    assert "timeout-minutes: 120" not in self_proving
    assert "|| 120" not in self_proving
    assert real_book.count(floored) == 1
    assert "timeout-minutes: 120" not in real_book


def test_gate_c_child_has_no_fixed_scenario_or_voice_cutoff() -> None:
    source = (ROOT / "src/aa/qualification/product_contract_live.py").read_text(encoding="utf-8")
    assert "scenario_deadline_at=time.time() + scenario_wait_s" not in source
    assert "loop.time() + scenario_wait_s" not in source
    assert "loop.time() + 120.0" not in source
    assert 'record_live_progress("scenario-response"' in source


def test_transport_has_no_total_wait_for_and_never_mutates_shared_timeout() -> None:
    source = (ROOT / "src/aa/opencode/client.py").read_text(encoding="utf-8")
    assert "asyncio.wait_for(\n                asyncio.to_thread" not in source
    assert "self._request_timeout = timeout" not in source
    assert "timeout=timeout" in source


def test_concurrent_sends_keep_independent_timeouts_and_reversed_completion() -> None:
    class ProbeClient(HttpOpenCodeClient):
        def __init__(self) -> None:
            super().__init__("http://127.0.0.1:1", request_timeout=3600.0)
            self.seen: list[tuple[str, float | None]] = []
            self.lock = threading.Lock()

        def _sync_request(
            self,
            method: str,
            path: str,
            body: dict[str, object] | None,
            timeout: float | None,
        ) -> object:
            if body is None:
                return None
            text = str(body["parts"][0]["text"])  # type: ignore[index]
            with self.lock:
                self.seen.append((text, timeout))
            # The slow request starts first but completes second.
            time.sleep(0.06 if text == "slow" else 0.01)
            return {
                "parts": [{"type": "text", "text": f"{text}-ok"}],
                "info": {},
            }

    async def scenario() -> tuple[str, str, ProbeClient]:
        client = ProbeClient()
        slow = asyncio.create_task(client.send_message("ses_slow", "slow", timeout=900.0))
        fast = asyncio.create_task(client.send_message("ses_fast", "fast", timeout=10.0))
        fast_reply = await fast
        slow_reply = await slow
        return slow_reply, fast_reply, client

    slow_reply, fast_reply, client = asyncio.run(scenario())
    assert slow_reply == "slow-ok"
    assert fast_reply == "fast-ok"
    assert sorted(client.seen) == [("fast", 10.0), ("slow", 900.0)]


def test_cancelled_request_does_not_contaminate_the_next_timeout() -> None:
    class ProbeClient(HttpOpenCodeClient):
        def __init__(self) -> None:
            super().__init__("http://127.0.0.1:1", request_timeout=3600.0)
            self.seen: list[tuple[str, float | None]] = []

        def _sync_request(
            self,
            method: str,
            path: str,
            body: dict[str, object] | None,
            timeout: float | None,
        ) -> object:
            if body is None:
                return None
            text = str(body["parts"][0]["text"])  # type: ignore[index]
            self.seen.append((text, timeout))
            time.sleep(0.05)
            return {"parts": [{"type": "text", "text": text}], "info": {}}

    async def scenario() -> ProbeClient:
        client = ProbeClient()
        task = asyncio.create_task(client.send_message("ses_a", "first", timeout=900.0))
        await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.01)
        assert await client.send_message("ses_b", "second", timeout=10.0) == "second"
        return client

    client = asyncio.run(scenario())
    assert ("first", 900.0) in client.seen
    assert ("second", 10.0) in client.seen


def test_no_outer_wall_clock_rejects_a_progressing_slow_request() -> None:
    class SlowButAlive(HttpOpenCodeClient):
        def _sync_request(
            self,
            method: str,
            path: str,
            body: dict[str, object] | None,
            timeout: float | None,
        ) -> object:
            # 0.05s of genuine work beyond the legacy 0.01s caller bound.
            time.sleep(0.05)
            return {"parts": [{"type": "text", "text": "late-ok"}], "info": {}}

    client = SlowButAlive("http://127.0.0.1:1", request_timeout=0.01)
    reply = asyncio.run(client.send_message("ses", "hello", timeout=0.01))
    assert reply == "late-ok"


def test_parent_ignores_fixed_scenario_deadline_when_real_progress_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress_path = tmp_path / "progress.json"
    monkeypatch.setenv(ENV_PROGRESS_PATH, str(progress_path))
    monkeypatch.delenv("AA_QUALIFICATION_JOB_STARTED_AT", raising=False)
    monkeypatch.delenv(ENV_JOB_STARTED_MONOTONIC, raising=False)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "5")
    monkeypatch.setenv(ENV_SCENARIO_INACTIVITY_S, "0.3")

    async def slow_progressing_lane(run_id: str) -> LaneResult:
        _ = run_id
        # A stale fixed deadline must not kill a progressing scenario.
        record_live_progress(
            "scenario-start",
            scenario_index=1,
            family="long-conversation",
            scenario_deadline_at=time.time() - 1.0,
            real_progress=True,
        )
        for _ in range(6):
            await asyncio.sleep(0.08)
            record_live_progress("scenario-stage", active_stage="answer", real_progress=True)
        return LaneResult(
            lane="live-telegram-evidence",
            status="PASS",
            passed=("slow-but-progressing",),
            metrics={"scenarios_executed": 1},
        )

    result = supervise_live_gate_c(
        slow_progressing_lane,
        run_id="slow-progress",
        main_sha=SHA,
        progress_path=progress_path,
    )
    assert result.status == "PASS"
    assert "watchdog_triggered" not in result.metrics


def test_fake_heartbeats_without_real_progress_are_still_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress_path = tmp_path / "progress.json"
    monkeypatch.setenv(ENV_PROGRESS_PATH, str(progress_path))
    monkeypatch.delenv("AA_QUALIFICATION_JOB_STARTED_AT", raising=False)
    monkeypatch.delenv(ENV_JOB_STARTED_MONOTONIC, raising=False)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "20")
    monkeypatch.setenv(ENV_SCENARIO_INACTIVITY_S, "0.4")

    async def heartbeat_only_lane(run_id: str) -> LaneResult:
        _ = run_id
        record_live_progress(
            "scenario-start",
            scenario_index=8,
            family="long-conversation",
            completed_scenario_count=7,
            scenario_total=18,
            real_progress=True,
        )
        while True:
            await asyncio.sleep(0.08)
            # The polling loop is alive, but no stage/response advanced.
            record_live_progress(
                "scenario-heartbeat",
                scenario_index=8,
                family="long-conversation",
                completed_scenario_count=7,
                scenario_total=18,
            )

    started = time.monotonic()
    result = supervise_live_gate_c(
        heartbeat_only_lane,
        run_id="heartbeat-only",
        main_sha=SHA,
        progress_path=progress_path,
    )
    assert time.monotonic() - started < 10.0
    assert result.status == "FAIL"
    assert result.metrics["watchdog_trigger"] == "no-real-progress"
    assert result.metrics["last_scenario_index"] == 8
    assert result.metrics["completed_scenario_count"] == 7
    persisted = read_live_progress(progress_path)
    assert persisted["last_real_progress_phase"] == "scenario-start"


def test_legacy_300s_env_value_cannot_reintroduce_a_scenario_cutoff() -> None:
    policy = TimePolicy.from_env(
        {
            "AA_QUALIFICATION_SCENARIO_WAIT_S": "300",
            "AA_TURN_TIME_LIMITS_ENABLED": "false",
        }
    )
    # The value is retained only as a projection; the campaign and real
    # progress supervision own termination.
    assert policy.qualification_campaign_s == 8100.0
    assert policy.qualification_scenario_inactivity_s > 300.0
    assert policy.turn_time_limits_enabled is False
