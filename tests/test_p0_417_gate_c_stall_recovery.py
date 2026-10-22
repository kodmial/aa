"""Regression coverage for the systemic Gate C stall recovery (kodmial/aa#417).

The confirmed incident (run 39715421348): three serial 30-minute
``no-real-progress`` stalls each left their abandoned turn running, late
replies contaminated the next scenario, and the watchdog timeout stayed
``unknown``. This module owns the behavioral contract:

- privacy-safe attribution through OpenCode request evidence, closed
  vocabulary, ``progress-unobservable`` honesty;
- no orphan work: quarantine late deliveries, cancel the abandoned
  per-chat turn, bounded shutdown, runtime restart only when
  non-cooperative;
- bounded recovery: the parent watchdog holds its trigger inside the
  child's declared recovery window and never while real stage progress
  advances;
- Gate C evidence carries the typed stall diagnosis (never unknown while
  stage evidence exists).
"""

from __future__ import annotations

import asyncio
import inspect
import os
import time
from pathlib import Path
from typing import Any

import pytest

from aa.qualification.live_turn_repair import (
    build_timeout_attribution,
    derive_timeout_category,
    quiesce_and_abort_stalled_turn,
    request_state_from_audit,
)
from aa.qualification.live_watchdog import (
    DEFAULT_RECOVERY_ALLOWANCE_S,
    ENV_RECOVERY_ALLOWANCE_S,
    ENV_SCENARIO_INACTIVITY_S,
    ENV_TOTAL_DEADLINE_S,
    read_live_progress,
    record_live_progress,
    recovery_allowance_s,
    supervise_live_gate_c,
)
from aa.qualification.product_contract_live import LaneResult
from aa.telegram.dispatcher import ChatTurnDispatcher
from aa.telegram.transport import TelegramIncoming

ROOT = Path(__file__).resolve().parents[1]
SRC_LIVE = ROOT / "src" / "aa" / "qualification" / "product_contract_live.py"
SHA = "a" * 40
CHAT = 920002


def _audit(
    operation: str,
    causality_id: str,
    *,
    success: bool | None = None,
    failure_category: str = "",
    latency_ms: float = 12.5,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "operation": operation,
        "latency_ms": latency_ms,
    }
    if success is not None:
        entry["success"] = success
    if success is False:
        entry["failure_category"] = failure_category
    if success is None:
        entry["state"] = "in-flight"
    return _causality_stub(entry, causality_id)


def _causality_stub(entry: dict[str, Any], causality_id: str) -> dict[str, Any]:
    entry = dict(entry)
    entry["causality_id"] = str(causality_id)[:64]
    return entry


# --- Attribution --------------------------------------------------------


def test_derive_timeout_category_is_closed_vocabulary() -> None:
    assert derive_timeout_category("planner") == "stuck-provider"
    assert derive_timeout_category("answer") == "stuck-provider"
    assert derive_timeout_category("whole_turn_judge") == "stuck-provider"
    assert derive_timeout_category("retrieval") == "stuck-retrieval"
    assert derive_timeout_category("validation") == "unknown"
    assert derive_timeout_category("") == "unknown"


def test_request_state_none_is_progress_unobservable() -> None:
    state = request_state_from_audit([], "turn-1")
    assert state["request_started"] is False
    assert state["request_state"] == "none"
    assert state["request_progress_observable"] == "progress-unobservable"


def test_request_state_completed_failed_cancelled_inflight() -> None:
    audit = _audit("message-text", "turn-1", success=True)
    completed = request_state_from_audit([audit], "turn-1")
    assert completed["request_started"] is True
    assert completed["request_state"] == "request-completed"
    assert completed["last_confirmed_request"] == "message-text"
    assert completed["request_progress_observable"] == "progress-unobservable"

    failed = request_state_from_audit(
        [_audit("message-text", "turn-1", success=False, failure_category="transient")],
        "turn-1",
    )
    assert failed["request_state"] == "request-failed"
    assert failed["last_request_failure_category"] == "transient"

    cancelled = request_state_from_audit(
        [_audit("message-text", "turn-1", success=False, failure_category="cancelled")],
        "turn-1",
    )
    assert cancelled["request_state"] == "request-cancelled"

    in_flight = request_state_from_audit([_audit("message-text", "turn-1", success=None)], "turn-1")
    assert in_flight["request_state"] == "unconfirmed-in-flight"


def test_request_state_ignores_other_causality_ids() -> None:
    audit = [
        _audit("message-text", "turn-1", success=True),
        _audit("message-text", "turn-2", success=False, failure_category="transient"),
    ]
    state = request_state_from_audit(audit, "turn-1")
    assert state["request_state"] == "request-completed"
    assert state["http_request_count"] == 1


def test_build_timeout_attribution_keeps_stage_evidence() -> None:
    attribution = build_timeout_attribution(
        causality_id="turn-1",
        family="long-conversation",
        last_active_stage="retrieval",
        stage_changed_at=time.monotonic() - 1.0,
        started_monotonic=time.monotonic() - 0.5,
        request_audit=[
            _audit("message-text", "turn-1", success=False, failure_category="cancelled")
        ],
        timeout_reason="no-real-progress",
    )
    assert attribution["timeout_category"] == "stuck-retrieval"
    assert attribution["timeout_reason"] == "no-real-progress"
    assert attribution["family"] == "long-conversation"
    assert attribution["request_state"] == "request-cancelled"
    assert attribution["stage_elapsed_ms"] >= 0.0
    # Privacy: the attribution block carries fixed-vocabulary labels and a
    # causality id, never prompt or reply bodies.
    blocked = {"prompt", "text", "reply", "payload", "body"}
    assert blocked.isdisjoint(attribution.keys())


# --- Dispatcher cancellation -------------------------------------------


async def _process_ok(incoming: Any) -> None:
    incoming.append("done")


async def test_cancel_chat_drains_queue_and_stops_worker() -> None:
    seen: list[Any] = []
    gate = asyncio.Event()

    async def process(incoming: Any) -> None:
        await gate.wait()
        seen.append(incoming)

    dispatcher = ChatTurnDispatcher(process, max_concurrent_turns=1, per_chat_queue_size=4)
    await dispatcher.start()
    await dispatcher.submit(_incoming("stall"))
    in_flight = dispatcher.in_flight(CHAT)
    for _ in range(3):
        await dispatcher.submit(_incoming("queued"))
    # Let the head turn claim the semaphore before cancelling.
    await asyncio.sleep(0.02)
    summary = await dispatcher.cancel_chat(CHAT, grace_s=2.0)
    assert in_flight is True
    assert summary["had_worker"] is True
    assert summary["dropped_queued_turns"] == 3
    assert summary["non_cooperative"] is False
    assert dispatcher.in_flight(CHAT) is False
    _ = _process_ok  # keep helper referenced


async def test_cancel_chat_without_worker_is_noop() -> None:
    dispatcher = ChatTurnDispatcher(_process_ok, max_concurrent_turns=1, per_chat_queue_size=2)
    await dispatcher.start()
    summary = await dispatcher.cancel_chat(CHAT, grace_s=1.0)
    assert summary["had_worker"] is False


async def test_cancel_chat_flags_non_cooperative_after_grace() -> None:
    ignored_cancellations = 0

    async def stubborn_once_then_cooperative(incoming: Any) -> None:
        nonlocal ignored_cancellations
        _ = incoming
        while True:
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                ignored_cancellations += 1
                if ignored_cancellations > 1:
                    raise

    dispatcher = ChatTurnDispatcher(
        stubborn_once_then_cooperative,
        max_concurrent_turns=1,
        per_chat_queue_size=2,
    )
    await dispatcher.start()
    await dispatcher.submit(_incoming("stall"))
    # Let the worker actually enter the stubborn turn (not queue.get).
    await asyncio.sleep(0.05)
    summary = await dispatcher.cancel_chat(CHAT, grace_s=0.1)
    assert summary["had_worker"] is True
    assert summary["stopped_within_grace"] is False
    assert summary["non_cooperative"] is True
    assert ignored_cancellations == 1


# --- Quiesce/abort orchestration ---------------------------------------


class _FakeQuiesceSeam:
    """Mirrors the lane's Telegram seam quarantine contract (counts only)."""

    def __init__(self) -> None:
        self.sent_texts: list[str] = []
        self.quiescing_reason = ""
        self.quarantined_texts: list[dict[str, Any]] = []

    def begin_quiescence(self, reason: str) -> None:
        self.quiescing_reason = str(reason or "unknown")[:32]

    def end_quiescence(self) -> None:
        self.quiescing_reason = ""


class _FakeApp:
    def __init__(self, *, had_worker: bool, summary: dict[str, Any] | None = None) -> None:
        self.running = True
        self.restart_count = 0
        self._had_worker = had_worker
        self._summary = summary
        self.cancelled = False

    async def cancel_pending_turns(self, chat_id: int, grace_s: float = 5.0) -> dict[str, Any]:
        self.cancelled = True
        if self._summary is not None:
            return dict(self._summary)
        return {
            "had_worker": self._had_worker,
            "stopped_within_grace": True,
            "dropped_queued_turns": 0,
            "non_cooperative": False,
        }

    def turn_in_flight(self, chat_id: int) -> bool:
        return False

    async def stop(self) -> None:
        self.running = False

    async def start(self) -> None:
        self.running = True


async def test_quiesce_abort_records_cleanup_without_restart() -> None:
    seam = _FakeQuiesceSeam()
    app = _FakeApp(had_worker=True)
    cleanup = await quiesce_and_abort_stalled_turn(
        app=app,
        chat_id=CHAT,
        seam=seam,
        grace_s=0.05,
        reason="no-real-progress",
    )
    assert seam.quiescing_reason != ""
    assert cleanup["quiesced"] is True
    assert cleanup["chat_cancelled"] is True
    assert cleanup["non_cooperative"] is False
    assert cleanup.get("runtime_retired") is not True


async def test_quiesce_abort_restarts_non_cooperative_runtime() -> None:
    seam = _FakeQuiesceSeam()
    app = _FakeApp(
        had_worker=True,
        summary={
            "had_worker": True,
            "stopped_within_grace": False,
            "non_cooperative": True,
            "dropped_queued_turns": 0,
        },
    )
    retire_calls: list[bool] = []

    async def retire() -> None:
        retire_calls.append(True)
        await app.stop()
        await app.start()

    cleanup = await quiesce_and_abort_stalled_turn(
        app=app, chat_id=CHAT, seam=seam, grace_s=0.05, reason="no-real-progress", restart=retire
    )
    assert retire_calls == [True]
    assert cleanup["non_cooperative"] is True
    assert cleanup["runtime_retired"] is True
    assert cleanup["runtime_restarted"] is True


async def test_quiesce_abort_survives_restart_failure() -> None:
    async def broken_retire() -> None:
        raise RuntimeError("restart failed")

    app = _FakeApp(
        had_worker=True,
        summary={
            "had_worker": True,
            "stopped_within_grace": False,
            "non_cooperative": True,
        },
    )
    cleanup = await quiesce_and_abort_stalled_turn(
        app=app,
        chat_id=CHAT,
        seam=_FakeQuiesceSeam(),
        grace_s=0.05,
        reason="no-real-progress",
        restart=broken_retire,
    )
    assert cleanup["runtime_retired"] is True
    assert cleanup["runtime_restarted"] is False
    # Recovery never raises: the lane keeps moving.
    assert "cleanup_failed" not in cleanup


# --- Lane wiring invariants (source-level, mirrors tests/test_p0_341) ---


def _lane_source() -> str:
    from aa.qualification import product_contract_live as live_mod

    lines, _ = inspect.getsourcelines(live_mod.run_live_telegram_evidence_lane)
    return "".join(lines)


def test_lane_quarantines_late_deliveries_and_resets_per_scenario() -> None:
    source = _lane_source()
    # Quarantine exists and is enforced on the seam path.
    assert "self.quiescing_reason" in source
    assert "self.quarantined_texts.append" in source
    # Every new scenario discards the previous scenario's abandoned-turn
    # quarantine window, so stale deliveries cannot satisfy scenario N+1.
    assert source.count("api.end_quiescence()") >= 2


def test_lane_emits_bounded_abort_lifecycle_markers() -> None:
    source = _lane_source()
    assert '"scenario-abort-start"' in source
    assert "stall_recovery_in_progress=True" in source
    assert '"scenario-abort-done"' in source
    assert "stall_recovery_in_progress=False" in source
    # The abort lifecycle appears after the timeout trigger and before
    # scenario-loop advance: cleanup runs inside the loop, not after it.
    assert source.index('"scenario-abort-start"') < source.index("quiesce_and_abort_stalled_turn")


def test_lane_censorship_links_timeout_sample_to_causality() -> None:
    source = _lane_source()
    assert '"outcome": "timeout"' in source
    assert '"censored": True' in source
    assert "scenario_causality_id" in source
    # The stage marker survives the timeout: the watchdog must be able to
    # classify identically from the persisted evidence.
    assert "clear_active_stage=False" in source


# --- Parent watchdog recovery allowance --------------------------------


def _incoming(text: str) -> TelegramIncoming:
    return TelegramIncoming(update_id=1, chat_id=CHAT, message_id=1, text=text)


def _progress_path(tmp_path: Path) -> Path:
    path = tmp_path / "p0417-progress.json"
    os.environ["AA_QUALIFICATION_LIVE_PROGRESS_PATH"] = str(path)
    return path


def _watchdog_lane_hold_then_kill(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> float:
    """Child scenario stalls, declares recovery; parent must hold, then kill."""
    progress_path = _progress_path(tmp_path)
    monkeypatch.delenv("AA_QUALIFICATION_JOB_STARTED_AT", raising=False)
    monkeypatch.delenv("AA_QUALIFICATION_JOB_STARTED_MONOTONIC", raising=False)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "30")
    monkeypatch.setenv(ENV_SCENARIO_INACTIVITY_S, "0.4")
    monkeypatch.setenv(ENV_RECOVERY_ALLOWANCE_S, "0.8")

    async def lane(run_id: str) -> LaneResult:
        _ = run_id
        record_live_progress(
            "scenario-start",
            scenario_index=1,
            family="ordinary-text",
            completed_scenario_count=0,
            scenario_total=18,
        )
        record_live_progress("scenario-response", scenario_index=1, real_progress=True)
        # Stall begins now (no further real progress). The child declares
        # the bounded recovery: normally this marker is written by the
        # scenario-abort path within the changed-stage attribution block.
        record_live_progress(
            "scenario-abort-start",
            scenario_index=1,
            stall_recovery_in_progress=True,
            stall_recovery_started_monotonic=time.monotonic(),
            real_progress=False,
        )
        await asyncio.sleep(2.0)
        return LaneResult(lane="live-telegram-evidence", status="FAIL")

    started = time.monotonic()
    result = supervise_live_gate_c(lane, run_id="p0417", main_sha=SHA, progress_path=progress_path)
    elapsed = time.monotonic() - started
    assert result.status == "FAIL"
    assert result.metrics["watchdog_trigger"] == "no-real-progress"
    # The trigger must land only after the declared recovery allowance —
    # the parent never kills inside the child's exposed bounded window
    # (0.4s inactivity + 0.8s allowance, well below 2s).
    assert elapsed >= 0.8, elapsed
    return elapsed


def test_watchdog_holds_allowance_then_kills(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _watchdog_lane_hold_then_kill(monkeypatch, tmp_path)


def test_recovery_allowance_default_and_env() -> None:
    assert DEFAULT_RECOVERY_ALLOWANCE_S > 0.0
    assert recovery_allowance_s() > 0.0


def test_watchdog_timeout_keeps_child_attribution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The watchdog evidence includes the child's last stage + request state."""
    progress_path = _progress_path(tmp_path)
    monkeypatch.delenv("AA_QUALIFICATION_JOB_STARTED_AT", raising=False)
    monkeypatch.delenv("AA_QUALIFICATION_JOB_STARTED_MONOTONIC", raising=False)
    monkeypatch.setenv(ENV_TOTAL_DEADLINE_S, "20")
    monkeypatch.setenv(ENV_SCENARIO_INACTIVITY_S, "0.3")
    monkeypatch.setenv(ENV_RECOVERY_ALLOWANCE_S, "0.2")

    async def lane(run_id: str) -> LaneResult:
        _ = run_id
        record_live_progress(
            "scenario-start",
            scenario_index=1,
            family="ordinary-text",
            completed_scenario_count=0,
            scenario_total=18,
        )
        record_live_progress(
            "scenario-response",
            scenario_index=1,
            family="ordinary-text",
            real_progress=True,
        )
        # Simulate the child's own typed abort evidence (as the fixed lane
        # now writes it) but never finish recovery: the parent must publish
        # the same typed diagnosis instead of an arbitrary unknown.
        record_live_progress(
            "scenario-abort-start",
            scenario_index=1,
            active_stage="retrieval",
            stall_recovery_in_progress=True,
            stall_recovery_started_monotonic=time.monotonic(),
            real_progress=False,
        )
        record_live_progress(
            "scenario-timeout",
            scenario_index=1,
            family="ordinary-text",
            timeout_category="stuck-retrieval",
            last_active_stage="retrieval",
            request_state="unconfirmed-in-flight",
            request_started=True,
            request_progress_observable="progress-unobservable",
            completed_scenario_count=1,
            scenario_total=18,
            scenario_diagnostics=[
                {
                    "scenario_index": 1,
                    "family": "ordinary-text",
                    "delivery_timeout": True,
                    "timeout_category": "stuck-retrieval",
                    "last_active_stage": "retrieval",
                    "request_state": "unconfirmed-in-flight",
                    "request_started": True,
                    "request_progress_observable": "progress-unobservable",
                }
            ],
            latency_samples=[
                {
                    "causality_id": "sim-1",
                    "family": "ordinary-text",
                    "mode": "text",
                    "outcome": "timeout",
                    "latency_ms": 15000.0,
                    "censored": True,
                }
            ],
            real_progress=False,
        )
        await asyncio.sleep(1.5)
        return LaneResult(lane="live-telegram-evidence", status="FAIL")

    result = supervise_live_gate_c(lane, run_id="p0417b", main_sha=SHA, progress_path=progress_path)
    assert result.failed == ("live-scenario-watchdog-timeout-stuck-retrieval",)
    metrics = result.metrics
    assert metrics["timeout_category"] == "stuck-retrieval"
    assert metrics["watchdog_last_active_stage"] == "retrieval"
    assert metrics["watchdog_request_state"] == "unconfirmed-in-flight"
    assert metrics["watchdog_request_progress_observable"] == "progress-unobservable"
    persisted = read_live_progress(progress_path) or {}
    diagnostics = [dict(item) for item in persisted.get("scenario_diagnostics", [])]
    # The child's own diagnostic entry survives with the recorded category,
    # stage and request evidence; the watchdog did not erase it.
    assert any(
        item.get("delivery_timeout") and item.get("timeout_category") == "stuck-retrieval"
        for item in diagnostics
    ), diagnostics
    samples = [dict(sample) for sample in persisted.get("latency_samples", [])]
    assert samples and samples[-1]["censored"] is True
    assert samples[-1]["outcome"] == "timeout"
