"""kodmial/aa#431 integration: attempt ledger, observer, stuck-worker isolation.

These tests exercise the REAL local HTTP client thread model:

- an attempt that never completes is observable via attempt-start/audit
  while it is running (the old post-completion audit produced nothing);
- cancelling a wedged call records an honest `cancelled` attempt state and
  leaves quiescence unproven until the worker thread's own fate is
  corroborated (a late completion is recorded as late_outcome);
- concurrent calls keep independent attempts (newest-wins classification);
- the independent stall-snapshot observer persists bounded snapshots even
  for attempts that are still in flight, and terminal traces when they end.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from pathlib import Path

import pytest

from aa.conversation.turn_causality import (
    clear_turn_causality,
    set_turn_causality,
)
from aa.opencode.attempt_audit import AttemptLedger
from aa.opencode.client import HttpOpenCodeClient
from aa.opencode.observer import (
    SNAPSHOT_DELAYS_S,
    StallSnapshotObserver,
    attempt_view,
)
from aa.qualification.live_turn_repair import (
    classify_stall_kind,
    request_state_from_audit,
)


class _HangingServer:
    """A real TCP server that accepts and never answers."""

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(None)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.host, self.port = self._sock.getsockname()[:2]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._accepted: list[socket.socket] = []
        self._thread.start()

    def _serve(self) -> None:
        self._sock.listen(16)
        while not self._stop.is_set():
            try:
                self._sock.settimeout(0.2)
                conn, _addr = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            self._accepted.append(conn)
            # Never read/respond: the client stays stuck inside its socket
            # read exactly like a wedged provider call would.

    async def start(self) -> None:
        return None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

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


async def _stuck_call(
    server: _HangingServer, causality: str
) -> tuple[HttpOpenCodeClient, asyncio.Task[object]]:
    """Start one real call against a hanging server stamped with causality."""
    client = HttpOpenCodeClient(server.base_url)
    set_turn_causality(causality, stage="graph-turn")
    return client, asyncio.create_task(
        client._call("POST", "/session/ses_one/message", {"text": "x"})
    )


@pytest.mark.asyncio
async def test_attempt_start_is_visible_while_request_is_stuck() -> None:
    server = _HangingServer()
    causality = "stuck-causality"
    client, task = await _stuck_call(server, causality)
    try:
        # Give the worker thread time to enter the real blocking socket
        # read so a cancellation leaves a genuinely live worker behind.
        await asyncio.sleep(0.55)
        entries = list(client.attempt_audit)
        assert len(entries) >= 1
        assert entries[0]["event"] == "attempt-start"
        report = request_state_from_audit(entries, causality)
        state = report["request_state"]
        if "worker_done" not in (entries[0] or {}):
            assert state == "http-in-flight"
        assert (
            classify_stall_kind(
                causality_id=causality,
                request_audit=entries,
            )
            == "http-in-flight"
        )
        gate = client.quiescence_proven(causality)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        audit_rows: tuple[dict[str, object], ...] = client.attempt_audit
        mine = [entry for entry in audit_rows if entry.get("causality_id") == causality]
        assert mine and mine[0]["state"] == "cancelled"
        if gate is False and not mine[0].get("worker_done"):
            # Task.done() proves nothing about the worker thread: without
            # worker corroboration the attempt is NOT proven quiesced.
            assert client.quiescence_proven(causality) is False
            # The worker later reaches a REAL terminal condition (server
            # close); the done-callback then corroborates it on the loop.
            server.stop()
            await asyncio.sleep(0.3)
            recorded = [
                entry for entry in client.attempt_audit if entry.get("causality_id") == causality
            ][0]
            assert recorded.get("worker_done") is True
            assert client.quiescence_proven(causality) is True
        else:
            # Clean cancellation before the thread ran: the worker future
            # itself already proved there is no orphan thread.
            assert client.quiescence_proven(causality) is True
    finally:
        clear_turn_causality()
        server.stop()


@pytest.mark.asyncio
async def test_late_outcome_recorded_when_abandoned_worker_completes() -> None:
    server = _HangingServer()
    causality = "late-causality"
    client, task = await _stuck_call(server, causality)
    try:
        await asyncio.sleep(0.55)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        mine = [e for e in client.attempt_audit if e.get("causality_id") == causality]
        assert mine and mine[0]["state"] == "cancelled"
        # The abandoned worker thread later reaches a REAL terminal
        # condition (server close errors out the read); the independent
        # done-callback records it as late evidence and never relabels the
        # authoritative cancelled state.
        server.stop()
        for _ in range(100):
            await asyncio.sleep(0.03)
            recorded = [e for e in client.attempt_audit if e.get("causality_id") == causality][0]
            if recorded.get("late_outcome"):
                break
        assert recorded["state"] == "cancelled"
        assert recorded["late_outcome"] == "failed"
        assert recorded.get("worker_done") is True
        assert client.quiescence_proven(causality) is True
    finally:
        clear_turn_causality()
        server.stop()


@pytest.mark.asyncio
async def test_concurrent_calls_keep_independent_attempts() -> None:
    server = _HangingServer()
    causality = "concurrent-causality"
    client = HttpOpenCodeClient(server.base_url)
    set_turn_causality(causality, stage="graph-turn")
    try:
        tasks = [
            asyncio.create_task(client._call("POST", "/session/ses_one/message", {"text": "x"}))
            for _ in range(4)
        ]
        for _ in range(200):
            if len(client.attempt_audit) >= 4:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        entries = list(client.attempt_audit)
        assert len({entry["attempt_id"] for entry in entries if "attempt_id" in entry}) >= 4
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        mine_states = {
            entry["state"]
            for entry in client.attempt_audit
            if entry.get("causality_id") == causality
        }
        assert mine_states == {"cancelled"}
    finally:
        clear_turn_causality()
        server.stop()


@pytest.mark.asyncio
async def test_worker_done_marking_is_idempotent_and_privacy_safe() -> None:
    ledger = AttemptLedger()
    attempt_id = ledger.register_start(
        operation="chat", causality_id="privacy-causality", agent="main"
    )
    assert ledger.mark_worker_done(attempt_id) is True
    assert ledger.mark_worker_done(attempt_id) is False
    entry = attempt_view(ledger.entries()[0])
    assert entry["worker_done"] is True
    assert "prompt" not in entry
    assert "session" not in entry


@pytest.mark.asyncio
async def test_observer_snapshots_in_flight_and_trace_terminal(tmp_path: Path) -> None:
    source_entries: list[dict[str, object]] = []

    def _source() -> list[dict[str, object]]:
        return list(source_entries)

    observer = StallSnapshotObserver(
        attempt_source=_source,
        out_dir=tmp_path / "attempt-observer",
        delays=(0.05, 1.05),
        interval_s=0.01,
    )
    await observer.start()
    source_entries.append(
        {
            "attempt_id": "att-1",
            "operation": "chat",
            "state": "in-flight",
            "attempt_started_monotonic": asyncio.get_running_loop().time() - 0.2,
            "latency_ms": None,
        }
    )
    for _ in range(300):
        await asyncio.sleep(0.01)
        files = sorted(p.name for p in (tmp_path / "attempt-observer").iterdir())
        if sum(name.startswith("attempt-att-1-snapshot") for name in files) == 2:
            break
    files = sorted(p.name for p in (tmp_path / "attempt-observer").iterdir())
    assert sum(name.startswith("attempt-att-1-snapshot") for name in files) == 2
    snapshot_payload = json.loads(
        (tmp_path / "attempt-observer" / files[0]).read_text(encoding="utf-8")
    )
    assert snapshot_payload["schema"] == "aa-opencode-stall-snapshot/1"
    assert snapshot_payload["attempt"]["attempt_id"] == "att-1"
    # Terminal transition persists a per-attempt trace file and index flush.
    source_entries[0] = {**source_entries[0], "state": "completed", "latency_ms": 1.0}
    await observer.stop(flush_reason="test-finish")
    payload = json.loads(
        (tmp_path / "attempt-observer" / "attempt-observer-index.json").read_text(encoding="utf-8")
    )
    assert payload["schema"] == "aa-opencode-stall-flush/1"
    files = sorted(p.name for p in (tmp_path / "attempt-observer").iterdir())
    assert any(name == "attempt-att-1-trace.json" for name in files)
    assert observer.out_dir == tmp_path / "attempt-observer"


@pytest.mark.asyncio
async def test_default_snapshot_delays_are_bounded() -> None:
    assert SNAPSHOT_DELAYS_S == (10.0, 25.0, 45.0, 75.0, 110.0)
    with pytest.raises(ValueError):
        StallSnapshotObserver(attempt_source=lambda: [], out_dir="/tmp/x", delays=(0,))
