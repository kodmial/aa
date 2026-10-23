"""kodmial/aa#448: live Gate C real-time diagnostics wiring regressions.

The confirmed Gate C run (39923515643, SHA 4db9874184139066a9db2db7ecdea9e2375ac07f)
showed the bot executing scenarios while the owner's stdout carried only
20-second timer lines. Three concrete wiring defects existed on main:

1. ``HttpOpenCodeClient.attempt_audit`` is a @property returning a tuple;
   the lane gated on ``callable(getattr(client, "attempt_audit"))`` so the
   ``StallSnapshotObserver`` was never created (0 snapshots, silent PASS).
2. ``LocalOpenCodeRuntime.runtime_state`` is a @property returning a dict;
   the lane passed that dict as ``process_state_source`` so a later call
   attempted to invoke a dict as a function, masking the actual process
   evidence as ``process_state_error=unavailable``.
3. The fast conversational ``generate_dialogue_reply`` call dropped the
   ``telemetry`` carrier, so the typed ``answer_failure_cause`` recorded
   for the fast ``conversational-generation-failed`` outcome was lost.

These tests run on the REAL interface types (a real loopback HTTP client
against a hanging socket server and the real ``LocalOpenCodeRuntime``
process supervision) with accelerated snapshot delays, and assert that:

- the observer is actually created through lazy fresh-read suppliers and
  persists the accelerated 10/25/45/75/110-style snapshots plus terminal
  traces and the flush index;
- process snapshots contain the runtime's actual alive/exit_code state;
- a restarted runtime instance is picked up by the re-reading supplier;
- attempt lifecycle events (start / finished / late outcome) surface as
  bounded privacy-safe payloads;
- heartbeat attempt snapshot fields, stage-change lines and the
  completed-line cause suffix carry the exact typed causes — never
  fabricated progress, chat ids, prompt or answer text.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from aa.conversation.answer_causality import (
    GENERATION_PROVIDER_TIMEOUT,
    GUARD_LANGUAGE,
    VERIFIER_UNAVAILABLE,
)
from aa.conversation.failures import TurnFailed
from aa.conversation.turn_causality import clear_turn_causality, set_turn_causality
from aa.conversation.turn_pipeline import run_v2_answer_turn
from aa.opencode.client import FakeOpenCodeClient, HttpOpenCodeClient
from aa.opencode.observer import StallSnapshotObserver
from aa.opencode.runtime import LocalOpenCodeRuntime, OpenCodeConfig
from aa.qualification.product_contract_live import (
    _completion_cause_suffix,
    _format_bounded_attempt_event,
    _format_bounded_live_fields,
    _live_attempt_snapshot_fields,
    _opencode_attempt_entries,
    _opencode_process_state,
    _print_live_stage_change,
    _snapshot_product_failure_reason,
)

# A distinctive prompt is sent through the real HTTP boundary once per
# hanging attempt; every diagnostic surface must stay free of it.
_PROMPT_SENTINEL = "PROMPTSENTINEL448"


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
            # read exactly like a wedged local OpenCode HTTP call.

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


def _pack_entry(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Поддержка рядом помогает пережить тягу сегодня.",
) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
    }


def _bound_runtime(client: HttpOpenCodeClient, base_url: str) -> LocalOpenCodeRuntime:
    """A real ``LocalOpenCodeRuntime`` bound to the shared client.

    ``manage_process=False`` keeps the runtime attached-in-memory for the
    attempt-ledger contract; the live process supervision contract is
    covered by its own dedicated test below.
    """
    config = OpenCodeConfig(
        base_url=base_url or "http://127.0.0.1:49999",
        command="opencode",
        workdir=".",
    )
    return LocalOpenCodeRuntime(config, client=client, manage_process=False)


def _start_in_flight_attempt(
    server: _HangingServer, causality: str, client: HttpOpenCodeClient
) -> asyncio.Task[object]:
    """Fire one real hung attempt stamped with the scenario causality."""
    set_turn_causality(causality, stage="graph-turn")
    return asyncio.create_task(
        client._call("POST", "/session/ses_one/message", {"text": _PROMPT_SENTINEL})
    )


async def _wait_for_entry(
    entries_source: Any, *, state: str | None = None, timeout_s: float = 5.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        for entry in reversed(entries_source()):
            last = dict(entry)
            if state is None or str(entry.get("state", "")) == state:
                return dict(entry)
        await asyncio.sleep(0.01)
    return last


def _cancel(task: asyncio.Task[object]) -> None:
    task.cancel()


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


# --- Observer wiring on the real interfaces ------------------------------


@pytest.mark.asyncio
async def test_old_callable_tuple_wiring_disabled_observation_and_lazy_supplier_restores_it(
    tmp_path: Path,
) -> None:
    """The exact pre-#448 binding defect versus the fixed lazy supplier.

    Old wiring: ``_audit_accessor = getattr(client, "attempt_audit", None)``
    followed by ``if callable(_audit_accessor)``. The real interface is a
    @property returning a tuple, so the observer was never created and no
    snapshot could ever exist. New wiring: the observer receives a lazy
    fresh-read supplier that re-reads the same property on every scan and
    therefore also follows a restarted runtime instance.
    """
    server = _HangingServer()
    causality = "p0448-supplier-causality"
    client = HttpOpenCodeClient(server.base_url)
    runtime = _bound_runtime(client, server.base_url)
    task = _start_in_flight_attempt(server, causality, client)
    try:
        # The old binding's false-negative: the accessor is not callable.
        accessor = getattr(client, "attempt_audit", None)
        assert isinstance(accessor, tuple)
        assert not callable(accessor)

        def entries_source() -> tuple[dict[str, Any], ...]:
            return _opencode_attempt_entries(runtime)

        started_entry = await _wait_for_entry(entries_source, state="in-flight")
        assert started_entry.get("event") == "attempt-start"
        assert started_entry.get("state") == "in-flight"
        assert started_entry.get("operation") == "message-text"
        assert started_entry.get("causality_id") == causality
        attempt_id = str(started_entry.get("attempt_id", ""))
        assert attempt_id

        out_dir = tmp_path / "attempt-observer"
        events: list[dict[str, Any]] = []

        def _sink(payload: Mapping[str, Any]) -> None:
            events.append(dict(payload))

        observer = StallSnapshotObserver(
            attempt_source=entries_source,
            out_dir=out_dir,
            delays=(0.05, 0.1),
            interval_s=0.01,
            event_sink=_sink,
        )
        await observer.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and observer.snapshot_file_count < 2:
            await asyncio.sleep(0.01)
        # Both accelerated snapshots exist for the in-flight attempt while
        # the worker thread stays wedged inside the real socket read.
        assert observer.snapshot_file_count == 2
        assert observer.snapshot_attempt_count == 1
        assert observer.has_snapshot_for(attempt_id)
        message_snapshots = sorted(out_dir.glob(f"attempt-{attempt_id}-snapshot-*.json"))
        assert len(message_snapshots) == 2
        payload = json.loads(message_snapshots[0].read_text(encoding="utf-8"))
        assert payload["schema"] == "aa-opencode-stall-snapshot/1"
        assert payload["attempt"]["state"] == "in-flight"
        assert payload["attempt"]["operation"] == "message-text"
        assert payload["attempt"]["causality_id"] == causality

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await _wait_for_entry(entries_source, state="cancelled")
        # Seconds matter: close the server so both hung workers (the
        # abandoned turn AND the recovery abort call it spawned) reach a
        # genuine terminal socket condition -> deterministic late outcome.
        server.stop()

        def _first_causality_entry() -> dict[str, Any]:
            for entry in entries_source():
                if str(entry.get("causality_id", "")) == causality:
                    return dict(entry)
            return {}

        settled = await asyncio.to_thread(_first_causality_entry)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and settled.get("late_outcome") != "failed":
            await asyncio.sleep(0.02)
            settled = await asyncio.to_thread(_first_causality_entry)
        assert settled.get("state") == "cancelled"
        assert settled.get("late_outcome") == "failed", settled
        assert settled.get("worker_done") is True

        await observer.stop(flush_reason="p0448-test-finish")
        trace_files = list(out_dir.glob(f"attempt-{attempt_id}-trace.json"))
        assert len(trace_files) == 1
        trace = json.loads(trace_files[0].read_text(encoding="utf-8"))
        assert trace["attempt"]["state"] == "cancelled"
        assert trace["attempt"]["late_outcome"] == "failed"
        index = json.loads((out_dir / "attempt-observer-index.json").read_text(encoding="utf-8"))
        assert index["schema"] == "aa-opencode-stall-flush/1"
        assert index["snapshot_attempts"] >= 2
        assert observer.traced_attempt_count >= 1

        # Deduplicated lifecycle events for THIS attempt, exactly once each.
        mine = [e for e in events if str(e.get("attempt_id", "")) == attempt_id]
        started_events = [e for e in mine if e.get("event") == "attempt-start"]
        finished_events = [e for e in mine if e.get("event") == "attempt-finished"]
        late_events = [e for e in mine if e.get("event") == "attempt-late-outcome"]
        persisted_events = [e for e in mine if e.get("event") == "stall-snapshot"]
        assert len(started_events) == 1
        assert len(finished_events) == 1
        assert len(late_events) == 1
        assert len(persisted_events) == 2
        assert started_events[0]["state"] == "in-flight"
        assert started_events[0]["causality_id"] == causality
        assert finished_events[0]["state"] == "cancelled"
        assert late_events[0]["late_outcome"] == "failed"

        # Privacy: snapshots, traces, emitted events, and the bounded
        # heartbeat summary never carry the prompt that crossed the same
        # actual HTTP boundary.
        exported = _dump(events) + "\n".join(
            p.read_text(encoding="utf-8") for p in out_dir.iterdir()
        )
        assert _PROMPT_SENTINEL not in exported
        summary_fields = _live_attempt_snapshot_fields(
            SimpleNamespace(opencode_runtime=runtime),
            causality_id=causality,
            last_active_stage="answer",
        )
        rendered = _format_bounded_live_fields(summary_fields)
        assert "attempt_state=cancelled" in rendered
        assert "attempt_operation=message-text" in rendered
        assert "worker_done=true" in rendered
        assert "late_outcome=failed" in rendered
        assert _PROMPT_SENTINEL not in rendered
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutdown only
            pass
        clear_turn_causality()
        server.stop()


@pytest.mark.asyncio
async def test_process_state_supplier_reports_real_alive_and_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DoD: process snapshots hold the runtime's actual alive/exit_code."""
    import aa.opencode.runtime as _oc_runtime

    class _FirstDownThenHealthyProbe:
        """First health probe fails (force spawn), later probes succeed."""

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, base_url: str, timeout: float) -> tuple[bool, str]:
            self.calls += 1
            _ = (base_url, timeout)
            return (True, "p0448") if self.calls > 1 else (False, "")

    monkeypatch.setattr(_oc_runtime, "_probe_health", _FirstDownThenHealthyProbe())

    serve_script = tmp_path / "opencode"
    serve_script.write_text("#!/bin/sh\nexec sleep 3\n")
    serve_script.chmod(0o755)
    runtime = LocalOpenCodeRuntime(
        OpenCodeConfig(
            base_url="http://127.0.0.1:49999",
            command=str(serve_script),
            workdir=str(tmp_path),
        ),
        manage_process=True,
        ready_timeout=2.0,
        shutdown_timeout=2.0,
    )
    try:
        await runtime.start()
        state = _opencode_process_state(runtime)
        assert state == {"owned_by_runtime": True, "alive": True, "exit_code": None}

        # The observer snapshot carries the ACTUAL runtime descriptors —
        # never the old ``process_state_error=unavailable`` masking that a
        # dict-as-callable caused.
        observer = StallSnapshotObserver(
            attempt_source=lambda: [],
            out_dir=tmp_path / "attempt-observer",
            process_state_source=lambda: _opencode_process_state(runtime),
        )
        path = await observer.snapshot_once(
            {
                "attempt_id": "att-448-process",
                "state": "in-flight",
                "attempt_started_monotonic": time.monotonic(),
            },
            delay=1.0,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["process"].get("alive") is True
        assert payload["process"].get("exit_code") is None
        assert "process_state_error" not in payload["process"]

        # Let the supervised process exit on its own; while still owned the
        # same lazy supplier reports the runtime's actual terminal verdict
        # (alive False with an integer exit code), never a generic error.
        await asyncio.sleep(3.5)
        state = _opencode_process_state(runtime)
        assert state["owned_by_runtime"] is True
        assert state["alive"] is False
        assert isinstance(state["exit_code"], int)
        path = await observer.snapshot_once(
            {
                "attempt_id": "att-448-process-exited",
                "state": "in-flight",
                "attempt_started_monotonic": time.monotonic(),
            },
            delay=1.0,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["process"].get("alive") is False
        assert isinstance(payload["process"].get("exit_code"), int)
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_attempt_supplier_reflects_restarted_runtime() -> None:
    """A lazy supplier reads the CURRENT runtime, not a captured instance."""
    server = _HangingServer()
    causality = "p0448-restart-causality"
    try:
        first_client = HttpOpenCodeClient(server.base_url)
        holder = {"runtime": _bound_runtime(first_client, server.base_url)}
        task = _start_in_flight_attempt(server, causality, first_client)

        def entries_source() -> tuple[dict[str, Any], ...]:
            return _opencode_attempt_entries(holder["runtime"])

        started_entry = await _wait_for_entry(entries_source, state="in-flight")
        assert started_entry.get("causality_id") == causality

        # Runtime restart: the fresh instance owns a fresh ledger because
        # the supplier re-reads through the holder on every call instead
        # of holding the stale reference. This is the no-orphan state the
        # stall recovery relies on: the restarted runtime publishes no
        # stale in-flight attempt for the abandoned turn's causality.
        fresh_client = HttpOpenCodeClient(server.base_url)
        holder["runtime"] = _bound_runtime(fresh_client, server.base_url)
        assert entries_source() == ()
        assert not any(
            entry.get("state") == "in-flight" and entry.get("causality_id") == causality
            for entry in _opencode_attempt_entries(holder["runtime"])
        )
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutdown only
            pass
        clear_turn_causality()
        server.stop()


# --- Stdout diagnostic helpers -------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_snapshot_fields_are_meaningful_for_a_stalled_attempt() -> None:
    """A 20-second waiting line must prove more than the timer."""
    server = _HangingServer()
    causality = "p0448-heartbeat-causality"
    client = HttpOpenCodeClient(server.base_url)
    try:
        runtime = _bound_runtime(client, server.base_url)
        task = _start_in_flight_attempt(server, causality, client)
        try:
            await _wait_for_entry(lambda: _opencode_attempt_entries(runtime), state="in-flight")
            fields = _live_attempt_snapshot_fields(
                SimpleNamespace(opencode_runtime=runtime),
                causality_id=causality,
                last_active_stage="answer",
                last_real_progress_age_s=19.5,
            )
            assert fields["attempt_state"] == "in-flight"
            assert fields["attempt_operation"] == "message-text"
            assert float(fields["attempt_elapsed_s"]) >= 0.0
            assert fields["last_active_stage"] == "answer"
            assert fields["last_real_progress_age_s"] == "19.5"
            rendered = _format_bounded_live_fields(fields)
            assert "attempt_state=in-flight" in rendered
            assert "attempt_operation=message-text" in rendered
            assert "last_active_stage=answer" in rendered
            assert "last_real_progress_age_s=19.5" in rendered
            # Honest uncertainty: without observed worker/process evidence
            # nothing server/provider-side is fabricated as progress.
            assert "worker_done=unobserved" in rendered
            assert "process_state=unobserved" in rendered
            assert _PROMPT_SENTINEL not in rendered
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        clear_turn_causality()
        server.stop()


@pytest.mark.asyncio
async def test_heartbeat_fields_report_unobserved_when_nothing_is_known() -> None:
    server = _HangingServer()
    try:
        stub_runtime = SimpleNamespace(client=FakeOpenCodeClient())
        fields = _live_attempt_snapshot_fields(
            SimpleNamespace(opencode_runtime=stub_runtime),
            causality_id="missing-causality",
        )
        assert fields["attempt_state"] == "unobserved"
        assert fields["attempt_operation"] == "unobserved"
        assert "attempt_elapsed_s" not in fields
        assert "worker_done" not in fields
        assert fields["process_state"] == "unobserved"
        assert fields["last_active_stage"] == "unobserved"
        rendered = _format_bounded_live_fields(fields)
        assert "attempt_state=unobserved" in rendered
        assert "attempt_operation=unobserved" in rendered
        assert "worker_done" not in rendered
        assert "process_state=unobserved" in rendered
        assert "last_active_stage=unobserved" in rendered
        assert _PROMPT_SENTINEL not in rendered
    finally:
        server.stop()


def test_stage_change_stdout_is_meaningful_and_bounded(capsys: pytest.CaptureFixture[str]) -> None:
    stage_state = {
        "scenario_index": 1,
        "scenario_total": 18,
        "causality_id": "p0448-stage-causality",
        "last_monotonic": time.monotonic() - 2.0,
    }
    _print_live_stage_change("answer", stage_state, SimpleNamespace(opencode_runtime=None))
    out = capsys.readouterr().out
    assert "phase=stage stage=answer" in out
    assert " 1/18" in out
    assert "last_active_stage=answer" in out
    assert "attempt_state=unobserved" in out
    assert "last_real_progress_age_s=2." in out
    assert _PROMPT_SENTINEL not in out


def test_attempt_event_line_is_bounded_and_resolves_scenario_identity() -> None:
    payload = {
        "event": "attempt-start",
        "attempt_id": "a" * 64,
        "operation": "message-text",
        "state": "in-flight",
        "causality_id": "p0448-event-causality",
        "stage": "graph-turn",
        "agent": "planner",
        "audit_agent": "planner",
        "model": "zen/best",
        "worker_done": None,
        # Hostile extra keys can never travel: the payload projection is
        # allow-listed upstream, and the renderer prints only known keys.
        _PROMPT_SENTINEL: _PROMPT_SENTINEL,
        "session_id": "ses_leak",
        "answer": "leak",
    }
    line = _format_bounded_attempt_event(payload, {"p0448-event-causality": 7})
    assert line.startswith("event=attempt-start")
    assert "operation=message-text" in line
    assert "causality_id=p0448-event-causality" in line
    assert "scenario_index=7" in line
    assert _PROMPT_SENTINEL not in line
    assert "session_id" not in line

    unknown = _format_bounded_attempt_event(payload, {"other-causality": 8})
    assert "scenario_index=unobserved" in unknown

    no_map = _format_bounded_attempt_event(payload, None)
    assert "scenario_index" not in no_map
    assert "event=attempt-start" in no_map


# --- Typed answer failure causes through the real pipeline ---------------


@pytest.mark.asyncio
async def test_fast_conversational_failure_carries_provider_exception_cause() -> None:
    """Fast-path ``conversational-generation-failed`` keeps the typed cause."""

    class _ExplodingModel:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            from aa.opencode.errors import OpenCodeTimeoutError

            raise OpenCodeTimeoutError("opencode request timed out")

    with pytest.raises(TurnFailed) as exc:
        await run_v2_answer_turn(
            user_message="Чем ты можешь быть полезен здесь?",
            summary="",
            recent=[],
            evidence_pack=[],
            answer_model=_ExplodingModel(),
            verifier_model=_ExplodingModel(),
            planner_model=None,
            retrieval_index=None,
            planner_mode="conversational",
            planner_reason="legitimate-glue",
            initial_query_count=0,
            whole_turn_judge_model=object(),
        )
    assert exc.value.category == "conversational-generation-failed"
    telemetry = dict(exc.value.telemetry)
    assert telemetry["answer_outcome"] == "conversational-generation-failed"
    cause = telemetry.get("answer_failure_cause")
    assert isinstance(cause, dict)
    assert cause["reason"] == GENERATION_PROVIDER_TIMEOUT
    assert cause["stage"] == "generation"
    assert cause["detail"] == "OpenCodeTimeoutError"
    # The delivered service-error reply must print the exact typed cause,
    # never the generic outcome as an invented root cause.
    printed = f"outcome={telemetry['answer_outcome'][:48]}" + _completion_cause_suffix(
        telemetry, "service_error"
    )
    assert printed.endswith("cause=provider-timeout")
    assert printed.split("cause=")[1] == "provider-timeout"


@pytest.mark.asyncio
async def test_fast_conversational_failure_carries_local_guard_cause() -> None:
    """A local safety guard rejection keeps its typed local cause."""

    class _EnglishOnlyModel:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            return AIMessage(content="Hello my friend, glad you wrote. Stay strong.")

    with pytest.raises(TurnFailed) as exc:
        await run_v2_answer_turn(
            user_message="Привет! Как обходиться с тягой вечером?",
            summary="",
            recent=[],
            evidence_pack=[],
            answer_model=_EnglishOnlyModel(),
            verifier_model=_EnglishOnlyModel(),
            planner_model=None,
            retrieval_index=None,
            planner_mode="conversational",
            planner_reason="legitimate-glue",
            initial_query_count=0,
            whole_turn_judge_model=object(),
        )
    assert exc.value.category == "conversational-generation-failed"
    cause = exc.value.telemetry.get("answer_failure_cause")
    assert isinstance(cause, dict)
    assert cause["reason"] == GUARD_LANGUAGE
    assert cause["source"] == "local"
    assert _completion_cause_suffix(exc.value.telemetry, "rejected") == " cause=guard-language"


@pytest.mark.asyncio
async def test_clarification_unavailable_keeps_the_first_recorded_cause() -> None:
    """Verifier outage: the terminal outcome keeps the PRIMARY typed cause."""

    class _DownVerifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            raise TimeoutError("verifier down")

    class _AnswerModel:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            return AIMessage(content="Поддержка рядом помогает спокойно разбирать тягу.")

    with pytest.raises(TurnFailed) as exc:
        await run_v2_answer_turn(
            user_message="Как обходиться с тягой вечером?",
            summary="",
            recent=[],
            evidence_pack=[_pack_entry()],
            answer_model=_AnswerModel(),
            verifier_model=_DownVerifier(),
            planner_model=None,
            retrieval_index=None,
            initial_query_count=12,
        )
    assert exc.value.category == "clarification-unavailable"
    telemetry = dict(exc.value.telemetry)
    assert telemetry["answer_outcome"] == "clarification-unavailable"
    cause = telemetry.get("answer_failure_cause")
    assert isinstance(cause, dict)
    assert cause["reason"] == VERIFIER_UNAVAILABLE
    # The stdout evidence prints the first recorded cause — NOT the
    # terminal outcome re-served as an invented root cause.
    rendered = _completion_cause_suffix(telemetry, "service_error")
    assert rendered == " cause=verifier-unavailable"
    assert "clarification-unavailable" not in rendered


def test_completion_cause_suffix_is_honest_unknown_without_records() -> None:
    assert _completion_cause_suffix({}, "service_error") == " cause=unknown"
    assert _completion_cause_suffix({}, "rejected") == " cause=unknown"
    assert _completion_cause_suffix({}, "answer_success") == ""
    assert _snapshot_product_failure_reason({}) == ""
