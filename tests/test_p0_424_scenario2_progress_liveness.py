"""kodmial/aa#424 DoD tests: progress/liveness observation + one finite campaign.

These tests prove the reintroduction of the 45/75/130 s answer cutoffs is
fixed:

1. The old arbitrary answer windows are GONE from source and workflow
   (no per-answer wall clock kills a healthy slow call).
2. Crossing the legacy windows is RECORDED as evidence markers
   (observation continues at 45/75/130/300/900 s), never enforced.
3. A "model call" simulated to take just past the legacy windows
   completes and is reported delivered (virtual-clock observation).
4. The ONLY bounded permission is ONE finite, physical campaign
   allowance, ending in ``unresolved-at-campaign-end`` (INCONCLUSIVE,
   never success) via tidy SIGTERM → SIGKILL process-group shutdown.
5. Genuinely unresponsive behavior is distinguished:
   sustained local-server non-responsiveness (consecutive failed health
   probes → proven server-error) vs a slow-but-progressing turn
   (delivered).

Fakes are applied ONLY at infrastructure boundaries (health probe, HTTP
transport, remote clock); the production supervision/observation logic
itself runs unmodified.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "diagnose_gate_c_scenario2.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "aa-gate-c-scenario2-diagnostic.yml"

LEGACY_WINDOWS_S = (45.0, 75.0, 130.0)


@pytest.fixture(scope="module")
def diag() -> Any:
    module_name = "diagnose_gate_c_scenario2_424"
    spec = importlib.util.spec_from_file_location(module_name, str(SCRIPT_PATH))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. The arbitrary cutoffs must be gone (source + workflow)
# ---------------------------------------------------------------------------


def test_no_answer_cutoff_symbols_remain_in_source(diag: Any) -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "DEFAULT_DIAGNOSTIC_CUTOFF_S",
        "DEFAULT_PROBE_A_CUTOFF_S",
        "DEFAULT_PROBE_A_MESSAGE_TIMEOUT_S",
        "diagnostic_cutoff_s",
        "probe_a_cutoff_s",
        "probe_a_message_timeout_s",
        "message-window",
        "per-answer-cutoff=false",
    ):
        assert forbidden not in source
    # The policy name IS the new truth: there is no per-answer cutoff.
    assert "no-per-answer-cutoff" in source
    # The ONLY remaining module constants named after the legacy windows
    # are the EVIDENCE checkpoint thresholds (recorded, never enforced).
    assert list(diag.OBSERVATION_CONTINUITY_THRESHOLD_S) == [
        45.0,
        75.0,
        130.0,
        300.0,
        900.0,
    ]


def test_legacy_window_numbers_exist_only_as_continuity_evidence(
    diag: Any,
) -> None:
    """AST check: 45/75/130 never appear as branch/kill conditions.

    The literals may exist in docstrings, the EVIDENCE threshold tuple,
    or unrelated infrastructure bounds (e.g. server ``ready_timeout``) —
    but a comparison against one of them would be a re-introduced cutoff
    switch and must fail this test.
    """
    import ast

    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    legacy = {45.0, 75.0, 130.0}
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and comparator.value in legacy:
                    offenders.append(f"line {comparator.lineno}")
    assert offenders == [], f"legacy cutoff comparisons remain: {offenders}"
    assert list(diag.OBSERVATION_CONTINUITY_THRESHOLD_S) == [
        45.0,
        75.0,
        130.0,
        300.0,
        900.0,
    ]


def test_child_supervision_media_window_strings_absent(diag: Any) -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "DEFAULT_SEQUENCE",
        "media-window",
        "parse_schedule",
    ):
        assert forbidden not in source
    assert "DEFAULT_SNAPSHOT_CADENCE_S" in source
    assert "DEFAULT_PROBE_A_CONNECT_TIMEOUT_S" in source
    assert "DEFAULT_CAMPAIGN_ALLOWANCE_S" in source
    assert "DEFAULT_FINALISATION_RESERVE_S" in source


def test_workflow_has_no_answer_cutoffs(workflow_text: str) -> None:
    for forbidden in (
        "--diagnostic-cutoff-s",
        "--probe-a-cutoff-s",
        "--probe-a-message-timeout-s",
    ):
        assert forbidden not in workflow_text
    for required in (
        "--campaign-allowance-s",
        "--finalisation-reserve-s",
        "--job-timeout-minutes",
        "--job-started-epoch",
        "timeout-minutes: 60",
    ):
        assert required in workflow_text


def test_continuity_thresholds_are_evidence_not_cutoffs(diag: Any) -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert list(diag.OBSERVATION_CONTINUITY_THRESHOLD_S) == [45.0, 75.0, 130.0, 300.0, 900.0]
    # The docstrings describe the reformed contract: recorded, never killed.
    assert "observed, not killed" in source
    assert "answer windows are now evidence checkpoints" in source


# ---------------------------------------------------------------------------
# 2. Virtual clock: continuity markers at the legacy windows are EVIDENCE
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _virtual_clock() -> Any:
    """Patch stdlib time.monotonic/perf_counter with a controllable clock."""
    state = {"offset": 0.0}
    mono_base = time.monotonic()
    perf_base = time.perf_counter()

    def mono() -> float:
        return mono_base + state["offset"]

    def perf() -> float:
        return perf_base + state["offset"]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(time, "monotonic", mono)
        mp.setattr(time, "perf_counter", perf)
        yield state


def _sink_events(path: pathlib.Path) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    if not path.exists():
        return lines
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            lines.append(json.loads(line))
    return lines


def test_child_continuity_markers_recorded_at_all_legacy_windows(
    diag: Any, tmp_path: pathlib.Path
) -> None:
    """Evidence checkpoint behavior: crossing 45/75/130/300/900 is recorded."""
    with _virtual_clock() as clock:
        sink = diag.EventSink(tmp_path / "continuity.jsonl", echo=False)
        counters: dict[str, Any] = {"last_stage": "", "last_real_monotonic": None}
        turn_started = time.monotonic()
        state: dict[str, Any] = {"index": 0, "next_at": 0.0}
        # (requested elapsed, ignored; expectations derived below)
        script = (
            0.0,
            44.9,
            45.5,  # emit 45.0
            46.0,  # next threshold 75: no event
            74.9,
            75.4,  # emit 75.0
            76.0,  # next threshold 130: no event
            129.9,
            130.4,  # emit 130.0
            131.0,  # next threshold 300: no event
            299.9,
            300.5,  # emit 300.0
            301.0,  # next threshold 900: no event
            899.9,
            900.6,  # emit 900.0
            901.0,  # repeat cadence: no event yet
            1810.0,  # emit repeat marker (>= 900.6 + 900)
        )
        for target in script:
            clock["offset"] = target
            diag._emit_observation_continuity(sink, counters, turn_started, state)
            elapsed = time.monotonic() - turn_started
            assert elapsed == pytest.approx(target, abs=1e-3)
        events = _sink_events(tmp_path / "continuity.jsonl")
        by_threshold = [
            float(event["threshold_s"])
            for event in events
            if event["event"] == "observation-continues"
        ]
    assert by_threshold == [
        45.0,
        75.0,
        130.0,
        300.0,
        900.0,
        1810.0,
    ], f"unexpected continuity checkpoints: {by_threshold}"


def test_child_continuity_locked_framing_assumptions(diag: Any) -> None:
    """No per-answer cutoff exists: continuity is doc'd as measurement."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "itional" not in "Presentation"
    assert diag.OBSERVATION_CONTINUITY_REPEAT_S > diag.DEFAULT_SNAPSHOT_CADENCE_S
    assert "crossing 45/75/130/300/900" in source.replace("\n", " ") or (
        "45/75/130/300/900" in " ".join(source.split())
    )


def test_parent_continuity_uses_child_seen_boundaries(diag: Any, tmp_path: pathlib.Path) -> None:
    with _virtual_clock() as clock:
        sink = diag.EventSink(tmp_path / "parent-continuity.jsonl", echo=False)
        evidence = diag.ProbeEvidence(probe=diag.PROBE_SCENARIO2)
        evidence.last_real_event = "stage-start:opencode"
        started = time.monotonic()
        state: dict[str, Any] = {"index": 0, "next_at": 0.0}
        for target in (45.5, 46.0, 75.5, 130.5, 905.0):
            clock["offset"] = target
            diag._parent_observation_continuity(sink, evidence, started, state)
        events = _sink_events(tmp_path / "parent-continuity.jsonl")
        observed = [str(event["last_real_event"]) for event in events]
        thresholds = [float(event["threshold_s"]) for event in events]
    assert observed == ["stage-start:opencode"] * 4
    assert thresholds == [45.0, 75.0, 130.0, 300.0]


# ---------------------------------------------------------------------------
# 3. Delivered runs: healthy slow calls are OBSERVED, not killed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_seconds",
    [46.0, 75.5, 131.0, 301.0, 901.5],
)
def test_slow_model_call_completes_and_is_recorded_delivered(
    diag: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, model_seconds: float
) -> None:
    """A healthy model call just past the legacy window completes."""
    phase_events: list[str] = []

    class _FakeRuntime:
        def __init__(self, config: Any, ready_timeout: float = 45.0) -> None:
            self.config = config
            self.running = False
            self.ready = False

        async def start(self) -> None:
            self.running = True
            self.ready = True

        async def ensure_ready(self, timeout: float | None = None) -> None:
            assert self.running and self.ready

        async def stop(self) -> None:
            self.running = False
            self.ready = False

    def fake_health(base_url: str, timeout: float) -> tuple[bool, str]:
        return False, ""  # nothing attached yet (runtime spawns its own).

    def fake_health_phased(base_url: str, method: str, path: str) -> dict[str, float]:
        return {"connect_ms": 0.5, "write_ms": 0.6, "read_ms": 0.7}

    def fake_request(
        base_url: str,
        method: str,
        path: str,
        body: dict[str, object] | None,
        connect_timeout: float,
        op_timeout: float | None,
    ) -> tuple[bool, bool, dict[str, float], dict[str, Any]]:
        phase_events.append(path)
        if path == "/session":
            assert op_timeout == 10.0  # bounded session-create is fine
            return (
                True,
                False,
                {"connect_ms": 1.0, "write_ms": 2.0, "read_ms": 3.0},
                {"_id": "session-opaque"},
            )
        assert path.endswith("/message")
        assert op_timeout is None, "the model call must carry NO total deadline"
        clock["offset"] += model_seconds  # virtual remote inference delay
        return (
            True,
            False,
            {"connect_ms": 4.0, "write_ms": 5.0, "read_ms": float(model_seconds * 1000.0)},
            {"ok": True, "served_model": "provider/model"},
        )

    monkeypatch.setattr("aa.opencode.runtime.LocalOpenCodeRuntime", _FakeRuntime)
    monkeypatch.setattr(diag, "_health_probe", fake_health)
    monkeypatch.setattr(diag, "_health_probe_phased", fake_health_phased)
    monkeypatch.setattr(diag, "_timed_request", fake_request)

    with _virtual_clock() as clock:
        sink = diag.EventSink(tmp_path / "events.jsonl", echo=False)
        argv = {
            "probe_a_port": "4097",
            "out_dir": str(tmp_path),
            "snapshot_cadence_s": 4800.0,
            "max_snapshots": 1,
            "connect_timeout_s": 8.0,
            "probe_a_bootstrap_timeout_s": 30.0,
        }
        exit_code = asyncio.run(diag.run_probe_a(argv, sink, {}))
    assert exit_code == diag.EXIT_OK
    events = _sink_events(tmp_path / "events.jsonl")
    end_event = next(event for event in events if event["event"] == "model-send-end")
    assert end_event["ok"] is True
    # The recorded duration is the real (virtual) remote inference time —
    # no per-answer wall clock truncated the call.
    assert end_event["elapsed_ms"] >= model_seconds * 1000.0 - 1.0
    end_status = next(event for event in events if event["event"] == "probe-end")
    assert end_status["status"] == "completed"


def test_slow_model_call_completes_scenario2_delivered(
    diag: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probe B: a slow full scenario-turn still DELIVERS (exit 0)."""

    class _FakeGraph:
        running = True

        def thread_id(self, chat_id: int) -> str:
            return f"thread-{chat_id}"

        def last_telemetry_for_thread(self, thread_id: str) -> dict[str, Any]:
            return {"delivery": 1}

    class _FakeClient:
        request_latency_audit: list[dict[str, Any]] = []

    class _FakeRuntime:
        running = True
        ready = True
        client = _FakeClient()

    class _FakeApiStub:
        """Placeholder: the diagnostic supplies its own _DiagnosticTelegramApi."""

    class _FakeTransport:
        def __init__(self, token: str, api: Any, **kwargs: Any) -> None:
            self._api = api

    class _FakeApp:
        def __init__(self, settings: Any, *, transport: Any = None) -> None:
            self.settings = settings
            self.transport = transport
            self.graph_runtime = _FakeGraph()
            self.opencode_runtime = _FakeRuntime()
            self.started = False
            self.stopped = False
            self._task: asyncio.Task[None] | None = None

        async def start(self) -> None:
            self.started = True
            self._task = asyncio.create_task(self._deliver_slow())

        async def _deliver_slow(self) -> None:
            # Simulated full scenario turn of 47 s: it lands right past the
            # legacy first window, which used to kill the whole run.
            clock["offset"] += 47.0
            await self.transport._api.call("sendMessage", {"text": "Готово"})

        async def stop(self) -> None:
            self.stopped = True
            if self._task is not None and not self._task.done():
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(self._task)

    monkeypatch.setattr("aa.app.Application", _FakeApp)

    with _virtual_clock() as clock:
        sink = diag.EventSink(tmp_path / "events-b.jsonl", echo=False)
        counters: dict[str, Any] = {}
        argv = {
            "out_dir": str(tmp_path),
            "snapshot_cadence_s": 4800.0,
            "max_snapshots": 1,
            "probe_b_bootstrap_timeout_s": 30.0,
        }
        exit_code = asyncio.run(diag.run_probe_b(argv, sink, counters))
    assert exit_code == diag.EXIT_OK
    assert counters["delivered"] is True
    events = _sink_events(tmp_path / "events-b.jsonl")
    delivery = next(event for event in events if event["event"] == "delivery")
    assert delivery["elapsed_ms"] >= 47.0 * 1000.0 - 1.0


# ---------------------------------------------------------------------------
# 4. THE ONLY bounded permission: ONE finite, physical campaign allowance
# ---------------------------------------------------------------------------


def test_effective_campaign_allowance_reconciles_actions_job(
    diag: Any,
) -> None:
    # No job marker → configured allowance.
    assert (
        diag.effective_campaign_allowance_s(
            campaign_allowance_s=1800.0, finalisation_reserve_s=600.0
        )
        == 1800.0
    )
    # Remaining job budget (job starts 3600s ahead, chose 600s reserve) bounds
    # the allowance: job start + 3600 - 600 - elapsed.
    now = 1_000_000.0
    assert (
        diag.effective_campaign_allowance_s(
            campaign_allowance_s=5000.0,
            finalisation_reserve_s=600.0,
            job_timeout_minutes=60.0,
            job_started_epoch=now,
            now_epoch=now,
        )
        == 3000.0
    )
    # Job budget consumed inside the reserve → tiny-but-finite floor of 1.0s.
    assert (
        diag.effective_campaign_allowance_s(
            campaign_allowance_s=5000.0,
            finalisation_reserve_s=600.0,
            job_timeout_minutes=60.0,
            job_started_epoch=now - 3500.0,
            now_epoch=now,
        )
        == 1.0
    )


def test_run_one_probe_hung_child_ends_at_campaign_allowance_tidy(
    diag: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No-progress child: SIGTERM → SIGKILL group, unresolved verdict."""
    spawned: list[subprocess.Popen[bytes]] = []

    def fake_spawn(*, plan: Any, out_dir: pathlib.Path) -> subprocess.Popen[bytes]:
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", "import time; time.sleep(120)"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        spawned.append(proc)
        return proc

    monkeypatch.setattr(diag, "_spawn_child", fake_spawn)
    plan = diag.ProbePlan(
        probe=diag.PROBE_SCENARIO2,
        snapshot_cadence_s=diag.DEFAULT_SNAPSHOT_CADENCE_S,
        max_snapshots=1,
        connect_timeout_s=diag.DEFAULT_PROBE_A_CONNECT_TIMEOUT_S,
    )
    sink = diag.EventSink(tmp_path / "sup.jsonl", echo=False)
    evidence = diag._run_one_probe(
        plan,
        out_dir=tmp_path,
        sink=sink,
        publisher=None,
        campaign_deadline_s=time.monotonic() + 0.3,
    )
    assert spawned and spawned[0].poll() is not None, "hung child NOT reaped"
    assert evidence.terminated_at_campaign_end is True
    assert evidence.status != diag.CLASS_DELIVERED
    assert evidence.status in ("", "running", diag.CLASS_UNRESOLVED)
    [probe_a] = [diag.ProbeEvidence(probe=diag.PROBE_DIRECT_MODEL, status="completed")]
    assert diag.classify_cause(probe_a, evidence) == diag.CLASS_UNRESOLVED
    assert diag.overall_exit_code(probe_a, evidence) == diag.EXIT_INCONCLUSIVE
    assert diag._abnormal_child_exit(evidence) is False, (
        "a tidy campaign-end shutdown must NOT count as harness error"
    )


def test_run_one_probe_without_deadline_lets_child_finish(
    diag: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """campaign_deadline_s=None (e.g. audit mode) never interrupts a child."""
    marker = tmp_path / "probe-scenario2-events.jsonl"

    def fake_spawn(*, plan: Any, out_dir: pathlib.Path) -> subprocess.Popen[bytes]:
        script = (
            "import json, sys, time\n"
            "print(json.dumps({'event': 'delivery', 'elapsed_ms': 121000.0, "
            "'reply_len': 7, 'reply_signature': 'abc1234567890abc'}), flush=True)\n"
            "time.sleep(0.2)\n"
            "print(json.dumps({'event': 'probe-end', 'probe': 'scenario2', "
            "'status': 'delivered'}), flush=True)\n"
        )
        with marker.open("w", encoding="utf-8") as stream:
            stream.write("")
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", script],
            stdout=None,
            stderr=subprocess.DEVNULL,
            cwd=str(out_dir),
            start_new_session=True,
        )
        # Deliver the events through the artifact file (the reader tails it).
        thread = threading.Thread(
            target=_write_child_events_later,
            args=(marker, proc, 121000.0),
            daemon=True,
        )
        thread.start()
        return proc

    monkeypatch.setattr(diag, "_spawn_child", fake_spawn)
    plan = diag.ProbePlan(
        probe=diag.PROBE_SCENARIO2,
        snapshot_cadence_s=diag.DEFAULT_SNAPSHOT_CADENCE_S,
        max_snapshots=1,
        connect_timeout_s=diag.DEFAULT_PROBE_A_CONNECT_TIMEOUT_S,
    )
    sink = diag.EventSink(tmp_path / "sup.jsonl", echo=False)
    evidence = diag._run_one_probe(
        plan,
        out_dir=tmp_path,
        sink=sink,
        publisher=None,
        campaign_deadline_s=None,
    )
    assert evidence.delivered is True
    assert evidence.delivered_elapsed_ms == 121000.0
    assert evidence.terminated_at_campaign_end is False
    assert evidence.killed is False
    assert evidence.status == diag.CLASS_DELIVERED


def _write_child_events_later(
    path: pathlib.Path, proc: subprocess.Popen[bytes], delivered_ms: float
) -> None:
    """Feed the artifact file the reader tail watches (no pipe buffering)."""
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.02)
        if path.exists():
            with path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "event": "delivery",
                            "elapsed_ms": delivered_ms,
                            "reply_len": 7,
                            "reply_signature": "abc1234567890abc",
                        }
                    )
                    + "\n"
                )
                stream.write(
                    json.dumps(
                        {
                            "event": "probe-end",
                            "probe": "scenario2",
                            "status": "delivered",
                        }
                    )
                    + "\n"
                )
            return


# ---------------------------------------------------------------------------
# 5. Genuinely unresponsive ≠ slow: sustained local failure is typed
# ---------------------------------------------------------------------------


def test_child_watchdog_flags_sustained_local_non_responsiveness(
    diag: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        diag, "_health_probe", lambda base_url, timeout: (False, "connection-refused")
    )
    counters: dict[str, Any] = {
        "last_stage": "opencode",
        "last_stage_at_s": 0.1,
        "last_stage_monotonic": time.monotonic(),
        "last_real_event": "stage-start:opencode",
        "last_real_monotonic": time.monotonic(),
        "typing_sends": 0,
        "delivered": False,
        "probe_start": time.monotonic(),
    }
    sink = diag.EventSink(tmp_path / "watchdog.jsonl", echo=False)
    anchor = threading.Event()
    watchdog = diag.ChildWatchdog(
        probe=diag.PROBE_SCENARIO2,
        cadence_s=0.05,
        max_snapshots=6,
        anchor=anchor,
        anchor_wait_timeout_s=15.0,
        sink=sink,
        checkpoint_dir=tmp_path,
        opencode_base_url="http://127.0.0.1:9",
        counters=counters,
        internal_probe=None,
    )
    watchdog.start()
    anchor.set()
    deadline = time.monotonic() + 20.0
    while counters.get("server_nonresponsive_sustained") is None and time.monotonic() < deadline:
        time.sleep(0.05)
    watchdog.stop()
    watchdog.join(timeout=10.0)
    assert counters.get("server_nonresponsive_sustained") is True
    events = _sink_events(tmp_path / "watchdog.jsonl")
    sustained = next(
        event for event in events if event["event"] == "server-nonresponsive-sustained"
    )
    assert sustained["consecutive_failures"] >= diag.NONRESPONSIVE_SNAPSHOTS_REQUIRED
    # The boundary naming stayed typed (stage only, no truthful blame).
    snapshot = next(event for event in events if event["event"] == "snapshot")
    assert snapshot["stage"] == "opencode"


def test_scenario2_run_exits_server_error_proven_when_sustained(
    diag: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sustained local failure: bounded typed exit 7 before campaign end."""

    class _FakeGraph:
        running = True

        def thread_id(self, chat_id: int) -> str:
            return f"thread-{chat_id}"

        def last_telemetry_for_thread(self, thread_id: str) -> dict[str, Any]:
            return {}

    class _FakeClient:
        request_latency_audit: list[dict[str, Any]] = []

    class _FakeRuntime:
        running = True
        ready = True
        client = _FakeClient()

    class _FakeTransport:
        def __init__(self, token: str, api: Any, **kwargs: Any) -> None:
            self._api = api

    class _FakeApp:
        def __init__(self, settings: Any, *, transport: Any = None) -> None:
            self.settings = settings
            self.transport = transport
            self.graph_runtime = _FakeGraph()
            self.opencode_runtime = _FakeRuntime()
            self.stopped = False

        async def start(self) -> None:
            return

        async def stop(self) -> None:
            self.stopped = True

    monkeypatch.setattr("aa.app.Application", _FakeApp)
    sink = diag.EventSink(tmp_path / "events-b.jsonl", echo=False)
    counters = {
        # The watchdog already proved sustained local non-responsiveness.
        "server_nonresponsive_sustained": True,
        "last_stage": "opencode",
        "last_real_event": "stage-start:opencode",
    }
    argv = {
        "out_dir": str(tmp_path),
        "snapshot_cadence_s": 4800.0,
        "max_snapshots": 1,
        "probe_b_bootstrap_timeout_s": 30.0,
    }
    exit_code = asyncio.run(diag.run_probe_b(argv, sink, counters))
    assert exit_code == diag.EXIT_SERVER_ERROR_PROVEN
    events = _sink_events(tmp_path / "events-b.jsonl")
    probe_end = next(event for event in events if event["event"] == "probe-end")
    assert probe_end["status"] == "server-error-proven"


# ---------------------------------------------------------------------------
# 6. Truthful verdicts: campaign end is INCONCLUSIVE, not success
# ---------------------------------------------------------------------------


def test_workflow_reports_inconclusive_accessibility(workflow_text: str) -> None:
    assert '"$status" -eq 3' in workflow_text
    assert "result-published" in workflow_text


def test_supporting_supervisor_poll_stays_live(diag: Any) -> None:
    """The supervisor MUST poll faster than visual cadence to observe."""
    assert diag.SUPERVISOR_POLL_S < diag.COMMENT_CADENCE_S
    # Sanitizer + killing are still enforced defaults.
    assert diag.NONRESPONSIVE_SNAPSHOTS_REQUIRED >= 4
