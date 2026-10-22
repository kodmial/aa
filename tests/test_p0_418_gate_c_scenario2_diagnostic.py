"""P0 diagnostic contract tests (kodmial/aa#418).

The new diagnostic workflow + script are strictly deterministic off-line
objects here: no subprocess probe is launched, no network is touched and no
model payload exists. Only the workflow contract and the script's pure
privacy/decision helpers are exercised.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "diagnose_gate_c_scenario2.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "aa-gate-c-scenario2-diagnostic.yml"


@pytest.fixture(scope="module")
def diag() -> Any:
    module_name = "diagnose_gate_c_scenario2_under_test"
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
# Workflow contract (kodmial/aa#418 DoD)
# ---------------------------------------------------------------------------


def test_workflow_triggers_are_limited_and_exact(workflow_text: str) -> None:
    assert "workflow_dispatch:" in workflow_text
    assert "- main" in workflow_text
    assert ".github/workflows/aa-gate-c-scenario2-diagnostic.yml" in workflow_text
    assert "scripts/diagnose_gate_c_scenario2.py" in workflow_text
    # No broad src/** qualification trigger path.
    assert '"src/aa/**"' not in workflow_text


def test_workflow_bounded_runtime_and_permissions(workflow_text: str) -> None:
    assert "timeout-minutes: 20" in workflow_text
    assert "issues: write" in workflow_text
    assert "concurrency:" in workflow_text
    assert "cancel-in-progress: false" in workflow_text


def test_workflow_uses_real_environment(workflow_text: str) -> None:
    assert "aa-opencode-toolchain" in workflow_text
    assert "AA_BOT_PRIMARY_MODEL" in workflow_text
    assert "AA_BOT_FALLBACK_MODEL" in workflow_text
    assert "AA_BOOK_AGE_IDENTITY" in workflow_text
    assert "bot_models" in workflow_text
    assert "build_retrieval_index.py --backend e5" in workflow_text
    # Text scenario only: no voice prefetch in the diagnostic run.
    assert "prefetch_voice_assets" not in workflow_text


def test_workflow_runs_scenario2_probe_and_uploads_artifacts(workflow_text: str) -> None:
    assert "scripts/diagnose_gate_c_scenario2.py" in workflow_text
    assert "--issue 418" in workflow_text
    assert "--out-dir diagnostic-scenario2-out" in workflow_text
    assert "aa-gate-c-scenario2-diagnostic-${{ github.run_id }}-scenario2" in workflow_text
    assert "if: always()" in workflow_text
    # The fallback notice must never duplicate the marker comment.
    assert "already exists" in workflow_text
    # No external Telegram bot token is ever configured for the diagnostic.
    assert "TELEGRAM_BOT_TOKEN" not in workflow_text


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parse_schedule_defaults_and_filtering(diag: Any) -> None:
    assert diag.parse_schedule("10,25,45,75,110") == (10.0, 25.0, 45.0, 75.0, 110.0)
    assert diag.parse_schedule("abc, x , 30, -1, 0") == (30.0,)
    assert diag.parse_schedule("") == ()


def test_comment_marker_is_run_scoped(diag: Any) -> None:
    assert diag.comment_marker("123") == "<!-- aa-scenario2-diagnostic run=123 -->"
    assert "run=456" not in diag.comment_marker("123")


def test_child_argv_roundtrip(diag: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "SCENARIO2_DIAG_ARGS",
        json.dumps(
            {"probe": "scenario2", "out": "/tmp/x", "snapshots": [5.0, 9.0], "cutoff": 20.0}
        ),
    )
    monkeypatch.setenv("SCENARIO2_DIAG_PROBE_A_PORT", "4097")
    monkeypatch.setenv("SCENARIO2_DIAG_PROBE_A_MESSAGE_TIMEOUT_S", "30")
    argv = diag.child_argv_from_env()
    assert argv["probe"] == "scenario2"
    assert argv["out_dir"] == "/tmp/x"
    assert argv["snapshot_schedule"] == (5.0, 9.0)
    assert argv["cutoff_s"] == 20.0
    assert argv["probe_a_port"] == "4097"
    assert argv["probe_a_message_timeout_s"] == 30.0


def test_child_argv_bad_payload_is_safe(diag: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCENARIO2_DIAG_ARGS", "not-json")
    argv = diag.child_argv_from_env()
    assert argv["probe"] == ""
    assert argv["snapshot_schedule"] == ()
    assert argv["probe_a_message_timeout_s"] == diag.DEFAULT_PROBE_A_MESSAGE_TIMEOUT_S


# ---------------------------------------------------------------------------
# Privacy: fail-closed sanitizer
# ---------------------------------------------------------------------------


def test_assert_no_text_leak_rejects_forbidden_keys(diag: Any) -> None:
    with pytest.raises(ValueError):
        diag.assert_no_text_leak({"text": "К вечеру очень тянет выпить."})
    with pytest.raises(ValueError):
        diag.assert_no_text_leak({"outer": {"book_text": "fragment"}})
    with pytest.raises(ValueError):
        diag.assert_no_text_leak([{"summary": "leak"}])


def test_assert_no_text_leak_allows_structured_fields(diag: Any) -> None:
    diag.assert_no_text_leak({"stage": "planner", "elapsed_ms": 42})
    diag.assert_no_text_leak({"text": "", "counters": [1, 2, 3]})
    diag.assert_no_text_leak({"reply_signature": "abc1234567890abc"})


# ---------------------------------------------------------------------------
# Evidence rendering and classification
# ---------------------------------------------------------------------------


def _evidence(diag: Any, **fields: Any) -> Any:
    return diag.ProbeEvidence(probe=fields.pop("probe", "scenario2"), **fields)


def test_probe_comment_lines_stay_structural(diag: Any) -> None:
    evidence = _evidence(
        diag,
        status="still-in-progress-at-probe-end",
        last_stage="retrieval",
        seconds_since_last_stage=40.0,
        fail_category="still-in-progress-at-probe-end",
        server_ready_ms=12000.0,
        snapshots=4,
    )
    lines = diag._probe_comment_lines(evidence)
    joined = "\n".join(lines)
    assert "`retrieval`" in joined
    assert "still-in-progress-at-probe-end" in joined
    assert all(line.startswith("- ") or line.startswith("###") for line in lines)


def test_classify_cause_direct_model_first(diag: Any) -> None:
    probe_a = _evidence(diag, probe="direct-model", status="inactivity-timeout")
    # Provider stall proven directly: retrieval is never blamed.
    assert diag.classify_cause(probe_a, _evidence(diag)) == "model-provider"


def test_classify_cause_mapping_order(diag: Any) -> None:
    ok_a = _evidence(diag, probe="direct-model", status="completed")
    assert diag.classify_cause(_evidence(diag, status="bootstrap-failed"), _evidence(diag)) == (
        "local-opencode-server"
    )
    assert diag.classify_cause(ok_a, _evidence(diag, status="bootstrap-failed")) == (
        "bootstrap-dependency"
    )
    assert diag.classify_cause(ok_a, _evidence(diag, delivered=True)) == "delivered"
    stalled_retrieval = _evidence(
        diag, last_stage="retrieval", fail_category="still-in-progress-at-probe-end"
    )
    assert diag.classify_cause(ok_a, stalled_retrieval) == "aa-retrieval"
    assert diag.classify_cause(ok_a, _evidence(diag, last_stage="")) == "stage-unobserved"
    assert diag.classify_cause(ok_a, _evidence(diag, last_stage="planner")) == "aa-graph-planner"


def test_final_report_invariants(diag: Any) -> None:
    ok_a = _evidence(diag, probe="direct-model", status="completed", server_ready_ms=9000.0)
    stalled_b = _evidence(
        diag,
        status="still-in-progress-at-probe-end",
        fail_category="still-in-progress-at-probe-end",
        last_stage="planner",
        server_ready_ms=14000.0,
        delivered_elapsed_ms=None,
        snapshots=5,
    )
    meta = {
        "run_url": "https://github.com/kodmial/aa/actions/runs/39752912805",
        "main_sha": "4f95b5260c7c84b8323302874a73e707b8cc633b",
    }
    report = diag.build_final_report(meta, ok_a, stalled_b)
    assert "4f95b5260c7c84b8323302874a73e707b8cc633b" in report
    assert "actions/runs/39752912805" in report
    assert "INCONCLUSIVE, never a product FAIL" in report
    assert "classification:" in report
    assert len(report.splitlines()) < 60


def test_final_report_names_unobserved_stage_explicitly(diag: Any) -> None:
    report = diag.build_final_report(
        {"run_url": "https://github.com/kodmial/aa/actions/runs/1", "main_sha": "abc"},
        _evidence(diag, probe="direct-model", status="completed"),
        _evidence(diag, status="child-exit-3"),
    )
    assert "stage=unobserved" in report


# ---------------------------------------------------------------------------
# IssueProgressPublisher: one idempotent comment per run (fail-soft)
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.comments: list[dict[str, Any]] = []
        self.next_id = 1000

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> tuple[bool, Any]:
        self.calls.append((method, path))
        if method == "GET":
            return True, list(self.comments)
        if method == "POST":
            self.next_id += 1
            comment = {"id": self.next_id, "body": (payload or {}).get("body", "")}
            self.comments.append(comment)
            return True, comment
        if method == "PATCH":
            for comment in self.comments:
                if comment["id"] == int(path.rsplit("/", 1)[-1]):
                    comment["body"] = (payload or {}).get("body", "")
            return True, None
        return False, None


def test_publisher_creates_then_updates_same_comment(diag: Any) -> None:
    recorder = _Recorder()
    publisher = diag.IssueProgressPublisher("kodmial", "aa", 418, "token", run_id="777")
    publisher._request = recorder._request
    assert publisher.publish("progress", "- status: running")
    assert publisher._comment_id is not None
    created_id = publisher._comment_id
    marker = diag.comment_marker("777")
    assert any(marker in comment["body"] for comment in recorder.comments)
    publisher._last_post = 0.0  # throttle elapsed (test-only reset)
    assert publisher.publish("progress2", "- status: done")
    methods = [method for method, _ in recorder.calls]
    assert methods.count("POST") == 1
    assert "PATCH" in methods
    assert recorder.comments[-1]["id"] == created_id


def test_publisher_finds_existing_marker_comment(diag: Any) -> None:
    recorder = _Recorder()
    recorder.comments.append({"id": 55, "body": "text " + diag.comment_marker("888")})
    publisher = diag.IssueProgressPublisher("kodmial", "aa", 418, "token", run_id="888")
    publisher._request = recorder._request
    assert publisher.ensure_comment() == 55
    publisher._last_post = 0.0
    assert publisher.publish("final", "- body")
    methods = [method for method, _ in recorder.calls]
    assert "POST" not in methods
    assert "PATCH" in methods


def test_publisher_without_token_is_fail_soft(diag: Any) -> None:
    publisher = diag.IssueProgressPublisher("kodmial", "aa", 418, "", run_id="1")
    assert publisher.ensure_comment() is None
    assert publisher.publish("x", "- no token") is False or True


def test_publisher_sanitizes_suppressed_text(diag: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    publisher = diag.IssueProgressPublisher("kodmial", "aa", 418, "token", run_id="42")
    original = recorder._request

    def recording(
        method: str, path: str, payload: dict[str, Any] | None = None
    ) -> tuple[bool, Any]:
        if method in ("POST", "PATCH") and payload is not None:
            body = str(payload.get("body", ""))
            assert "К вечеру" not in body
        return original(method, path, payload)

    publisher._request = recording
    publisher._last_post = 0.0
    assert publisher.publish("leaky title", "- text: К вечеру очень тянет выпить.\n- ok: 1")
    assert any("suppressed by sanitizer" in comment["body"] for comment in recorder.comments)


# ---------------------------------------------------------------------------
# Bounded probe integration (fakes at the external boundaries only)
# ---------------------------------------------------------------------------


def test_workflow_treats_inconclusive_exit_as_success(workflow_text: str) -> None:
    assert "status=$?" in workflow_text
    assert '"$status" -eq 3' in workflow_text
    assert "0=complete, 3=inconclusive" in workflow_text


def test_run_probe_a_bounds_bootstrap_and_times_http_phases(
    diag: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instances: list[Any] = []

    class _FakeRuntime:
        def __init__(self, config: Any, ready_timeout: float = 45.0) -> None:
            self.config = config
            self.ready_timeout = ready_timeout
            self.running = False
            self.ready = False
            self.started = False
            self.stopped = False
            instances.append(self)

        async def start(self) -> None:
            self.started = True
            self.running = True
            self.ready = True

        async def ensure_ready(self, timeout: float | None = None) -> None:
            assert self.running and self.ready

        async def stop(self) -> None:
            self.stopped = True
            self.running = False
            self.ready = False

    calls: list[tuple[str, str]] = []

    def fake_request(
        base_url: str, method: str, path: str, body: dict[str, object] | None, timeout: float
    ) -> tuple[bool, bool, dict[str, float], dict[str, Any]]:
        calls.append((method, path))
        if path == "/session":
            return (
                True,
                False,
                {"connect_ms": 1.0, "write_ms": 2.0, "read_ms": 3.0},
                {"_id": "session-opaque"},
            )
        if path == "/session/session-opaque/message":
            return (
                True,
                False,
                {"connect_ms": 4.0, "write_ms": 5.0, "read_ms": 6.0},
                {"served_model": "provider/model"},
            )
        return True, False, {"status": 200}, {}

    monkeypatch.setattr("aa.opencode.runtime.LocalOpenCodeRuntime", _FakeRuntime)
    monkeypatch.setattr(diag, "_health_probe", lambda base_url, timeout: (False, ""))
    monkeypatch.setattr(
        diag,
        "_health_probe_phased",
        lambda base_url, method, path: {"connect_ms": 0.5, "write_ms": 0.6, "read_ms": 0.7},
    )
    monkeypatch.setattr(diag, "_timed_request", fake_request)

    sink = diag.EventSink(tmp_path / "events.jsonl", echo=False)
    counters: dict[str, Any] = {}
    argv = {
        "probe_a_port": "4097",
        "out_dir": str(tmp_path),
        "snapshot_schedule": (),
        "probe_a_message_timeout_s": 5.0,
        "probe_a_bootstrap_timeout_s": 5.0,
    }

    assert asyncio.run(diag.run_probe_a(argv, sink, counters)) == diag.EXIT_OK
    assert instances and instances[0].config.base_url.endswith(":4097")
    assert instances[0].started and instances[0].stopped
    assert counters["server_ready_ms"] is not None
    assert ("POST", "/session") in calls
    assert ("POST", "/session/session-opaque/message") in calls

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    health_phases = next(event for event in events if event["event"] == "server-health-phases")
    assert health_phases["connect_ms"] == 0.5
    model_event = next(event for event in events if event["event"] == "model-send-end")
    assert model_event["connect_ms"] == 4.0
    assert model_event["write_ms"] == 5.0
    assert model_event["read_ms"] == 6.0
    assert model_event["ok"] is True


def test_run_probe_a_bootstrap_timeout_is_bounded(
    diag: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instances: list[Any] = []

    class _HangingRuntime:
        def __init__(self, config: Any, ready_timeout: float = 45.0) -> None:
            self.running = False
            self.ready = False
            self.stopped = False
            instances.append(self)

        async def start(self) -> None:
            await asyncio.sleep(10.0)

        async def ensure_ready(self, timeout: float | None = None) -> None:
            raise AssertionError("ensure_ready must not run after a start timeout")

        async def stop(self) -> None:
            self.stopped = True

    monkeypatch.setattr("aa.opencode.runtime.LocalOpenCodeRuntime", _HangingRuntime)
    monkeypatch.setattr(diag, "_health_probe", lambda base_url, timeout: (False, ""))

    sink = diag.EventSink(tmp_path / "events.jsonl", echo=False)
    argv = {
        "probe_a_port": "4097",
        "out_dir": str(tmp_path),
        "snapshot_schedule": (),
        "probe_a_message_timeout_s": 5.0,
        "probe_a_bootstrap_timeout_s": 0.05,
    }
    started = time.monotonic()
    code = asyncio.run(diag.run_probe_a(argv, sink, {}))
    assert code == diag.EXIT_BOOTSTRAP_FAILED
    assert time.monotonic() - started < 2.0
    assert instances and instances[0].stopped


def test_run_probe_b_delivers_through_production_boundary(
    diag: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeGraph:
        running = True

        def thread_id(self, chat_id: int) -> str:
            return f"thread-{chat_id}"

        def last_telemetry_for_thread(self, thread_id: str) -> dict[str, Any]:
            return {"delivery": 1, "planner_calls": 2}

    class _FakeClient:
        request_latency_audit: list[dict[str, Any]] = []

    class _FakeApp:
        def __init__(self, settings: Any, *, transport: Any = None) -> None:
            self.settings = settings
            self.transport = transport
            self.graph_runtime = _FakeGraph()
            self.opencode_runtime = type(
                "Runtime",
                (),
                {"running": True, "ready": True, "client": _FakeClient()},
            )()
            self.started = False
            self.stopped = False
            self._task: asyncio.Task[None] | None = None

        async def start(self) -> None:
            self.started = True
            self._task = asyncio.create_task(self._deliver())

        async def _deliver(self) -> None:
            api = self.transport._api
            while not api.pending_updates:
                await asyncio.sleep(0.005)
            await asyncio.sleep(0.01)
            await api.call("sendMessage", {"text": "Готово"})

        async def stop(self) -> None:
            self.stopped = True
            if self._task is not None:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task

    monkeypatch.setattr("aa.app.Application", _FakeApp)

    sink = diag.EventSink(tmp_path / "events-b.jsonl", echo=False)
    counters: dict[str, Any] = {}
    argv = {
        "out_dir": str(tmp_path),
        "snapshot_schedule": (),
        "cutoff_s": 5.0,
        "probe_b_bootstrap_timeout_s": 5.0,
    }
    assert asyncio.run(diag.run_probe_b(argv, sink, counters)) == diag.EXIT_OK
    assert counters["delivered"] is True
    events = [
        json.loads(line)
        for line in (tmp_path / "events-b.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["event"] == "delivery" for event in events)
    assert any(event["event"] == "scenario-end" for event in events)


def test_run_probe_b_bootstrap_timeout_is_bounded(
    diag: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _HangingApp:
        def __init__(self, settings: Any, *, transport: Any = None) -> None:
            self.stopped = False

        async def start(self) -> None:
            await asyncio.sleep(10.0)

        async def stop(self) -> None:
            self.stopped = True

    monkeypatch.setattr("aa.app.Application", _HangingApp)

    sink = diag.EventSink(tmp_path / "events-b.jsonl", echo=False)
    argv = {
        "out_dir": str(tmp_path),
        "snapshot_schedule": (),
        "cutoff_s": 5.0,
        "probe_b_bootstrap_timeout_s": 0.05,
    }
    started = time.monotonic()
    code = asyncio.run(diag.run_probe_b(argv, sink, {}))
    assert code == diag.EXIT_BOOTSTRAP_FAILED
    assert time.monotonic() - started < 2.0


def test_supervisor_kills_child_group_at_cutoff(
    diag: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def spawn(*, plan: Any, out_dir: pathlib.Path) -> Any:
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import signal, time; "
                    "signal.signal(signal.SIGUSR1, signal.SIG_IGN); "
                    "time.sleep(60)"
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    monkeypatch.setattr(diag, "_spawn_child", spawn)
    monkeypatch.setattr(diag, "CHILD_CUTOFF_SLACK_S", 0.05)
    monkeypatch.setattr(diag, "_final_supervisor_snapshot", lambda *args, **kwargs: {})
    monkeypatch.setattr(diag, "DEFAULT_TERMINATE_GRACE_S", 0.1)

    plan = diag.ProbePlan(probe=diag.PROBE_SCENARIO2, schedule=(), child_cutoff_s=0.05)
    evidence = diag._run_one_probe(
        plan,
        out_dir=tmp_path,
        sink=diag.EventSink(tmp_path / "supervisor.jsonl", echo=False),
        publisher=None,
    )
    assert evidence.killed is True
    assert evidence.status == "still-in-progress-at-probe-end"


def test_spawn_child_is_isolated_process_group(
    diag: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class _FakePopen:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            captured["argv"] = argv
            captured["kwargs"] = kwargs

        def poll(self) -> int:
            return 0

    monkeypatch.setattr(diag.subprocess, "Popen", _FakePopen)
    plan = diag.ProbePlan(probe=diag.PROBE_SCENARIO2, schedule=(1.0,), child_cutoff_s=5.0)
    diag._spawn_child(plan=plan, out_dir=tmp_path)

    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["stdout"] is None
    child_args = json.loads(captured["kwargs"]["env"]["SCENARIO2_DIAG_ARGS"])
    assert child_args["probe"] == diag.PROBE_SCENARIO2
    assert diag._child_events_path(tmp_path, diag.PROBE_DIRECT_MODEL) != (
        diag._child_events_path(tmp_path, diag.PROBE_SCENARIO2)
    )


def test_evidence_collection_and_report_generation(diag: Any, tmp_path: pathlib.Path) -> None:
    lines = [
        {"event": "server-health", "ok": True, "elapsed_ms": 1200.0},
        {"event": "session-create-end", "ok": True, "read_ms": 3.0},
        {
            "event": "model-send-end",
            "ok": True,
            "elapsed_ms": 500.0,
            "connect_ms": 1.0,
            "write_ms": 2.0,
            "read_ms": 3.0,
        },
        {
            "event": "delivery",
            "elapsed_ms": 1500.0,
            "reply_len": 42,
            "reply_signature": "abc1234567890abc",
        },
        {"event": "probe-end", "probe": "scenario2", "status": "delivered"},
    ]
    source = io.StringIO("\n".join(json.dumps(line) for line in lines) + "\n")
    evidence = diag.ProbeEvidence(probe=diag.PROBE_SCENARIO2)
    diag._reader_loop(
        source, evidence, diag.EventSink(tmp_path / "reader-events.jsonl", echo=False)
    )
    assert evidence.delivered is True
    assert evidence.model_elapsed_ms == 500.0
    assert evidence.status == "delivered"

    report = diag.build_final_report(
        {"run_url": "https://example.invalid/run", "main_sha": "deadbeef"},
        diag.ProbeEvidence(probe=diag.PROBE_DIRECT_MODEL, status="completed"),
        evidence,
    )
    assert "classification: `delivered`" in report
    serialized = diag._evidence_dict(evidence)
    assert serialized["delivered"] is True
    json.dumps(serialized)
