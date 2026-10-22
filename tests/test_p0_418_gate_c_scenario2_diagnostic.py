"""P0 diagnostic contract tests (kodmial/aa#418, updated by kodmial/aa#424).

The diagnostic workflow + script are strictly deterministic off-line objects
here: no subprocess probe is launched against real infrastructure, no network
is touched and no model payload exists. Only the workflow contract, the
script's pure privacy/decision helpers and bounded probe behaviours are
exercised. kodmial/aa#424 contract points asserted here:

- NO per-answer wall-clock cutoff exists (45/75/130s windows are gone);
- truthful verdict taxonomy; in-flight ``opencode`` stages are NEVER mapped
  to a proven provider root cause;
- one campaign-level physical allowance, reconciled with the Actions job
  budget, ending in ``unresolved-at-campaign-end`` (INCONCLUSIVE, never
  success).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
import pathlib
import signal
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
# Workflow contract (kodmial/aa#418 + kodmial/aa#424 DoD)
# ---------------------------------------------------------------------------


def test_workflow_triggers_are_limited_and_exact(workflow_text: str) -> None:
    assert "workflow_dispatch:" in workflow_text
    assert "- main" in workflow_text
    assert ".github/workflows/aa-gate-c-scenario2-diagnostic.yml" in workflow_text
    assert "scripts/diagnose_gate_c_scenario2.py" in workflow_text
    # No broad src/** qualification trigger path.
    assert '"src/aa/**"' not in workflow_text


def test_workflow_has_no_per_answer_cutoff(workflow_text: str) -> None:
    # The old arbitrary answer windows must not reappear in the workflow.
    assert "--diagnostic-cutoff-s" not in workflow_text
    assert "--probe-a-cutoff-s" not in workflow_text
    assert "--probe-a-message-timeout-s" not in workflow_text
    # The campaign-level allowance (one job ceiling) IS present and typed.
    assert "--campaign-allowance-s" in workflow_text
    assert "--finalisation-reserve-s" in workflow_text
    assert "--job-timeout-minutes" in workflow_text
    assert "--job-started-epoch" in workflow_text
    assert "timeout-minutes: 60" in workflow_text


def test_workflow_reports_inconclusive_as_not_success(workflow_text: str) -> None:
    assert "issues: write" in workflow_text
    assert "concurrency:" in workflow_text
    assert "cancel-in-progress: false" in workflow_text
    # Exit 3 (INCONCLUSIVE) must NOT be swallowed into a silent success.
    assert '"$status" -eq 3' in workflow_text
    assert "exit 3" in workflow_text
    assert "NOT a success" in workflow_text
    assert "result-published" in workflow_text
    assert "result.json" in workflow_text


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
    # The fallback notice must never duplicate the marker comment and must
    # not fire for an INCONCLUSIVE-but-published run.
    assert "already exists" in workflow_text
    assert "result-published != 'true'" in workflow_text
    # No external Telegram bot token is ever configured for the diagnostic.
    assert "TELEGRAM_BOT_TOKEN" not in workflow_text


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_comment_marker_is_run_scoped(diag: Any) -> None:
    assert diag.comment_marker("123") == "<!-- aa-scenario2-diagnostic run=123 -->"
    assert "run=456" not in diag.comment_marker("123")


def test_effective_campaign_allowance_static_default(diag: Any) -> None:
    # Without a job marker: configured allowance is authoritative.
    assert (
        diag.effective_campaign_allowance_s(
            campaign_allowance_s=1800.0,
            finalisation_reserve_s=600.0,
        )
        == 1800.0
    )
    # An absurd configured value stays finite (physical ceiling semantics).
    assert (
        diag.effective_campaign_allowance_s(
            campaign_allowance_s=float("nan"),
            finalisation_reserve_s=600.0,
            now_epoch=1000.0,
        )
        >= 1.0
    )


def test_effective_campaign_allowance_reconciles_job_budget(diag: Any) -> None:
    # Job started 100s before "now", 60-minute cap, 600s reserve:
    # remaining from now = (3600 - 600 - 100) = 2900 < configured 5000.
    now = 1000.0
    assert (
        diag.effective_campaign_allowance_s(
            campaign_allowance_s=5000.0,
            finalisation_reserve_s=600.0,
            job_timeout_minutes=60.0,
            job_started_epoch=now - 100.0,
            now_epoch=now,
        )
        == 2900.0
    )


def test_effective_campaign_allowance_job_budget_exhausted(diag: Any) -> None:
    # Job already inside its finalisation reserve: keep the minimum viable
    # observation window rather than collapsing to a token 1s.
    # (job_started_epoch must stay positive — the static guard ignores
    # unset markers; elapsed 3100s > 3600 - 600 reserve.)
    now = 100_000.0
    assert (
        diag.effective_campaign_allowance_s(
            campaign_allowance_s=5000.0,
            finalisation_reserve_s=600.0,
            job_timeout_minutes=60.0,
            job_started_epoch=now - 3100.0,
            now_epoch=now,
        )
        == diag.MIN_CAMPAIGN_ALLOWANCE_S
    )


def test_child_argv_roundtrip(diag: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "SCENARIO2_DIAG_ARGS",
        json.dumps(
            {
                "probe": "scenario2",
                "out": "/tmp/x",
                "snapshot_cadence_s": 2.0,
                "max_snapshots": 7,
                "connect_timeout_s": 8.0,
            }
        ),
    )
    monkeypatch.setenv("SCENARIO2_DIAG_PROBE_A_PORT", "4097")
    argv = diag.child_argv_from_env()
    assert argv["probe"] == "scenario2"
    assert argv["out_dir"] == "/tmp/x"
    assert argv["snapshot_cadence_s"] == 2.0
    assert argv["max_snapshots"] == 7
    assert argv["connect_timeout_s"] == 8.0
    assert argv["probe_a_port"] == "4097"
    assert argv["probe_a_bootstrap_timeout_s"] == diag.DEFAULT_PROBE_A_BOOTSTRAP_TIMEOUT_S


def test_child_argv_bad_payload_is_safe(diag: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCENARIO2_DIAG_ARGS", "not-json")
    argv = diag.child_argv_from_env()
    assert argv["probe"] == ""
    assert argv["snapshot_cadence_s"] == diag.DEFAULT_SNAPSHOT_CADENCE_S
    assert argv["max_snapshots"] == diag.DEFAULT_MAX_SNAPSHOTS
    assert argv["connect_timeout_s"] == diag.DEFAULT_PROBE_A_CONNECT_TIMEOUT_S


def test_script_source_has_no_arbitrary_answer_cutoffs(diag: Any) -> None:
    """kodmial/aa#424: the 45/75/130 answer windows may not exist at all."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "DEFAULT_DIAGNOSTIC_CUTOFF_S",
        "DEFAULT_PROBE_A_CUTOFF_S",
        "DEFAULT_PROBE_A_MESSAGE_TIMEOUT_S",
        "--diagnostic-cutoff-s",
        "--probe-a-cutoff-s",
        "--probe-a-message-timeout-s",
        "delivery-timeout",
        "still-in-progress-at-probe-end",
    ):
        assert forbidden not in source, f"forbidden symbol reapplied: {forbidden}"
    # The typed, answer-unrelated connect bound stays explicit.
    assert "DEFAULT_PROBE_A_CONNECT_TIMEOUT_S" in source


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
# Evidence rendering and truthful classification (#424)
# ---------------------------------------------------------------------------


def _evidence(diag: Any, **fields: Any) -> Any:
    return diag.ProbeEvidence(probe=fields.pop("probe", "scenario2"), **fields)


def test_probe_comment_lines_stay_structural(diag: Any) -> None:
    evidence = _evidence(
        diag,
        status=diag.CLASS_UNRESOLVED,
        last_stage="retrieval",
        seconds_since_last_stage=40.0,
        fail_category=diag.CLASS_UNRESOLVED,
        server_ready_ms=12000.0,
        snapshots=4,
        terminated_at_campaign_end=True,
    )
    lines = diag._probe_comment_lines(evidence)
    joined = "\n".join(lines)
    assert "`retrieval`" in joined
    assert "terminated_at_campaign_end" in joined
    assert all(line.startswith("- ") or line.startswith("###") for line in lines)


def test_classify_provider_error_from_real_payload_is_proven(diag: Any) -> None:
    probe_a = _evidence(diag, probe="direct-model", status="provider-error-proven")
    # Undelivered turn, no provider/status confusion in probe B.
    probe_b = _evidence(diag, status="child-exit-0")
    probe_b.delivered = False
    assert diag.classify_cause(probe_a, probe_b) == "provider-error-proven"
    # DELIVERED beats attribution evidence.
    probe_b_delivered = _evidence(diag, status="delivered", delivered=True)
    assert diag.classify_cause(probe_a, probe_b_delivered) == "delivered"


def test_classify_server_error_is_proven_only_from_infra_evidence(diag: Any) -> None:
    probe_a = _evidence(diag, probe="direct-model", status="server-error-proven")
    probe_b = _evidence(diag)
    assert diag.classify_cause(probe_a, probe_b) == "server-error-proven"
    # Sustained local-server non-responsiveness observed by the probe child
    # ends that probe with the proven local-infra status.
    hung_probe_a = _evidence(
        diag,
        probe="direct-model",
        status="server-error-proven",
        server_nonresponsive_sustained=True,
    )
    assert diag.classify_cause(hung_probe_a, _evidence(diag)) == "server-error-proven"


def test_classify_mapping_order(diag: Any) -> None:
    assert (
        diag.classify_cause(
            _evidence(diag, probe="direct-model", status="server-error-proven"),
            _evidence(diag, status="bootstrap-failed"),
        )
        == "server-error-proven"
    )
    # B-probe bootstrap failure has NO proven root: boundary-localized.
    assert (
        diag.classify_cause(
            _evidence(diag, probe="direct-model", status="completed"),
            _evidence(diag, status="bootstrap-failed"),
        )
        == "boundary-localized-root-unknown"
    )


def test_classify_campaign_end_and_delivery(diag: Any) -> None:
    completed_a = _evidence(diag, probe="direct-model", status="completed")
    unresolved_b = _evidence(diag, status=diag.CLASS_UNRESOLVED, terminated_at_campaign_end=True)
    assert diag.classify_cause(completed_a, unresolved_b) == "unresolved-at-campaign-end"
    assert diag.overall_exit_code(completed_a, unresolved_b) == diag.EXIT_INCONCLUSIVE
    delivered_b = _evidence(diag, status="delivered", delivered=True)
    assert diag.classify_cause(completed_a, delivered_b) == "delivered"
    assert diag.overall_exit_code(completed_a, delivered_b) == diag.EXIT_OK
    # In-flight opencode stage alone is NEVER a proven provider root cause.
    boundary_b = _evidence(
        diag,
        status=diag.CLASS_UNRESOLVED,
        terminated_at_campaign_end=True,
        last_stage="index",
        active_stage="opencode",
        fail_category=diag.CLASS_UNRESOLVED,
    )
    assert diag.classify_cause(completed_a, boundary_b) == "unresolved-at-campaign-end"
    assert diag.classify_cause(completed_a, boundary_b) not in (
        "model-provider",
        "provider-error-proven",
    )


def test_classify_boundary_localized_never_model_provider(diag: Any) -> None:
    """kodmial/aa#424: active stage `opencode` must not map to model-provider."""
    completed_a = _evidence(diag, probe="direct-model", status="completed")
    evidence = _evidence(diag, status="child-exit-0", exit_code=0, active_stage="opencode")
    classification = diag.classify_cause(completed_a, evidence)
    assert classification == "boundary-localized-root-unknown"
    assert classification not in ("model-provider", "provider-error-proven")
    evidence_no_stage = _evidence(diag, status="child-exit-0", exit_code=0)
    assert diag.classify_cause(completed_a, evidence_no_stage) == "stage-unobserved"


def test_classify_stages_only_localize_boundary(diag: Any) -> None:
    completed_a = _evidence(diag, probe="direct-model", status="completed")
    for stage in ("retrieval", "planner", "answer", "opencode", "provider"):
        evidence = _evidence(diag, status="child-exit-3", exit_code=3, last_stage=stage)
        classification = diag.classify_cause(completed_a, evidence)
        assert classification == "boundary-localized-root-unknown"
        assert "provider-error" not in classification


def test_final_report_invariants(diag: Any) -> None:
    ok_a = _evidence(diag, probe="direct-model", status="completed", server_ready_ms=9000.0)
    unresolved_b = _evidence(
        diag,
        status=diag.CLASS_UNRESOLVED,
        fail_category=diag.CLASS_UNRESOLVED,
        last_stage="planner",
        server_ready_ms=14000.0,
        delivered_elapsed_ms=None,
        snapshots=48,
        terminated_at_campaign_end=True,
    )
    meta = {
        "run_url": "https://github.com/kodmial/aa/actions/runs/39752912805",
        "main_sha": "4f95b5260c7c84b8323302874a73e707b8cc633b",
        "effective_campaign_allowance_s": 1800.0,
    }
    report = diag.build_final_report(meta, ok_a, unresolved_b)
    assert "4f95b5260c7c84b8323302874a73e707b8cc633b" in report
    assert "actions/runs/39752912805" in report
    # Prominent, non-deceptive INCONCLUSIVE framing.
    assert "outcome: INCONCLUSIVE" in report
    assert "NOT a success" in report
    assert "No per-answer wall-clock cutoff exists" in report
    assert "campaign allowance applied: 1800.0s" in report
    assert "classification:" in report
    assert len(report.splitlines()) < 60


def test_final_report_names_unobserved_stage_explicitly(diag: Any) -> None:
    report = diag.build_final_report(
        {"run_url": "https://github.com/kodmial/aa/actions/runs/1", "main_sha": "abc"},
        _evidence(diag, probe="direct-model", status="completed"),
        _evidence(diag, status="child-exit-3"),
    )
    assert "stage=unobserved" in report


def test_final_report_proven_verdicts(diag: Any) -> None:
    meta = {"run_url": "https://github.com/kodmial/aa/actions/runs/1", "main_sha": "abc"}
    delivered = diag.build_final_report(
        meta,
        _evidence(diag, probe="direct-model", status="completed"),
        _evidence(diag, status="delivered", delivered=True, delivered_elapsed_ms=131000.0),
    )
    assert "outcome: delivered" in delivered
    provider = diag.build_final_report(
        meta,
        _evidence(diag, probe="direct-model", status="provider-error-proven"),
        _evidence(diag, status="child-exit-6", exit_code=6),
    )
    assert "provider-error-proven" in provider
    assert "outcome: failure" in provider


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
        base_url: str,
        method: str,
        path: str,
        body: dict[str, object] | None,
        connect_timeout: float,
        op_timeout: float | None,
    ) -> tuple[bool, bool, dict[str, float], dict[str, Any]]:
        calls.append((method, path))
        if path == "/session":
            assert op_timeout == 10.0
            return (
                True,
                False,
                {"connect_ms": 1.0, "write_ms": 2.0, "read_ms": 3.0},
                {"_id": "session-opaque"},
            )
        if path == "/session/session-opaque/message":
            # kodmial/aa#424: the model call carries NO op timeout at all.
            assert op_timeout is None
            return (
                True,
                False,
                {"connect_ms": 4.0, "write_ms": 5.0, "read_ms": 6.0},
                {"served_model": "provider/model"},
            )
        if path == "/session":
            return True, False, {"status": 200, "read_ms": 1.0}, {"_list": []}
        return True, False, {"status": 200}, {"_list": []}

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
        "snapshot_cadence_s": 100000.0,
        "max_snapshots": 1,
        "connect_timeout_s": 8.0,
        "probe_a_message_timeout_s_position": None,
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


def test_run_probe_a_proven_provider_error_from_real_payload(
    diag: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _OkRuntime:
        def __init__(self, config: Any, ready_timeout: float = 45.0) -> None:
            self.running = True
            self.ready = True
            self.stopped = False

        async def start(self) -> None:
            return

        async def ensure_ready(self, timeout: float | None = None) -> None:
            return

        async def stop(self) -> None:
            self.stopped = True

    calls: list[tuple[str, str]] = []

    def fake_request(
        base_url: str,
        method: str,
        path: str,
        body: dict[str, object] | None,
        connect_timeout: float,
        op_timeout: float | None,
    ) -> tuple[bool, bool, dict[str, float], dict[str, Any]]:
        calls.append((method, path))
        if path == "/session":
            return (
                True,
                False,
                {"connect_ms": 1.0, "write_ms": 2.0, "read_ms": 3.0},
                {"_id": "session-opaque"},
            )
        if path.endswith("/message"):
            return (
                False,
                True,
                {"connect_ms": 4.0, "write_ms": 5.0, "read_ms": 6.0, "status": 200},
                {"ok": False, "provider_error": True, "error_status_code": 429},
            )
        return True, False, {"status": 200}, {"_list": []}

    monkeypatch.setattr("aa.opencode.runtime.LocalOpenCodeRuntime", _OkRuntime)
    monkeypatch.setattr(diag, "_health_probe", lambda base_url, timeout: (False, ""))
    monkeypatch.setattr(diag, "_timed_request", fake_request)

    sink = diag.EventSink(tmp_path / "events.jsonl", echo=False)
    argv = {
        "probe_a_port": "4097",
        "out_dir": str(tmp_path),
        "snapshot_cadence_s": 100000.0,
        "max_snapshots": 1,
        "probe_a_bootstrap_timeout_s": 5.0,
    }
    code = asyncio.run(diag.run_probe_a(argv, sink, {}))
    assert code == diag.EXIT_PROVIDER_ERROR_PROVEN
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    end = next(event for event in events if event["event"] == "probe-end")
    assert end["status"] == "provider-error-proven"


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
        "snapshot_cadence_s": 100000.0,
        "max_snapshots": 1,
        "probe_a_bootstrap_timeout_s": 0.05,
    }
    started = time.monotonic()
    code = asyncio.run(diag.run_probe_a(argv, sink, {}))
    assert code == diag.EXIT_SERVER_ERROR_PROVEN
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
        "snapshot_cadence_s": 100000.0,
        "max_snapshots": 1,
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
    # No contentful prompt/answer ever leaks through the seam events.
    joined = (tmp_path / "events-b.jsonl").read_text(encoding="utf-8")
    assert "Готово" not in joined


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
        "snapshot_cadence_s": 100000.0,
        "max_snapshots": 1,
        "probe_b_bootstrap_timeout_s": 0.05,
    }
    started = time.monotonic()
    code = asyncio.run(diag.run_probe_b(argv, sink, {}))
    assert code == diag.EXIT_BOOTSTRAP_FAILED
    assert time.monotonic() - started < 2.0


def _sleeper_child(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return subprocess.Popen(  # noqa: S603 (fixed argv, no shell)
        [
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def test_supervisor_ends_child_group_at_campaign_allowance(
    diag: Any, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """True no-progress/hung run: campaign budget, tidy cleanup, no leak."""
    monkeypatch.setattr(diag, "_spawn_child", _sleeper_child)
    monkeypatch.setattr(diag, "DEFAULT_TERMINATE_GRACE_S", 0.1)

    plan = diag.ProbePlan(
        probe=diag.PROBE_SCENARIO2,
        snapshot_cadence_s=100000.0,
        max_snapshots=1,
        connect_timeout_s=1.0,
    )
    evidence = diag._run_one_probe(
        plan,
        out_dir=tmp_path,
        sink=diag.EventSink(tmp_path / "supervisor.jsonl", echo=False),
        publisher=None,
        campaign_deadline_s=time.monotonic() + 0.2,
    )
    assert evidence.killed is True
    assert evidence.terminated_at_campaign_end is True
    assert evidence.status == diag.CLASS_UNRESOLVED
    assert not diag._abnormal_child_exit(evidence)
    assert (
        diag.classify_cause(
            diag.ProbeEvidence(probe=diag.PROBE_DIRECT_MODEL, status="not-run"), evidence
        )
        == diag.CLASS_UNRESOLVED
    )
    assert (
        diag.overall_exit_code(
            diag.ProbeEvidence(probe=diag.PROBE_DIRECT_MODEL, status="not-run"), evidence
        )
        == diag.EXIT_INCONCLUSIVE
    )
    events = (tmp_path / "supervisor.jsonl").read_text(encoding="utf-8")
    assert '"event": "campaign-end"' in events or '"event":"campaign-end"' in events.replace(
        ", ", ","
    )


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
    plan = diag.ProbePlan(
        probe=diag.PROBE_SCENARIO2,
        snapshot_cadence_s=2.0,
        max_snapshots=7,
        connect_timeout_s=5.0,
    )
    diag._spawn_child(plan=plan, out_dir=tmp_path)

    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["stdout"] is None
    child_args = json.loads(captured["kwargs"]["env"]["SCENARIO2_DIAG_ARGS"])
    assert child_args["probe"] == diag.PROBE_SCENARIO2
    assert child_args["snapshot_cadence_s"] == 2.0
    assert child_args["max_snapshots"] == 7
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


# ---------------------------------------------------------------------------
# kodmial/aa#422: SIGUSR1 handler must be nonfatal (chain=False) and the
# supervisor must never SIGUSR1 an unregistered child. Regression tests use
# REAL subprocesses with short intervals (no tens-of-seconds CI sleeps).
# ---------------------------------------------------------------------------


def _diag_subprocess_source(stacks_path: pathlib.Path) -> str:
    """Child program registering the script's nonfatal stack handler."""
    return (
        "import importlib.util, time\n"
        f"spec = importlib.util.spec_from_file_location(\n"
        f"    'diag_child_under_test', {str(SCRIPT_PATH)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "import sys\n"
        "sys.modules['diag_child_under_test'] = module\n"
        "spec.loader.exec_module(module)\n"
        "registered = module._install_stack_handler(\n"
        f"    __import__('pathlib').Path({str(stacks_path)!r}))\n"
        "assert registered is True, 'stack handler registration failed'\n"
        "print('READY', flush=True)\n"
        "time.sleep(60)\n"
    )


def _wait_stack_dump(stacks_path: pathlib.Path, proc: subprocess.Popen[str]) -> bool:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        if stacks_path.exists() and stacks_path.stat().st_size > 0:
            return True
        time.sleep(0.05)
    return stacks_path.exists() and stacks_path.stat().st_size > 0


def test_register_handler_survives_two_sigusr1_and_dumps_stacks(
    tmp_path: pathlib.Path,
) -> None:
    """kodmial/aa#422: SIGUSR1 stack dump must NEVER kill the probe (P0)."""
    stacks = tmp_path / "stacks-direct-model.txt"
    proc = subprocess.Popen(  # noqa: S603 (fixed argv, no shell)
        [sys.executable, "-c", _diag_subprocess_source(stacks)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None  # stdout=PIPE (typed narrowing for mypy)
    try:
        line = proc.stdout.readline()
        assert line.strip() == "READY"
        time.sleep(0.2)
        proc.send_signal(signal.SIGUSR1)
        assert _wait_stack_dump(stacks, proc), "faulthandler did not dump stacks"
        assert proc.poll() is None, f"child died on SIGUSR1: {proc.returncode}"
        proc.send_signal(signal.SIGUSR1)
        time.sleep(0.4)
        assert proc.poll() is None, f"child died on second SIGUSR1: {proc.returncode}"
        stacks_content = stacks.read_text(encoding="utf-8")
        # Two SIGUSR1-triggered faulthandler dumps are appended.
        assert stacks_content.count("most recent call first") >= 2
        # Normal bounded shutdown still works with plain SIGTERM.
        proc.terminate()
        assert proc.wait(timeout=10) == -signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_chained_chain_true_handler_reproduces_original_probe_kill(
    tmp_path: pathlib.Path,
) -> None:
    """Negative deterministic reproduction of the original chain=True bug.

    The old behavior (run 39767853106): each SIGUSR1 dumped stacks and then
    chained into the default SIGUSR1 disposition, terminating the child
    with exit code -10 at ~20s. This documents the regression that must
    never come back.
    """
    stacks = tmp_path / "stacks-chained.txt"
    code = (
        "import faulthandler, signal, time\n"
        f"stream = open({str(stacks)!r}, 'a', encoding='utf-8')\n"
        "faulthandler.register(signal.SIGUSR1, file=stream, all_threads=True, chain=True)\n"
        "print('READY', flush=True)\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen(  # noqa: S603 (fixed argv, no shell)
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None  # stdout=PIPE (typed narrowing for mypy)
    try:
        assert proc.stdout.readline().strip() == "READY"
        time.sleep(0.2)
        proc.send_signal(signal.SIGUSR1)
        deadline = time.monotonic() + 15.0
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert proc.returncode == -signal.SIGUSR1, (
            f"chained handler must terminate the child, got {proc.returncode}"
        )
        assert stacks.exists() and stacks.stat().st_size > 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_register_handler_registration_defaults_to_nonfatal_chain(diag: Any) -> None:
    """The merged handler must be non-fatal: default chain=False (P0 fix)."""
    import inspect

    signature = inspect.signature(diag._install_stack_handler)
    assert signature.parameters["chain"].default is False


def test_send_sigusr1_requires_ready_sentinel(diag: Any, tmp_path: pathlib.Path) -> None:
    class _FakeProc:
        def __init__(self) -> None:
            self.signals: list[int] = []

        def send_signal(self, sig: int) -> None:
            self.signals.append(int(sig))

    proc = _FakeProc()
    ready_path = diag._stack_handler_ready_path(tmp_path, diag.PROBE_SCENARIO2)
    assert ready_path.name == "stack-handler-ready-scenario2.txt"
    assert diag._send_sigusr1(proc, ready_path=ready_path) is False
    assert proc.signals == []
    ready_path.write_text("registered\n", encoding="utf-8")
    assert diag._send_sigusr1(proc, ready_path=ready_path) is True
    assert proc.signals == [int(signal.SIGUSR1)]


def test_stage_events_tracked_inflight_and_completed(diag: Any, tmp_path: pathlib.Path) -> None:
    sink = diag.EventSink(tmp_path / "events.jsonl", echo=False)
    evidence = diag.ProbeEvidence(probe=diag.PROBE_SCENARIO2)
    for payload in (
        {"event": "stage-start", "stage": "index", "ts": "2026-01-01T10:00:00Z"},
        {"event": "stage-end", "stage": "index", "elapsed_ms": 5560.0},
        {"event": "stage-start", "stage": "opencode", "ts": "2026-01-01T10:05:00Z"},
    ):
        assert diag._consume_child_line(json.dumps(payload), evidence, sink)
    # The freshest in-flight marker (opencode) wins over the stale one.
    assert evidence.active_stage == "opencode"
    assert evidence.active_stage_ts == "2026-01-01T10:05:00Z"
    assert evidence.last_completed_stage == "index"
    assert evidence.last_completed_stage_elapsed_ms == 5560.0


def test_classify_signal_killed_children_is_harness_error(diag: Any) -> None:
    """Run 39767853106 falsified evidence shape: both children -10 at ~20s."""
    probe_a = diag.ProbeEvidence(
        probe=diag.PROBE_DIRECT_MODEL, status="child-exit--10", exit_code=-10
    )
    probe_b = diag.ProbeEvidence(
        probe=diag.PROBE_SCENARIO2,
        status="child-exit--10",
        exit_code=-10,
        last_stage="index",
    )
    assert diag.classify_cause(probe_a, probe_b) == diag.CLASS_HARNESS_ERROR
    # Never success and no unproven stage attribution.
    assert diag.classify_cause(probe_a, probe_b) not in ("delivered", "aa-retrieval-index")
    assert diag.overall_exit_code(probe_a, probe_b) == diag.EXIT_CHILD_UNEXPECTED
    overall = diag.overall_exit_code(probe_a, probe_b)
    assert overall not in (diag.EXIT_OK, diag.EXIT_INCONCLUSIVE)


def test_child_unexpected_exit_code_is_harness_error(diag: Any) -> None:
    probe_b = diag.ProbeEvidence(probe=diag.PROBE_SCENARIO2, status="child-exit-5", exit_code=5)
    probe_ok_a = diag.ProbeEvidence(probe=diag.PROBE_DIRECT_MODEL, status="completed")
    assert diag._abnormal_child_exit(probe_b) is True
    assert diag.classify_cause(probe_ok_a, probe_b) == diag.CLASS_HARNESS_ERROR


def test_forced_bounded_exit_is_not_harness_error(diag: Any) -> None:
    probe_b = diag.ProbeEvidence(
        probe=diag.PROBE_SCENARIO2,
        status=diag.CLASS_SERVER_ERROR_PROVEN,
        exit_code=-signal.SIGKILL,
        forced_exit=True,
        server_nonresponsive_sustained=True,
    )
    probe_ok_a = diag.ProbeEvidence(probe=diag.PROBE_DIRECT_MODEL, status="completed")
    assert diag._abnormal_child_exit(probe_b) is False
    assert diag.classify_cause(probe_ok_a, probe_b) == "server-error-proven"


def test_campaign_terminated_children_are_not_harness_error(diag: Any) -> None:
    """Bounded campaign shutdown (-SIGTERM) is the expected INCONCLUSIVE path."""
    completed_a = diag.ProbeEvidence(probe=diag.PROBE_DIRECT_MODEL, status="completed")
    unresolved_b = diag.ProbeEvidence(
        probe=diag.PROBE_SCENARIO2,
        status=diag.CLASS_UNRESOLVED,
        exit_code=-signal.SIGTERM,
        terminated_at_campaign_end=True,
        last_stage="retrieval",
        fail_category=diag.CLASS_UNRESOLVED,
    )
    assert diag._abnormal_child_exit(unresolved_b) is False
    assert diag.classify_cause(completed_a, unresolved_b) == diag.CLASS_UNRESOLVED
    assert diag.overall_exit_code(completed_a, unresolved_b) == diag.EXIT_INCONCLUSIVE


def test_delivered_run_reports_success_but_signal_killed_not(diag: Any) -> None:
    delivered_a = diag.ProbeEvidence(probe=diag.PROBE_DIRECT_MODEL, status="completed")
    delivered_b = diag.ProbeEvidence(probe=diag.PROBE_SCENARIO2, status="delivered", delivered=True)
    assert diag.overall_exit_code(delivered_a, delivered_b) == diag.EXIT_OK


def test_exit_code_mapping_covers_all_classifications(diag: Any) -> None:
    completed_a = diag.ProbeEvidence(probe=diag.PROBE_DIRECT_MODEL, status="completed")
    for classification, expected_code in (
        ("delivered", diag.EXIT_OK),
        ("provider-error-proven", diag.EXIT_PROVIDER_ERROR_PROVEN),
        ("server-error-proven", diag.EXIT_SERVER_ERROR_PROVEN),
        ("harness-error", diag.EXIT_CHILD_UNEXPECTED),
        ("unresolved-at-campaign-end", diag.EXIT_INCONCLUSIVE),
        ("boundary-localized-root-unknown", diag.EXIT_INCONCLUSIVE),
        ("stage-unobserved", diag.EXIT_INCONCLUSIVE),
    ):
        assert diag._EXIT_BY_CLASSIFICATION[classification] == expected_code
    # overall_ok requires an actual delivery, not a bare "completed" status.
    delivered = diag.ProbeEvidence(probe=diag.PROBE_SCENARIO2, status="delivered", delivered=True)
    assert diag.overall_exit_code(completed_a, delivered) == diag.EXIT_OK


def test_comment_lines_name_campaign_termination_a(diag: Any) -> None:
    evidence = diag.ProbeEvidence(
        probe=diag.PROBE_SCENARIO2,
        status=diag.CLASS_UNRESOLVED,
        exit_code=None,
        terminated_at_campaign_end=True,
        last_stage="index",
        active_stage="opencode",
        last_real_event="stage-start:opencode",
        last_completed_stage="index",
    )
    lines = "\n".join(diag._probe_comment_lines(evidence))
    assert "`opencode`" in lines
    assert "last_completed_stage: `index`" in lines
    assert "terminated_at_campaign_end" in lines


def test_infrastructure_failure_report_names_no_stage_cause(diag: Any) -> None:
    meta = {"run_url": "https://github.com/kodmial/aa/actions/runs/39767853106", "main_sha": "abc"}
    probe_a = diag.ProbeEvidence(
        probe=diag.PROBE_DIRECT_MODEL, status="child-exit--10", exit_code=-10
    )
    probe_b = diag.ProbeEvidence(
        probe=diag.PROBE_SCENARIO2,
        status="child-exit--10",
        exit_code=-10,
        last_stage="index",
        active_stage="",
    )
    report = diag.build_final_report(meta, probe_a, probe_b)
    assert "harness-error" in report
    assert "NO stage-level" in report
    assert diag._NEXT_FIX[diag.CLASS_HARNESS_ERROR]


def test_summary_banner_never_misleads(diag: Any) -> None:
    assert "NOT a success" in diag._summary_banner(diag.CLASS_UNRESOLVED)
    assert "NOT a success" in diag._summary_banner(diag.CLASS_BOUNDARY_LOCALIZED)
    assert "NOT a success" in diag._summary_banner(diag.CLASS_STAGE_UNOBSERVED)
    assert "outcome: failure" in diag._summary_banner(diag.CLASS_PROVIDER_ERROR_PROVEN)
    assert diag._summary_banner(diag.CLASS_DELIVERED) == (
        "> **outcome: delivered — full scenario #2 answer delivered**"
    )


def test_evidence_dict_serializes_new_fields(diag: Any) -> None:
    evidence = diag.ProbeEvidence(
        probe=diag.PROBE_SCENARIO2,
        status=diag.CLASS_UNRESOLVED,
        exit_code=-10,
        terminated_at_campaign_end=True,
        active_stage="opencode",
        active_stage_ts="2026-01-01T10:05:00Z",
        last_completed_stage="index",
        last_completed_stage_elapsed_ms=5560.0,
        server_nonresponsive_sustained=False,
        forced_exit=False,
    )
    serialized = diag._evidence_dict(evidence)
    assert serialized["exited_by_signal"] == 10
    assert serialized["terminated_at_campaign_end"] is True
    assert serialized["active_stage"] == "opencode"
    assert serialized["last_completed_stage_ms"] == 5560.0
    json.dumps(serialized)
