"""kodmial/aa#425 DoD tests: scenario #2 diagnostic triage + recovery.

Proves the AA-owned, deterministic, GitHub-only post-run controller for
the single Gate C scenario #2 diagnostic (kodmial/aa#418/#424):

- truthful typed outcomes (DELIVERED / ROOT_CAUSE_PROVED /
  BOUNDARY_LOCALIZED_CAUSE_UNKNOWN / HARNESS_FAILURE / BOOTSTRAP_BLOCKED /
  INCONCLUSIVE / RUN_MISSING) with synthetic fixtures;
- a normal slow (>130/300/900 s) but completing turn is reported as
  NORMAL OBSERVED SLOW WORK, never a hang, and never as a root cause;
- evidence cannot fabricate a PASS: corrupted/sensitive artifacts,
  missing/expired artifacts and identity mismatch all fail closed to
  HARNESS_FAILURE evidence codes;
- exactly one idempotent triage comment per (sha, run) on issue #419, one
  repair issue per proven instrumentation fingerprint, and one
  bounded-careful diagnostic dispatch (``target_sha`` only) for a missing
  run; workflow_run + cron + dispatch overlaps reconcile idempotently.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from aa.control import scenario2_triage as triage

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "reconcile_scenario2_diagnostic.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "aa-scenario2-diagnostic-reconcile.yml"
DIAG_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "aa-gate-c-scenario2-diagnostic.yml"

SHA_MAIN = "b" * 40
SHA_424 = "c" * 40
RUN_ID = 39824142258


@pytest.fixture(scope="module", name="reconciler")
def reconciler_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "reconcile_scenario2_diagnostic", str(SCRIPT_PATH)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["reconcile_scenario2_diagnostic"] = module
    spec.loader.exec_module(module)
    return module


def _run(
    run_id: int = RUN_ID,
    sha: str = SHA_424,
    status: str = "completed",
    conclusion: str | None = "success",
    created: str = "2026-10-22T17:36:09Z",
    updated: str = "2026-10-22T17:55:30Z",
) -> triage.DiagnosticRun:
    return triage.DiagnosticRun(
        run_id=run_id,
        head_sha=sha,
        event="push",
        status=status,
        conclusion=conclusion,
        created_at=created,
        updated_at=updated,
        run_number=3,
        html_url=f"https://github.com/kodmial/aa/actions/runs/{run_id}",
    )


def _event(
    event: str,
    mono: float,
    fields: dict[str, Any] | None = None,
    *,
    ts: str = "2026-10-22T17:40:20.123456+00:00",
) -> dict[str, Any]:
    payload: dict[str, Any] = {"ts": ts, "mono": mono, "event": event}
    payload.update(fields or {})
    return payload


def _jsonl(events: list[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(e) + "\n").encode("utf-8") for e in events)


def _scenario_events(with_delivery: bool = False) -> list[dict[str, Any]]:
    """Realistic sanitized probe-scenario2 child event stream."""
    events = [
        _event("probe-start", 0.0, {"probe": "scenario2", "family": "substantive-drinking"}),
        _event("bootstrap-start", 0.01, {"probe": "scenario2"}),
        _event("bootstrap-end", 2.4, {"ok": True, "elapsed_ms": 2380.0, "opencode_ready": True}),
        _event("scenario-start", 2.5, {"family": "substantive-drinking"}),
        _event("stage-start", 2.6, {"stage": "index"}),
        _event("stage-end", 5.2, {"stage": "index", "elapsed_ms": 2600.0}),
        _event("stage-start", 5.3, {"stage": "retrieval"}),
        _event("stage-end", 14.9, {"stage": "retrieval", "elapsed_ms": 9600.0}),
        _event("stage-start", 18.2, {"stage": "opencode"}),
    ]
    if with_delivery:
        events.extend(
            [
                _event("model-send-start", 18.3, {"ok": False}),
                _event(
                    "delivery",
                    158.3,
                    {"elapsed_ms": 155800.0, "reply_len": 412, "typing_sends": 9},
                ),
                _event("stage-end", 158.9, {"stage": "opencode", "elapsed_ms": 140700.0}),
                _event("probe-end", 159.0, {"probe": "scenario2", "status": "delivered"}),
            ]
        )
    return events


def _direct_events() -> list[dict[str, Any]]:
    return [
        _event("probe-start", 0.0, {"probe": "direct-model"}),
        _event("server-health", 2.0, {"ok": True, "elapsed_ms": 1980.0, "attached": False}),
        _event("session-create-start", 2.1, {"probe": "direct-model"}),
        _event(
            "session-create-end",
            3.4,
            {"ok": True, "connect_ms": 12.0, "read_ms": 1100.0, "status": 200},
        ),
        _event("model-send-start", 3.5, {"model": "gpt-5.1", "read_inactivity_s": 1800.0}),
        _event("model-send-end", 6.6, {"ok": True, "elapsed_ms": 3100.0, "status": 200}),
    ]


def _result(
    *,
    classification: str = "delivered",
    outcome: str = "delivered",
    delivered: bool = True,
    delivered_ms: float = 158300.0,
    direct_model_ms: float = 6500.0,
    sha: str = SHA_MAIN,
    run_id: int = RUN_ID,
) -> dict[str, Any]:
    return {
        "schema": triage.DIAGNOSTIC_SCHEMA_VERSION,
        "run_id": run_id,
        "main_sha": sha,
        "run_url": f"https://github.com/kodmial/aa/actions/runs/{run_id}",
        "target_issue": 418,
        "classification": classification,
        "outcome": outcome,
        "exit_code": 0,
        "probe_direct_model": {
            "probe": "direct-model",
            "status": "completed",
            "exit_code": 0,
            "delivered": True,
            "delivered_elapsed_ms": None,
            "model_request_total_ms": direct_model_ms,
        },
        "probe_scenario2": {
            "probe": "scenario2",
            "status": "delivered" if delivered else "running",
            "exit_code": 0,
            "delivered": delivered,
            "delivered_elapsed_ms": delivered_ms,
        },
    }


def _write_artifact(
    tmp_path: Path,
    result_bytes: bytes,
    *,
    scenario_events: bytes = b"",
    supervisor_events: bytes = b"",
    direct_events: bytes = b"",
    include_members: tuple[str, ...] | None = None,
) -> Path:
    members: dict[str, bytes] = {
        triage.REQUIRED_RESULT_MEMBER: result_bytes,
        triage.REQUIRED_SUPERVISOR_MEMBER: supervisor_events,
        triage.REQUIRED_SCENARIO_EVENTS_MEMBER: scenario_events,
        "probe-direct-model-events.jsonl": direct_events,
        triage.OPTIONAL_REPORT_MEMBER: (
            "> **outcome: delivered — full scenario #2 answer delivered**\n".encode()
        ),
    }
    if include_members is not None:
        members = {name: members[name] for name in include_members if name in members}
    path = tmp_path / f"{triage.artifact_name_for_run(RUN_ID)}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


# ---------------------------------------------------------------------------
# 1. Synthetic DELIVERED + normal slow work
# ---------------------------------------------------------------------------


def test_delivered_result_maps_to_typed_delivered() -> None:
    outcome = triage.map_typed_outcome(_result())
    assert outcome.outcome == triage.OUTCOME_DELIVERED
    assert outcome.is_success is True
    assert triage.product_gate_fulfilled(outcome.outcome) is True


@pytest.mark.parametrize(
    "delivered_ms",
    [130_500.0, 300_400.0, 900_600.0, 1_200_000.0],
)
def test_slow_completing_turn_is_normal_slow_work_not_hang(delivered_ms: float) -> None:
    """A >130/300/900 s slow-but-real-completing turn is never an anomaly."""
    crossings = triage.slow_work_crossings(_result(delivered_ms=delivered_ms))
    assert crossings, "slow delivery must be recorded as evidence"
    joined = "\n".join(crossings)
    assert "NORMAL OBSERVED SLOW WORK, NOT a provider hang" in joined
    # Slow work must never fabricate a root cause or downgrade DELIVERED.
    outcome = triage.map_typed_outcome(_result(delivered_ms=delivered_ms))
    assert outcome.outcome == triage.OUTCOME_DELIVERED
    comment = triage.build_triage_comment(
        run=_run(sha=SHA_MAIN),
        artifact_name=triage.artifact_name_for_run(RUN_ID),
        triage=outcome,
        timeline_lines=[],
        crossings=crossings,
    )
    assert comment.count("NORMAL OBSERVED SLOW WORK") >= 1
    # No fabricated attribution beyond the recorded facts.
    assert comment.count("PROVEN") <= 1  # only the typed-outcome banner at most


def test_short_fast_delivery_has_no_slow_note() -> None:
    assert triage.slow_work_crossings(_result(delivered_ms=20000.0)) == ()


# ---------------------------------------------------------------------------
# 2. Truthful inconclusive / boundary outcomes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("classification", "expected"),
    [
        ("unresolved-at-campaign-end", triage.OUTCOME_INCONCLUSIVE),
        (
            "boundary-localized-root-unknown",
            triage.OUTCOME_BOUNDARY_LOCALIZED_CAUSE_UNKNOWN,
        ),
        ("stage-unobserved", triage.OUTCOME_BOUNDARY_LOCALIZED_CAUSE_UNKNOWN),
    ],
)
def test_inconclusive_families_never_count_as_done(classification: str, expected: str) -> None:
    outcome = triage.map_typed_outcome(
        _result(classification=classification, outcome="inconclusive")
    )
    assert outcome.outcome == expected
    # A green Actions workflow plus an inconclusive result is NOT done.
    assert triage.product_gate_fulfilled(outcome.outcome) is False
    assert triage.is_done(outcome.outcome) is False


def test_boundary_vs_root_cause_separation_is_explicit() -> None:
    boundary = triage.map_typed_outcome(_result(classification="boundary-localized-root-unknown"))
    comment = triage.build_triage_comment(
        run=_run(),
        artifact_name=triage.artifact_name_for_run(RUN_ID),
        triage=boundary,
        timeline_lines=[],
        crossings=(),
    )
    assert "BOUNDARY LOCALIZED" in comment
    assert "no component" in comment
    # Root cause requires concrete proof from the provider layer.
    assert (
        triage.map_typed_outcome(
            _result(classification="provider-error-proven", outcome="failure")
        ).outcome
        == triage.OUTCOME_ROOT_CAUSE_PROVED
    )


def test_provider_429_is_root_cause_proved() -> None:
    result = _result(classification="provider-error-proven", outcome="failure")
    result["probe_direct_model"]["status"] = "provider-error-proven"
    result["probe_direct_model"]["fail_category"] = "http-429"
    outcome = triage.map_typed_outcome(result)
    assert outcome.outcome == triage.OUTCOME_ROOT_CAUSE_PROVED
    comment = triage.build_triage_comment(
        run=_run(),
        artifact_name=triage.artifact_name_for_run(RUN_ID),
        triage=outcome,
        timeline_lines=[],
        crossings=(),
    )
    assert "concrete error" in comment


def test_bootstrap_blocked_outcome() -> None:
    result = _result()
    result["classification"] = ""
    result["probe_scenario2"]["status"] = "bootstrap-failed"
    outcome = triage.map_typed_outcome(result)
    assert outcome.outcome == triage.OUTCOME_BOOTSTRAP_BLOCKED


def test_unknown_classification_refuses_attribution() -> None:
    outcome = triage.map_typed_outcome(_result(classification="mystery-state"))
    assert outcome.outcome == triage.OUTCOME_INCONCLUSIVE
    assert "refuses attribution" in outcome.detail


# ---------------------------------------------------------------------------
# 3. Evidence integrity: missing/corrupt/sensitive artifacts
# ---------------------------------------------------------------------------


def test_missing_artifact_classifies_blocked_never_pass() -> None:
    outcome = triage.map_typed_outcome(None, blocked_code=triage.BLOCK_ARTIFACT_MISSING)
    assert outcome.outcome == triage.OUTCOME_HARNESS_FAILURE
    assert outcome.blocked_code == triage.BLOCK_ARTIFACT_MISSING
    assert triage.product_gate_fulfilled(outcome.outcome) is False
    # A missing-artifact run must spawn exactly one instrumentation issue.
    plan = triage.build_repair_issue_plan(
        triage=outcome,
        run=_run(),
        artifact_name=triage.artifact_name_for_run(RUN_ID),
        fingerprint_key=triage.fingerprint_key_for_triage(outcome, {}),
        existing_fingerprints=set(),
    )
    assert plan.should_create is True
    assert plan.labels == ("priority:p0", "automation:ready")


def test_missing_required_telemetry_is_harness_failure(reconciler: Any, tmp_path: Path) -> None:
    path = _write_artifact(
        tmp_path,
        json.dumps(_result()).encode("utf-8"),
        include_members=(triage.REQUIRED_RESULT_MEMBER, triage.OPTIONAL_REPORT_MEMBER),
    )
    outcome, timeline, gaps, crossings, banner = reconciler.evaluate_artifact(path, _run())
    assert outcome.outcome == triage.OUTCOME_HARNESS_FAILURE
    assert outcome.blocked_code == triage.BLOCK_TELEMETRY_MISSING
    assert timeline == []
    assert crossings == ()
    joined = "\n".join(gaps)
    assert "supervisor-events.jsonl" in joined
    assert banner is not None


def test_corrupt_artifact_fails_closed(reconciler: Any, tmp_path: Path) -> None:
    path = tmp_path / "corrupt.zip"
    path.write_bytes(b"this is not a zip archive")
    outcome, _timeline, gaps, _crossings, _banner = reconciler.evaluate_artifact(path, _run())
    assert outcome.outcome == triage.OUTCOME_HARNESS_FAILURE
    assert outcome.blocked_code == triage.BLOCK_ARTIFACT_CORRUPT
    assert "artifact unreadable" in "\n".join(gaps)


def test_corrupt_result_json_fails_closed(reconciler: Any, tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, b"{ not json at all")
    outcome, _timeline, gaps, _c, _b = reconciler.evaluate_artifact(path, _run())
    assert outcome.blocked_code == triage.BLOCK_ARTIFACT_CORRUPT
    assert "result.json unreadable" in "\n".join(gaps)


def test_artifact_member_path_traversal_rejected(reconciler: Any, tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, json.dumps(_result()).encode("utf-8"))
    with zipfile.ZipFile(path, "w") as archive:
        # Write the unsafe name first: _read_zip_member must reject it even
        # when scanning for the first wanted member.
        archive.writestr("../escape.txt", b"payload")
        archive.writestr(triage.REQUIRED_RESULT_MEMBER, json.dumps(_result()))
        archive.writestr(triage.REQUIRED_SUPERVISOR_MEMBER, b"")
        archive.writestr(triage.REQUIRED_SCENARIO_EVENTS_MEMBER, b"")
    with pytest.raises(ValueError, match="unsafe artifact member path"):
        reconciler.evaluate_artifact(path, _run())


def test_sensitive_content_is_rejected_by_sanitization() -> None:
    for payload in (
        "blob with ghp_0123456789abcdefghij inside",
        "bearer token: gho_ABCDEFGHIJKLMNOPQRSTUV234567",
        "https://api.example.com/v1/endpoint?token=abcdef123456789",
        "81234567:AAEfkEfkABCDEFGHIJKLMNOPQRSTUVWX1234567890ABCDEFGHI",
        "-----BEGIN RSA PRIVATE KEY-----",
    ):
        with pytest.raises(triage.TriageSanitizationError):
            triage.assert_sanitized(payload, owner="t")


def test_identity_mismatch_fails_closed(reconciler: Any, tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, json.dumps(_result(run_id=999)).encode("utf-8"))
    outcome, _timeline, gaps, _crossings, _banner = reconciler.evaluate_artifact(
        path, _run(run_id=RUN_ID)
    )
    assert outcome.blocked_code == triage.BLOCK_IDENTITY_MISMATCH
    assert "result.json run_id" in "\n".join(gaps)


def test_stale_sha_run_never_selected_for_target() -> None:
    stale = _run(sha=SHA_424)
    selection = triage.select_diagnostic_run([stale], allowed_shas=(SHA_MAIN,))
    assert selection.action == "missing"


def test_full_evidence_flow_renders_single_delivered_comment(
    reconciler: Any, tmp_path: Path
) -> None:
    path = _write_artifact(
        tmp_path,
        json.dumps(_result()).encode("utf-8"),
        supervisor_events=_jsonl(
            [
                _event(
                    "supervisor-start",
                    0.0,
                    {"sha": SHA_MAIN, "policy": "no-per-answer-cutoff"},
                )
            ]
        ),
        scenario_events=_jsonl(_scenario_events(with_delivery=True)),
        direct_events=_jsonl(_direct_events()),
    )
    outcome, timeline, gaps, crossings, banner = reconciler.evaluate_artifact(
        path, _run(sha=SHA_MAIN)
    )
    assert outcome.outcome == triage.OUTCOME_DELIVERED
    assert crossings, "slow delivered run must carry the slow-work note"
    comment = triage.build_triage_comment(
        run=_run(sha=SHA_MAIN),
        artifact_name=triage.artifact_name_for_run(RUN_ID),
        triage=outcome,
        timeline_lines=triage.render_timeline(timeline),
        crossings=crossings,
        evidence_gaps=gaps,
    )
    assert "`stage-start`" in comment and "`delivery`" in comment
    assert "stage=opencode" in comment
    assert banner is not None and "delivered" in banner
    assert comment.startswith(triage.triage_marker(SHA_MAIN, RUN_ID))


# ---------------------------------------------------------------------------
# 4. Missing-run dispatch decisions (dedup/backoff/bounded retries)
# ---------------------------------------------------------------------------


def test_missing_run_dispatches_with_target_sha_only() -> None:
    plan = triage.decide_dispatch([], target_sha=SHA_MAIN, markers=[], now_epoch=1_000_000.0)
    assert plan.should_dispatch is True
    assert plan.attempt == 1
    inputs = triage.build_dispatch_inputs(SHA_MAIN)
    assert inputs == {"target_sha": SHA_MAIN}
    assert "issue_number" not in plan.marker_body


def test_active_run_blocks_duplicate_dispatch() -> None:
    active = triage.parse_runs(
        [
            {
                "id": 2,
                "head_sha": SHA_MAIN,
                "event": "workflow_dispatch",
                "status": "in_progress",
                "conclusion": None,
                "created_at": "2026-10-22T18:00:00Z",
                "updated_at": None,
            }
        ]
    )
    plan = triage.decide_dispatch(active, target_sha=SHA_MAIN, markers=[], now_epoch=1_000_000.0)
    assert plan.should_dispatch is False
    assert "already active" in plan.reason


def test_completed_run_blocks_dispatch() -> None:
    plan = triage.decide_dispatch(
        [_run(sha=SHA_MAIN)], target_sha=SHA_MAIN, markers=[], now_epoch=1_000_000.0
    )
    assert plan.should_dispatch is False


def test_dispatch_backoff_then_bounded_retry_cap() -> None:
    markers = [triage.DispatchMarker(SHA_MAIN, 900_000.0, 1)]
    plan = triage.decide_dispatch(
        [], target_sha=SHA_MAIN, markers=markers, now_epoch=900_000.0 + 60.0
    )
    assert plan.should_dispatch is False  # inside the 30-minute backoff
    plan2 = triage.decide_dispatch(
        [], target_sha=SHA_MAIN, markers=markers, now_epoch=900_000.0 + 1900.0
    )
    assert plan2.should_dispatch is True
    assert plan2.attempt == 2
    capped = [
        triage.DispatchMarker(SHA_MAIN, 100.0, 1),
        triage.DispatchMarker(SHA_MAIN, 200.0, 2),
        triage.DispatchMarker(SHA_MAIN, 300.0, 3),
    ]
    plan3 = triage.decide_dispatch([], target_sha=SHA_MAIN, markers=capped, now_epoch=1_000_000.0)
    assert plan3.should_dispatch is False
    assert "bounded retry cap" in plan3.reason


def test_dispatch_markers_unique_shas_dont_interfere() -> None:
    other = [triage.DispatchMarker("a" * 40, 900_000.0, 3)]
    plan = triage.decide_dispatch([], target_sha=SHA_MAIN, markers=other, now_epoch=1_000_000.0)
    assert plan.should_dispatch is True
    assert plan.attempt == 1


# ---------------------------------------------------------------------------
# 5. Single idempotent triage comment per (sha, run)
# ---------------------------------------------------------------------------


def test_triage_comment_idempotent_retries() -> None:
    comment = triage.build_triage_comment(
        run=_run(sha=SHA_MAIN),
        artifact_name=triage.artifact_name_for_run(RUN_ID),
        triage=triage.map_typed_outcome(_result(sha=SHA_MAIN)),
        timeline_lines=[],
        crossings=(),
    )
    existing = [(1, comment)]
    action = triage.decide_triage_comment(existing, sha=SHA_MAIN, run_id=RUN_ID)
    assert action.kind == "update"
    assert action.comment_id == 1
    # Overlapping cron tick: still update, never create a duplicate.
    tripled = triage.decide_triage_comment(existing * 3, sha=SHA_MAIN, run_id=RUN_ID)
    assert tripled.kind == "update"
    assert triage.decide_triage_comment([], sha=SHA_MAIN, run_id=RUN_ID).kind == "create"
    assert comment.startswith(triage.triage_marker(SHA_MAIN, RUN_ID))


def test_parse_triage_markers_round_trip() -> None:
    bodies = [
        triage.build_triage_comment(
            run=_run(sha=SHA_MAIN),
            artifact_name="x",
            triage=triage.map_typed_outcome(_result(sha=SHA_MAIN)),
            timeline_lines=[],
            crossings=(),
        ),
        triage.build_triage_comment(
            run=_run(sha=SHA_424, run_id=42),
            artifact_name="y",
            triage=triage.map_typed_outcome(
                _result(sha=SHA_424, run_id=42, outcome="inconclusive")
            ),
            timeline_lines=[],
            crossings=(),
        ),
    ]
    markers = triage.parse_triage_markers(bodies)
    assert (SHA_MAIN, str(RUN_ID)) in markers
    assert (SHA_424, "42") in markers


# ---------------------------------------------------------------------------
# 6. Repair issue planning: one per proven fingerprint
# ---------------------------------------------------------------------------


def test_repair_issue_fingerprint_dedup() -> None:
    outcome = triage.map_typed_outcome(None, blocked_code=triage.BLOCK_ARTIFACT_MISSING)
    key = triage.fingerprint_key_for_triage(outcome, {})
    plan = triage.build_repair_issue_plan(
        triage=outcome,
        run=_run(),
        artifact_name=triage.artifact_name_for_run(RUN_ID),
        fingerprint_key=key,
        existing_fingerprints=set(),
    )
    assert plan.should_create is True
    assert plan.title.startswith("P0: aa-scenario2-triage ")
    assert plan.body.count("target_sha") >= 1
    assert "never patch product semantic logic" in plan.body
    dedup = triage.build_repair_issue_plan(
        triage=outcome,
        run=_run(),
        artifact_name=triage.artifact_name_for_run(RUN_ID),
        fingerprint_key=key,
        existing_fingerprints={plan.fingerprint},
    )
    assert dedup.should_create is False
    assert "never loop an identical fingerprint" in dedup.reason


def test_repair_issue_never_for_slow_or_boundary_work() -> None:
    outcomes = (
        triage.map_typed_outcome(_result(delivered_ms=1_200_000.0)),
        triage.map_typed_outcome(_result(classification="boundary-localized-root-unknown")),
        triage.map_typed_outcome(_result(classification="unresolved-at-campaign-end")),
    )
    for outcome in outcomes:
        plan = triage.build_repair_issue_plan(
            triage=outcome,
            run=_run(),
            artifact_name="x",
            fingerprint_key="whatever",
            existing_fingerprints=set(),
        )
        assert plan.should_create is False


def test_harness_truncation_fingerprint_names_the_probes() -> None:
    outcome = triage.TriageOutcome(
        triage.OUTCOME_HARNESS_FAILURE,
        "harness-error",
        blocked_code=triage.BLOCK_HARNESS_TRUNCATED,
    )
    key = triage.fingerprint_key_for_triage(outcome, {})
    assert "probe_scenario2" in key and "probe_direct_model" in key
    fingerprint = triage.repair_fingerprint(outcome.blocked_code or "x", key)
    assert len(fingerprint) == 12
    assert triage.parse_issue_fingerprints(
        [f"nonsense <!-- aa-scenario2-triage-fp fp={fingerprint} -->"]
    ) == {fingerprint}


def test_same_defect_new_run_reuses_fingerprint() -> None:
    """Identical defect on a different run must NOT mint a second issue."""
    outcome_a = triage.map_typed_outcome(None, missing_files=("result.json",))
    outcome_b = triage.map_typed_outcome(
        None, missing_files=("result.json", "supervisor-events.jsonl")
    )
    first = triage.build_repair_issue_plan(
        triage=outcome_a,
        run=_run(run_id=1),
        artifact_name="x",
        fingerprint_key=triage.fingerprint_key_for_triage(outcome_a, {}),
        existing_fingerprints=set(),
    )
    assert first.should_create is True
    second = triage.build_repair_issue_plan(
        triage=outcome_b,
        run=_run(run_id=1),
        artifact_name="x",
        fingerprint_key=triage.fingerprint_key_for_triage(outcome_b, {}),
        existing_fingerprints=set(),
    )
    assert second.should_create is True
    third = triage.build_repair_issue_plan(
        triage=outcome_b,
        run=_run(run_id=2),
        artifact_name="x",
        fingerprint_key=triage.fingerprint_key_for_triage(outcome_b, {}),
        existing_fingerprints={second.fingerprint},
    )
    assert third.should_create is False


# ---------------------------------------------------------------------------
# 7. Run-inventory parsing / selection / ancestor backfill
# ---------------------------------------------------------------------------


def test_parse_runs_filters_and_types() -> None:
    runs = triage.parse_runs(
        {
            "workflow_runs": [
                {
                    "id": RUN_ID,
                    "head_sha": SHA_424.upper(),
                    "event": "push",
                    "status": "completed",
                    "conclusion": "success",
                    "created_at": "2026-10-22T17:36:09Z",
                    "updated_at": "2026-10-22T17:55:30Z",
                    "html_url": f"https://github.com/kodmial/aa/actions/runs/{RUN_ID}",
                    "run_number": 3,
                },
                {
                    "id": 7,
                    "head_sha": SHA_MAIN,
                    "event": "workflow_dispatch",
                    "status": "queued",
                    "conclusion": None,
                    "created_at": "2026-10-22T18:00:00Z",
                    "updated_at": None,
                },
            ]
        }
    )
    assert len(runs) == 2
    assert runs[0].head_sha == SHA_424
    assert runs[0].is_completed
    assert runs[0].run_number == 3
    assert runs[1].is_active


def test_selection_prefers_newest_completed_run() -> None:
    older = _run(run_id=1, created="2026-10-22T17:36:09Z")
    newer = _run(run_id=2, created="2026-10-22T19:00:00Z")
    selection = triage.select_diagnostic_run([older, newer], allowed_shas=(SHA_424,))
    assert selection.action == "reconcile"
    assert selection.run is not None and selection.run.run_id == 2


def test_active_diagnostic_run_waits() -> None:
    active = _run(status="in_progress", conclusion=None)
    selection = triage.select_diagnostic_run([active], allowed_shas=(SHA_424,))
    assert selection.action == "wait"


def test_machinery_identical_ancestor_is_accepted(
    reconciler: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reconciler, "machinery_unchanged_between", lambda *a, **k: True)
    decision, ancestor = reconciler.select_with_ancestor_backfill(
        [_run(sha=SHA_424)],
        owner="kodmial",
        repo="aa",
        allowed=(SHA_MAIN,),
        target_sha=SHA_MAIN,
        resolve_ancestor=True,
    )
    assert ancestor == SHA_424
    assert decision.action == "reconcile"


def test_machinery_differing_ancestor_is_rejected(
    reconciler: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reconciler, "machinery_unchanged_between", lambda *a, **k: False)
    _decision, ancestor = reconciler.select_with_ancestor_backfill(
        [_run(sha=SHA_424)],
        owner="kodmial",
        repo="aa",
        allowed=(SHA_MAIN,),
        target_sha=SHA_MAIN,
        resolve_ancestor=True,
    )
    assert ancestor == ""


# ---------------------------------------------------------------------------
# 8. Workflow contract
# ---------------------------------------------------------------------------


def test_reconcile_workflow_contract() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert 'workflows: ["AA Gate C scenario2 diagnostic"]' in text
    assert "types: [completed]" in text
    assert 'cron: "*/15 * * * *"' in text
    assert "workflow_dispatch" in text
    assert "actions: read" in text
    assert "issues: write" in text
    assert "contents: read" in text
    assert "cancel-in-progress: false" in text
    assert "reconcile_scenario2_diagnostic.py" in text
    assert "--post" in text
    assert "--triage-issue 419" in text
    assert "TRIAGE_EVENT_SHA: ${{ github.event.workflow_run.head_sha }}" in text
    # Never starts the 18-case qualification or another Telegram bot here.
    assert "aa-self-proving-qualification" not in text
    assert "aa-telegram" not in text
    assert "issue_number" not in text


def test_diagnostic_workflow_name_is_reconcile_target() -> None:
    assert "name: AA Gate C scenario2 diagnostic" in DIAG_WORKFLOW_PATH.read_text(encoding="utf-8")


def test_module_constants_wired_to_the_task() -> None:
    assert triage.TRIAGE_ISSUE_NUMBER == 419
    assert triage.DIAGNOSTIC_WORKFLOW_FILE == "aa-gate-c-scenario2-diagnostic.yml"
    assert triage.artifact_name_for_run(123) == ("aa-gate-c-scenario2-diagnostic-123-scenario2")
    assert triage.SLOW_WORK_THRESHOLDS_S == (45.0, 75.0, 130.0, 300.0, 900.0)
    assert triage.MAX_DISPATCH_ATTEMPTS == 3
    assert triage.DISPATCH_RETRY_BACKOFF_S == 1800.0


# ---------------------------------------------------------------------------
# 9. Timeline sanitization details
# ---------------------------------------------------------------------------


def test_timeline_skips_snapshot_noise_and_unknown_fields() -> None:
    events = [
        _event("snapshot", 1.0, {"stage": "opencode", "typing_sends": 1}),
        _event(
            "stage-end",
            2.0,
            {"stage": "opencode", "elapsed_ms": 100.0, "session_id": "SECRETISH"},
        ),
    ]
    entries = triage.build_timeline({"probe-scenario2": events})
    rendered = "\n".join(triage.render_timeline(entries))
    assert "snapshot" not in rendered
    assert "SECRETISH" not in rendered
    assert "stage=opencode" in rendered


def test_timeline_cap_respected() -> None:
    events = [
        _event("stage-start", float(index), {"stage": f"stage-{index}"}) for index in range(300)
    ]
    entries = triage.build_timeline({"probe-scenario2": events})
    assert len(entries) == triage.MAX_TIMELINE_ENTRIES


def test_run_wall_time_formatting() -> None:
    assert triage.format_elapsed("2026-10-22T17:36:09Z", "2026-10-22T17:55:30Z") == "1161.0s"
    assert triage.format_elapsed(None, "2026-10-22T17:55:30Z") == "unknown"


# ---------------------------------------------------------------------------
# 10. CLI integration (pure local inventory; no network)
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def argparse_stub(
    *,
    runs_file: str,
    comments_file: str,
    open_issues_file: str,
    main_sha: str,
    post: bool,
) -> Any:
    """Minimal stand-in for the CLI argparse namespace (hermetic)."""
    return type(
        "_TriageArgs",
        (),
        {
            "owner": "kodmial",
            "repo": "aa",
            "triage_issue": 419,
            "main_sha": main_sha,
            "event_sha": [],
            "now_epoch": 2_000_000.0,
            "runs_file": runs_file,
            "comments_file": comments_file,
            "open_issues_file": open_issues_file,
            "artifact_file": "",
            "post": post,
        },
    )()


def test_cli_dispatches_missing_run_once(
    reconciler: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_path = _write_json(tmp_path / "runs.json", {"workflow_runs": []})
    comments_path = _write_json(tmp_path / "comments.json", [])
    issues_path = _write_json(tmp_path / "issues.json", [])
    dispatch_calls: list[str] = []
    marker_posts: list[str] = []
    monkeypatch.setattr(reconciler, "collect_main_sha", lambda owner, repo: SHA_MAIN)
    monkeypatch.setattr(
        reconciler,
        "post_dispatch_marker",
        lambda owner, repo, issue, body: marker_posts.append(body),
    )
    monkeypatch.setattr(
        reconciler,
        "dispatch_diagnostic",
        lambda owner, repo, target_sha: dispatch_calls.append(target_sha),
    )
    args = argparse_stub(
        runs_file=runs_path,
        comments_file=comments_path,
        open_issues_file=issues_path,
        main_sha=SHA_MAIN,
        post=True,
    )
    exit_code = reconciler.run(args)
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exit_code == reconciler.EXIT_OK
    assert summary["phase"] == "dispatched"
    assert summary["dispatched"] is True
    assert dispatch_calls == [SHA_MAIN]
    assert len(marker_posts) == 1
    assert "--now-epoch" not in marker_posts[0]


def test_cli_blocked_reconcile_on_missing_artifact_never_says_pass(
    reconciler: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_path = _write_json(
        tmp_path / "runs.json",
        {
            "workflow_runs": [
                {
                    "id": RUN_ID,
                    "head_sha": SHA_MAIN,
                    "event": "push",
                    "status": "completed",
                    "conclusion": "success",
                    "created_at": "2026-10-22T17:36:09Z",
                    "updated_at": "2026-10-22T17:55:30Z",
                }
            ]
        },
    )
    comments_path = _write_json(tmp_path / "comments.json", [])
    issues_path = _write_json(tmp_path / "issues.json", [])
    monkeypatch.setattr(reconciler, "collect_main_sha", lambda owner, repo: SHA_MAIN)
    monkeypatch.setattr(
        reconciler,
        "collect_artifact",
        lambda owner, repo, run_id: (None, True),
    )
    monkeypatch.setattr(reconciler, "_open_bodies", lambda args: [])
    posted: list[str] = []

    def fake_post(owner: str, repo: str, issue: int, body: str) -> str:
        posted.append(body)
        return "created comment on #419"

    monkeypatch.setattr(
        reconciler,
        "post_triage_comment",
        lambda owner, repo, issue, action, b: fake_post(owner, repo, issue, b),
    )
    created_issues: list[str] = []

    def fake_create(owner: str, repo: str, plan: Any) -> int:
        created_issues.append(plan.title)
        return 500

    monkeypatch.setattr(reconciler, "create_repair_issue", fake_create)
    args = argparse_stub(
        runs_file=runs_path,
        comments_file=comments_path,
        open_issues_file=issues_path,
        main_sha=SHA_MAIN,
        post=True,
    )
    exit_code = reconciler.run(args)
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exit_code == reconciler.EXIT_OK  # truthful blocked reconcile
    assert summary["outcome"] == triage.OUTCOME_HARNESS_FAILURE
    assert summary["blocked_code"] == triage.BLOCK_ARTIFACT_MISSING
    assert summary["comment"]["action"] == "create"
    assert summary["repair_issue"]["created"] == 500
    assert created_issues and created_issues[0].startswith("P0: aa-scenario2-triage ")


def test_cli_named_missing_permission_failure(
    reconciler: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(owner: str, repo: str) -> str:
        raise reconciler.GhError(
            f"gh api repos/{owner}/{repo}/branches/main failed: HTTP 403 Forbidden"
        )

    monkeypatch.setattr(reconciler, "collect_main_sha", boom)
    monkeypatch.setattr(sys, "argv", ["reconcile_scenario2_diagnostic.py", "--runs-file", "x"])
    exit_code = reconciler.main()
    out = capsys.readouterr().out
    assert exit_code == reconciler.EXIT_GITHUB_ERROR
    assert "missing-permission" in out
    assert "actions:read" in out and "issues:write" in out
