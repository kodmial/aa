"""kodmial/aa#483 DoD tests: EVIDENCE_IDENTITY_MISMATCH evidence path.

Proves the repaired AA-owned diagnostic evidence path for the exact defect
fingerprint escalated from reconciler run #40971756236:

- Reproducing state: run record head_sha = ``SHA_RECORD`` (the Actions
  record of a ``workflow_dispatch`` always names the dispatch-ref head),
  while the run checked out and diagnosed ``SHA_CHECKOUT`` — the artifact
  result.json named ``d81425c46ed7db0d9e072569410665ee2c95bd0b`` — so the
  reconciler triaged ``EVIDENCE_IDENTITY_MISMATCH`` and the diagnostic
  verdict was not evaluable (fingerprint ``1dabf0fdd79a`` on issue
  kodmial/aa#483). The blocked-code classification and the exact
  one-issue-per-fingerprint-must-not-duplicate behavior are pinned;
- Repaired state: the diagnostic run publishes a durable declared
  identity marker on #419; the reconciler binds artifact evidence to the
  SHA the run actually diagnosed, so healthy evidence reconciles to a
  real typed verdict and is never a fabricated PASS.

Zero ChatGPT/external-LLM judgement: everything is deterministic,
network-free and unit-tested here.
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
DIAG_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "aa-gate-c-scenario2-diagnostic.yml"

# Exact provenance pinned by the issue body (frozen task snapshot).
RUN_ID = 40971756236
SHA_RECORD = "3dcabb5190f367be59d2e8ac94e7073791e1e2c5"
SHA_CHECKOUT = "d81425c46ed7db0d9e072569410665ee2c95bd0b"
PINNED_FINGERPRINT = "1dabf0fdd79a"
BLOCKED_CODE = "EVIDENCE_IDENTITY_MISMATCH"


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


def _defect_run(
    run_id: int = RUN_ID,
    record_sha: str = SHA_RECORD,
    declared_sha: str = "",
    status: str = "completed",
    conclusion: str | None = "failure",
    created: str = "2026-10-29T23:14:29Z",
    updated: str = "2026-10-30T00:13:57Z",
) -> triage.DiagnosticRun:
    """The exact run shape pinned by issue kodmial/aa#483."""
    return triage.DiagnosticRun(
        run_id=run_id,
        head_sha=record_sha,
        event="workflow_dispatch",
        status=status,
        conclusion=conclusion,
        created_at=created,
        updated_at=updated,
        run_number=4,
        html_url=f"https://github.com/kodmial/aa/actions/runs/{run_id}",
        declared_sha=declared_sha,
    )


def _push_run(run_id: int, sha: str) -> triage.DiagnosticRun:
    return triage.parse_runs(
        [
            {
                "id": run_id,
                "head_sha": sha,
                "event": "push",
                "status": "completed",
                "conclusion": "success",
                "created_at": "2026-10-22T17:36:09Z",
                "updated_at": "2026-10-22T17:55:30Z",
            }
        ]
    )[0]


def _defect_result(
    *,
    classification: str = "unresolved-at-campaign-end",
    outcome: str = "inconclusive",
    run_id: int = RUN_ID,
    main_sha: str = SHA_CHECKOUT,
) -> dict[str, Any]:
    """Minimal result.json identity shape as recorded by the pinned run."""
    return {
        "schema": triage.DIAGNOSTIC_SCHEMA_VERSION,
        "run_id": run_id,
        "main_sha": main_sha,
        "run_url": f"https://github.com/kodmial/aa/actions/runs/{run_id}",
        "target_issue": 418,
        "classification": classification,
        "outcome": outcome,
        "exit_code": 3,
        "probe_direct_model": {
            "probe": "direct-model",
            "status": "completed",
            "delivered": True,
            "model_request_total_ms": 6500.0,
        },
        "probe_scenario2": {
            "probe": "scenario2",
            "status": "undelivered-at-campaign-end",
            "delivered": False,
            "delivered_elapsed_ms": None,
        },
    }


def _write_artifact(tmp_path: Path, result: dict[str, Any], *, run_id: int = RUN_ID) -> Path:
    path = tmp_path / (triage.artifact_name_for_run(run_id) + ".zip")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(triage.REQUIRED_RESULT_MEMBER, json.dumps(result).encode("utf-8"))
        archive.writestr(triage.REQUIRED_SUPERVISOR_MEMBER, b"")
        archive.writestr(triage.REQUIRED_SCENARIO_EVENTS_MEMBER, b"")
    return path


# ---------------------------------------------------------------------------
# 1. Exact defect reproduction: blocked-code classification (fail closed)
# ---------------------------------------------------------------------------


def test_exact_issue483_defect_classifies_blocked_never_pass(
    reconciler: Any, tmp_path: Path
) -> None:
    path = _write_artifact(tmp_path, _defect_result())
    outcome, _timeline, gaps, _crossings, _banner = reconciler.evaluate_artifact(
        path, _defect_run()
    )
    assert outcome.outcome == triage.OUTCOME_HARNESS_FAILURE
    assert outcome.blocked_code == triage.BLOCK_IDENTITY_MISMATCH
    assert outcome.is_success is False
    # Blocked evidence is never a fabricated product success.
    assert triage.product_gate_fulfilled(outcome.outcome) is False
    joined = "\n".join(gaps)
    assert SHA_CHECKOUT in joined
    assert SHA_RECORD in joined


def test_defect_fingerprint_is_deterministic_and_pinned() -> None:
    outcome = triage.map_typed_outcome(None, blocked_code=triage.BLOCK_IDENTITY_MISMATCH)
    key = triage.fingerprint_key_for_triage(outcome, {})
    assert key == BLOCKED_CODE
    fingerprint = triage.repair_fingerprint(outcome.blocked_code or "x", key)
    assert fingerprint == PINNED_FINGERPRINT


def test_repair_issue_plan_matches_issue483_spec() -> None:
    outcome = triage.map_typed_outcome(None, blocked_code=triage.BLOCK_IDENTITY_MISMATCH)
    plan = triage.build_repair_issue_plan(
        triage=outcome,
        run=_defect_run(),
        artifact_name=triage.artifact_name_for_run(RUN_ID),
        fingerprint_key=triage.fingerprint_key_for_triage(outcome, {}),
        existing_fingerprints=set(),
    )
    assert plan.should_create is True
    assert plan.title == (
        "P0: aa-scenario2-triage EVIDENCE_IDENTITY_MISMATCH — proven "
        "diagnostic instrumentation/evidence gap"
    )
    assert plan.fingerprint == PINNED_FINGERPRINT
    assert plan.labels == ("priority:p0", "automation:ready")
    assert f"aa-scenario2-triage-fp fp={PINNED_FINGERPRINT}" in plan.body
    assert f"`{SHA_RECORD}`" in plan.body
    assert "#" + str(RUN_ID) in plan.body
    assert "never patch product semantic logic" in plan.body
    # Fingerprint round-trip: the open-issue scanner sees the marker.
    assert triage.parse_issue_fingerprints([plan.body]) == {PINNED_FINGERPRINT}


def test_same_fingerprint_never_reopens_a_second_issue() -> None:
    """New runs of the identical defect must not mint duplicate issues."""
    outcome = triage.map_typed_outcome(None, blocked_code=triage.BLOCK_IDENTITY_MISMATCH)
    deduped = triage.build_repair_issue_plan(
        triage=outcome,
        run=_defect_run(run_id=99999),
        artifact_name=triage.artifact_name_for_run(99999),
        fingerprint_key=triage.fingerprint_key_for_triage(outcome, {}),
        existing_fingerprints={PINNED_FINGERPRINT},
    )
    assert deduped.should_create is False
    assert "never loop an identical fingerprint" in deduped.reason


def test_missing_telemetry_outranks_identity_blocked_code() -> None:
    """Precedence: evidence defects outrank any diagnostic verdict."""
    outcome = triage.map_typed_outcome(
        None,
        missing_files=("result.json",),
        blocked_code=triage.BLOCK_IDENTITY_MISMATCH,
    )
    assert outcome.blocked_code == triage.BLOCK_TELEMETRY_MISSING
    assert outcome.outcome == triage.OUTCOME_HARNESS_FAILURE


# ---------------------------------------------------------------------------
# 2. Declared identity: healthy evidence reconciles to a real verdict
# ---------------------------------------------------------------------------


def test_record_head_sha_alone_is_the_identity_when_undeclared() -> None:
    run = _defect_run()
    assert run.identity_sha == SHA_RECORD
    error = triage.validate_result_identity(_defect_result(), run)
    assert error is not None


def test_declared_identity_binds_artifact_to_the_diagnosed_sha(
    reconciler: Any, tmp_path: Path
) -> None:
    run = _defect_run(declared_sha=SHA_CHECKOUT)
    assert run.identity_sha == SHA_CHECKOUT
    path = _write_artifact(tmp_path, _defect_result())
    outcome, timeline, gaps, crossings, _banner = reconciler.evaluate_artifact(path, run)
    joined = "\n".join(gaps)
    assert BLOCKED_CODE not in joined
    # The diagnostic verdict IS evaluable again: inconclusive-by-evidence,
    # never a harness/evidence block and never a fabricated PASS.
    assert outcome.outcome == triage.OUTCOME_INCONCLUSIVE
    assert outcome.blocked_code is None
    assert triage.product_gate_fulfilled(outcome.outcome) is False
    assert timeline == []
    assert crossings == ()
    comment = triage.build_triage_comment(
        run=run,
        artifact_name=triage.artifact_name_for_run(RUN_ID),
        triage=outcome,
        timeline_lines=triage.render_timeline(timeline),
        crossings=crossings,
        evidence_gaps=gaps,
    )
    # The triage comment represents the declared diagnosed SHA, not the
    # Actions record head_sha, via the exact per-run marker.
    assert comment.startswith(triage.triage_marker(SHA_CHECKOUT, RUN_ID))


def test_delivered_declared_run_maps_truthful_success(reconciler: Any, tmp_path: Path) -> None:
    """Healthy identity + delivered classification = truthful DELIVERED."""
    run = _defect_run(declared_sha=SHA_CHECKOUT, conclusion="success")
    result = _defect_result(classification="delivered", outcome="delivered")
    result["probe_scenario2"].update(
        {"status": "delivered", "delivered": True, "delivered_elapsed_ms": 158300.0}
    )
    path = _write_artifact(tmp_path, result)
    outcome, _timeline, gaps, crossings, _banner = reconciler.evaluate_artifact(path, run)
    assert outcome.outcome == triage.OUTCOME_DELIVERED
    assert outcome.blocked_code is None
    assert triage.product_gate_fulfilled(outcome.outcome) is True
    # A genuinely slow completing turn is recorded as observed slow work.
    assert crossings


# ---------------------------------------------------------------------------
# 3. Declared-target selection and dispatch guard
# ---------------------------------------------------------------------------


def test_declared_marker_round_trip_and_malformed_rejection() -> None:
    marker = triage.declared_target_marker(RUN_ID, SHA_CHECKOUT)
    declared = triage.parse_declared_targets([marker, "noise", ""])
    assert declared == {RUN_ID: SHA_CHECKOUT}
    # Malformed, truncated or re-ordered markers never mint identity.
    bad = [
        "<!-- aa-scenario2-diagnostic-declared run=abc sha=zz -->",
        "<!-- aa-scenario2-diagnostic-declared sha=" + "a" * 40 + " run=1 -->",
        "<!-- aa-scenario2-diagnostic-declared run=2 sha=" + "a" * 39 + " -->",
    ]
    assert triage.parse_declared_targets(bad) == {}
    # First valid marker per run wins (deterministic).
    duplicated = [
        marker,
        triage.declared_target_marker(RUN_ID, "b" * 40),
    ]
    assert triage.parse_declared_targets(duplicated) == {RUN_ID: SHA_CHECKOUT}


def test_attach_declared_targets_folds_identity_into_runs() -> None:
    runs = [
        _defect_run(run_id=RUN_ID),
        _push_run(777, SHA_RECORD),
    ]
    attached = triage.attach_declared_targets(runs, {RUN_ID: SHA_CHECKOUT, 777: SHA_RECORD})
    assert attached[0].declared_sha == SHA_CHECKOUT
    assert attached[1].declared_sha == ""  # identity equals record; nothing added


def test_server_authoritative_run_title_supplies_declared_identity() -> None:
    runs = triage.parse_runs(
        [
            {
                "id": RUN_ID,
                "head_sha": SHA_RECORD,
                "display_title": f"AA scenario2 diagnostic target={SHA_CHECKOUT}",
                "event": "workflow_dispatch",
                "status": "completed",
                "conclusion": "failure",
                "created_at": "2026-10-29T23:14:29Z",
                "updated_at": "2026-10-30T00:13:57Z",
            }
        ]
    )
    assert runs[0].declared_sha == SHA_CHECKOUT
    assert runs[0].identity_sha == SHA_CHECKOUT
    decision = triage.select_declared_run(runs, allowed_shas=(SHA_CHECKOUT,))
    assert decision.action == "reconcile"


def test_run_title_is_not_overridden_by_comment_marker() -> None:
    runs = triage.parse_runs(
        [
            {
                "id": RUN_ID,
                "head_sha": SHA_RECORD,
                "display_title": f"AA scenario2 diagnostic target={SHA_CHECKOUT}",
                "event": "workflow_dispatch",
                "status": "completed",
                "conclusion": "failure",
            }
        ]
    )
    attached = triage.attach_declared_targets(runs, {RUN_ID: "a" * 40})
    assert attached[0].declared_sha == SHA_CHECKOUT


def test_only_automation_authors_may_mint_identity_markers() -> None:
    assert triage.is_trusted_identity_author("github-actions[bot]")
    assert triage.is_trusted_identity_author("GitHub-Actions[bot]")
    assert not triage.is_trusted_identity_author("random-user")
    assert not triage.is_trusted_identity_author("")


def test_declared_selection_accepts_target_sha_run_record_mismatch() -> None:
    runs = [_defect_run(declared_sha=SHA_CHECKOUT)]
    decision = triage.select_declared_run(runs, allowed_shas=(SHA_CHECKOUT,))
    assert decision.action == "reconcile"
    assert decision.run is not None and decision.run.run_id == RUN_ID
    # Unrelated declared sha still resolves missing (never fabricate a match).
    missing = triage.select_declared_run(runs, allowed_shas=("f" * 40,))
    assert missing.action == "missing"


def test_declared_selection_waits_for_active_declared_run() -> None:
    active = _defect_run(declared_sha=SHA_CHECKOUT, status="in_progress", conclusion=None)
    decision = triage.select_declared_run([active], allowed_shas=(SHA_CHECKOUT,))
    assert decision.action == "wait"
    assert "never dispatch a duplicate" in decision.reason


def test_dispatch_guard_ignores_unrelated_declared_runs() -> None:
    declared_other = triage.DiagnosticRun(
        run_id=888,
        head_sha="f" * 40,
        event="workflow_dispatch",
        status="completed",
        conclusion="success",
        created_at="2026-10-29T20:00:00Z",
        updated_at=None,
        run_number=1,
        declared_sha="a" * 40,
    )
    plan = triage.decide_dispatch(
        [declared_other], target_sha=SHA_RECORD, markers=[], now_epoch=1_000_000.0
    )
    assert plan.should_dispatch is True


def test_dispatch_guard_skips_declared_completed_run() -> None:
    plan = triage.decide_dispatch(
        [_defect_run(declared_sha=SHA_CHECKOUT)],
        target_sha=SHA_CHECKOUT,
        markers=[],
        now_epoch=1_000_000.0,
    )
    assert plan.should_dispatch is False
    assert "completed diagnostic run already exists" in plan.reason


def test_dispatch_guard_skips_declared_active_run() -> None:
    active = _defect_run(declared_sha=SHA_CHECKOUT, status="in_progress", conclusion=None)
    plan = triage.decide_dispatch(
        [active], target_sha=SHA_CHECKOUT, markers=[], now_epoch=1_000_000.0
    )
    assert plan.should_dispatch is False
    assert "already active" in plan.reason


def test_inflight_dispatch_without_declared_marker_never_duplicates() -> None:
    """The dispatched run may not have published its marker yet (or ever).

    Its Actions record names the dispatch-ref head, so correlate the durable
    attempt marker with the newer active workflow_dispatch run.
    """
    marker_epoch = triage.parse_iso_epoch("2026-10-30T01:00:00Z")
    assert marker_epoch is not None
    active = triage.parse_runs(
        [
            {
                "id": RUN_ID,
                "head_sha": SHA_RECORD,
                "event": "workflow_dispatch",
                "status": "in_progress",
                "conclusion": None,
                "created_at": "2026-10-30T01:01:00Z",
                "updated_at": None,
            }
        ]
    )
    plan = triage.decide_dispatch(
        active,
        target_sha=SHA_CHECKOUT,
        markers=[triage.DispatchMarker(SHA_CHECKOUT, marker_epoch, 1)],
        now_epoch=marker_epoch + 7200.0,
    )
    assert plan.should_dispatch is False
    assert "still active" in plan.reason


def test_selection_precedence_record_first_declared_second_ancestor_third(
    reconciler: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1) Direct record match beats declared and ancestor discovery.
    decision, ancestor = reconciler.select_with_ancestor_backfill(
        [_defect_run()],
        owner="kodmial",
        repo="aa",
        allowed=(SHA_RECORD,),
        target_sha=SHA_RECORD,
        resolve_ancestor=True,
    )
    assert decision.action == "reconcile"
    assert ancestor == ""
    # 2) Declared identity beats the machinery-identical ancestor heuristic.
    monkeypatch.setattr(reconciler, "machinery_unchanged_between", lambda *a, **k: True)
    decision, ancestor = reconciler.select_with_ancestor_backfill(
        [_defect_run(declared_sha=SHA_CHECKOUT)],
        owner="kodmial",
        repo="aa",
        allowed=(SHA_CHECKOUT,),
        target_sha=SHA_CHECKOUT,
        resolve_ancestor=True,
    )
    assert decision.action == "reconcile"
    assert decision.run is not None and decision.run.run_id == RUN_ID
    assert ancestor == ""  # resolved by declared identity, not ancestor logic
    # 3) No record identities in allowed and machinery differing at the
    #    only candidate: truthful missing, never a fabricated selection.
    clean = triage.attach_declared_targets([_push_run(2, SHA_RECORD)], {})
    monkeypatch.setattr(reconciler, "machinery_unchanged_between", lambda *a, **k: False)
    decision, ancestor = reconciler.select_with_ancestor_backfill(
        clean,
        owner="kodmial",
        repo="aa",
        allowed=(SHA_CHECKOUT,),
        target_sha=SHA_CHECKOUT,
        resolve_ancestor=True,
    )
    assert decision.action == "missing"
    assert ancestor == ""


# ---------------------------------------------------------------------------
# 4. CLI integration (pure local inventory; no network)
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def argparse_stub(
    *, runs_file: str, comments_file: str, open_issues_file: str, main_sha: str, post: bool
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


def test_cli_reconciles_declared_dispatch_run_against_target_sha(
    reconciler: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The workflow_dispatch run whose record head_sha not in allowed is
    still reconciled for its declared target SHA, and its verdict is
    evaluable; the known fingerprint on #483 is never re-opened."""
    runs_path = _write_json(
        tmp_path / "runs.json",
        {
            "workflow_runs": [
                {
                    "id": RUN_ID,
                    "head_sha": SHA_RECORD,
                    "event": "workflow_dispatch",
                    "status": "completed",
                    "conclusion": "failure",
                    "created_at": "2026-10-29T23:14:29Z",
                    "updated_at": "2026-10-30T00:13:57Z",
                }
            ]
        },
    )
    declared_comment = {
        "id": 6510000001,
        "user": {"login": "github-actions[bot]", "type": "Bot"},
        "body": triage.declared_target_marker(RUN_ID, SHA_CHECKOUT)
        + "\nDiagnostic run identity declared.",
    }
    comments_path = _write_json(tmp_path / "comments.json", [declared_comment])
    open_issues_path = _write_json(
        tmp_path / "issues.json",
        [
            {
                "title": "P0: pinned issue483",
                "body": f"<!-- aa-scenario2-triage-fp fp={PINNED_FINGERPRINT} -->",
            }
        ],
    )
    artifact_path = _write_artifact(tmp_path, _defect_result())
    monkeypatch.setattr(reconciler, "collect_main_sha", lambda owner, repo: SHA_CHECKOUT)
    monkeypatch.setattr(
        reconciler,
        "collect_artifact",
        lambda owner, repo, run_id: (12943562273, False),
    )

    def fake_download(owner: str, repo: str, artifact_id: int) -> Path:
        return artifact_path

    monkeypatch.setattr(reconciler, "download_artifact", fake_download)
    posted: list[str] = []

    def fake_post(owner: str, repo: str, issue: int, body: str) -> str:
        posted.append(body)
        return "updated comment in place"

    monkeypatch.setattr(
        reconciler,
        "post_triage_comment",
        lambda owner, repo, issue, action, body: fake_post(owner, repo, issue, body),
    )
    args = argparse_stub(
        runs_file=runs_path,
        comments_file=comments_path,
        open_issues_file=open_issues_path,
        main_sha=SHA_CHECKOUT,
        post=True,
    )
    exit_code = reconciler.run(args)
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exit_code == reconciler.EXIT_OK
    assert summary["phase"] == "reconciled"
    assert summary["run_id"] == RUN_ID
    # The identity gap is resolved: result.json is evaluable and typed.
    assert summary["outcome"] == triage.OUTCOME_INCONCLUSIVE
    assert summary["blocked_code"] is None
    assert summary["repair_issue"]["created"] is None
    # Healthy evidence is NOT a proven instrumentation defect: the pinned
    # issue stays closed-for-repair (the same fingerprint never re-opens
    # for new runs of healthy machinery).
    assert "not a proven instrumentation defect" in summary["repair_issue"]["reason"]
    assert posted and triage.triage_marker(SHA_CHECKOUT, RUN_ID) in posted[0]


def test_cli_still_dispatches_when_neither_record_nor_declared_matches(
    reconciler: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_path = _write_json(tmp_path / "runs.json", {"workflow_runs": []})
    comments_path = _write_json(tmp_path / "comments.json", [])
    open_issues_path = _write_json(tmp_path / "issues.json", [])
    dispatch_calls: list[str] = []
    monkeypatch.setattr(reconciler, "collect_main_sha", lambda owner, repo: SHA_CHECKOUT)
    monkeypatch.setattr(
        reconciler, "dispatch_diagnostic", lambda owner, repo, sha: dispatch_calls.append(sha)
    )
    monkeypatch.setattr(reconciler, "post_dispatch_marker", lambda *a, **k: None)
    args = argparse_stub(
        runs_file=runs_path,
        comments_file=comments_path,
        open_issues_file=open_issues_path,
        main_sha=SHA_CHECKOUT,
        post=True,
    )
    exit_code = reconciler.run(args)
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exit_code == reconciler.EXIT_OK
    assert summary["phase"] == "dispatched"
    assert dispatch_calls == [SHA_CHECKOUT]


def test_cli_ignores_forged_declared_marker_from_untrusted_author(
    reconciler: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An arbitrary commenter cannot stall diagnostics for a SHA."""
    runs_path = _write_json(
        tmp_path / "runs.json",
        {
            "workflow_runs": [
                {
                    "id": RUN_ID,
                    "head_sha": SHA_RECORD,
                    "event": "workflow_dispatch",
                    "status": "in_progress",
                    "conclusion": None,
                    "created_at": "2026-10-30T00:00:00Z",
                    "updated_at": None,
                }
            ]
        },
    )
    forged = {
        "id": 6510000099,
        "user": {"login": "untrusted-commenter", "type": "User"},
        "body": triage.declared_target_marker(RUN_ID, SHA_CHECKOUT),
    }
    comments_path = _write_json(tmp_path / "comments.json", [forged])
    open_issues_path = _write_json(tmp_path / "issues.json", [])
    dispatch_calls: list[str] = []
    monkeypatch.setattr(reconciler, "collect_main_sha", lambda owner, repo: SHA_CHECKOUT)
    monkeypatch.setattr(
        reconciler, "dispatch_diagnostic", lambda owner, repo, sha: dispatch_calls.append(sha)
    )
    monkeypatch.setattr(reconciler, "post_dispatch_marker", lambda *a, **k: None)
    args = argparse_stub(
        runs_file=runs_path,
        comments_file=comments_path,
        open_issues_file=open_issues_path,
        main_sha=SHA_CHECKOUT,
        post=True,
    )
    exit_code = reconciler.run(args)
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exit_code == reconciler.EXIT_OK
    assert summary["phase"] == "dispatched"
    assert dispatch_calls == [SHA_CHECKOUT]


# ---------------------------------------------------------------------------
# 5. Workflow contract
# ---------------------------------------------------------------------------


def test_diagnostic_workflow_publishes_declared_identity_step() -> None:
    text = DIAG_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "Publish declared exact-SHA run identity" in text
    assert "aa-scenario2-diagnostic-declared run=" in text
    assert '"${{ steps.sha.outputs.main-sha }}"' in text
    assert "issue_number: 419" in text
    assert "core.warning" in text  # best-effort, never fatal
    assert "actions/checkout@v4" in text
    # Identity is also in the server-authoritative Actions run record, so it
    # is available before any asynchronous comment lands (and survives a
    # failed best-effort publish).
    assert "run-name:" in text
    assert "github.event.inputs.target_sha || github.sha" in text
    # Dedup scans every comment page, not just the first 100.
    assert text.count("github.paginate(github.rest.issues.listComments") == 2
    # The declared marker format matches the reconciler's parser exactly.
    marker = triage.declared_target_marker(123, SHA_CHECKOUT)
    assert triage.parse_declared_targets([marker]) == {123: SHA_CHECKOUT}
