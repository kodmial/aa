"""kodmial/aa#481 DoD tests: EVIDENCE_ARTIFACT_MISSING fingerprint regression.

Proves the AA-owned, deterministic diagnostic evidence path for the exact
proven defect fingerprint published by the scenario #2 reconciler
(``fp=80317ed1a437``, source run 39824142258, SHA
``d81425c46ed7db0d9e072569410665ee2c95bd0b``):

- an Actions-green diagnostic run whose evidence artifact is absent from
  the AA evidence store classifies as HARNESS_FAILURE with blocked code
  ``EVIDENCE_ARTIFACT_MISSING`` — never a fabricated product PASS;
- the reconciler mints exactly ONE repair issue for that fingerprint,
  with the exact published title/body shape, and never mints or re-opens
  a duplicate while the same fingerprint already has an open issue;
- given NEW healthy evidence (a fresh completed run whose artifact is
  intact and evaluable) the reconciler classifies the defect fixed
  (typed DELIVERED / honest verdict) and touches no repair issue;
- the diagnostic workflow itself now publishes durable evaluable
  evidence: maximum Actions retention (90 days) and a fail-loud upload
  (``if-no-files-found: error``) so a green run can never stand without
  its evidence artifact.

Pure, hermetic, network-free; the reconciler module is exercised through
its own decision core and CLI entry points with monkeypatched ``gh``
callers only.
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

# Live defect identity (pinned by the reconciler-published issue #481).
SOURCE_RUN_ID = 39_824_142_258
SOURCE_SHA = "d81425c46ed7db0d9e072569410665ee2c95bd0b"
EXPECTED_FINGERPRINT = "80317ed1a437"
EXPECTED_TITLE = (
    "P0: aa-scenario2-triage EVIDENCE_ARTIFACT_MISSING — "
    "proven diagnostic instrumentation/evidence gap"
)
EXPECTED_ARTIFACT_NAME = "aa-gate-c-scenario2-diagnostic-39824142258-scenario2"

# A fresh healthy run representing post-fix evidence under future main.
HEALTHY_RUN_ID = 40_000_000_001
HEALTHY_SHA = "a" * 40

FP_MARKER = f"<!-- aa-scenario2-triage-fp fp={EXPECTED_FINGERPRINT} -->"


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


def source_run() -> triage.DiagnosticRun:
    """The exact reconciler-selected run as recorded on #419 (2026-10-22)."""
    return triage.DiagnosticRun(
        run_id=SOURCE_RUN_ID,
        head_sha=SOURCE_SHA,
        event="push",
        status="completed",
        conclusion="success",
        created_at="2026-10-22T17:36:09Z",
        updated_at="2026-10-22T17:45:02Z",
        run_number=3,
        html_url=f"https://github.com/kodmial/aa/actions/runs/{SOURCE_RUN_ID}",
    )


def blocked_outcome() -> triage.TriageOutcome:
    """The exact classification path the reconciler takes for a missing artifact."""
    return triage.map_typed_outcome(None, blocked_code=triage.BLOCK_ARTIFACT_MISSING)


def healthy_result(run_id: int = HEALTHY_RUN_ID, sha: str = HEALTHY_SHA) -> dict[str, Any]:
    """Fresh evaluable `result.json` from a healthy post-fix diagnostic run."""
    return {
        "schema": triage.DIAGNOSTIC_SCHEMA_VERSION,
        "run_id": run_id,
        "main_sha": sha,
        "run_url": f"https://github.com/kodmial/aa/actions/runs/{run_id}",
        "target_issue": 418,
        "classification": triage.CLASS_DELIVERED,
        "outcome": "delivered",
        "exit_code": 0,
        "probe_direct_model": {
            "probe": "direct-model",
            "status": "completed",
            "exit_code": 0,
            "delivered": True,
            "delivered_elapsed_ms": None,
            "model_request_total_ms": 6100.0,
        },
        "probe_scenario2": {
            "probe": "scenario2",
            "status": "delivered",
            "exit_code": 0,
            "delivered": True,
            "delivered_elapsed_ms": 152400.0,
        },
    }


def write_healthy_artifact(tmp_path: Path, run_id: int, sha: str) -> Path:
    path = tmp_path / f"{triage.artifact_name_for_run(run_id)}.zip"
    members: dict[str, bytes] = {
        triage.REQUIRED_RESULT_MEMBER: (json.dumps(healthy_result(run_id, sha)) + "\n").encode(),
        triage.REQUIRED_SUPERVISOR_MEMBER: (
            json.dumps(
                {
                    "ts": "2026-10-30T00:00:00.000000+00:00",
                    "mono": 0.1,
                    "event": "supervisor-start",
                    "sha": sha,
                    "policy": "no-per-answer-cutoff",
                    "effective_campaign_allowance_s": 1800.0,
                }
            )
            + "\n"
        ).encode(),
        triage.REQUIRED_SCENARIO_EVENTS_MEMBER: (
            json.dumps(
                {
                    "ts": "2026-10-30T00:00:01.000000+00:00",
                    "mono": 1.0,
                    "event": "delivery",
                    "elapsed_ms": 152400.0,
                    "reply_len": 388,
                    "typing_sends": 8,
                }
            )
            + "\n"
        ).encode(),
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def issue_481_open_body() -> str:
    """Open-issue body shape carrying the live fingerprint marker."""
    return "\n".join(
        (
            EXPECTED_TITLE,
            "<!-- continuum-task:implementation -->",
            FP_MARKER,
        )
    )


def _write_json(path: Path, payload: Any) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


@pytest.fixture(name="hermetic_post")
def hermetic_post_fixture(reconciler: Any, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record and re-surface every issued-triage write instead of posting it.

    Guardrail: the reconciler is exercised with --post on, yet nothing in
    this module may reach the live GitHub API. Both write helpers are
    replaced with recording stubs, and any raw ``gh`` subprocess attempt
    fails the test loudly. (A earlier unguarded run of this suite posted
    synthetic triage comments to the live ops issue; this fixture exists
    so that class of leak can never happen again.)
    """
    posted: list[str] = []

    def fake_post_triage_comment(owner: str, repo: str, issue: int, action: Any, body: str) -> str:
        posted.append(body)
        if action.kind == "update" and action.comment_id is not None:
            return f"updated comment {action.comment_id} in place"
        return f"created comment on #{issue}"

    def fake_post_dispatch_marker(owner: str, repo: str, issue: int, marker_body: str) -> None:
        posted.append(marker_body)

    def forbidden_subprocess(args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"hermetic test attempted a subprocess call: {args}")

    monkeypatch.setattr(reconciler, "post_triage_comment", fake_post_triage_comment)
    monkeypatch.setattr(reconciler, "post_dispatch_marker", fake_post_dispatch_marker)
    monkeypatch.setattr(reconciler.subprocess, "run", forbidden_subprocess)
    return posted


def argparse_stub(
    *,
    runs_file: str,
    comments_file: str,
    open_issues_file: str,
    main_sha: str,
    post: bool,
    artifact_file: str = "",
) -> Any:
    """Minimal hermetic stand-in for the reconciler CLI namespace."""
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
            "artifact_file": artifact_file,
            "post": post,
        },
    )()


# ---------------------------------------------------------------------------
# 1. Blocked-code classification for the exact live defect
# ---------------------------------------------------------------------------


def test_missing_artifact_classifies_blocked_never_pass() -> None:
    outcome = blocked_outcome()
    assert outcome.outcome == triage.OUTCOME_HARNESS_FAILURE
    assert outcome.blocked_code == triage.BLOCK_ARTIFACT_MISSING
    assert outcome.classification == ""
    assert outcome.is_success is False
    assert triage.product_gate_fulfilled(outcome.outcome) is False
    assert triage.is_done(outcome.outcome) is False
    assert "artifact/evidence defect" in outcome.detail


def test_blocked_comment_names_harness_defect_not_product_cause() -> None:
    comment = triage.build_triage_comment(
        run=source_run(),
        artifact_name=EXPECTED_ARTIFACT_NAME,
        triage=blocked_outcome(),
        timeline_lines=[],
        crossings=(),
        evidence_gaps=["AA-owned artifact `…` absent: diagnostic evidence cannot be re-read"],
    )
    assert "instrumentation/harness defect" in comment
    assert triage.OUTCOME_ROOT_CAUSE_PROVED not in comment
    assert f"run={SOURCE_RUN_ID}" in comment


# ---------------------------------------------------------------------------
# 2. Exact fingerprint regression (fp=80317ed1a437)
# ---------------------------------------------------------------------------


def test_exact_fingerprint_matches_issue_481() -> None:
    outcome = blocked_outcome()
    key = triage.fingerprint_key_for_triage(outcome, {})
    assert key == "EVIDENCE_ARTIFACT_MISSING"
    fingerprint = triage.repair_fingerprint(outcome.blocked_code or "", key)
    assert fingerprint == EXPECTED_FINGERPRINT
    assert triage.parse_issue_fingerprints([issue_481_open_body()]) == {EXPECTED_FINGERPRINT}


def test_first_observation_publishes_exactly_the_live_issue_shape() -> None:
    plan = triage.build_repair_issue_plan(
        triage=blocked_outcome(),
        run=source_run(),
        artifact_name=EXPECTED_ARTIFACT_NAME,
        fingerprint_key="EVIDENCE_ARTIFACT_MISSING",
        existing_fingerprints=set(),
    )
    assert plan.should_create is True
    assert plan.fingerprint == EXPECTED_FINGERPRINT
    assert plan.title == EXPECTED_TITLE
    assert plan.labels == ("priority:p0", "automation:ready")
    assert FP_MARKER in plan.body
    assert f"Source run: #{SOURCE_RUN_ID}" in plan.body
    assert f"Exact SHA: `{SOURCE_SHA}`" in plan.body
    assert "`EVIDENCE_ARTIFACT_MISSING`" in plan.body
    assert f"Artifact examined: `{EXPECTED_ARTIFACT_NAME}`" in plan.body
    assert "never patch product semantic logic" in plan.body
    assert "never introduce an arbitrary answer cutoff" in plan.body
    assert "must not re-open this issue for new runs" in plan.body


def test_open_issue_with_same_fingerprint_never_duplicates() -> None:
    plan = triage.build_repair_issue_plan(
        triage=blocked_outcome(),
        run=source_run(),
        artifact_name=EXPECTED_ARTIFACT_NAME,
        fingerprint_key="EVIDENCE_ARTIFACT_MISSING",
        existing_fingerprints=triage.parse_issue_fingerprints([issue_481_open_body()]),
    )
    assert plan.fingerprint == EXPECTED_FINGERPRINT
    assert plan.should_create is False
    assert "already has an open repair issue" in plan.reason
    assert plan.title == "" and plan.body == ""


# ---------------------------------------------------------------------------
# 3. CLI end-to-end: artifact-absent run triages blocked, never duplicates
# ---------------------------------------------------------------------------


def runs_payload(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {"workflow_runs": runs}


def run_payload(run: triage.DiagnosticRun, *, run_id_override: int | None = None) -> dict[str, Any]:
    """Actions-shaped payload form of a typed run (API key is ``id``)."""
    data = dict(run.__dict__)
    run_id = data.pop("run_id")
    if not isinstance(run_id, int):
        raise TypeError("typed runs always carry an integer run_id")
    data["id"] = run_id if run_id_override is None else run_id_override
    return data


def test_cli_artifact_absent_reconcile_blocked_and_single_issue(
    reconciler: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hermetic_post: list[str],
) -> None:
    runs_path = _write_json(tmp_path / "runs.json", runs_payload([run_payload(source_run())]))
    comments_path = _write_json(tmp_path / "comments.json", [])
    issues_path = _write_json(tmp_path / "issues.json", [issue_481_open_body()])
    monkeypatch.setattr(reconciler, "collect_main_sha", lambda owner, repo: SOURCE_SHA)
    monkeypatch.setattr(reconciler, "collect_artifact", lambda owner, repo, run_id: (None, False))
    monkeypatch.setattr(reconciler, "_open_bodies", lambda args: [issue_481_open_body()])
    created_titles: list[str] = []

    def fake_create(owner: str, repo: str, plan: Any) -> int:
        created_titles.append(plan.title)
        return 999

    monkeypatch.setattr(reconciler, "create_repair_issue", fake_create)
    args = argparse_stub(
        runs_file=runs_path,
        comments_file=comments_path,
        open_issues_file=issues_path,
        main_sha=SOURCE_SHA,
        post=True,
    )
    exit_code = reconciler.run(args)
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exit_code == reconciler.EXIT_OK
    assert summary["outcome"] == triage.OUTCOME_HARNESS_FAILURE
    assert summary["blocked_code"] == triage.BLOCK_ARTIFACT_MISSING
    assert summary["artifact"]["source"] == "absent"
    assert summary["repair_issue"]["created"] is None
    assert "already has an open repair issue" in summary["repair_issue"]["reason"]
    assert created_titles == []
    # One truthful, idempotent #419 comment for the exact live (sha, run).
    assert len(hermetic_post) == 1
    assert triage.BLOCK_ARTIFACT_MISSING in hermetic_post[0]


def test_cli_retried_same_defect_after_issue_opens_never_creates_second_issue(
    reconciler: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hermetic_post: list[str],
) -> None:
    """Same fingerprint, later run: reconciler records on #419 but re-opens nothing."""
    runs_path = _write_json(
        tmp_path / "runs.json",
        runs_payload([run_payload(source_run(), run_id_override=SOURCE_RUN_ID + 1)]),
    )
    comments_path = _write_json(tmp_path / "comments.json", [])
    issues_path = _write_json(tmp_path / "issues.json", [issue_481_open_body()])
    monkeypatch.setattr(reconciler, "collect_main_sha", lambda owner, repo: SOURCE_SHA)
    monkeypatch.setattr(reconciler, "collect_artifact", lambda owner, repo, run_id: (None, False))
    monkeypatch.setattr(reconciler, "_open_bodies", lambda args: [issue_481_open_body()])
    created_titles: list[str] = []

    def fake_create(owner: str, repo: str, plan: Any) -> int:
        created_titles.append(plan.title)
        return 999

    monkeypatch.setattr(reconciler, "create_repair_issue", fake_create)
    args = argparse_stub(
        runs_file=runs_path,
        comments_file=comments_path,
        open_issues_file=issues_path,
        main_sha=SOURCE_SHA,
        post=True,
    )
    exit_code = reconciler.run(args)
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exit_code == reconciler.EXIT_OK
    assert summary["blocked_code"] == triage.BLOCK_ARTIFACT_MISSING
    assert summary["repair_issue"]["created"] is None
    assert created_titles == []
    # The retried run gets its own (sha, run) comment, still no second issue.
    assert len(hermetic_post) == 1
    assert f"run={SOURCE_RUN_ID + 1}" in hermetic_post[0]


# ---------------------------------------------------------------------------
# 4. New healthy evidence classifies the defect fixed
# ---------------------------------------------------------------------------


def test_healthy_outcome_is_honest_delivered_without_repair_issue() -> None:
    outcome = triage.map_typed_outcome(healthy_result())
    assert outcome.outcome == triage.OUTCOME_DELIVERED
    assert outcome.blocked_code is None
    assert outcome.requires_repair_issue is False
    assert triage.product_gate_fulfilled(outcome.outcome) is True


def test_cli_new_healthy_run_classifies_the_gap_fixed(
    reconciler: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hermetic_post: list[str],
) -> None:
    healthy_run = triage.DiagnosticRun(
        run_id=HEALTHY_RUN_ID,
        head_sha=HEALTHY_SHA,
        event="push",
        status="completed",
        conclusion="success",
        created_at="2026-10-30T00:10:00Z",
        updated_at="2026-10-30T00:21:00Z",
        run_number=4,
        html_url=f"https://github.com/kodmial/aa/actions/runs/{HEALTHY_RUN_ID}",
    )
    healthy_artifact = write_healthy_artifact(tmp_path, HEALTHY_RUN_ID, HEALTHY_SHA)
    runs_path = _write_json(
        tmp_path / "runs.json",
        runs_payload([run_payload(healthy_run), run_payload(source_run())]),
    )
    comments_path = _write_json(tmp_path / "comments.json", [])
    # The escalated issue from the defect remains open while its repair lands.
    issues_path = _write_json(tmp_path / "issues.json", [issue_481_open_body()])
    monkeypatch.setattr(reconciler, "collect_main_sha", lambda owner, repo: HEALTHY_SHA)
    monkeypatch.setattr(
        reconciler, "collect_artifact", lambda owner, repo, run_id: (124_000_000_001, False)
    )
    monkeypatch.setattr(
        reconciler, "download_artifact", lambda owner, repo, artifact_id: healthy_artifact
    )
    monkeypatch.setattr(reconciler, "_open_bodies", lambda args: [issue_481_open_body()])
    created_titles: list[str] = []

    def fake_create(owner: str, repo: str, plan: Any) -> int:
        created_titles.append(plan.title)
        return 999

    monkeypatch.setattr(reconciler, "create_repair_issue", fake_create)
    args = argparse_stub(
        runs_file=runs_path,
        comments_file=comments_path,
        open_issues_file=issues_path,
        main_sha=HEALTHY_SHA,
        post=True,
    )
    exit_code = reconciler.run(args)
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exit_code == reconciler.EXIT_OK
    assert summary["run_id"] == HEALTHY_RUN_ID
    assert summary["outcome"] == triage.OUTCOME_DELIVERED
    assert summary["blocked_code"] is None
    assert summary["artifact"]["source"] == "downloaded"
    assert summary["repair_issue"]["created"] is None
    assert created_titles == []
    # The fresh healthy run publishes exactly ONE idempotent triage comment
    # on #419 with an honest typed DELIVERED verdict — the gap is resolved.
    assert len(hermetic_post) == 1
    assert hermetic_post[0].startswith(
        f"<!-- aa-scenario2-triage sha={HEALTHY_SHA} run={HEALTHY_RUN_ID} -->"
    )
    assert f"Typed outcome: `{triage.OUTCOME_DELIVERED}`" in hermetic_post[0]
    assert triage.BLOCK_ARTIFACT_MISSING not in hermetic_post[0]


def test_cli_blocked_comment_shape_is_deterministic_and_idempotent(
    reconciler: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hermetic_post: list[str],
) -> None:
    runs_path = _write_json(tmp_path / "runs.json", runs_payload([run_payload(source_run())]))
    comments_path = _write_json(tmp_path / "comments.json", [])
    issues_path = _write_json(tmp_path / "issues.json", [])
    monkeypatch.setattr(reconciler, "collect_main_sha", lambda owner, repo: SOURCE_SHA)
    monkeypatch.setattr(reconciler, "collect_artifact", lambda owner, repo, run_id: (None, False))
    monkeypatch.setattr(reconciler, "_open_bodies", lambda args: [])
    created_titles: list[str] = []

    def fake_create(owner: str, repo: str, plan: Any) -> int:
        created_titles.append(plan.title)
        return 999

    monkeypatch.setattr(reconciler, "create_repair_issue", fake_create)
    args = argparse_stub(
        runs_file=runs_path,
        comments_file=comments_path,
        open_issues_file=issues_path,
        main_sha=SOURCE_SHA,
        post=True,
    )
    exit_code_1 = reconciler.run(args)
    capsys.readouterr()
    assert exit_code_1 == reconciler.EXIT_OK
    assert len(hermetic_post) == 1
    assert hermetic_post[0].startswith(
        f"<!-- aa-scenario2-triage sha={SOURCE_SHA} run={SOURCE_RUN_ID} -->"
    )
    assert "EVIDENCE_ARTIFACT_MISSING" in hermetic_post[0]
    assert triage.OUTCOME_HARNESS_FAILURE in hermetic_post[0]
    # First observation of the fingerprint minted exactly ONE repair issue.
    assert created_titles == [EXPECTED_TITLE]
    # Second pass over the identical (sha, run) with the issue now open:
    # comment repaired in place, no duplicate issue creation.
    monkeypatch.setattr(reconciler, "_open_bodies", lambda args: [issue_481_open_body()])
    args_with_comment = argparse_stub(
        runs_file=runs_path,
        comments_file=_write_json(
            tmp_path / "comments2.json",
            [
                {
                    "id": 63_478,
                    "user": {"login": "github-actions[bot]", "type": "Bot"},
                    "body": hermetic_post[0],
                }
            ],
        ),
        open_issues_file=issues_path,
        main_sha=SOURCE_SHA,
        post=True,
    )
    exit_code_2 = reconciler.run(args_with_comment)
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exit_code_2 == reconciler.EXIT_OK
    assert summary["comment"]["action"] == "update"
    assert summary["comment"]["comment_id"] == 63_478
    assert summary["repair_issue"]["created"] is None
    assert len(created_titles) == 1


# ---------------------------------------------------------------------------
# 5. Diagnostic workflow publishes durable evaluable evidence (kodmial/aa#481)
# ---------------------------------------------------------------------------


def test_diagnostic_workflow_artifact_is_durable_and_fails_loud() -> None:
    text = DIAG_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "retention-days: 90" in text
    assert "retention-days: 7" not in text
    assert "if-no-files-found: error" in text
    assert "if-no-files-found: warn" not in text
    upload_section = text.split("Upload diagnostic artifacts", 1)[1]
    assert "if: always()" in upload_section
    uploaded_name = "aa-gate-c-scenario2-diagnostic-${{ github.run_id }}-scenario2"
    assert uploaded_name in text
    # The reconciler consumes exactly this canonical AA-owned artifact name.
    assert (
        triage.artifact_name_for_run("${{ github.run_id }}")
        == "aa-gate-c-scenario2-diagnostic-${{ github.run_id }}-scenario2"
    )


def test_declared_identity_github_script_input_uses_valid_action_schema() -> None:
    text = DIAG_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "        uses: actions/github-script@v7\n        with:\n          script: |\n" in text
    assert "        uses: actions/github-script@v7\n        script: |\n" not in text


def test_diagnostic_workflow_still_pushes_and_dispatches_evidence() -> None:
    text = DIAG_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert 'aa-gate-c-scenario2-diagnostic.yml"' in text
    assert "scripts/diagnose_gate_c_scenario2.py" in text
    assert "workflow_dispatch" in text
    assert "target_sha" in text
