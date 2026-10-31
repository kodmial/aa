"""kodmial/aa#491 DoD tests: EVIDENCE_ARTIFACT_MISSING occurrence regression.

Proves the AA-owned, deterministic diagnostic evidence path for the exact
proven defect occurrence published by the scenario #2 reconciler
(``fp=80317ed1a437``, source run 41156665024, exact SHA
``4cebab2a9307417948fba4e6b368297841927fa8``):

- the failed diagnostic run's absent evidence artifact classifies as
  HARNESS_FAILURE with blocked code ``EVIDENCE_ARTIFACT_MISSING`` — never
  a fabricated product PASS — and the reconciler comment stays truthful;
- exactly ONE repair issue exists for the fingerprint: a later run of the
  same defect fingerprint must NOT mint a duplicate issue while kodmial/
  aa#491 (or any open issue) carries the fingerprint marker;
- given NEW healthy evidence (the post-fix push-triggered diagnostic run
  whose artifact is intact and evaluable) the reconciler classifies the
  gap resolved (typed DELIVERED) and touches no repair issue;
- the root cause that starved run 41156665024 of its artifact is
  structurally dead: the declared-identity ``actions/github-script``
  step now nests ``script`` under ``with`` (an invalid step schema makes
  the whole workflow fail startup, so no artifact can ever exist), and
  the evidence upload stays durable and fail-loud.

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
import yaml

from aa.control import scenario2_triage as triage

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "reconcile_scenario2_diagnostic.py"
DIAG_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "aa-gate-c-scenario2-diagnostic.yml"

# kodmial/aa#491 (reconciler escalation comment on #419).
SOURCE_RUN_ID = 41_156_665_024
SOURCE_SHA = "4cebab2a9307417948fba4e6b368297841927fa8"
EXPECTED_FINGERPRINT = "80317ed1a437"
EXPECTED_TITLE = (
    "P0: aa-scenario2-triage EVIDENCE_ARTIFACT_MISSING — "
    "proven diagnostic instrumentation/evidence gap"
)
EXPECTED_ARTIFACT_NAME = "aa-gate-c-scenario2-diagnostic-41156665024-scenario2"

# The healthy post-fix push-triggered run that resolved the occurrence on
# #419 (kodmial/aa#483 declared identity names its diagnosed SHA).
HEALTHY_RUN_ID = 41_161_688_889
HEALTHY_SHA = "ca81337d50f40a18a606bdc33897d70512d442a7"

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
    """The exact reconciler-selected run as recorded on #419 (2026-10-31)."""
    return triage.DiagnosticRun(
        run_id=SOURCE_RUN_ID,
        head_sha=SOURCE_SHA,
        event="push",
        status="completed",
        conclusion="failure",
        created_at="2026-10-31T01:42:50Z",
        updated_at="2026-10-31T01:44:05Z",
        run_number=7,
        html_url=f"https://github.com/kodmial/aa/actions/runs/{SOURCE_RUN_ID}",
    )


def healthy_run() -> triage.DiagnosticRun:
    """The fresh completed push-run whose DELIVERED triage resolved #491."""
    return triage.DiagnosticRun(
        run_id=HEALTHY_RUN_ID,
        head_sha=HEALTHY_SHA,
        event="push",
        status="completed",
        conclusion="success",
        created_at="2026-10-31T02:49:23Z",
        updated_at="2026-10-31T02:57:53Z",
        run_number=8,
        html_url=f"https://github.com/kodmial/aa/actions/runs/{HEALTHY_RUN_ID}",
    )


def blocked_outcome() -> triage.TriageOutcome:
    """The exact classification path the reconciler takes for a missing artifact."""
    return triage.map_typed_outcome(None, blocked_code=triage.BLOCK_ARTIFACT_MISSING)


def healthy_result(run_id: int = HEALTHY_RUN_ID, sha: str = HEALTHY_SHA) -> dict[str, Any]:
    """Fresh evaluable `result.json` from the healthy post-fix diagnostic run."""
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
            "model_request_total_ms": 17279.0,
        },
        "probe_scenario2": {
            "probe": "scenario2",
            "status": "delivered",
            "exit_code": 0,
            "delivered": True,
            "delivered_elapsed_ms": 174600.0,
        },
    }


def write_healthy_artifact(tmp_path: Path, run_id: int, sha: str) -> Path:
    path = tmp_path / f"{triage.artifact_name_for_run(run_id)}.zip"
    members: dict[str, bytes] = {
        triage.REQUIRED_RESULT_MEMBER: (json.dumps(healthy_result(run_id, sha)) + "\n").encode(),
        triage.REQUIRED_SUPERVISOR_MEMBER: (
            json.dumps(
                {
                    "ts": "2026-10-31T02:49:30.000000+00:00",
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
                    "ts": "2026-10-31T02:52:25.000000+00:00",
                    "mono": 175.0,
                    "event": "delivery",
                    "elapsed_ms": 174600.0,
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


def issue_491_open_body() -> str:
    """The live open-issue body of kodmial/aa#491 carrying the fingerprint."""
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
    fails the test loudly.
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


# ---------------------------------------------------------------------------
# 1. Blocked-code classification for the exact occurrence
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
        evidence_gaps=[
            f"AA-owned artifact `{EXPECTED_ARTIFACT_NAME}` absent: "
            "diagnostic evidence cannot be re-read for this run"
        ],
    )
    assert "instrumentation/harness defect" in comment
    assert triage.OUTCOME_ROOT_CAUSE_PROVED not in comment
    assert f"run={SOURCE_RUN_ID}" in comment
    assert f"sha={SOURCE_SHA}" in comment
    assert EXPECTED_ARTIFACT_NAME in comment


# ---------------------------------------------------------------------------
# 2. Exact fingerprint regression (fp=80317ed1a437) and duplicate guard
# ---------------------------------------------------------------------------


def test_exact_fingerprint_matches_issue_491() -> None:
    outcome = blocked_outcome()
    key = triage.fingerprint_key_for_triage(outcome, {})
    assert key == "EVIDENCE_ARTIFACT_MISSING"
    fingerprint = triage.repair_fingerprint(outcome.blocked_code or "", key)
    assert fingerprint == EXPECTED_FINGERPRINT
    assert triage.parse_issue_fingerprints([issue_491_open_body()]) == {EXPECTED_FINGERPRINT}
    # The escalated plan mirrors the live issue shape of kodmial/aa#491.
    plan = triage.build_repair_issue_plan(
        triage=blocked_outcome(),
        run=source_run(),
        artifact_name=EXPECTED_ARTIFACT_NAME,
        fingerprint_key=key,
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


def test_open_issue_491_with_same_fingerprint_never_duplicates() -> None:
    """A later run of the same fingerprint must never mint a second issue."""
    plan = triage.build_repair_issue_plan(
        triage=blocked_outcome(),
        run=source_run(),
        artifact_name=EXPECTED_ARTIFACT_NAME,
        fingerprint_key="EVIDENCE_ARTIFACT_MISSING",
        existing_fingerprints=triage.parse_issue_fingerprints([issue_491_open_body()]),
    )
    assert plan.fingerprint == EXPECTED_FINGERPRINT
    assert plan.should_create is False
    assert "already has an open repair issue" in plan.reason
    assert plan.title == "" and plan.body == ""


def test_retried_same_fingerprint_run_reconciles_without_issue_mint(
    reconciler: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hermetic_post: list[str],
) -> None:
    """Same fingerprint, later artifact-absent run: recorded on #419 only."""
    runs_path = _write_json(
        tmp_path / "runs.json",
        runs_payload([run_payload(source_run(), run_id_override=SOURCE_RUN_ID + 1)]),
    )
    comments_path = _write_json(tmp_path / "comments.json", [])
    issues_path = _write_json(tmp_path / "issues.json", [issue_491_open_body()])
    monkeypatch.setattr(reconciler, "collect_main_sha", lambda owner, repo: SOURCE_SHA)
    monkeypatch.setattr(reconciler, "collect_artifact", lambda owner, repo, run_id: (None, False))
    monkeypatch.setattr(reconciler, "_open_bodies", lambda args: [issue_491_open_body()])
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
    assert summary["comment"]["action"] == "create"
    assert created_titles == []
    assert len(hermetic_post) == 1
    assert triage.BLOCK_ARTIFACT_MISSING in hermetic_post[0]
    assert f"run={SOURCE_RUN_ID + 1}" in hermetic_post[0]


def test_idempotent_triage_comment_update_never_reopens_issue(
    reconciler: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hermetic_post: list[str],
) -> None:
    """A retried reconcile of the identical (sha, run) repairs in place."""
    runs_path = _write_json(tmp_path / "runs.json", runs_payload([run_payload(source_run())]))
    issues_path = _write_json(tmp_path / "issues.json", [issue_491_open_body()])
    monkeypatch.setattr(reconciler, "collect_main_sha", lambda owner, repo: SOURCE_SHA)
    monkeypatch.setattr(reconciler, "collect_artifact", lambda owner, repo, run_id: (None, False))
    monkeypatch.setattr(reconciler, "_open_bodies", lambda args: [issue_491_open_body()])
    created_titles: list[str] = []

    def fake_create(owner: str, repo: str, plan: Any) -> int:
        created_titles.append(plan.title)
        return 999

    monkeypatch.setattr(reconciler, "create_repair_issue", fake_create)
    args_first = argparse_stub(
        runs_file=runs_path,
        comments_file=_write_json(tmp_path / "comments.json", []),
        open_issues_file=issues_path,
        main_sha=SOURCE_SHA,
        post=True,
    )
    assert reconciler.run(args_first) == reconciler.EXIT_OK
    capsys.readouterr()
    assert len(hermetic_post) == 1
    first_comment = hermetic_post[0]
    assert first_comment.startswith(
        f"<!-- aa-scenario2-triage sha={SOURCE_SHA} run={SOURCE_RUN_ID} -->"
    )
    # Second pass with the comment already on #419: update in place.
    args_again = argparse_stub(
        runs_file=runs_path,
        comments_file=_write_json(
            tmp_path / "comments2.json",
            [
                {
                    "id": 65_355_122,
                    "user": {"login": "github-actions[bot]", "type": "Bot"},
                    "body": first_comment,
                }
            ],
        ),
        open_issues_file=issues_path,
        main_sha=SOURCE_SHA,
        post=True,
    )
    exit_code = reconciler.run(args_again)
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert exit_code == reconciler.EXIT_OK
    assert summary["comment"]["action"] == "update"
    assert summary["comment"]["comment_id"] == 65_355_122
    assert summary["repair_issue"]["created"] is None
    assert created_titles == []


# ---------------------------------------------------------------------------
# 3. New healthy evidence classifies the occurrence resolved
# ---------------------------------------------------------------------------


def test_healthy_outcome_is_honest_delivered_without_repair_issue() -> None:
    outcome = triage.map_typed_outcome(healthy_result())
    assert outcome.outcome == triage.OUTCOME_DELIVERED
    assert outcome.blocked_code is None
    assert outcome.requires_repair_issue is False
    assert triage.product_gate_fulfilled(outcome.outcome) is True


def test_cli_new_healthy_run_classifies_issue_491_resolved(
    reconciler: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hermetic_post: list[str],
) -> None:
    """The post-fix push run's DELIVERED triage resolves the open occurrence.

    Mirrors the live post-merge state: the healthy run for the fixed
    machinery sha reconciles while kodmial/aa#491 is still open, and the
    reconciler records ONLY the honest triage comment — no repair issue
    is minted or re-opened for the resolved fingerprint.
    """
    healthy_artifact = write_healthy_artifact(tmp_path, HEALTHY_RUN_ID, HEALTHY_SHA)
    runs_path = _write_json(
        tmp_path / "runs.json",
        runs_payload([run_payload(healthy_run()), run_payload(source_run())]),
    )
    comments_path = _write_json(tmp_path / "comments.json", [])
    issues_path = _write_json(tmp_path / "issues.json", [issue_491_open_body()])
    monkeypatch.setattr(reconciler, "collect_main_sha", lambda owner, repo: HEALTHY_SHA)
    monkeypatch.setattr(
        reconciler, "collect_artifact", lambda owner, repo, run_id: (124_400_000_001, False)
    )
    monkeypatch.setattr(
        reconciler, "download_artifact", lambda owner, repo, artifact_id: healthy_artifact
    )
    monkeypatch.setattr(reconciler, "_open_bodies", lambda args: [issue_491_open_body()])
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
    # Exactly one idempotent, truthful triage comment; the gap is resolved.
    assert len(hermetic_post) == 1
    assert hermetic_post[0].startswith(
        f"<!-- aa-scenario2-triage sha={HEALTHY_SHA} run={HEALTHY_RUN_ID} -->"
    )
    assert f"Typed outcome: `{triage.OUTCOME_DELIVERED}`" in hermetic_post[0]
    assert triage.BLOCK_ARTIFACT_MISSING not in hermetic_post[0]


# ---------------------------------------------------------------------------
# 4. Workflow root-cause regression (the schema defect that produced the
#    startup-failure run without any artifact)
# ---------------------------------------------------------------------------


def diagnostic_steps() -> list[dict[str, Any]]:
    """Parsed `jobs.diagnose` steps of the AA-owned diagnostic workflow."""
    parsed = yaml.safe_load(DIAG_WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    jobs = parsed.get("jobs")
    assert isinstance(jobs, dict)
    diagnose = jobs.get("diagnose")
    assert isinstance(diagnose, dict)
    steps = diagnose.get("steps")
    assert isinstance(steps, list)
    return steps


def test_github_script_steps_nest_script_input_under_with() -> None:
    """Invalid step schema fails the workflow at startup, before any upload.

    Source run 41156665024 starved the reconciler of its artifact because
    the declared-identity ``actions/github-script`` step declared
    ``script`` at step level (not under ``with``), which fails whole-
    workflow schema validation: Actions records a startup failure with
    no jobs and no artifact, and the open issue #491 is born. Every
    github-script step must therefore nest ``script`` under ``with`` and
    must never carry it as a bare step key.
    """
    occurrences = 0
    for step in diagnostic_steps():
        if not str(step.get("uses", "")).startswith("actions/github-script"):
            continue
        occurrences += 1
        assert "script" not in step, "script must be nested under with (startup-failure class)"
        with_block = step.get("with")
        assert isinstance(with_block, dict), "github-script step must declare a with: block"
        assert "script" in with_block
    assert occurrences >= 2  # declared-identity publisher + degraded-notice fallback


def test_evidence_upload_stays_durable_and_fails_loud() -> None:
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


def test_diagnostic_workflow_declared_identity_starts_before_any_job() -> None:
    """Server-authoritative identity still binds evidence to the exact SHA.

    Run 41161688889 resolved #491 with a run-name bound to the checked-out
    SHA and the declared-identity marker; both must remain part of the
    workflow so later reconciles can never misbind evidence.
    """
    parsed = yaml.safe_load(DIAG_WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    run_name = str(parsed.get("run-name", ""))
    assert "target=" in run_name
    steps = diagnostic_steps()
    names = [str(step.get("name", "")) for step in steps]
    assert "Checkout exact SHA" in names
    assert "Verify exact SHA" in names
    assert "Upload diagnostic artifacts" in names
