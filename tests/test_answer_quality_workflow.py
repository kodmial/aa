"""Workflow contract tests for the #73 answer-quality evaluator workflow."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "aa-answer-quality-eval.yml"


def _read() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_file_exists_with_least_privilege() -> None:
    text = _read()
    assert "contents: read" in text
    # issues:write only for grading/remediation issue state; actions:write
    # only for the explicit downstream scheduler dispatch; read-only elsewhere.
    assert "issues: write" in text
    assert "actions: write" in text
    assert "issues: read" in text
    assert "actions: read" in text


def test_workflow_supports_dispatch_run_close_and_push_catchup() -> None:
    text = _read()
    assert "workflow_dispatch:" in text
    assert "benchmark_run_id" in text
    assert "workflow_run:" in text
    assert "AA conversation eval" in text
    assert "issues:" in text
    assert "closed" in text
    assert "push:" in text
    assert "qualification/ru_answer_quality_rubric.v2.json" in text
    assert ".github/workflows/aa-answer-quality-eval.yml" in text


def test_workflow_readiness_wakes_on_62_and_73_with_revalidation() -> None:
    text = _read()
    assert "[62, 73]" in text or ("62" in text and "73" in text)
    assert "no-op" in text
    assert "closing #73" in text
    assert "predates" in text


def test_workflow_binds_frozen_rubric_by_checksum() -> None:
    text = _read()
    assert "sha256" in text
    assert "ru_answer_quality_rubric.v2.sha256" in text
    assert "sidecar" in text
    assert "version bump" in text


def test_workflow_grades_each_complete_tuple_exactly_once() -> None:
    text = _read()
    assert "grade exactly once" in text or "exactly-once" in text
    assert "already has a #63 result" in text
    assert "aa-answer-quality-result" in text
    assert "issue=63" in text
    assert "currency" in text
    assert "current" in text and "stale" in text


def test_workflow_consumes_trusted_artifact_privately() -> None:
    text = _read()
    assert "AA_BOOK_AGE_IDENTITY" in text
    assert "never" in text and "log" in text
    for line in text.splitlines():
        if "AA_BOOK_AGE_IDENTITY" in line:
            assert "echo" not in line
    assert "corpus" in text and "checksum" in text
    assert "completeness" in text
    assert "diagnostic" in text


def test_workflow_runs_calibration_before_grading() -> None:
    text = _read()
    assert "--calibration" in text
    assert "run_answer_quality_eval" in text


def test_workflow_remediates_idempotently_and_reconnects_capability_6() -> None:
    text = _read()
    assert "fingerprint" in text
    assert "reopened capability #6" in text or "reopen" in text.lower()
    assert "automation-blocked-by" in text
    assert "continuum-issue-scheduler" in text
    assert "workflow_dispatch" in text
    assert "one remediation issue per root cause" in text or "per root cause" in text


def test_workflow_states_same_model_limitation() -> None:
    text = _read()
    assert "provisional" in text
    assert "independent clinical validation" in text


def test_workflow_never_grades_product_behavior() -> None:
    text = _read()
    assert "never" in text
    assert "product behavior" in text or "Evaluation infrastructure only" in text
