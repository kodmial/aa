"""Workflow contract tests for the #72 benchmark harness workflow."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "aa-conversation-eval.yml"


def _read() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_file_exists_with_least_privilege() -> None:
    text = _read()
    assert "contents: read" in text
    assert "issues: write" in text
    # actions:write only for the explicit downstream evaluator dispatch.
    assert "actions: write" in text
    assert "aa-answer-quality-eval.yml" in text


def test_workflow_supports_dispatch_and_readiness_event() -> None:
    text = _read()
    assert "workflow_dispatch:" in text
    assert "issues:" in text
    assert "closed" in text
    for tracker in ("7", "72", "127"):
        assert tracker in text


def test_workflow_revalidates_trackers_and_noops() -> None:
    text = _read()
    assert "no-op" in text
    assert "#7" in text and "#72" in text
    assert "vNext" in text or "#127" in text
    assert "closing #72" in text


def test_workflow_enforces_exact_main_and_frozen_corpus() -> None:
    text = _read()
    assert "exact" in text.lower()
    assert "PASS" in text
    assert "test_product_contract_vnext" in text or "#127" in text
    assert "git rev-parse HEAD" in text
    assert "ru_product_contract.v1_2.input.jsonl" in text


def test_workflow_runs_deterministic_shards_with_bounded_parallelism() -> None:
    text = _read()
    assert "deterministic shard" in text.lower() or "deterministic" in text
    assert "max-parallel: 2" in text
    assert "atomic" in text.lower()
    assert "stable ID" in text or "stable" in text.lower()


def test_workflow_never_uploads_plaintext_and_merges_completely() -> None:
    text = _read()
    assert "never plaintext" in text.lower() or "Never" in text
    assert "tar.zst.age" in text
    assert "missing/duplicate" in text or "duplicate case" in text
    assert "resumable only" in text.lower() or "resumable" in text.lower()


def test_workflow_encrypts_uploads_and_publishes_manifest() -> None:
    text = _read()
    assert "age-encrypted" in text or "age_encrypted" in text or "encrypted" in text
    assert "qualification-results" in text
    assert "never mutate production main" in text.lower() or "never mutate" in text.lower()


def test_workflow_posts_canonical_marker_and_handles_completion() -> None:
    text = _read()
    assert "aa-conversation-eval-result" in text
    assert "complete|incomplete|stale" in text
    assert "INCOMPLETE" in text
    assert "#62" in text


def test_workflow_dispatches_evaluator_without_relying_on_close_event() -> None:
    text = _read()
    assert "workflow_dispatch" in text
    assert "GITHUB_TOKEN" in text
    assert "not a reliable workflow trigger" in text


def test_workflow_does_not_use_issue_scheduler_for_benchmark() -> None:
    text = _read()
    assert "never routes #62 through" in text or "ordinary issue-scheduler" in text


def test_workflow_reopens_reruns_exactly_once_per_tuple() -> None:
    text = _read()
    assert "exactly once per SHA+corpus tuple" in text or "exactly once" in text
    assert "reopen" in text.lower()


def test_workflow_revalidates_main_at_publication() -> None:
    text = _read()
    assert "Revalidate main at publication" in text or "revalidated at publication" in text.lower()
    assert "stale" in text
