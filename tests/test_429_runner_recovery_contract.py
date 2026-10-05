"""Regression contract for OpenCode 429 hosted-runner recovery."""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_429_recovery_watches_all_live_aa_opencode_workflows() -> None:
    workflow = _text(".github/workflows/aa-real-book-429-recovery.yml")
    assert '"AA real-book retrieval qualification"' in workflow
    assert '"AA bot runtime"' in workflow
    assert "run_attempt < 4" in workflow
    assert "reRunWorkflow" in workflow
    assert "OPENCODE_429_RESTART_REQUIRED" in workflow


def test_runtime_emits_runner_restart_marker_and_exit_code() -> None:
    workflow = _text(".github/workflows/aa-runtime.yml")
    assert 'if [ "$status" -eq 75 ]' in workflow
    assert '"OPENCODE_429_RESTART_REQUIRED"' in workflow
    assert "aa-runtime-429-recovery" in workflow
    assert "exit 75" in workflow


def test_real_book_qualification_uses_same_restart_signal() -> None:
    runner = _text("scripts/run_real_book_retrieval_qualification.py")
    assert "EXIT_RUNNER_RESTART_REQUIRED = 75" in runner
    assert '"OPENCODE_429_RESTART_REQUIRED"' in runner
    assert "except OpenCodeRateLimitError" in runner


def test_429_is_not_a_local_retry_policy() -> None:
    adapter = _text("src/aa/conversation/model_adapter.py")
    orchestrator = _text("src/aa/conversation/orchestrator.py")
    assert adapter.count("except OpenCodeRateLimitError:") >= 2
    assert orchestrator.count("except OpenCodeRateLimitError:") >= 2
