"""Runtime workflow control-plane regression tests (no network).

Issue #6 requires the bounded runtime to never run two authoritative
pollers: a dedicated Actions concurrency group plus an explicit
active-run check, preferring duplicate rejection over replacement.
Qualification #7 scenarios 10/11 exercise exactly this behavior.
"""

from __future__ import annotations

import pathlib


def _workflows_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _read_workflow(name: str) -> str:
    return (_workflows_root() / name).read_text(encoding="utf-8")


def test_runtime_workflow_has_dedicated_concurrency_group() -> None:
    text = _read_workflow("aa-runtime.yml")
    assert "concurrency:" in text
    assert "group: aa-runtime" in text
    assert "cancel-in-progress: false" in text


def test_runtime_workflow_rejects_duplicate_run_explicitly() -> None:
    text = _read_workflow("aa-runtime.yml")
    assert "Reject duplicate runtime" in text
    assert "listWorkflowRuns" in text
    assert "already active; rejecting duplicate" in text


def test_runtime_control_workflow_keeps_control_concurrency() -> None:
    text = _read_workflow("aa-runtime-control.yml")
    assert "group: aa-runtime-control" in text
    assert "cancel-in-progress: false" in text
    assert "already" in text
