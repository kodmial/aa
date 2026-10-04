"""Runtime workflow control-plane regression tests (no network).

Issue #6 requires the bounded runtime to never run two authoritative
pollers (a dedicated Actions concurrency group plus an explicit
active-run check) and to restore the canonical artifact through the
production encrypted-snapshot path from #28 with no silent plaintext
fallback. Qualification #7 scenarios 10/11/18 exercise this behavior.
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


def test_runtime_uses_encrypted_snapshot_restore() -> None:
    text = _read_workflow("aa-runtime.yml")
    assert "scripts/restore_canonical.py" in text
    assert "AA_BOOK_AGE_IDENTITY" in text


def test_runtime_has_no_silent_plaintext_fallback() -> None:
    text = _read_workflow("aa-runtime.yml")
    assert "scripts/fetch_aa_source.py" not in text
    assert "scripts/build_canonical.py" not in text
    assert "--allow-network-fallback" not in text


def test_runtime_forbids_network_fetch_fallback() -> None:
    text = _read_workflow("aa-runtime.yml")
    assert 'AA_ALLOW_NETWORK_FETCH: "0"' in text
