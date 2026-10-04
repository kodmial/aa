"""Continuum qualification caller contract tests (issue #69).

Keeps AA's live ``@main`` callers compatible with the qualification
lifecycle: the OpenCode caller must accept ``qualification`` mode with the
immutable run-identity triple, and the scheduler caller must wake on
trusted qualification-result comments without letting public comments
drive state transitions.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENCODE_CALLER = REPO_ROOT / ".github" / "workflows" / "continuum-opencode.yml"
SCHEDULER_CALLER = REPO_ROOT / ".github" / "workflows" / "continuum-issue-scheduler.yml"

TRUSTED_ASSOCIATIONS = ("'OWNER'", "'MEMBER'", "'COLLABORATOR'")
RESULT_MARKERS = (
    "continuum-qualification-result",
    "continuum-docker-qualification-result",
    "continuum-render-qualification-result",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_opencode_caller_accepts_qualification_mode() -> None:
    body = _read(OPENCODE_CALLER)
    assert "- qualification" in body
    for name in ("capability_number", "qualification_number", "required_sha"):
        assert re.search(rf"^\s+{name}:", body, re.MULTILINE) is not None
        assert f"inputs.{name}" in body


def test_opencode_caller_stays_on_main_without_pin() -> None:
    body = _read(OPENCODE_CALLER)
    assert "kodmial/continuum/.github/workflows/continuum-opencode.yml@main" in body
    assert "continuum_ref: main" in body
    assert not re.search(r"continuum.*@[0-9a-f]{40}", body)
    assert "pinned" not in body.lower()


def test_opencode_caller_preserves_aa_knobs() -> None:
    body = _read(OPENCODE_CALLER)
    # Bare-passthrough contract synced with Continuum main: every knob is
    # forwarded as `"${{ inputs.<knob> }}"` with no `|| '<literal>'` default.
    # An empty passthrough lets the engine fall back to vars.AUTOMATION_*;
    # pinning a literal here would override that repository variable.
    # (`conflict_strategy` keeps its `|| 'merge'` because the callee input
    # is a choice that must receive a listed value on non-dispatch events.)
    for knob in ("max_dispatch_attempts", "pause_on_failure"):
        assert f'"${{{{ inputs.{knob} }}}}"' in body
        assert f"inputs.{knob} ||" not in body


def test_scheduler_wakes_on_trusted_qualification_results() -> None:
    body = _read(SCHEDULER_CALLER)
    # Owner-only manual commands stay intact.
    assert "github.actor == github.repository_owner" in body
    assert "contains(github.event.comment.body, '/oc')" in body
    assert "contains(github.event.comment.body, '/opencode')" in body
    # Trusted qualification-result wake uses the current Continuum rule.
    for association in TRUSTED_ASSOCIATIONS:
        assert f"github.event.comment.author_association == {association}" in body
    assert "github.actor == 'github-actions[bot]'" in body
    for marker in RESULT_MARKERS:
        assert marker in body


def test_scheduler_rejects_untrusted_comments_by_construction() -> None:
    body = _read(SCHEDULER_CALLER)
    gate = body[body.index("jobs:") :]
    assert "author_association" in gate
    # A bare marker match without the trust conjunction would let any
    # public commenter wake reconciliation; the trust check must gate it.
    trust_pos = gate.index("author_association")
    marker_pos = gate.index("continuum-qualification-result")
    assert trust_pos < marker_pos
    assert "&&" in gate


def test_scheduler_preserves_aa_knobs_and_main_ref() -> None:
    body = _read(SCHEDULER_CALLER)
    assert "kodmial/continuum/.github/workflows/continuum-issue-scheduler.yml@main" in body
    assert "continuum_ref: main" in body
    # Bare-passthrough contract synced with Continuum main: every knob is
    # forwarded as `"${{ inputs.<knob> }}"` with no `|| '<literal>'` default.
    # An empty passthrough lets the engine fall back to vars.AUTOMATION_*;
    # pinning a literal here would override that repository variable.
    for knob in (
        "wip_limit",
        "max_dispatch_attempts",
        "require_priority_label",
        "opencode_dispatch",
        "count_open_prs_as_wip",
        "pause_on_failure",
    ):
        assert f'"${{{{ inputs.{knob} }}}}"' in body
        assert f"inputs.{knob} ||" not in body
    assert not re.search(r"continuum.*@[0-9a-f]{40}", body)
