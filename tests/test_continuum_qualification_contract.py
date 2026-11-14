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
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
OPENCODE_CALLER = WORKFLOWS / "continuum-opencode.yml"
OPENCODE_WATCHDOG = WORKFLOWS / "continuum-opencode-watchdog.yml"
SCHEDULER_CALLER = WORKFLOWS / "continuum-issue-scheduler.yml"
SCHEDULER_ORIGIN_MARKER = "<!-- continuum-origin role=continuum component=issue-scheduler -->"

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


def _caller_pause_on_failure_expression() -> str:
    data = yaml.safe_load(_read(OPENCODE_CALLER))
    expression = data["jobs"]["call"]["with"]["pause_on_failure"]
    assert isinstance(expression, str)
    return expression


def _scheduler_with_inputs() -> dict[str, Any]:
    data = yaml.safe_load(_read(SCHEDULER_CALLER))
    with_inputs = data["jobs"]["call"]["with"]
    assert isinstance(with_inputs, dict)
    return with_inputs


def _scheduler_dispatch_marker() -> str:
    expression = _scheduler_with_inputs()["dispatch_marker"]
    assert isinstance(expression, str)
    if "${{" not in expression:
        return expression.strip()
    match = re.search(r"\|\|\s*'([^']*)'\s*\}\}", expression)
    assert match is not None, expression
    return match.group(1)


def _scheduler_pause_on_failure() -> str:
    value = _scheduler_with_inputs()["pause_on_failure"]
    assert isinstance(value, str)
    assert value in {"true", "false"}
    return value


def _scheduler_dispatch_body(marker: str) -> str:
    return f"/oc\n\n{marker}\n{SCHEDULER_ORIGIN_MARKER}"


def _evaluate_pause_on_failure(
    *,
    comment_body: str | None,
    dispatch_marker: str,
    pause_on_failure: str,
    vars_dispatch_marker: str = "",
    comment_user_login: str | None = "github-actions[bot]",
    repository_owner: str = "owner",
) -> str:
    """Evaluate the caller's GitHub expression without a live workflow run."""
    expression = _caller_pause_on_failure_expression()
    assert expression.startswith("${{") and expression.endswith("}}")
    py = expression[3:-2].strip()
    py = py.replace("github.event.comment.user.login", "COMMENT_USER_LOGIN")
    py = py.replace("github.event.comment.body", "COMMENT_BODY")
    py = py.replace("github.repository_owner", "REPOSITORY_OWNER")
    py = py.replace("inputs.dispatch_marker", "DISPATCH_MARKER")
    py = py.replace("vars.AUTOMATION_DISPATCH_MARKER", "VARS_DISPATCH_MARKER")
    py = py.replace("inputs.pause_on_failure", "PAUSE_ON_FAILURE")
    py = py.replace("&&", " and ").replace("||", " or ")

    def contains(haystack: str, needle: str) -> bool:
        return needle in haystack

    result: Any = eval(  # noqa: S307 - expression text comes from this repository only
        py,
        {"__builtins__": {}, "contains": contains},
        {
            "COMMENT_BODY": comment_body,
            "COMMENT_USER_LOGIN": comment_user_login,
            "DISPATCH_MARKER": dispatch_marker,
            "PAUSE_ON_FAILURE": pause_on_failure,
            "REPOSITORY_OWNER": repository_owner,
            "VARS_DISPATCH_MARKER": vars_dispatch_marker,
        },
    )
    assert result in {"true", "false"}
    return str(result)


def test_opencode_caller_preserves_aa_knobs() -> None:
    body = _read(OPENCODE_CALLER)
    assert "\"${{ inputs.max_dispatch_attempts || '5' }}\"" in body
    # Normal implementation runs keep the bounded dispatch budget. The
    # scheduler-generated dispatch path is separately pinned non-terminal by
    # test_opencode_caller_matches_scheduler_pause_policy.
    assert "inputs.pause_on_failure || 'true'" in body


def test_opencode_caller_matches_scheduler_pause_policy() -> None:
    # kodmial/aa#506: the scheduler is non-terminal (`pause_on_failure:
    # "false"`). A scheduler-generated `/oc` comment must inherit that policy
    # instead of adding a terminal pause the scheduler then clears, which
    # produced a pause/unpause/re-dispatch loop. Read both the marker and the
    # policy from the scheduler caller so this contract cannot silently drift.
    scheduler_marker = _scheduler_dispatch_marker()
    scheduler_pause = _scheduler_pause_on_failure()
    assert scheduler_marker in _caller_pause_on_failure_expression()
    assert SCHEDULER_ORIGIN_MARKER in _caller_pause_on_failure_expression()
    assert (
        _evaluate_pause_on_failure(
            comment_body=_scheduler_dispatch_body(scheduler_marker),
            dispatch_marker="",
            pause_on_failure="",
            comment_user_login="owner",
            repository_owner="owner",
        )
        == scheduler_pause
    )
    # The reusable scheduler may also post through the workflow token.
    assert (
        _evaluate_pause_on_failure(
            comment_body=_scheduler_dispatch_body(scheduler_marker),
            dispatch_marker="",
            pause_on_failure="",
            comment_user_login="github-actions[bot]",
            repository_owner="owner",
        )
        == scheduler_pause
    )
    # A custom repository marker is honoured on issue_comment runs, where the
    # workflow_dispatch `inputs` context is unavailable. This matches the
    # scheduler, which resolves the same AUTOMATION_DISPATCH_MARKER variable.
    assert (
        _evaluate_pause_on_failure(
            comment_body=_scheduler_dispatch_body("<!-- custom-marker -->"),
            dispatch_marker="",
            pause_on_failure="",
            vars_dispatch_marker="<!-- custom-marker -->",
            comment_user_login="owner",
            repository_owner="owner",
        )
        == scheduler_pause
    )
    # An explicit dispatch marker still wins over the repository variable.
    assert (
        _evaluate_pause_on_failure(
            comment_body=_scheduler_dispatch_body("<!-- input-marker -->"),
            dispatch_marker="<!-- input-marker -->",
            pause_on_failure="",
            vars_dispatch_marker="<!-- other-marker -->",
            comment_user_login="owner",
            repository_owner="owner",
        )
        == scheduler_pause
    )
    # Manual owner `/oc` comments are not scheduler dispatches and keep the
    # bounded terminal-pause default, even when they quote the public marker.
    assert (
        _evaluate_pause_on_failure(
            comment_body=f"/oc\n\n> {scheduler_marker}",
            dispatch_marker="",
            pause_on_failure="",
            comment_user_login="owner",
            repository_owner="owner",
        )
        == "true"
    )
    # The public marker alone is not automation identity: a non-owner comment
    # cannot claim the scheduler's non-terminal policy merely by copying it.
    assert (
        _evaluate_pause_on_failure(
            comment_body=_scheduler_dispatch_body(scheduler_marker),
            dispatch_marker="",
            pause_on_failure="",
            comment_user_login="random-user",
            repository_owner="owner",
        )
        == "true"
    )
    # A workflow_dispatch call has no comment body and keeps its explicit
    # override (or the bounded default when unset).
    assert (
        _evaluate_pause_on_failure(
            comment_body=None,
            dispatch_marker="",
            pause_on_failure="",
            comment_user_login=None,
        )
        == "true"
    )
    assert (
        _evaluate_pause_on_failure(
            comment_body=None,
            dispatch_marker="",
            pause_on_failure="false",
            comment_user_login=None,
        )
        == "false"
    )


def test_opencode_watchdog_bounded_and_non_terminal() -> None:
    data = yaml.safe_load(_read(OPENCODE_WATCHDOG))
    call = data["jobs"]["call"]
    assert call["uses"] == (
        "kodmial/continuum/.github/workflows/continuum-opencode-watchdog.yml@main"
    )
    with_inputs = call["with"]
    assert with_inputs["continuum_ref"] == "main"
    assert with_inputs["watched_workflow"] == "OpenCode agent"
    # Keep real finite retry budgets: the watchdog must never retry forever,
    # and genuine 429 recovery must stay armed with the reviewed upstream
    # fresh-runner restart budget.
    assert with_inputs["max_recovery_retries"] == "3"
    assert with_inputs["max_429_runner_restarts"] == "3"
    # The scheduler owns the terminal/non-terminal decision and is pinned
    # non-terminal; the watchdog must not override it with a human pause.
    assert with_inputs["pause_on_failure"] == "false"
    # Duplicate watcher loops would duplicate recovery comments/dispatches.
    watching = sorted(
        path.name
        for path in (*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml"))
        if "continuum-opencode-watchdog.yml@" in path.read_text(encoding="utf-8")
    )
    assert watching == [OPENCODE_WATCHDOG.name]
    # PR #505 proposed a 15-minute job timeout after an old stacked run was
    # cancelled mid-recovery. Current accepted watchdog runs finish in
    # seconds and do not hit the upstream 5-minute default; no timeout
    # override is needed and adding one would only mask a future hang.
    assert "timeout_minutes" not in with_inputs


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
    # AA intentionally runs three concurrent implementation slots: wip_limit
    # defaults to '3' so scheduled and event-driven reconciliation does not
    # depend on repository variable drift. Issue #146 mandates non-terminal
    # convergence semantics, so pause_on_failure is pinned to hard "false"
    # (a bare inputs passthrough would let repository variable drift rearm
    # terminal pause). All other knobs stay as bare passthroughs so an empty
    # value lets the engine fall back to vars.AUTOMATION_*.
    assert "\"${{ inputs.wip_limit || '3' }}\"" in body
    assert 'pause_on_failure: "false"' in body
    for knob in (
        "max_dispatch_attempts",
        "require_priority_label",
        "opencode_dispatch",
        "count_open_prs_as_wip",
    ):
        assert f'"${{{{ inputs.{knob} }}}}"' in body
        assert f"inputs.{knob} ||" not in body
    assert not re.search(r"continuum.*@[0-9a-f]{40}", body)
