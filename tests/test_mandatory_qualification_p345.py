"""Issue #345: ARG_MAX-safe paginated tracker and shared-SHA dispatch.

Proves the post-merge lifecycle no longer crashes on issue volume,
paginates both lists, runs truly idempotently under concurrency, and
recovers deterministically on infrastructure failure:

- >200 open issues with huge Cyrillic bodies: file-based inventory parses
  the whole set with zero argv failure, finds qualifying items on late
  pages, and never ingests pull requests;
- >500 tracker comments with a trusted PASS beyond the first page: full
  history is honored, forged markers are ignored, corrupt pages are typed
  BLOCKED without triggering another run;
- two capabilities on one main commit: exactly one shared dispatch; new
  SHAs restart with new evidence;
- runner states (success/failure/429/stale/canceled/403/5xx) map to
  bounded idempotent outcomes with no double worker or ghost slot;
- the previous Actions ARG_MAX input size is reproduced and shown healthy
  on the file-based path;
- Product Contract #110 and the #290 breaker stay intact.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from aa.qualification import mandatory_lifecycle as lifecycle

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "aa-mandatory-qualification-lifecycle.yml"
SCRIPT = REPO_ROOT / "scripts" / "reconcile_mandatory_qualification.py"

SHA_A = "a" * 40
SHA_B = "b" * 40

QUALIFYING_BODY = "<!-- automation-qualification: #7 -->\nProve exact-main live Gate C/E."
PLAIN_BODY = "Refactor one helper with unit tests; no live gate."
CYRILLIC_CHUNK = "Описание задачи qualification с кириллицей. " * 40

# Linux single-argument ceiling that the old `python - "$open_json"`
# pattern exceeded (MAX_ARG_STRLEN = 128 KiB).
MAX_ARG_STRLEN = 128 * 1024


def _big_issue(number: int, *, qualifying: bool = False, pr: bool = False) -> dict[str, Any]:
    base = QUALIFYING_BODY if qualifying else PLAIN_BODY
    body = base + "\n" + CYRILLIC_CHUNK
    item: dict[str, object] = {"number": number, "body": body}
    if pr:
        item["pull_request"] = {"url": "https://api.github.com/repos/x/y/pulls/1"}
        item["html_url"] = "https://github.com/x/y/pull/1"
    return item


def _comment(body: str, association: str = "NONE", login: str = "someone") -> dict[str, object]:
    return {"body": body, "association": association, "login": login}


def test_inventory_pages_find_late_qualifying_items_and_skip_prs() -> None:
    pages = [
        [_big_issue(n) for n in range(100, 200)],
        [_big_issue(n) for n in range(200, 300)],
        [_big_issue(336, qualifying=True), _big_issue(337, qualifying=True)]
        + [_big_issue(n) for n in range(300, 380)],
        [_big_issue(n, pr=True) for n in range(400, 430)]
        + [_big_issue(500, qualifying=True, pr=True)],
    ]
    # A pull request carrying the marker must never count, even on page 4.
    found = lifecycle.collect_qualifying_capabilities_from_pages(pages)
    assert found == [336, 337]
    total_bodies = sum(len(str(item["body"])) for page in pages for item in page)
    assert total_bodies > 150 * 1024


def test_inventory_partial_page_fails_closed() -> None:
    with pytest.raises(lifecycle.MandatoryLifecycleError) as exc_info:
        lifecycle.collect_qualifying_capabilities_from_pages(
            [[{"number": 336, "body": QUALIFYING_BODY}], None]
        )
    assert lifecycle.BLOCKED_INVENTORY_INCOMPLETE in str(exc_info.value)


def test_arg_max_repro_old_argv_would_crash_but_file_path_is_healthy(
    tmp_path: Path,
) -> None:
    issues = [_big_issue(n) for n in range(100, 340)]
    issues.append(_big_issue(336, qualifying=True))
    payload = json.dumps([{"number": item["number"], "body": item["body"]} for item in issues])
    # Reproduce the previous Actions failure size: the old single process
    # argument exceeded MAX_ARG_STRLEN deterministically at this volume.
    assert len(payload.encode("utf-8")) > MAX_ARG_STRLEN
    inventory = tmp_path / "inventory.json"
    inventory.write_text(payload, encoding="utf-8")
    inventory.chmod(0o600)
    assert lifecycle.load_capability_inventory_file(inventory) == [336]
    # JSONL streaming variant carries the same inventory safely.
    jsonl = tmp_path / "inventory.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in issues),
        encoding="utf-8",
    )
    jsonl.chmod(0o600)
    assert lifecycle.load_capability_inventory_file(jsonl) == [336]


def test_inventory_file_corrupt_fails_closed(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text('[{"number": 336, ', encoding="utf-8")
    with pytest.raises(lifecycle.MandatoryLifecycleError) as exc_info:
        lifecycle.load_capability_inventory_file(broken)
    assert lifecycle.BLOCKED_INVENTORY_INCOMPLETE in str(exc_info.value)
    missing = tmp_path / "missing.json"
    with pytest.raises(lifecycle.MandatoryLifecycleError):
        lifecycle.load_capability_inventory_file(missing)


def test_full_tracker_history_trusted_pass_beyond_first_page() -> None:
    comments: list[dict[str, object]] = [_comment(f"note {n}") for n in range(540)]
    old_fail = f"<!-- continuum-qualification-result issue=7 sha={SHA_A} result=fail -->"
    comments[10] = _comment(old_fail, association="OWNER", login="owner")
    late_pass = f"<!-- continuum-qualification-result issue=7 sha={SHA_A} result=pass -->"
    comments[520] = _comment(late_pass, association="MEMBER", login="member")
    slim = tmp_path_comments(comments)
    trusted = lifecycle.parse_trusted_results(lifecycle.tracker_comments_to_triples(slim))
    assert trusted[-1] == lifecycle.TrustedResult(sha=SHA_A, result="pass")
    completion = lifecycle.gate_capability_completion(
        capability=336, current_main_sha=SHA_A, results=trusted
    )
    assert completion.may_close is True
    # The old FAIL plus the trusted PASS means no fresh dispatch for SHA_A.
    dispatches: list[lifecycle.MandatoryQualification] = []
    lease = lifecycle.classify_tracker_lease(
        has_in_progress_label=False, has_active_run=False, age_s=0.0
    )
    plan = lifecycle.plan_shared_qualification_dispatch(
        capabilities=[336, 337],
        current_main_sha=SHA_A,
        dispatches=dispatches,
        results=trusted,
        lease=lease,
    )
    assert plan.should_dispatch is False
    assert "trusted pass" in plan.reason


def tmp_path_comments(comments: list[dict[str, object]]) -> list[lifecycle.TrackerComment]:
    return [
        lifecycle.TrackerComment(
            body=str(item["body"]),
            association=str(item["association"]),
            login=str(item["login"]),
        )
        for item in comments
    ]


def test_forged_late_pass_never_counts_and_stale_pass_never_qualifies() -> None:
    comments = [_comment(f"note {n}") for n in range(510)]
    comments[505] = _comment(
        f"<!-- continuum-qualification-result issue=7 sha={SHA_A} result=pass -->",
        association="NONE",
        login="mallory",
    )
    trusted = lifecycle.parse_trusted_results(
        lifecycle.tracker_comments_to_triples(tmp_path_comments(comments))
    )
    assert trusted == []
    completion = lifecycle.gate_capability_completion(
        capability=336, current_main_sha=SHA_B, results=trusted
    )
    assert completion.may_close is False


def test_corrupt_tracker_page_is_typed_blocked(tmp_path: Path) -> None:
    with pytest.raises(lifecycle.MandatoryLifecycleError) as exc_info:
        lifecycle.assemble_comment_history_from_pages(
            [[{"body": "x", "authorAssociation": "OWNER", "author": {"login": "o"}}], None]
        )
    assert lifecycle.BLOCKED_HISTORY_INCOMPLETE in str(exc_info.value)
    broken = tmp_path / "comments.json"
    broken.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(lifecycle.MandatoryLifecycleError) as exc_info2:
        lifecycle.load_tracker_comments_file(broken)
    assert lifecycle.BLOCKED_HISTORY_INCOMPLETE in str(exc_info2.value)


def test_shared_sha_exactly_one_dispatch_and_new_sha_restarts() -> None:
    idle = lifecycle.classify_tracker_lease(
        has_in_progress_label=False, has_active_run=False, age_s=0.0
    )
    first = lifecycle.plan_shared_qualification_dispatch(
        capabilities=[336, 337],
        current_main_sha=SHA_A,
        dispatches=[],
        results=[],
        lease=idle,
    )
    assert first.should_dispatch is True
    # Deterministic single claimant: smallest capability owns the run.
    assert first.owner_capability == 336
    assert first.waiting_capabilities == (336, 337)
    marker = lifecycle.format_shared_dispatch_marker([336, 337], SHA_A)
    assert "capabilities=336,337" in marker and SHA_A in marker
    parsed = lifecycle.parse_shared_dispatch_markers([marker])
    assert parsed == [(SHA_A, [336, 337])]
    # Duplicate push / PR-close / tracker-comment events reuse the run.
    recorded = [lifecycle.build_dispatch_tuple(336, SHA_A)]
    for duplicate_caps in ([337], [336, 337], [337, 336]):
        repeat = lifecycle.plan_shared_qualification_dispatch(
            capabilities=duplicate_caps,
            current_main_sha=SHA_A,
            dispatches=recorded,
            results=[],
            lease=idle,
        )
        assert repeat.should_dispatch is False
        assert "no duplicate" in repeat.reason
    # An active lease also refuses a second concurrent dispatch.
    active = lifecycle.classify_tracker_lease(
        has_in_progress_label=True, has_active_run=True, age_s=5.0
    )
    held = lifecycle.plan_shared_qualification_dispatch(
        capabilities=[337], current_main_sha=SHA_A, dispatches=[], results=[], lease=active
    )
    assert held.should_dispatch is False
    # A new code SHA properly restarts with new evidence.
    restart = lifecycle.plan_shared_qualification_dispatch(
        capabilities=[336, 337],
        current_main_sha=SHA_B,
        dispatches=recorded,
        results=[],
        lease=idle,
    )
    assert restart.should_dispatch is True


def test_runner_states_bounded_idempotent() -> None:
    live = lifecycle.classify_runner_state(status="in_progress")
    assert (live.action, live.retry_allowed) == ("retain", False)
    queued = lifecycle.classify_runner_state(status="queued")
    assert queued.action == "retain"
    limited = lifecycle.classify_runner_state(status="rate_limited_429", attempt=0)
    assert (limited.action, limited.retry_allowed) == ("retry-bounded", True)
    exhausted = lifecycle.classify_runner_state(status="rate_limited_429", attempt=3)
    assert (exhausted.action, exhausted.retry_allowed) == ("blocked-terminal", False)
    postponed = lifecycle.classify_runner_state(status="postponed", attempt=1)
    assert postponed.action == "retry-bounded"
    forbidden = lifecycle.classify_runner_state(status="completed", http_status=403, attempt=0)
    assert forbidden.action == "retry-bounded"
    server = lifecycle.classify_runner_state(status="runner_error", http_status=503, attempt=2)
    assert server.action == "retry-bounded"
    stale = lifecycle.classify_runner_state(status="canceled", sha_matches_head=False)
    assert stale.action == "re-dispatch-exact-main"
    same_cancel = lifecycle.classify_runner_state(status="canceled", sha_matches_head=True)
    assert same_cancel.action == "retry-bounded"
    success = lifecycle.classify_runner_state(status="completed", conclusion="success")
    assert (success.action, success.retry_allowed) == ("none", False)
    failure = lifecycle.classify_runner_state(status="completed", conclusion="failure")
    assert failure.action == "none"
    wrong = lifecycle.classify_runner_state(status="wrong_attempt")
    assert (wrong.action, wrong.retry_allowed) == ("retain", False)
    unknown = lifecycle.classify_runner_state(status="mystery")
    assert unknown.action == "blocked-terminal"
    assert lifecycle.should_retry_infra_failure(failure_kind="terminal", attempt=0) is False
    assert lifecycle.should_retry_infra_failure(failure_kind="rate_limited_429", attempt=2) is True
    # Cron recovery is bounded and respects the #290 breaker.
    assert lifecycle.needs_scheduled_recovery(failure_kind="rate_limited_429", attempt=0) is True
    assert lifecycle.needs_scheduled_recovery(failure_kind="rate_limited_429", attempt=9) is False
    assert lifecycle.needs_scheduled_recovery(failure_kind="terminal", attempt=0) is False
    for fingerprint in lifecycle.EXHAUSTED_BREAKER_FINGERPRINTS:
        assert (
            lifecycle.needs_scheduled_recovery(
                failure_kind="rate_limited_429", attempt=0, fingerprint=fingerprint
            )
            is False
        )


def test_genuinely_active_run_vs_ghost_slot() -> None:
    assert lifecycle.is_genuinely_active_run(has_active_run=False) is False
    assert (
        lifecycle.is_genuinely_active_run(has_active_run=True, run_sha=SHA_A, current_sha=SHA_A)
        is True
    )
    assert (
        lifecycle.is_genuinely_active_run(has_active_run=True, run_sha=SHA_B, current_sha=SHA_A)
        is False
    )
    assert lifecycle.is_genuinely_active_run(has_active_run=True, run_conclusion="success") is False
    assert (
        lifecycle.is_genuinely_active_run(has_active_run=True, run_conclusion="cancelled") is False
    )


def test_dispatch_record_roundtrip_is_privacy_safe() -> None:
    record = lifecycle.build_dispatch_record(
        capabilities=[336, 337],
        sha=SHA_A,
        actor="github-actions[bot]",
        run_id="123",
        run_url="https://github.com/x/y/actions/runs/123",
    )
    assert record["capabilities"] == [336, 337]
    assert record["qualification"] == 7
    assert record["contract"] == lifecycle.CONTRACT_VERSION
    blob = lifecycle.serialize_record(record)
    assert "кириллиц" not in blob and "SECRET" not in blob
    assert lifecycle.parse_record_text(blob)["sha"] == SHA_A
    blocked = lifecycle.build_blocked_record(
        capabilities=[336],
        sha=SHA_A,
        blocked_code=lifecycle.BLOCKED_HISTORY_INCOMPLETE,
        detail="page 6 unavailable",
    )
    assert "never PASS" in blocked["blocked_reason"]
    with pytest.raises(lifecycle.MandatoryLifecycleError):
        lifecycle.parse_record_text('{"capabilities": [336]}')


def test_workflow_is_arg_max_safe_paginated_and_shared() -> None:
    body = WORKFLOW.read_text(encoding="utf-8")
    # The crashing pattern is gone: no giant JSON in argv or shell capture.
    assert 'open_json="$(gh issue list' not in body
    assert "sys.argv[1]" not in body
    assert "--limit 200 --json number,body" not in body
    assert "gh issue view 7 --comments" not in body
    # Paginated REST reads into chmod-600 files, loaded from files only.
    assert "--paginate" in body or "per_page=100&page=" in body
    assert "chmod 600" in body
    assert "--inventory-file" in body
    assert "--comments-file" in body
    assert "--record-file" in body
    # Shared single-claimant dispatch plus deterministic cron recovery.
    assert "concurrency:" in body
    assert "cancel-in-progress: false" in body
    assert "cron:" in body
    assert "schedule" in body
    assert "aa-self-proving-qualification.yml" in body
    assert "required_sha" in body
    assert "Closes #" not in body
    assert "Fixes #" not in body


def test_reconcile_script_shared_modes(tmp_path: Path) -> None:
    inventory = tmp_path / "inv.jsonl"
    rows = [_big_issue(n) for n in range(100, 120)]
    rows.extend([_big_issue(336, qualifying=True), _big_issue(337, qualifying=True)])
    inventory.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    comments = tmp_path / "comments.json"
    comments.write_text(json.dumps([]))
    record = tmp_path / "record.json"

    def run(*extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--sha",
                SHA_A,
                "--inventory-file",
                str(inventory),
                "--comments-file",
                str(comments),
                "--record-file",
                str(record),
                "--dry-run",
                *extra,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    first = run()
    assert first.returncode == 0, first.stderr
    assert '"dispatch": true' in first.stdout
    assert "owner" in first.stdout
    saved = json.loads(record.read_text(encoding="utf-8"))
    assert saved["capabilities"] == [336, 337]
    # Recording the shared dispatch suppresses the duplicate second event.
    marker = lifecycle.format_dispatch_marker(lifecycle.build_dispatch_tuple(336, SHA_A))
    comments.write_text(json.dumps([{"body": marker, "association": "", "login": ""}]))
    second = run()
    assert second.returncode == 0, second.stderr
    assert '"dispatch": false' in second.stdout
    # Corrupt history is typed BLOCKED and never dispatches.
    comments.write_text("not json")
    third = run()
    assert third.returncode == 2
    assert "blocked" in third.stdout.lower()
    assert 'dispatch": true' not in third.stdout


def test_contract_and_breaker_preserved() -> None:
    assert lifecycle.CONTRACT_VERSION == "product-contract-110/immutable"
    assert lifecycle.QUALIFICATION_ISSUE_NUMBER == 7
    for fingerprint in lifecycle.EXHAUSTED_BREAKER_FINGERPRINTS:
        assert lifecycle.is_repair_allowed(fingerprint) is False
    assert lifecycle.is_repair_allowed("C:some-new-component:live-production-path") is True
    reason = lifecycle.typed_blocker_reason(
        lifecycle.BLOCKER_TELEGRAM_TOKEN, detail="no token in CI"
    )
    assert "never PASS" in reason
