"""Mandatory exact-main qualification lifecycle regression matrix (#336).

Proves the IMPLEMENTED -> QUALIFYING -> QUALIFIED lifecycle is wired as a
mandatory post-merge gate instead of GitHub auto-close:

- declared product capabilities stay OPEN as ``automation:qualifying``
  with non-closing PR references until trusted exact-main #7 PASS;
- dispatch is observable/idempotent on the immutable tuple
  (capability, #7, exact SHA, contract);
- reusable tracker #7 leases never deadlock dispatch;
- only trusted exact-SHA PASS may complete a capability;
- external blockers stay typed FAIL/BLOCKED, never synthetic PASS;
- the #290 convergence breaker stays engaged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aa.qualification import mandatory_lifecycle as lifecycle

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "aa-mandatory-qualification-lifecycle.yml"
SELF_PROVING_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "aa-self-proving-qualification.yml"

SHA_A = "a" * 40
SHA_B = "b" * 40

MANDATORY_BODY = """<!-- continuum-task:implementation -->
<!-- automation-qualification: #7 -->
<!-- product-contract: #110 -->

## DoD
Prove exact-main live qualification Gate C/E on the exact merged SHA.
"""

CODE_ONLY_BODY = """<!-- continuum-task:implementation -->

## DoD
Refactor one helper with unit tests; no live qualification.
"""


def test_mandatory_vs_code_only_reference_and_state() -> None:
    assert lifecycle.pr_reference_kind(MANDATORY_BODY) == "Relates to"
    assert lifecycle.pr_reference_kind(CODE_ONLY_BODY) == "Closes"
    assert lifecycle.post_merge_state(MANDATORY_BODY) == "automation:qualifying"
    assert lifecycle.post_merge_state(CODE_ONLY_BODY) == "closed"


def test_no_bypass_via_mutable_pr_text() -> None:
    ok, _ = lifecycle.validate_pr_body_for_capability(MANDATORY_BODY, "Relates to #336", 336)
    assert ok is True
    for closing in ("Closes #336", "Fixes #336", "Resolves #336", "closes #336"):
        blocked, reason = lifecycle.validate_pr_body_for_capability(MANDATORY_BODY, closing, 336)
        assert blocked is False
        assert "Relates to" in reason
    # An unrelated PR must not close the declared capability.
    allowed, _ = lifecycle.validate_pr_body_for_capability(MANDATORY_BODY, "Closes #999", 336)
    assert allowed is True
    # Ordinary tasks keep the closing path.
    ok_code, _ = lifecycle.validate_pr_body_for_capability(CODE_ONLY_BODY, "Closes #100", 100)
    assert ok_code is True


def test_original_pinned_spec_still_validates() -> None:
    ok, reason = lifecycle.validate_task_admission(MANDATORY_BODY)
    assert ok is True
    assert "#7" in reason


def test_missing_marker_is_setup_error_not_silent_close() -> None:
    body = "## DoD\nProve exact-main live Gate C qualification on the merged SHA.\n"
    assert lifecycle.requires_exact_main_qualification(body) is True
    assert lifecycle.parse_automation_qualification(body) is None
    ok, reason = lifecycle.validate_task_admission(body)
    assert ok is False
    assert reason.startswith("setup-error")
    assert lifecycle.requires_exact_main_qualification(CODE_ONLY_BODY) is False


def test_wrong_tracker_is_rejected() -> None:
    body = MANDATORY_BODY.replace("#7", "#8")
    ok, _ = lifecycle.validate_task_admission(body)
    assert ok is False


def test_idempotent_dispatch_single_new_sha() -> None:
    first = lifecycle.should_dispatch_qualification(
        capability=336, current_main_sha=SHA_A, dispatches=[], results=[]
    )
    assert first.dispatch is True
    marker = lifecycle.format_dispatch_marker(lifecycle.build_dispatch_tuple(336, SHA_A))
    assert "capability=336" in marker and SHA_A in marker and "contract=" in marker
    recorded = lifecycle.parse_dispatch_markers([marker])
    second = lifecycle.should_dispatch_qualification(
        capability=336, current_main_sha=SHA_A, dispatches=recorded, results=[]
    )
    assert second.dispatch is False
    assert "no duplicate" in second.reason


def test_old_sha_result_never_satisfies_new_main() -> None:
    old_pass = [lifecycle.TrustedResult(sha=SHA_A, result="pass")]
    decision = lifecycle.should_dispatch_qualification(
        capability=336, current_main_sha=SHA_B, dispatches=[], results=old_pass
    )
    assert decision.dispatch is True
    completion = lifecycle.gate_capability_completion(
        capability=336, current_main_sha=SHA_B, results=old_pass
    )
    assert completion.may_close is False
    assert "stale" in completion.reason


def test_only_trusted_exact_sha_pass_may_close() -> None:
    current_pass = [lifecycle.TrustedResult(sha=SHA_A, result="pass")]
    assert (
        lifecycle.gate_capability_completion(
            capability=336, current_main_sha=SHA_A, results=current_pass
        ).may_close
        is True
    )
    assert (
        lifecycle.gate_capability_completion(
            capability=336,
            current_main_sha=SHA_A,
            results=[lifecycle.TrustedResult(sha=SHA_A, result="fail")],
        ).may_close
        is False
    )
    assert (
        lifecycle.gate_capability_completion(
            capability=336, current_main_sha=SHA_A, results=[]
        ).may_close
        is False
    )


def test_forged_untrusted_marker_never_counts() -> None:
    comments = [
        (
            f"<!-- continuum-qualification-result issue=7 sha={SHA_A} result=pass -->",
            "NONE",
            "mallory",
        ),
    ]
    assert lifecycle.parse_trusted_results(comments) == []
    trusted = [
        (
            f"<!-- continuum-qualification-result issue=7 sha={SHA_A} result=pass -->",
            "OWNER",
            "owner",
        ),
        (
            f"<!-- continuum-qualification-result issue=7 sha={SHA_A} result=fail -->",
            "NONE",
            "mallory",
        ),
    ]
    parsed = lifecycle.parse_trusted_results(trusted)
    assert parsed == [lifecycle.TrustedResult(sha=SHA_A, result="pass")]
    assert lifecycle.is_trusted_author(association="MEMBER", login="someone") is True
    assert lifecycle.is_trusted_author(association="NONE", login="github-actions[bot]") is True
    assert lifecycle.is_trusted_author(association="NONE", login="mallory") is False


def test_tracker_lease_active_vs_orphaned_recovery() -> None:
    active = lifecycle.classify_tracker_lease(
        has_in_progress_label=True, has_active_run=True, age_s=10.0
    )
    assert active.state == "active"
    assert lifecycle.orphaned_lease_recovery(active).action == "retain"
    orphaned = lifecycle.classify_tracker_lease(
        has_in_progress_label=True, has_active_run=False, age_s=7200.0
    )
    assert orphaned.state == "orphaned"
    recovery = lifecycle.orphaned_lease_recovery(orphaned)
    assert recovery.action == "re-dispatch-exact-main"
    assert "/oc" not in recovery.reason
    idle = lifecycle.classify_tracker_lease(
        has_in_progress_label=False, has_active_run=False, age_s=0.0
    )
    assert idle.state == "idle"
    assert lifecycle.orphaned_lease_recovery(idle).action == "none"
    with pytest.raises(lifecycle.MandatoryLifecycleError):
        lifecycle.classify_tracker_lease(
            has_in_progress_label=True, has_active_run=False, age_s=-1.0
        )


def test_concurrent_same_sha_merges_deduplicate() -> None:
    dispatches = [lifecycle.build_dispatch_tuple(336, SHA_A)]
    decision = lifecycle.should_dispatch_qualification(
        capability=337, current_main_sha=SHA_A, dispatches=dispatches, results=[]
    )
    # A second capability on the same SHA still dispatches once for itself,
    # but the identical capability/SHA pair never dispatches twice.
    assert decision.dispatch is True
    repeat = lifecycle.should_dispatch_qualification(
        capability=336, current_main_sha=SHA_A, dispatches=dispatches, results=[]
    )
    assert repeat.dispatch is False


def test_typed_external_blockers_never_fabricate_pass() -> None:
    for code in sorted(lifecycle.TYPED_BLOCKERS):
        reason = lifecycle.typed_blocker_reason(code, detail="live peer unreachable")
        assert "blocked" in reason
        assert "never PASS" in reason
    with pytest.raises(lifecycle.MandatoryLifecycleError):
        lifecycle.typed_blocker_reason("MADE_UP_BLOCKER")


def test_convergence_breaker_remains_engaged() -> None:
    for fingerprint in lifecycle.EXHAUSTED_BREAKER_FINGERPRINTS:
        assert lifecycle.is_repair_allowed(fingerprint) is False
    assert lifecycle.is_repair_allowed("C:some-new-component:live-production-path") is True


def test_mandatory_workflow_is_observable_idempotent_trigger() -> None:
    body = WORKFLOW.read_text(encoding="utf-8")
    assert "aa-mandatory-qualification-lifecycle" in body
    assert "pull_request_target" in body
    assert "reconcile_mandatory_qualification.py" in body
    assert "required_sha" in body
    assert "mandatory_lifecycle" in body
    # Post-merge fan-out targets the repository-owned A-F workflow with the
    # exact SHA, never a generic no-secret runner.
    assert "aa-self-proving-qualification.yml" in body
    assert "cancel-in-progress: false" in body


def test_self_proving_lane_owns_live_secrets_and_privacy() -> None:
    body = SELF_PROVING_WORKFLOW.read_text(encoding="utf-8")
    assert "TELEGRAM_BOT_TOKEN" in body
    assert "AA_BOOK_AGE_IDENTITY" in body
    assert "required_sha" in body
    assert "exact current main" in body
    # Privacy-safe logs: evidence carries digests/counts, never raw text.
    assert "privacy-safe" in body


def test_no_qualification_bypass_keywords_in_mandatory_workflow() -> None:
    body = WORKFLOW.read_text(encoding="utf-8")
    assert "Closes #" not in body
    assert "Fixes #" not in body
