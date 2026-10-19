"""Prevents provider outages, false gate errors, and changed fingerprints from
creating unbounded AA source-code mutation PRs. Final qualification stays strict.
"""

import json
import subprocess
import sys
from pathlib import Path

from aa.qualification.repair_policy import plan_repair


def failure(gate, category, component="live-production-path"):
    return {
        "gate": gate,
        "status": "FAIL",
        "failure_category": category,
        "component": component,
    }


def test_provider_access_error_is_not_a_code_repair():
    result = plan_repair([failure("C", "live-summarizer-probe-OpenCodeProviderAccessError")])
    assert result.actionable == ()
    assert result.held == (("C", "provider-or-infrastructure"),)


def test_semantic_failure_is_actionable_without_global_breaker():
    result = plan_repair([failure("C", "live-book-grounding-substantive-drinking-2")])
    assert result.actionable == (0,)


def test_global_breaker_stops_new_c_fingerprints_but_not_independent_a():
    cases = [failure("A", "model-policy-mismatch", "model-policy"),
             failure("C", "live-book-grounding-substantive-drinking-2")]
    result = plan_repair(cases, gate_c_breaker=True)
    assert result.actionable == (0,)
    assert result.held == (("C", "global-gate-c-breaker"),)


def test_qualified_model_403_and_broken_performance_are_not_code_mutations():
    cases = [failure("C", "OpenCodeProviderAccessError"),
             failure("E", "ordinary-turn-failed", "slo")]
    result = plan_repair(cases)
    assert result.actionable == ()
    assert result.held == (
        ("C", "provider-or-infrastructure"),
        ("E", "latency-inconclusive-while-c-unavailable"),
    )


def test_clean_standalone_e_latency_regression_remains_actionable():
    assert plan_repair([failure("E", "ordinary-turn-failed", "slo")]).actionable == (0,)


def test_ambiguous_c_or_harness_failure_cannot_spawn_a_new_p0():
    cases = [failure("C", "workflow-step-failed", "workflow-step"),
             failure("B", "evidence-invalid", "unknown"),
             failure("C", "live-unclassified-abort")]
    assert plan_repair(cases).actionable == ()


def test_only_one_aa_repair_in_flight_regardless_of_fingerprint():
    result = plan_repair(
        [failure("A", "model-policy-mismatch", "model-policy"),
         failure("C", "live-book-grounding-substantive-drinking-2")],
        in_flight_repair=True,
    )
    assert result.actionable == ()
    assert len(result.held) == 2


def test_gate_d_optional_external_telegram_never_creates_code_repair():
    assert plan_repair([failure("D", "external-telegram-down")]).actionable == ()


def test_cli_uses_same_policy_and_fails_closed_on_malformed_input():
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "scripts" / "plan_qualification_repair.py")]
    payload = {"failures": [failure("C", "OpenCodeProviderAccessError")]}
    run = subprocess.run(cmd, input=json.dumps(payload), text=True, capture_output=True, check=True)
    assert json.loads(run.stdout)["actionable"] == []
    malformed = subprocess.run(cmd, input='{"failures": "not-a-list"}', text=True, capture_output=True)
    assert malformed.returncode != 0


def test_workflow_uses_policy_before_publishing_code_repair():
    workflow = (Path(__file__).resolve().parents[1] /
                ".github" / "workflows" / "aa-self-proving-qualification.yml").read_text()
    assert "scripts/plan_qualification_repair.py" in workflow
    assert "repair-policy decision" in workflow
