"""Provider-free contract tests for kodmial/aa#553 validation-only A/B.

The bounded real-book A/B is exercised only through pure helpers and fake
child supervision. No OpenCode server, provider, Telegram network, corpus or
model call is started here. The tests pin the Definition of Done:

- OFF/ON environment isolation inside each child process;
- identical scenario/config identity for both arms;
- no duplicate provider run for the same exact SHA/config;
- bounded execution (two existing 300 s live budgets, 600 s preflight);
- stop before the ON arm on timeout/429/active backoff/insufficient budget;
- honest COMPLETE/INCOMPLETE/INCONCLUSIVE/BLOCKED classification;
- no raw prompt/reply/session/secret leakage in report or result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Any

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "diagnose_gate_c_scenario2.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "aa-gate-c-scenario2-diagnostic.yml"
TRIGGER_PATH = REPO_ROOT / "qualification" / "one_shot_retrieval_ab_553.trigger"


@pytest.fixture(scope="module")
def diag() -> Any:
    module_name = "diagnose_gate_c_scenario2_553"
    spec = importlib.util.spec_from_file_location(module_name, str(SCRIPT_PATH))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _complete_arm(diag: Any, arm: str) -> dict[str, Any]:
    expected = diag.expected_ab_env_flag(arm)
    return {
        "arm": arm,
        "probe": f"scenario2-{arm}",
        "mode": diag.MODE_ONE_SHOT_AB,
        "status": "delivered",
        "delivered": True,
        "timed_out": False,
        "typed_failure": "",
        "env_flag_raw": expected,
        "expected_env_flag": expected,
        "effective_flag": expected == "1",
        "scenario": diag.ab_scenario_identity(),
        "budget": diag._ab_budget(),
        "preflight_elapsed_ms": 1000.0,
        "live_elapsed_ms": 2000.0,
        "delivery_ms": 2000.0,
        "retrieval": {
            "calls_observed": 1,
            "metadata_present": True,
            "retrieval_mode": "one_shot" if expected == "1" else "legacy",
            "fastpath_attempted": expected == "1",
            "fallback_reason": "",
            "coverage_status": "covered",
            "LLM_call_count": 1,
        },
        "attempts": {
            "present": True,
            "model_http_attempts": 4,
            "by_agent": {"retrieval": {"attempts": 1}},
            "failure_categories": [],
            "observed_models": ["provider/model"],
            "request_audit_present": True,
            "provider_429": False,
            "active_backoff": False,
        },
        "telemetry": {
            "answer_outcome": "delivered",
            "verifier_outcome": "supported",
            "verified_book_units": 2,
            "adequacy_verdict": "adequate",
            "planner_latency_ms": 100.0,
            "retrieval_latency_ms": 200.0,
            "answer_latency_ms": 300.0,
            "verifier_latency_ms": 400.0,
        },
        "product": {
            "product_outcome": "answer_success",
            "failure_reason": "",
            "grounded_substantive": True,
            "reply_len": 120,
            "reply_signature": "abc123",
        },
        "provider_429": False,
        "active_backoff": False,
    }


# ---------------------------------------------------------------------------
# Workflow / trigger contract
# ---------------------------------------------------------------------------


def test_workflow_keeps_existing_scenario2_and_adds_opt_in_ab(
    workflow_text: str,
) -> None:
    assert "--mode one-shot-ab" in workflow_text
    assert "--ab-issue 553" in workflow_text
    assert "steps.mode.outputs.mode == 'scenario2'" in workflow_text
    assert "steps.mode.outputs.mode == 'one-shot-ab'" in workflow_text
    # The existing standalone command and artifact remain available.
    assert "--issue 418" in workflow_text
    assert "aa-gate-c-scenario2-diagnostic-${{ github.run_id }}-scenario2" in workflow_text
    # The new artifact is clearly separate and bounded by the same job cap.
    assert "aa-gate-c-scenario2-diagnostic-${{ github.run_id }}-one-shot-ab" in workflow_text
    assert "timeout-minutes: 60" in workflow_text
    # No endless schedule and no broad trigger.
    assert "schedule:" not in workflow_text
    assert '"src/aa/**"' not in workflow_text
    # No separate bot/token is introduced for the A/B.
    assert "TELEGRAM_BOT_TOKEN" not in workflow_text


def test_workflow_mode_selector_is_narrow(workflow_text: str) -> None:
    assert "qualification/one_shot_retrieval_ab_553.trigger" in workflow_text
    assert "workflow_dispatch:" in workflow_text
    assert "mode:" in workflow_text
    assert "one-shot-ab" in workflow_text
    assert "default: scenario2" in workflow_text
    assert "Determine diagnostic mode" in workflow_text
    assert "github.event.before" in workflow_text


def test_trigger_file_is_validation_only() -> None:
    text = TRIGGER_PATH.read_text(encoding="utf-8")
    assert "entry=one-shot-ab" in text
    assert "issue=553" in text
    assert "validation evidence only" in text
    assert "never changes the" in text


# ---------------------------------------------------------------------------
# Identity, isolation and bounds
# ---------------------------------------------------------------------------


def test_arm_environment_is_explicit_and_isolated(diag: Any) -> None:
    off = diag.ab_arm_environment(diag.AB_ARM_OFF)
    on = diag.ab_arm_environment(diag.AB_ARM_ON)
    assert off == {diag.ONE_SHOT_FASTPATH_ENV: "0"}
    assert on == {diag.ONE_SHOT_FASTPATH_ENV: "1"}
    assert off is not on
    with pytest.raises(ValueError):
        diag.ab_arm_environment("unknown")


def test_spawn_child_sets_only_its_arm_flag(
    diag: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    captured: list[dict[str, str]] = []

    class _FakeProcess:
        pass

    def fake_popen(*args: Any, **kwargs: Any) -> Any:
        captured.append(dict(kwargs.get("env") or {}))
        return _FakeProcess()

    monkeypatch.setattr(diag.subprocess, "Popen", fake_popen)
    monkeypatch.delenv(diag.ONE_SHOT_FASTPATH_ENV, raising=False)
    for arm in (diag.AB_ARM_OFF, diag.AB_ARM_ON):
        diag._spawn_child(
            plan=diag.ProbePlan(
                probe=f"scenario2-{arm}",
                snapshot_cadence_s=1.0,
                max_snapshots=1,
                connect_timeout_s=1.0,
                arm_env=diag.ab_arm_environment(arm),
                ab_mode=True,
                ab_arm=arm,
            ),
            out_dir=tmp_path,
        )
    assert captured[0][diag.ONE_SHOT_FASTPATH_ENV] == "0"
    assert captured[1][diag.ONE_SHOT_FASTPATH_ENV] == "1"
    # The parent process never receives the flag from the A/B wiring.
    assert diag.ONE_SHOT_FASTPATH_ENV not in diag.os.environ


def test_ab_identity_is_identical_for_both_arms(diag: Any) -> None:
    scenario = diag.ab_scenario_identity()
    first = diag.ab_config_identity("provider/primary", "provider/fallback")
    second = diag.ab_config_identity("provider/primary", "provider/fallback")
    assert first == second
    assert first["scenario"] == scenario
    assert first["arms"] == [diag.AB_ARM_OFF, diag.AB_ARM_ON]
    assert first["config_digest"] == second["config_digest"]
    # The scenario digest is the production canary prompt, never raw text.
    assert "prompt" not in scenario
    assert len(str(scenario["prompt_sha256"])) == 64


def test_ab_budget_matches_existing_canary_and_is_bounded(diag: Any) -> None:
    budget = diag._ab_budget()
    assert budget["live_budget_s"] == 300.0
    assert budget["preflight_budget_s"] == 600.0
    assert budget["cleanup_reserve_s"] == 20.0
    assert budget["live_model_budget_s"] == 280.0
    assert budget["total_real_model_budget_cap_s"] == 600.0
    assert budget["total_real_model_budget_cap_s"] == 2 * budget["live_budget_s"]
    # Absurd values never create an unbounded arm.
    clamped = diag._ab_budget(1.0, 1.0, 999.0)
    assert clamped["cleanup_reserve_s"] <= clamped["live_budget_s"] / 4.0
    assert clamped["live_model_budget_s"] >= 1.0


# ---------------------------------------------------------------------------
# Honest classification and stop conditions
# ---------------------------------------------------------------------------


def test_complete_requires_every_observed_field(diag: Any) -> None:
    off = _complete_arm(diag, diag.AB_ARM_OFF)
    on = _complete_arm(diag, diag.AB_ARM_ON)
    assert diag._ab_missing_fields(off) == []
    assert diag._ab_missing_fields(on) == []
    assert diag.classify_ab_outcome([off, on]) == diag.AB_STATUS_COMPLETE


def test_absent_telemetry_is_incomplete_not_pass(diag: Any) -> None:
    arm = _complete_arm(diag, diag.AB_ARM_OFF)
    arm["telemetry"] = {}
    arm["retrieval"] = {}
    arm["attempts"] = {}
    arm["product"] = {}
    missing = diag._ab_missing_fields(arm)
    assert "telemetry.snapshot" in missing
    assert "retrieval.metadata" in missing
    assert "attempts.ledger" in missing
    assert "product.product_outcome" in missing
    status = diag.classify_ab_outcome([arm, _complete_arm(diag, diag.AB_ARM_ON)])
    assert status == diag.AB_STATUS_INCOMPLETE
    assert status != diag.AB_STATUS_COMPLETE


def test_effective_flag_must_match_the_arm(diag: Any) -> None:
    arm = _complete_arm(diag, diag.AB_ARM_ON)
    arm["effective_flag"] = False
    assert "effective_flag_mismatch" in diag._ab_missing_fields(arm)
    arm = _complete_arm(diag, diag.AB_ARM_OFF)
    arm["env_flag_raw"] = "1"
    assert "env_flag_effective" in diag._ab_missing_fields(arm)


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        ({"provider_429": True}, "confirmed-429"),
        ({"active_backoff": True}, "active-backoff"),
        ({"timed_out": True}, "timeout"),
        ({"status": "timeout"}, "timeout"),
        ({"status": "bootstrap-failed"}, "bootstrap-failed"),
        ({"status": "failed", "delivered": False}, "arm-not-delivered"),
    ],
)
def test_stop_before_on_arm_on_failure(diag: Any, update: dict[str, Any], expected: str) -> None:
    arm = _complete_arm(diag, diag.AB_ARM_OFF)
    arm.update(update)
    stop, reason = diag._ab_should_stop_after_arm(arm)
    assert stop is True
    assert reason == expected
    stop, reason = diag._ab_should_stop_after_arm(_complete_arm(diag, diag.AB_ARM_OFF))
    assert stop is False
    assert reason == ""


def test_classification_blocks_or_stays_inconclusive(diag: Any) -> None:
    off = _complete_arm(diag, diag.AB_ARM_OFF)
    assert (
        diag.classify_ab_outcome([off], second_arm_skipped_reason="insufficient-job-budget")
        == diag.AB_STATUS_BLOCKED
    )
    assert (
        diag.classify_ab_outcome([off], second_arm_skipped_reason="arm-not-delivered")
        == diag.AB_STATUS_INCONCLUSIVE
    )
    assert diag.classify_ab_outcome([], deduplicated=True) == diag.AB_STATUS_DEDUPLICATED
    assert diag.classify_ab_outcome([]) == diag.AB_STATUS_INCONCLUSIVE
    failed = _complete_arm(diag, diag.AB_ARM_ON)
    failed["status"] = "failed"
    failed["delivered"] = False
    assert diag.classify_ab_outcome([off, failed]) == diag.AB_STATUS_INCONCLUSIVE


def test_durable_complete_marker_deduplicates_same_sha_and_config(diag: Any) -> None:
    config = diag.ab_config_identity("provider/primary", "provider/fallback")
    marker = diag.ab_complete_marker("a" * 40, config["config_digest"])

    class _Publisher:
        def __init__(self, body: str) -> None:
            self.body = body

        def list_comments(self) -> list[dict[str, Any]]:
            return [{"body": self.body}]

    assert diag.ab_already_complete(_Publisher(marker), "a" * 40, config["config_digest"])
    assert not diag.ab_already_complete(_Publisher(marker), "b" * 40, config["config_digest"])
    assert not diag.ab_already_complete(_Publisher(marker), "a" * 40, "different-config")
    assert not diag.ab_already_complete(_Publisher("no marker"), "a" * 40, "x")


# ---------------------------------------------------------------------------
# Parent supervision bounds (fake child, no provider)
# ---------------------------------------------------------------------------


def _ab_args(diag: Any, out_dir: pathlib.Path) -> argparse.Namespace:
    parsed: argparse.Namespace = diag.build_parser().parse_args(
        ["--mode", "one-shot-ab", "--out-dir", str(out_dir)]
    )
    return parsed


def test_deduplicated_run_never_starts_a_child(
    diag: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr(diag, "ab_already_complete", lambda *args, **kwargs: True)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("deduplicated A/B must not start a provider child")

    monkeypatch.setattr(diag, "_run_one_probe", forbidden)
    exit_code = diag._parent_ab_main(_ab_args(diag, tmp_path), tmp_path, 1800.0, "")
    assert exit_code == diag.EXIT_OK
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == diag.AB_STATUS_DEDUPLICATED
    assert result["arms"] == []
    assert result["outcome"] == "deduplicated"


def test_off_then_on_is_sequential_and_complete(
    diag: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    calls: list[str] = []

    def fake_probe(plan: Any, **kwargs: Any) -> Any:
        calls.append(plan.ab_arm)
        evidence = diag.ProbeEvidence(probe=plan.probe, status="delivered")
        evidence.delivered = True
        evidence.ab_result = _complete_arm(diag, plan.ab_arm)
        return evidence

    monkeypatch.setattr(diag, "_run_one_probe", fake_probe)
    exit_code = diag._parent_ab_main(_ab_args(diag, tmp_path), tmp_path, 1800.0, "")
    assert calls == [diag.AB_ARM_OFF, diag.AB_ARM_ON]
    assert exit_code == diag.EXIT_OK
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == diag.AB_STATUS_COMPLETE
    assert [arm["arm"] for arm in result["arms"]] == [
        diag.AB_ARM_OFF,
        diag.AB_ARM_ON,
    ]
    assert result["scenario"] == diag.ab_scenario_identity()
    assert result["arms"][0]["scenario"] == result["arms"][1]["scenario"]


def test_timeout_arm_stops_before_on_arm(
    diag: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    calls: list[str] = []

    def fake_probe(plan: Any, **kwargs: Any) -> Any:
        calls.append(plan.ab_arm)
        evidence = diag.ProbeEvidence(probe=plan.probe, status="timeout")
        evidence.ab_result = {
            "arm": plan.ab_arm,
            "probe": plan.probe,
            "mode": diag.MODE_ONE_SHOT_AB,
            "status": "timeout",
            "delivered": False,
            "timed_out": True,
            "typed_failure": "timeout",
            "env_flag_raw": diag.expected_ab_env_flag(plan.ab_arm),
            "expected_env_flag": diag.expected_ab_env_flag(plan.ab_arm),
            "effective_flag": plan.ab_arm == diag.AB_ARM_ON,
        }
        return evidence

    monkeypatch.setattr(diag, "_run_one_probe", fake_probe)
    exit_code = diag._parent_ab_main(_ab_args(diag, tmp_path), tmp_path, 1800.0, "")
    assert calls == [diag.AB_ARM_OFF]
    assert exit_code == diag.EXIT_BOOTSTRAP_FAILED
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == diag.AB_STATUS_BLOCKED
    assert result["second_arm_skipped_reason"] == "timeout"
    assert len(result["arms"]) == 1


def test_insufficient_budget_skips_on_arm(
    diag: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    calls: list[str] = []

    def fake_probe(plan: Any, **kwargs: Any) -> Any:
        calls.append(plan.ab_arm)
        evidence = diag.ProbeEvidence(probe=plan.probe, status="delivered")
        evidence.delivered = True
        evidence.ab_result = _complete_arm(diag, plan.ab_arm)
        return evidence

    monkeypatch.setattr(diag, "_run_one_probe", fake_probe)
    exit_code = diag._parent_ab_main(_ab_args(diag, tmp_path), tmp_path, 1.0, "")
    assert calls == [diag.AB_ARM_OFF]
    assert exit_code == diag.EXIT_BOOTSTRAP_FAILED
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == diag.AB_STATUS_BLOCKED
    assert result["second_arm_skipped_reason"] == "insufficient-job-budget"


# ---------------------------------------------------------------------------
# Privacy: no raw prompt/reply/session/secret in persisted evidence
# ---------------------------------------------------------------------------


def test_product_evidence_persists_only_length_and_signature(diag: Any) -> None:
    reply = "SECRET_RAW_REPLY_TEXT that must never be persisted"
    snapshot = {"product_outcome": "answer_success", "failure_reason": ""}
    product = diag._ab_product_evidence(reply, snapshot)
    rendered = json.dumps(product, ensure_ascii=False)
    assert "SECRET_RAW_REPLY_TEXT" not in rendered
    assert product["reply_len"] == len(reply)
    assert product["reply_signature"]


def test_ab_evidence_contains_no_raw_text_or_secrets(diag: Any, tmp_path: pathlib.Path) -> None:
    meta = {
        "schema": diag.AB_SCHEMA_VERSION,
        "mode": diag.MODE_ONE_SHOT_AB,
        "main_sha": "a" * 40,
        "run_id": "12345",
        "run_url": "https://github.com/kodmial/aa/actions/runs/12345",
        "scenario": diag.ab_scenario_identity(),
        "config": diag.ab_config_identity("provider/primary", "provider/fallback"),
        "budget": diag._ab_budget(),
        "effective_campaign_allowance_s": 1800.0,
    }
    arms = [
        _complete_arm(diag, diag.AB_ARM_OFF),
        _complete_arm(diag, diag.AB_ARM_ON),
    ]
    result, report = diag._write_ab_evidence(tmp_path, meta, arms, diag.AB_STATUS_COMPLETE)
    rendered = json.dumps(result, ensure_ascii=False) + report
    assert diag.SCENARIO_2_PROMPT not in rendered
    assert "SECRET_RAW_REPLY_TEXT" not in rendered
    assert "session_id" not in rendered
    assert "TELEGRAM_BOT_TOKEN" not in rendered
    assert "token" not in rendered.casefold()
    # The complete status is evidence-only and explicitly not a product PASS.
    assert "NOT a product PASS" in report
    assert "not a qualification verdict" in report.casefold()
    diag.assert_no_text_leak(result)
