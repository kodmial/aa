"""Gate C independent unseen-dialogue evaluation (kodmial/aa#341).

Proves the two-tier Gate C semantic evaluator: C-regression (stable
visible deterministic cases, including the preserved 18 fixed turns) plus
C-heldout (sealed unseen inputs provisioned post-code-freeze). A visible
checked-in fixture alone never makes held-out PASS; a missing unseen
source, missing judge, unreadable corpus or provider outage is typed
BLOCKED (an explicit external prerequisite), never product FAIL and never
a silent reuse of regression strings.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from aa.qualification.gate_c_heldout import (
    BLOCKED_BASELINE_MISMATCH,
    BLOCKED_BASELINE_UNPINNED,
    BLOCKED_GENERATOR_UNAVAILABLE,
    BLOCKED_JUDGE_UNAVAILABLE,
    GATE_C_EVALUATOR_VERSION,
    GATE_C_HELDOUT_SCHEMA_VERSION,
    HeldoutSourceProvenance,
    build_gate_c_semantic_report,
    build_heldout_tier_report,
    build_regression_tier_report,
    check_fixture_novelty,
    decide_gate_c_overall,
    describe_independence,
    evaluate_heldout_turn,
    fixture_digest,
    generate_unseen_fixture,
    resolve_heldout_source,
    run_calibration_controls,
    validate_heldout_fixture,
    write_baseline_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _passing_book_snapshot() -> dict[str, Any]:
    return {
        "answer_outcome": "served",
        "verifier_outcome": "passed",
        "verifier_unavailable_units": 0,
        "turn_budget_exceeded": False,
        "planner_query_count": 3,
        "retrieval_passages": 4,
        "verified_book_units": 2,
        "response_units": 3,
        "adequacy_verdict": "pass",
        "answers_request": True,
        "technically_grounded": True,
        "qualified": True,
        "answer_relevant": True,
        "failure_category": "",
        "planner_reason": "",
    }


def _scripted_synth(category: str, thread: str, index: int, seed_hex: str) -> dict[str, Any]:
    safety = "ordinary"
    if category == "emergency":
        safety = "emergency"
    elif category == "out-of-scope":
        safety = "out_of_book"
    return {
        "prompt": f"synth-{seed_hex[:8]}-{index}-{category}",
        "context": f"thread-{thread}",
        "gold_intent_category": category,
        "safety_expectation": safety,
        "requires_book_evidence": safety == "ordinary",
    }


def test_calibration_controls_all_pass() -> None:
    result = run_calibration_controls(regression_prompts=["visible regression turn"])
    assert result.all_passed, result.failed
    assert result.to_dict()["controls_executed"] >= 10


def test_relevant_grounded_passes_while_unrelated_and_unsupported_fail() -> None:
    helpful = "Steady support reply with grounded guidance."
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-001",
        reply=helpful,
        snapshot=_passing_book_snapshot(),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
    )
    assert verdict.outcome == "PASS"
    assert verdict.judge_available is True
    assert verdict.relevance_to_resolved_intent is True
    assert verdict.unsupported_claim_count == 0
    # Self-affirming telemetry with unrelated text fails.
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-002",
        reply=helpful,
        snapshot=dict(_passing_book_snapshot(), adequacy_verdict="fail", answers_request=False),
        judge_helpful=False,
        judge_addresses_intent=False,
        judge_model="probe-model",
    )
    assert verdict.outcome == "FAIL"
    # Polished but book-unsupported advice fails.
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-003",
        reply="Polished generic advice without any book evidence.",
        snapshot=dict(
            _passing_book_snapshot(),
            verified_book_units=0,
            retrieval_passages=0,
            planner_query_count=0,
        ),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
    )
    assert verdict.outcome == "FAIL"


def test_elliptical_continuation_passes_while_topic_confusion_fails() -> None:
    helpful = "Steady support reply with grounded guidance."
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-004",
        reply=helpful,
        snapshot=_passing_book_snapshot(),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
    )
    assert verdict.outcome == "PASS"
    assert verdict.dialogue_continuity_ok is True
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-005",
        reply=helpful,
        snapshot=dict(_passing_book_snapshot(), answer_relevant=False),
        judge_helpful=False,
        judge_addresses_intent=False,
        judge_model="probe-model",
    )
    assert verdict.outcome == "FAIL"
    # Broken dialogue continuity fails even with positive telemetry.
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-006",
        reply=helpful,
        snapshot=_passing_book_snapshot(),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
        continuity_broken=True,
    )
    assert verdict.outcome == "FAIL"
    assert verdict.dialogue_continuity_ok is False


def test_multineed_partial_and_quote_boundary_false_positives_fail() -> None:
    helpful = "Steady support reply with grounded guidance."
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-007",
        reply=helpful,
        snapshot=dict(_passing_book_snapshot(), response_units=5, verified_book_units=2),
        judge_helpful=False,
        judge_addresses_intent=False,
        judge_model="probe-model",
    )
    assert verdict.outcome == "FAIL"
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-008",
        reply="«Поддержка рядом помогает пережить тягу спокойно и уверенно»",
        snapshot=_passing_book_snapshot(),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
    )
    assert verdict.outcome == "FAIL"
    assert verdict.failure_category == "quote-only-no-guidance"


def test_filler_link_only_and_internal_detail_fail_while_glue_passes() -> None:
    helpful = "Steady support reply with grounded guidance."
    # Generic filler dissented by the independent judge fails.
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-009",
        reply=helpful,
        snapshot=_passing_book_snapshot(),
        judge_helpful=False,
        judge_addresses_intent=False,
        judge_model="probe-model",
    )
    assert verdict.outcome == "FAIL"
    assert verdict.failure_category == "judge-negative"
    # Unhelpful link-only response fails.
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-010",
        reply="https://example.invalid/pc-s-0001",
        snapshot=_passing_book_snapshot(),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
    )
    assert verdict.outcome == "FAIL"
    # User-visible internal infrastructure detail fails.
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-011",
        reply="Retrieval corpus evidence pack verifier outcome passed.",
        snapshot=_passing_book_snapshot(),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
    )
    assert verdict.outcome == "FAIL"
    # Ordinary non-substantive glue without AA doctrine needs no citation.
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-012",
        reply="I hear you. What feels most important right now?",
        snapshot={
            "answer_outcome": "served",
            "verifier_outcome": "passed",
            "verifier_unavailable_units": 0,
            "turn_budget_exceeded": False,
            "planner_query_count": 0,
            "retrieval_passages": 0,
            "verified_book_units": 0,
            "response_units": 0,
            "adequacy_verdict": "pass",
            "answers_request": True,
            "technically_grounded": True,
            "qualified": True,
            "answer_relevant": True,
            "failure_category": "",
            "planner_reason": "",
        },
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_contains_substantive_claim=False,
        judge_model="probe-model",
        requires_book_evidence=False,
    )
    assert verdict.outcome == "PASS"


def test_judge_unavailable_is_blocked_never_pass_or_fail() -> None:
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-013",
        reply="Steady support reply with grounded guidance.",
        snapshot=_passing_book_snapshot(),
        judge_helpful=None,
        judge_model="",
    )
    assert verdict.outcome == "BLOCKED"
    assert verdict.judge_available is False
    assert verdict.failure_category == BLOCKED_JUDGE_UNAVAILABLE


def test_emergency_and_out_of_book_must_not_fabricate_book_support() -> None:
    helpful = "Steady bounded safety reply."
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-014",
        reply=helpful,
        snapshot=_passing_book_snapshot(),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
        requires_book_evidence=False,
        safety_expectation="emergency",
    )
    assert verdict.outcome == "FAIL"
    assert verdict.failure_category == "fabricated-book-support"
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-015",
        reply=helpful,
        snapshot=dict(_passing_book_snapshot(), verified_book_units=0),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
        requires_book_evidence=False,
        safety_expectation="emergency",
    )
    assert verdict.outcome == "PASS"


def test_no_unseen_source_is_blocked_with_exact_missing_prerequisite() -> None:
    # The repository carries no sealed fixture and no pinned baseline, so
    # the held-out tier is BLOCKED. The public held_out_v1.json descriptors
    # must never count as sealed proof.
    assert (REPO_ROOT / "qualification" / "held_out_v1.json").is_file()
    provenance, fixture = resolve_heldout_source(repo_root=REPO_ROOT, run_id="test")
    assert fixture is None
    assert provenance.source == "unavailable"
    assert provenance.blocked_reason == BLOCKED_BASELINE_UNPINNED
    assert provenance.scenario_count == 0
    assert provenance.evaluator_version == GATE_C_EVALUATOR_VERSION
    report = build_heldout_tier_report(provenance=provenance, verdicts=())
    assert report.status == "BLOCKED"
    assert report.to_dict()["failure_categories"] == {BLOCKED_BASELINE_UNPINNED: 1}


def test_visible_fixture_echo_is_overt_leakage_never_heldout_pass() -> None:
    visible = ["Some visible regression turn phrasing here"]
    novelty = check_fixture_novelty(
        fixture_prompts=["  some VISIBLE regression turn phrasing here "],
        regression_prompts=visible,
    )
    assert novelty["novel"] is False
    assert novelty["echoed_count"] == 1
    novelty = check_fixture_novelty(
        fixture_prompts=["Entirely novel unseen dialogue turn."],
        regression_prompts=visible,
    )
    assert novelty["novel"] is True


def test_two_independent_seeds_give_distinct_digests_on_stable_rubric() -> None:
    rubric_hex = "a" * 64
    first, first_prov = generate_unseen_fixture(
        seed=b"0123456789abcdef-seed-one....",
        rubric_hex=rubric_hex,
        synthesizer=_scripted_synth,
        regression_prompts=["visible regression turn"],
        scenario_count=12,
        run_id="seed-one",
    )
    second, second_prov = generate_unseen_fixture(
        seed=b"fedcba9876543210-seed-two....",
        rubric_hex=rubric_hex,
        synthesizer=_scripted_synth,
        regression_prompts=["visible regression turn"],
        scenario_count=12,
        run_id="seed-two",
    )
    assert fixture_digest(first) != fixture_digest(second)
    assert first["seed_sha256"] != second["seed_sha256"]
    for payload in (first, second):
        ok, missing = validate_heldout_fixture(payload)
        assert ok, missing
        assert payload["version"] == GATE_C_HELDOUT_SCHEMA_VERSION
    assert first_prov.generation_success is True
    assert second_prov.generation_success is True
    assert first_prov.independence_level == "procedural_only"


def test_missing_generator_is_blocked_never_fabricated() -> None:
    from aa.qualification.gate_c_heldout import GateCHeldoutBlockedError

    try:
        generate_unseen_fixture(
            seed=b"0123456789abcdef-seed-one....",
            rubric_hex="b" * 64,
            synthesizer=None,
            scenario_count=12,
            run_id="offline",
        )
    except GateCHeldoutBlockedError as exc:
        assert exc.reason == BLOCKED_GENERATOR_UNAVAILABLE
    else:
        raise AssertionError("missing generator must raise typed BLOCKED")


def test_baseline_mismatch_blocks_while_matching_baseline_reports_generator() -> None:
    from aa.qualification.gate_c_heldout import evaluator_source_sha, rubric_digest

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "qualification").mkdir()
        shutil.copy(
            REPO_ROOT / "qualification" / "ru_answer_quality_rubric.v2.json",
            root / "qualification" / "ru_answer_quality_rubric.v2.json",
        )
        baseline_path = root / "qualification" / "gate_c_evaluator_baseline.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "version": "aa-gate-c-evaluator-baseline/1",
                    "evaluator_version": GATE_C_EVALUATOR_VERSION,
                    "evaluator_source_sha": "0" * 64,
                    "rubric_version": "ru-answer-quality-rubric-v2",
                    "rubric_digest": rubric_digest(REPO_ROOT),
                    "run_id": "test",
                }
            ),
            encoding="utf-8",
        )
        os.environ["AA_GATE_C_HELDOUT_BASELINE"] = str(baseline_path)
        try:
            provenance, fixture = resolve_heldout_source(repo_root=root, run_id="test")
        finally:
            del os.environ["AA_GATE_C_HELDOUT_BASELINE"]
        assert fixture is None
        assert provenance.blocked_reason == BLOCKED_BASELINE_MISMATCH
        # A matching baseline with no generated fixture yet reports the
        # exact generator prerequisite instead of passing.
        payload = write_baseline_file(repo_root=REPO_ROOT, path=baseline_path)
        assert payload["evaluator_source_sha"] == evaluator_source_sha(REPO_ROOT)
        os.environ["AA_GATE_C_HELDOUT_BASELINE"] = str(baseline_path)
        try:
            provenance, fixture = resolve_heldout_source(repo_root=root, run_id="test")
        finally:
            del os.environ["AA_GATE_C_HELDOUT_BASELINE"]
        assert fixture is None
        assert provenance.blocked_reason == BLOCKED_GENERATOR_UNAVAILABLE


def test_delivery_receipts_separated_d_never_proves_c() -> None:
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-016",
        reply="Steady support reply with grounded guidance.",
        snapshot=_passing_book_snapshot(),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
        receipt_kind="external-confirmed",
    )
    assert verdict.outcome == "FAIL"
    assert verdict.failure_category == "external-receipt-mislabel"
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-017",
        reply="Steady support reply with grounded guidance.",
        snapshot=_passing_book_snapshot(),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
        receipt_kind="simulated-in-process",
    )
    assert verdict.outcome == "PASS"


def test_independence_reported_honestly() -> None:
    assert (
        describe_independence(
            judge_models=["opencode/muse-spark"],
            generator_models=["opencode/muse-spark"],
        )
        == "procedural_only"
    )
    assert describe_independence(judge_models=[], generator_models=[]) == "procedural_only"
    assert (
        describe_independence(
            judge_models=["other-family/model-a"],
            generator_models=["opencode/muse-spark"],
        )
        == "family_diverse"
    )


def test_overall_requires_regression_pass_plus_unseen_heldout_pass() -> None:
    assert decide_gate_c_overall(regression_status="PASS", heldout_status="PASS") == "PASS"
    assert decide_gate_c_overall(regression_status="PASS", heldout_status="BLOCKED") == "BLOCKED"
    assert decide_gate_c_overall(regression_status="BLOCKED", heldout_status="BLOCKED") == "BLOCKED"
    assert decide_gate_c_overall(regression_status="FAIL", heldout_status="PASS") == "FAIL"
    assert decide_gate_c_overall(regression_status="PASS", heldout_status="FAIL") == "FAIL"
    regression = build_regression_tier_report(scenario_ids=["c-regression-001"], failed_ids=[])
    provenance = HeldoutSourceProvenance(
        source="unavailable",
        fixture_digest="",
        generation_success=False,
        evaluator_version=GATE_C_EVALUATOR_VERSION,
        evaluator_source_sha="e" * 64,
        rubric_version="ru-answer-quality-rubric-v2",
        rubric_digest="r" * 64,
        run_id="test",
        independence_level="procedural_only",
        reproducibility="heldout-unavailable-no-reproducible-fixture",
        blocked_reason=BLOCKED_BASELINE_UNPINNED,
        scenario_count=0,
    )
    heldout = build_heldout_tier_report(provenance=provenance, verdicts=())
    report = build_gate_c_semantic_report(regression=regression, heldout=heldout)
    assert report.overall == "BLOCKED"
    assert report.to_dict()["schema_version"] == "aa-gate-c-semantic-report/1"


def test_tier_reports_carry_no_prompt_text_or_seed() -> None:
    sentinel_prompt = "sentinel-unseen-prompt-text-341"
    sentinel_seed = "sentinel-seed-341"
    regression = build_regression_tier_report(scenario_ids=["c-regression-001"], failed_ids=[])
    provenance = HeldoutSourceProvenance(
        source="post_freeze_generated",
        fixture_digest="d" * 64,
        generation_success=True,
        evaluator_version=GATE_C_EVALUATOR_VERSION,
        evaluator_source_sha="e" * 64,
        rubric_version="ru-answer-quality-rubric-v2",
        rubric_digest="r" * 64,
        run_id="test",
        independence_level="procedural_only",
        reproducibility="run-unique-seed-single-use-fixture",
        blocked_reason="",
        scenario_count=1,
    )
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-018",
        reply="Steady support reply with grounded guidance.",
        snapshot=_passing_book_snapshot(),
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
    )
    heldout = build_heldout_tier_report(provenance=provenance, verdicts=[verdict])
    report = build_gate_c_semantic_report(regression=regression, heldout=heldout)
    dumped = json.dumps(report.to_dict(), ensure_ascii=False)
    assert sentinel_prompt not in dumped
    assert sentinel_seed not in dumped
    assert "prompt" not in json.dumps(provenance.to_dict())


def test_regression_preserved_18_with_explicit_tier_separation() -> None:
    import aa.qualification.product_contract_live as live_mod
    from aa.qualification.gate_cd_boundary import owner_gate_for_check

    assert live_mod.GATE_C_REGRESSION_SCENARIO_COUNT == 18
    source = inspect.getsource(live_mod.run_live_telegram_evidence_lane)
    assert "core_scenarios" in source
    assert "held_out_scenarios" in source
    assert "scenarios = core_scenarios + held_out_scenarios" in source
    assert "c-regression-preserved-18" in source
    assert "c-heldout-blocked-" in source
    assert "C-regression" in source
    assert owner_gate_for_check("c-regression-preserved-18") == "C"
    assert owner_gate_for_check("c-heldout-blocked-external-live-prerequisites-unavailable") == "C"
    assert owner_gate_for_check("c-heldout-tier-fail") == "C"


async def test_live_lane_offline_is_incomplete_with_blocked_tiers(monkeypatch: Any) -> None:
    from aa.qualification.product_contract_live import (
        assert_no_text_leak,
        run_live_telegram_evidence_lane,
    )

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("AA_BOOK_AGE_IDENTITY", raising=False)
    result = await run_live_telegram_evidence_lane()
    assert result.lane == "live-telegram-evidence"
    assert result.status == "INCOMPLETE"
    assert "c-heldout-blocked-external-live-prerequisites-unavailable" in result.incomplete
    assert result.metrics["gate_c_heldout_status"] == "BLOCKED"
    assert result.metrics["gate_c_regression_status"] == "BLOCKED"
    assert result.metrics["gate_c_overall"] == "BLOCKED"
    assert (
        result.metrics["gate_c_heldout_blocked_reason"] == "EXTERNAL_LIVE_PREREQUISITES_UNAVAILABLE"
    )
    assert result.metrics["gate_c_independence_level"] == "procedural_only"
    assert_no_text_leak(result.to_dict())


async def test_delegate_graph_telemetry_never_makes_heldout_pass() -> None:
    from aa.conversation.graph_runtime import GraphTurnRuntime

    async def _delegate(thread: str, text: str) -> str:
        _ = thread
        return "Понял вас. Давайте разберём спокойно."

    runtime = GraphTurnRuntime(delegate=_delegate)
    await runtime.start()
    try:
        snapshot = runtime.last_telemetry_for_thread(runtime.thread_id(923001))
    finally:
        await runtime.stop()
    assert snapshot == {}
    verdict = evaluate_heldout_turn(
        scenario_id="c-heldout-019",
        reply="Понял вас. Давайте разберём спокойно.",
        snapshot=snapshot,
        judge_helpful=True,
        judge_addresses_intent=True,
        judge_model="probe-model",
    )
    assert verdict.outcome == "FAIL"
    assert verdict.failure_category == "missing-turn-telemetry"


def test_gate_c_scripts_prove_blocked_paths() -> None:
    scripts = REPO_ROOT / "scripts"
    pin = subprocess.run(
        [
            sys.executable,
            str(scripts / "pin_gate_c_evaluator_baseline.py"),
            "--out",
            "/tmp/opencode/gate_c_baseline_test.json",
            "--run-id",
            "test-341",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert pin.returncode == 0, pin.stderr
    payload = json.loads(pin.stdout)
    assert len(payload["evaluator_source_sha"]) == 16
    assert len(payload["rubric_digest"]) == 16
    baseline = json.loads(Path("/tmp/opencode/gate_c_baseline_test.json").read_text())
    assert baseline["version"] == "aa-gate-c-evaluator-baseline/1"
    gen = subprocess.run(
        [
            sys.executable,
            str(scripts / "generate_gate_c_heldout.py"),
            "--out",
            "/tmp/opencode/gate_c_heldout_test.json",
            "--offline-blocked",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert gen.returncode == 2
    assert json.loads(gen.stdout)["reason"] == "EXTERNAL_SEMANTIC_PROVIDER_UNAVAILABLE"
    assert not Path("/tmp/opencode/gate_c_heldout_test.json").exists()
