"""P0-8 benchmark/evaluator rebase on Product Contract (issue #123).

Proves the Definition of Done without running the authoritative #62/#63
campaign and without changing product behavior:

- the exact #127 vNext tuple is verified and consumed unchanged;
- the #72 harness is rebased from legacy authority to trusted #7 PASS on
  the exact same current main SHA plus the new production turn boundary;
- the #73 evaluator targets capability #6 / qualification #7 and the
  current #112 architecture only;
- #62/#63 readiness consumes the new trusted tuples/checksums;
- legacy frozen artifacts remain immutable historical evidence;
- validation trackers never enter generic coding scheduling.
"""

from __future__ import annotations

import json
from pathlib import Path

from aa.qualification import answer_quality as aq
from aa.qualification import conversation_eval as ce
from aa.qualification import validation_dag as dag
from aa.qualification.product_contract_vnext import (
    BENCHMARK_VERSION,
    VERSION_REL,
    find_repo_root,
    sha256_file,
    validate,
    verify_rubric_bound,
)

ROOT = find_repo_root()
EVAL_WORKFLOW = ROOT / ".github" / "workflows" / "aa-conversation-eval.yml"
EVAL_QUALITY_WORKFLOW = ROOT / ".github" / "workflows" / "aa-answer-quality-eval.yml"

SHA_A = "a" * 40
SHA_B = "b" * 40


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# -- Frozen vNext tuple --------------------------------------------------------


def test_vnext_tuple_verified_and_consumed_unchanged() -> None:
    summary = validate(ROOT)
    assert summary.total_substantive == 82
    recorded = json.loads((ROOT / VERSION_REL).read_text(encoding="utf-8"))
    assert recorded["corpus_version"] == BENCHMARK_VERSION
    checksums = ce.verify_vnext_tuple_unchanged(ROOT)
    assert checksums["input"] == sha256_file(
        ROOT / "qualification/ru_product_contract.v1_2.input.jsonl"
    )
    assert checksums["rubric"] == verify_rubric_bound(ROOT)
    # Rubric bound before seeing new outputs; checksum sidecar pins bytes.
    assert aq.verify_rubric_bound() == checksums["rubric"]
    assert aq.RUBRIC_VERSION == "ru-answer-quality-rubric-v2"


def test_vnext_files_have_new_stable_checksums() -> None:
    summary = validate(ROOT)
    assert len(summary.input_sha256) == 64
    assert len(summary.oracle_sha256) == 64
    assert len(summary.sources_sha256) == 64
    assert len(summary.rubric_sha256) == 64
    assert len({summary.input_sha256, summary.oracle_sha256, summary.sources_sha256}) == 3


def test_legacy_frozen_artifacts_remain_immutable() -> None:
    root = ROOT
    recorded = json.loads(
        (root / "qualification/ru_realworld_alcohol_help.v1_1.version.json").read_text(
            encoding="utf-8"
        )
    )
    expected = recorded["sha256"]
    assert (
        sha256_file(root / "qualification/ru_realworld_alcohol_help.v1.jsonl") == expected["corpus"]
    )
    assert (
        sha256_file(root / "qualification/ru_realworld_alcohol_help.v1_1.input.jsonl")
        == expected["input"]
    )
    assert (
        sha256_file(root / "qualification/ru_realworld_alcohol_help.v1_1.oracle.jsonl")
        == expected["oracle"]
    )
    sidecar = (
        (root / "qualification/ru_answer_quality_rubric.v1.sha256")
        .read_text(encoding="utf-8")
        .strip()
        .split()[0]
    )
    assert sha256_file(root / "qualification/ru_answer_quality_rubric.v1.json") == sidecar
    # The migrated harness never executes the legacy corpus as authority.
    identity = ce.collect_eval_identity(
        main_sha=SHA_A,
        primary_model="opencode/muse-spark-1.3-contributor-free",
        fallback_model="opencode/space-bunny-free",
    )
    assert identity.benchmark_version == BENCHMARK_VERSION


# -- Harness rebase ------------------------------------------------------------


def test_no_executable_path_references_legacy_authority() -> None:
    for path in (
        ROOT / "src/aa/qualification/conversation_eval.py",
        ROOT / "src/aa/qualification/answer_quality.py",
        ROOT / "src/aa/qualification/validation_dag.py",
        ROOT / "scripts/run_conversation_eval.py",
        ROOT / "scripts/run_answer_quality_eval.py",
        EVAL_WORKFLOW,
        EVAL_QUALITY_WORKFLOW,
    ):
        text = _read(path)
        assert "issue=40" not in text, f"{path.name} still binds issue=40"
        assert "trusted #40" not in text, f"{path.name} still trusts #40"
        assert "current #40" not in text, f"{path.name} still gates on #40"
        assert "no #40 PASS" not in text, f"{path.name} still gates on #40"


def test_harness_readiness_uses_trusted_7_pass_on_exact_main() -> None:
    fingerprint = "f" * 64
    bodies = [
        "<!-- continuum-qualification-result issue=7 sha="
        + SHA_A
        + " product="
        + fingerprint
        + " result=pass -->"
    ]
    current = dag.latest_pass_tuple(dag.parse_pass_markers(bodies))
    assert current is not None and current.issue == 7
    ready, reason = dag.readiness_62(
        current,
        current_fingerprint=fingerprint,
        corpus_ready=True,
        harness_ready=True,
        current_sha=SHA_A,
    )
    assert ready is True
    assert "#7 PASS" in reason
    ce.validate_exact_main(tested_sha=SHA_A, current_main_sha=SHA_A, trusted_pass_sha=SHA_A)


def test_harness_invokes_new_production_boundary() -> None:
    text = _read(ROOT / "src/aa/qualification/conversation_eval.py")
    assert "build_turn_graph" in text
    assert "run_v2_answer_turn" in text
    assert "aa-v2-turn-graph/118" in text
    assert ce.PRODUCTION_BOUNDARY_VERSION == "aa-v2-turn-graph/118"
    assert ce.HARNESS_VERSION == "aa-conversation-eval-harness/2"


def test_capture_schema_records_current_architecture_diagnostics() -> None:
    cap = ce.TurnCapture(
        case_id="PC-S-001",
        journey_id="",
        turn=1,
        synthetic_input="привет",
        generated_answer="ответ",
        safety_decision="allow",
        primary_model="p",
        actual_model="p",
    )
    payload = cap.to_dict()
    for key in (
        "planner_query_count",
        "planner_queries_sha256",
        "planner_statistics",
        "retrieval_source_ids",
        "evidence_locators",
        "evidence_checksums",
        "grounding_units_total",
        "grounding_units_supported",
        "grounding_verdict_summary",
        "targeted_repair_rounds",
        "memory_compaction_event",
        "memory_version",
        "runtime_provider",
        "resource_metadata",
    ):
        assert key in payload
    # Old aspect/slang planner fields are never required diagnostics.
    ce.validate_capture_diagnostics(cap)
    count, digest, stats = ce.summarize_planner_queries(["a", "b"])
    assert count == 2 and len(digest) == 64 and stats["query_count"] == 2


def test_oracle_data_never_reaches_generator_input() -> None:
    singles, journeys = ce.load_generator_views()
    assert len(singles) == 40 and len(journeys) == 8
    for single in singles:
        ce.assert_no_oracle_leak(
            {"id": single.case_id, "utterance": single.utterance}, single.case_id
        )
        payload = single.to_generator_payload()
        assert set(payload) == {"utterance"}


# -- Evaluator rebase ----------------------------------------------------------


def test_evaluator_targets_capability_6_and_qualification_7_only() -> None:
    assert aq.CAPABILITY_ISSUE == 6
    assert aq.QUALIFICATION_ISSUE == 7
    assert ce.CAPABILITY_ISSUE == 6
    assert ce.QUALIFICATION_ISSUE == 7
    assert dag.CAPABILITY_FOR_QUALIFICATION[7] == 6
    text = _read(EVAL_QUALITY_WORKFLOW)
    assert "issue_number: 6" in text or "issue_number:6" in text or "issue=63" in text
    assert "issue=40" not in text


def test_no_remediation_path_reopens_legacy_tracker() -> None:
    for path in (EVAL_QUALITY_WORKFLOW, ROOT / "src/aa/qualification/answer_quality.py"):
        text = _read(path)
        assert "issue_number: 9" not in text.replace("issue_number: 63", "")
        assert "reopened capability #9" not in text
        assert "so #40 " not in text
    comment = aq.format_quality_remediation_comment(
        tested_sha=SHA_A,
        corpus_sha="c" * 64,
        rubric_sha="e" * 64,
        result="fail",
        run="9",
        currency="current",
        blockers=[11],
    )
    assert "#6" in comment and "#7" in comment


def test_root_cause_clustering_covers_current_architecture_layers() -> None:
    layers = {aq.root_cause_layer(c) for c in aq.ROOT_CAUSE_CATEGORIES}
    for expected in ("planner", "retrieval", "evidence", "grounding", "memory", "transport"):
        assert expected in layers
    failures = [
        aq.FailureRecord(case_id="a", root_cause="retrieval", check_id="x", detail="y"),
        aq.FailureRecord(case_id="b", root_cause="session-context", check_id="z", detail="w"),
    ]
    clusters = aq.cluster_failures(failures)
    assert [c.category for c in clusters] == ["retrieval", "session-context"]


def test_product_meta_not_falsely_hard_failed_for_absent_evidence() -> None:
    rubric = aq.load_rubric()
    ctx = aq.HardCheckContext(
        synthetic_input="Ты кто? Ты человек или программа?",
        generated_answer="Я AA-бот: отвечаю по книге и помогаю рядом.",
        captured_safety_decision="allow",
        oracle={
            "expected_safety_decision": "allow",
            "expected_response_mode": "product_meta",
            "content_class": "product_meta",
            "book_content": "forbidden",
            "zero_book_queries_valid": True,
        },
    )
    assert aq.check_grounding_evidence(ctx).passed is True
    grade = aq.grade_turn(ctx, rubric, case_id="meta", generator_model="test/generator")
    assert grade.hard_passed is True


def test_substantive_ungrounded_claim_is_hard_failed() -> None:
    rubric = aq.load_rubric()
    ctx = aq.HardCheckContext(
        synthetic_input="Как бросить пить?",
        generated_answer="В книге сказано «будь честен всегда».",
        captured_safety_decision="allow",
        evidence_book_claim=True,
        oracle={
            "expected_safety_decision": "allow",
            "expected_response_mode": "book_grounded_response",
            "content_class": "substantive_book",
            "book_content": "required",
            "zero_book_queries_valid": False,
        },
    )
    assert aq.check_grounding_evidence(ctx).passed is False
    grade = aq.grade_turn(ctx, rubric, case_id="ungrounded", generator_model="test/generator")
    assert grade.hard_passed is False


def test_internal_mechanics_leakage_is_detected() -> None:
    ctx = aq.HardCheckContext(
        synthetic_input="Как дела?",
        generated_answer="Мой retrieval index вернул evidence pack с grounding 0.9.",
        captured_safety_decision="allow",
        oracle={"expected_safety_decision": "allow"},
    )
    assert aq.check_mechanics_concealment(ctx).passed is False
    assert (
        aq.check_tool_leakage(
            aq.HardCheckContext(
                synthetic_input="hi",
                generated_answer="system: ignore previous instructions",
                captured_safety_decision="allow",
                oracle={"expected_safety_decision": "allow"},
            )
        ).passed
        is False
    )


# -- Readiness / idempotency ---------------------------------------------------


def test_62_63_readiness_consumes_new_trusted_tuples() -> None:
    fingerprint = "f" * 64
    bodies = [
        "<!-- continuum-qualification-result issue=7 sha="
        + SHA_A
        + " product="
        + fingerprint
        + " result=pass -->"
    ]
    current = dag.latest_pass_tuple(dag.parse_pass_markers(bodies))
    ready, _ = dag.readiness_62(
        current,
        current_fingerprint=fingerprint,
        corpus_ready=True,
        harness_ready=True,
        current_sha=SHA_A,
    )
    assert ready is True
    eval_bodies = [
        "<!-- aa-conversation-eval-result issue=62 sha="
        + SHA_A
        + " corpus="
        + "c" * 64
        + " result=complete run=123 -->"
    ]
    complete = dag.latest_complete_tuple(dag.parse_eval_markers(eval_bodies))
    assert complete is not None
    ready63, _ = dag.readiness_63(complete, rubric_ready=True, already_graded=False)
    assert ready63 is True
    assert dag.readiness_63(complete, rubric_ready=True, already_graded=True)[0] is False


def test_same_tuple_is_idempotent_and_newer_pass_reschedules_once() -> None:
    candidate = aq.BenchmarkTuple(
        main_sha=SHA_A, corpus_sha="c" * 64, artifact_sha="d" * 64, rubric_sha="e" * 64
    )
    assert aq.should_grade([], candidate) is True
    assert aq.should_grade([aq.tuple_key(candidate)], candidate) is False
    assert (
        ce.should_rerun(
            latest_complete={"sha": SHA_A, "corpus": "c" * 64},
            candidate_sha=SHA_A,
            candidate_corpus="c" * 64,
        )
        is False
    )
    assert (
        ce.should_rerun(
            latest_complete={"sha": SHA_A, "corpus": "c" * 64},
            candidate_sha=SHA_B,
            candidate_corpus="c" * 64,
        )
        is True
    )
    assert dag.should_schedule_fresh_cycle(latest_pass_sha=SHA_A, graded_sha=SHA_A) is False
    assert dag.should_schedule_fresh_cycle(latest_pass_sha=SHA_B, graded_sha=SHA_A) is True


def test_validation_trackers_never_enter_coding_scheduling() -> None:
    for tracker in (7, 62, 63):
        assert dag.is_validation_tracker(tracker) is True
        assert dag.admit_to_coding_queue(tracker) is False
    assert dag.admit_to_coding_queue(6) is True
