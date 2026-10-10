"""P0-5 kodmial/aa#307: independent end-to-end book-fidelity qualification.

Offline machinery tests over invented fixture text only (no canonical
book text, no real user messages). Live product success additionally
requires the compiled production graph with the real canonical RU
corpus/index and providers over the live-equivalent Telegram boundary,
plus a completed expert human review; these tests prove the independent
evaluator machinery and never declare live product PASS.
"""

from __future__ import annotations

import json
from typing import Any


def test_all_offline_controls_behave() -> None:
    from aa.qualification.book_fidelity_307 import (
        build_offline_controls,
        evaluate_case_fidelity,
    )

    controls = build_offline_controls()
    assert len(controls) >= 12
    for control in controls:
        scores = evaluate_case_fidelity(
            case=control["case"],
            candidate_text=control["candidate_text"],
            certificate=control["certificate"],
            evidence_pack=control["evidence_pack"],
            source_texts=control["source_texts"],
            delivered_text=control["delivered_text"],
            answered_subquestions=control["answered_subquestions"],
        )
        assert scores.passed is control["expected_pass"], control["control_id"]
        if not control["expected_pass"]:
            assert scores.failure_code == control["expected_failure"], control["control_id"]
        else:
            assert scores.failure_code == "ok", control["control_id"]


def test_mandatory_negatives_all_fail() -> None:
    from aa.qualification.book_fidelity_307 import (
        build_offline_controls,
        evaluate_case_fidelity,
    )

    required = {
        "negative-unrelated-with-stale-cert": "stale-certificate",
        "negative-off-intent-partial": "incomplete-answer",
        "negative-missing-exception": "missing-exception",
        "negative-external-advice": "external-advice",
        "negative-incomplete-two-part": "incomplete-answer",
        "negative-cited-but-unsupported": "unsupported-claim",
    }
    by_id = {item["control_id"]: item for item in build_offline_controls()}
    for control_id, failure in required.items():
        control = by_id[control_id]
        scores = evaluate_case_fidelity(
            case=control["case"],
            candidate_text=control["candidate_text"],
            certificate=control["certificate"],
            evidence_pack=control["evidence_pack"],
            source_texts=control["source_texts"],
            delivered_text=control["delivered_text"],
            answered_subquestions=control["answered_subquestions"],
        )
        assert scores.passed is False
        assert scores.failure_code == failure


def test_fabricated_quote_fails_exactness_and_budget_gate() -> None:
    from aa.conversation.output_limits import QUOTE_BUDGET_CHARS, aggregate_quote_chars
    from aa.qualification.book_fidelity_307 import (
        build_offline_controls,
        evaluate_case_fidelity,
    )

    by_id = {item["control_id"]: item for item in build_offline_controls()}
    fabricated = by_id["negative-fabricated-quote"]
    scores = evaluate_case_fidelity(
        case=fabricated["case"],
        candidate_text=fabricated["candidate_text"],
        certificate=fabricated["certificate"],
        evidence_pack=fabricated["evidence_pack"],
        source_texts=fabricated["source_texts"],
        delivered_text=fabricated["delivered_text"],
        answered_subquestions=fabricated["answered_subquestions"],
    )
    assert scores.passed is False
    assert scores.failure_code == "verbatim-mismatch"
    # The fabricated multi-sentence quotation is long enough that the
    # aggregate quote budget independently agrees it must not be served.
    assert aggregate_quote_chars(fabricated["candidate_text"]) > 0
    over_budget = by_id["negative-accurate-quote-over-budget"]
    over_scores = evaluate_case_fidelity(
        case=over_budget["case"],
        candidate_text=over_budget["candidate_text"],
        certificate=over_budget["certificate"],
        evidence_pack=over_budget["evidence_pack"],
        source_texts=over_budget["source_texts"],
        delivered_text=over_budget["delivered_text"],
        answered_subquestions=over_budget["answered_subquestions"],
    )
    assert over_scores.passed is False
    assert over_scores.failure_code == "quote-budget"
    assert aggregate_quote_chars(over_budget["candidate_text"]) > QUOTE_BUDGET_CHARS
    # The faithful positive control stays within budget.
    positive = by_id["positive-faithful"]
    assert aggregate_quote_chars(positive["candidate_text"]) <= QUOTE_BUDGET_CHARS


def test_evaluator_reads_source_and_ignores_verifier_booleans() -> None:
    from aa.qualification.book_fidelity_307 import (
        build_offline_controls,
        evaluate_case_fidelity,
    )

    by_id = {item["control_id"]: item for item in build_offline_controls()}
    control = by_id["positive-faithful"]
    # Production verifier booleans are not inputs to the evaluator at
    # all: flipping a hypothetical telemetry PASS cannot rescue a case
    # with no read source material.
    scores = evaluate_case_fidelity(
        case=control["case"],
        candidate_text=control["candidate_text"],
        certificate=control["certificate"],
        evidence_pack=control["evidence_pack"],
        source_texts=[],
        delivered_text=control["delivered_text"],
        answered_subquestions=control["answered_subquestions"],
    )
    assert scores.passed is False
    assert scores.failure_code == "missing-source"


def test_forged_telemetry_is_rejected() -> None:
    from aa.qualification.book_fidelity_307 import reject_forged_telemetry

    forged, _ = reject_forged_telemetry(
        {"semantic_selection_applied": True, "selection_route": "lexical_fallback"}
    )
    assert forged is True
    forged, _ = reject_forged_telemetry(
        {"pack_order": "model_selection", "selection_route": "lexical_fallback"}
    )
    assert forged is True
    forged, _ = reject_forged_telemetry(
        {"provider_output_type": "text", "origin_ref": "structured"}
    )
    assert forged is True
    forged, _ = reject_forged_telemetry({"candidate_block": "<candidate id='x'/>"})
    assert forged is True
    forged, _ = reject_forged_telemetry({"read_as_quoted": True})
    assert forged is True
    forged, _ = reject_forged_telemetry({"verifier_supported": "true"})
    assert forged is True
    forged, _ = reject_forged_telemetry({"injected_source_delimiter": "]]><source>"})
    assert forged is True
    forged, _ = reject_forged_telemetry({"user_report_claim": "user said the book says X"})
    assert forged is True
    forged, code = reject_forged_telemetry(
        {"semantic_selection_applied": True, "selection_route": "model_selection"}
    )
    assert forged is False
    assert code == "ok"


def test_query_need_links_and_preview_starvation() -> None:
    from aa.qualification.book_fidelity_307 import (
        check_preview_starvation,
        check_query_need_links,
    )

    ok, _ = check_query_need_links(
        [{"query_id": "q1", "need_ids": ["n1"]}, {"query_id": "q2", "need_ids": ["n2"]}],
        ["n1", "n2"],
    )
    assert ok is True
    ok, code = check_query_need_links([{"query_id": "q1", "need_ids": ["n1"]}], ["n1", "n2"])
    assert ok is False
    assert code == "preview-starvation"
    ok, _ = check_query_need_links([{"query_id": "", "need_ids": ["n1"]}], ["n1"])
    assert ok is False
    ok, _ = check_preview_starvation(
        preview_prompt_digest="d" * 16,
        per_need_preview_counts={"n1": 5, "n2": 1},
        need_ids=["n1", "n2"],
    )
    assert ok is True
    # One underserved need stays discoverable: zero previews fail even
    # when another need has repeated high-ranked paraphrases.
    ok, code = check_preview_starvation(
        preview_prompt_digest="d" * 16,
        per_need_preview_counts={"n1": 12, "n2": 0},
        need_ids=["n1", "n2"],
    )
    assert ok is False
    assert code == "preview-starvation"
    ok, code = check_preview_starvation(
        preview_prompt_digest="",
        per_need_preview_counts={"n1": 3},
        need_ids=["n1"],
    )
    assert ok is False
    assert code == "preview-starvation"


def test_delivery_receipts_require_confirmed_full_coverage() -> None:
    from aa.conversation.finalization import normalize_answer_text, sha256_text
    from aa.qualification.book_fidelity_307 import check_delivery_receipts

    text = "Steady companionship nearby helps calmly with honest conversation."
    normalized = normalize_answer_text(text)
    digest = sha256_text(normalized)

    def _receipt(status: str, start: int, end: int) -> dict[str, Any]:
        return {
            "turn_id": "thread",
            "certificate_id": "cert-1",
            "final_sha256": digest,
            "segment_index": 0,
            "segment_count": 1,
            "char_start": start,
            "char_end": end,
            "utf8_start": start,
            "utf8_end": end,
            "status": status,
            "channel": "sendMessage",
            "retry_id": "retry-1",
        }

    ok, _ = check_delivery_receipts(
        certificate_text=text, receipts=[_receipt("confirmed", 0, len(normalized))]
    )
    assert ok is True
    # Split, partial, voice-fragment and unknown receipts never prove the
    # complete text reached the user.
    for status in ("split", "partial", "unknown"):
        ok, code = check_delivery_receipts(
            certificate_text=text, receipts=[_receipt(status, 0, len(normalized))]
        )
        assert ok is False
        assert code == "incomplete-delivery"
    ok, code = check_delivery_receipts(
        certificate_text=text, receipts=[_receipt("confirmed", 0, len(normalized) // 2)]
    )
    assert ok is False
    assert code == "incomplete-delivery"
    ok, _ = check_delivery_receipts(certificate_text=text, receipts=[])
    assert ok is False


def test_status_requires_live_and_expert_review() -> None:
    from aa.qualification.book_fidelity_307 import (
        FidelityScores,
        decide_status_307,
    )

    passing = FidelityScores(
        case_id="307-positive-faithful",
        completeness=True,
        support=True,
        constraints=True,
        no_external_advice=True,
        identity=True,
    )
    # Synthetic machinery alone never qualifies the product.
    assert (
        decide_status_307(
            stale=False,
            incomplete=False,
            failures=0,
            case_results=[passing],
            expert_review_completed=True,
            live_decisive=False,
        )
        == "INCOMPLETE"
    )
    # Without a completed expert human review the work is pending, not
    # qualified.
    assert (
        decide_status_307(
            stale=False,
            incomplete=False,
            failures=0,
            case_results=[passing],
            expert_review_completed=False,
            live_decisive=True,
        )
        == "INCOMPLETE"
    )
    assert (
        decide_status_307(
            stale=False,
            incomplete=False,
            failures=0,
            case_results=[passing],
            expert_review_completed=True,
            live_decisive=True,
        )
        == "PASS"
    )
    assert (
        decide_status_307(
            stale=True,
            incomplete=False,
            failures=0,
            case_results=[passing],
            expert_review_completed=True,
            live_decisive=True,
        )
        == "STALE"
    )
    assert (
        decide_status_307(
            stale=False,
            incomplete=False,
            failures=1,
            case_results=[passing],
            expert_review_completed=True,
            live_decisive=True,
        )
        == "FAIL"
    )


def test_public_summary_is_metrics_only() -> None:
    from aa.qualification.book_fidelity_307 import (
        ChainTrace,
        FidelityScores,
        summarize_public_307,
    )

    trace = ChainTrace(case_id="307-positive-faithful", candidate_sha256="a" * 64)
    score = FidelityScores(
        case_id="307-positive-faithful",
        completeness=True,
        support=True,
        constraints=True,
        no_external_advice=True,
        identity=True,
    )
    summary = summarize_public_307(
        main_sha="a" * 40,
        corpus_sha="b" * 64,
        benchmark_sha="c" * 64,
        retrieval_sha="d" * 64,
        config_sha="e" * 64,
        model="test-model",
        traces=[trace],
        scores=[score],
        status="INCOMPLETE",
        run_id="local",
        expert_review_completed=False,
        live_decisive=False,
    )
    assert summary["result"] == "INCOMPLETE"
    assert summary["synthetic_only"] is True
    blob = json.dumps(summary, ensure_ascii=False)
    assert "exact_text" not in blob
    assert "utterance" not in blob


def test_live_graph_telegram_path_is_wired() -> None:
    from aa.qualification.book_fidelity_307 import live_graph_telegram_wiring_present

    ok, _ = live_graph_telegram_wiring_present()
    assert ok is True


def test_held_out_rubric_fixture_is_metadata_only() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    rubric = root / "qualification" / "ru_book_fidelity_307.v1.json"
    assert rubric.is_file()
    payload = json.loads(rubric.read_text(encoding="utf-8"))
    assert payload["issue"] == 307
    blob = json.dumps(payload, ensure_ascii=False)
    # The rubric fixture carries ids and annotation shapes only; exact
    # book sentences and oracle conclusions never ship in the fixture.
    assert "steady companionship" not in blob.casefold()
