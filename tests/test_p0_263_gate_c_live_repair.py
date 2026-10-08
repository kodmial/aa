"""P0 kodmial/aa#268: Gate C live-production-path failures, model-driven.

Proven product failures on exact main 83e2717 (run 37798921806):

- C:live-book-grounding-substantive-drinking-11 plus
  C:live-answer-relevance-substantive-drinking-11 (a held-out
  colloquial/typo craving turn collapsed to a generic fallback while
  its correctly spelled sibling passed);
- C:live-book-grounding-long-conversation-17 (a generic four-token
  continuation in an ongoing recovery thread failed grounding while
  relevance passed);
- C:live-answer-no-generic-collapse plus
  C:live-substantive-grounded-book-answer (6/8 grounded).

Model-driven repair (#268): no domain tables, regexes, step-number
parsing, or token-bound rescue heuristics remain. The hidden planner
resolves intent (mode + resolved_intent + queries); the unified
structured verifier judges groundedness and answer relevance
(addresses_intent / answer_relevant) in the same invocation.
Application code only validates schema and cardinality.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any


def _pack_entry(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Поддержка рядом помогает пережить тягу сегодня.",
) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _book_grounding(
    passage_id: str = "chapter-3#exp0000",
    unit_text: str = "Поддержка рядом помогает пережить тягу сегодня.",
    *,
    addresses_intent: bool = True,
    answer_relevant: bool = True,
) -> dict[str, Any]:
    return {
        "verified": True,
        "all_required_supported": True,
        "answer_relevant": answer_relevant,
        "units": [
            {
                "unit_id": "u1",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": [passage_id],
                "addresses_intent": addresses_intent,
                "text": unit_text,
            }
        ],
    }


def test_recovery_queries_pass_recovery_validation() -> None:
    from aa.conversation.answer_adequacy import build_recovery_queries
    from aa.retrieval.evidence import validate_recovery_queries

    queries = build_recovery_queries(
        "Вечером тяжело пережить тягу, что помогает?",
        summary="",
        recent=[],
    )
    assert 1 <= len(queries) <= 6
    assert validate_recovery_queries(queries) == [item.strip() for item in queries]


def test_recovery_validator_rejects_bad_lists() -> None:
    from aa.retrieval.evidence import EvidenceError, validate_recovery_queries

    assert validate_recovery_queries([]) == []
    try:
        validate_recovery_queries([""])
    except EvidenceError:
        pass
    else:
        raise AssertionError("empty-string recovery query must fail closed")
    try:
        validate_recovery_queries([f"query {index}" for index in range(17)])
    except EvidenceError:
        pass
    else:
        raise AssertionError("17 recovery queries must fail closed")


def test_planner_validator_still_requires_ten_to_sixteen() -> None:
    from aa.retrieval.evidence import EvidenceError, validate_planner_queries

    try:
        validate_planner_queries(["only one query"])
    except EvidenceError:
        pass
    else:
        raise AssertionError("planner 1-query list must still fail closed")


def test_recovery_retrieval_accepts_single_query(tmp_path: pathlib.Path) -> None:
    from aa.conversation.answer_adequacy import build_recovery_queries
    from aa.retrieval.evidence import retrieve_evidence_for_recovery
    from tests.test_hybrid_retrieval import _build_index

    index = _build_index(tmp_path)
    queries = build_recovery_queries("Hero started to drink every evening")
    assert 1 <= len(queries) <= 6
    pack = retrieve_evidence_for_recovery(index, queries)
    assert pack.passages


def test_generic_continuation_relevance_is_model_driven() -> None:
    """Generic continuation relevance comes from the verifier, not tokens."""
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    prior = "Вечером тяжело пережить тягу, как обходиться?"
    generic_followup = "И что отсюда следует прямо сейчас?"
    pack = [_pack_entry()]
    reply = "Поддержка рядом помогает пережить тягу сегодня."

    relevant = assess_turn_adequacy(
        user_message=generic_followup,
        reply=reply,
        evidence_pack=pack,
        grounding_result=_book_grounding(answer_relevant=True, addresses_intent=True),
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent=prior,
        prior_user_messages=[prior],
    )
    assert relevant.verdict == "pass"
    assert relevant.answers_request is True

    # Prior messages are accepted for backward compatibility but ignored:
    # the same model verdict passes with or without them.
    without_prior = assess_turn_adequacy(
        user_message=generic_followup,
        reply=reply,
        evidence_pack=pack,
        grounding_result=_book_grounding(answer_relevant=True, addresses_intent=True),
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent=prior,
    )
    assert without_prior.verdict == "pass"

    irrelevant = assess_turn_adequacy(
        user_message=generic_followup,
        reply=reply,
        evidence_pack=pack,
        grounding_result=_book_grounding(answer_relevant=False, addresses_intent=False),
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent=prior,
        prior_user_messages=[prior],
    )
    assert irrelevant.verdict == "fail"
    assert irrelevant.failure_category == "irrelevant-citation"


def test_explicit_pivot_fails_on_model_relevance() -> None:
    """An explicit new topic fails when the verifier marks it irrelevant."""
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    prior = "Вечером тяжело пережить тягу, как обходиться?"
    pivot = "Другое дело ночью мысли крутятся спать плохо"
    pack = [_pack_entry()]
    reply = "Поддержка рядом помогает пережить тягу сегодня."

    verdict = assess_turn_adequacy(
        user_message=pivot,
        reply=reply,
        evidence_pack=pack,
        grounding_result=_book_grounding(answer_relevant=False, addresses_intent=False),
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent=pivot,
        prior_user_messages=[prior],
    )
    assert verdict.verdict == "fail"
    assert verdict.failure_category == "irrelevant-citation"


def test_unrelated_citation_still_fails_with_context() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    ev_text = "Финансовое планирование помогает вести бюджет спокойно."
    pack = [_pack_entry(passage_id="chapter-9#exp0001", text=ev_text)]
    grounding = {
        "verified": True,
        "all_required_supported": True,
        "answer_relevant": False,
        "units": [
            {
                "unit_id": "u1",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": ["chapter-9#exp0001"],
                "addresses_intent": False,
                "text": ev_text,
            }
        ],
    }
    verdict = assess_turn_adequacy(
        user_message="И что отсюда следует прямо сейчас?",
        reply=ev_text,
        evidence_pack=pack,
        grounding_result=grounding,
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent="Вечером тяжело пережить тягу, как обходиться?",
        prior_user_messages=["Вечером тяжело пережить тягу, как обходиться?"],
    )
    assert verdict.verdict == "fail"
    assert verdict.failure_category == "irrelevant-citation"


def test_qualification_relevance_uses_rubric_telemetry_and_judge() -> None:
    """Qualification rescue is a rubric/telemetry judgment, not a token bound."""
    from aa.qualification.product_contract_live import assess_reply_relevance_with_rubric

    prompt = "Вечером тяжело пережить тягу, как обходиться?"
    reply = "Поддержка рядом помогает пережить тягу сегодня."
    passing_telemetry = {
        "adequacy_verdict": "pass",
        "answers_request": True,
        "technically_grounded": True,
        "answer_relevant": True,
    }
    failing_telemetry = {
        "adequacy_verdict": "fail",
        "answers_request": False,
        "technically_grounded": False,
        "answer_relevant": False,
    }
    # Telemetry is authoritative when present.
    assert assess_reply_relevance_with_rubric(prompt, reply, telemetry=passing_telemetry) is True
    assert assess_reply_relevance_with_rubric(prompt, reply, telemetry=failing_telemetry) is False
    # Without telemetry an injected stub judge decides; no signal fails closed.
    assert assess_reply_relevance_with_rubric(prompt, reply, judge=lambda p, r, c: True) is True
    assert assess_reply_relevance_with_rubric(prompt, reply, judge=lambda p, r, c: False) is False
    assert assess_reply_relevance_with_rubric(prompt, reply) is False


def test_no_domain_hardcoding_in_production_or_qualification() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "answer_adequacy.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "planner_node.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "verifier.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "qualification" / "product_contract_live.py").read_text(
            encoding="utf-8"
        ),
    ]
    for source in sources:
        for fragment in (
            "_STEP_WORD_RE",
            "_RECOVERY_DOMAIN_STEMS",
            "_GREETING_VOCABULARY",
            "_FOLLOWUP_REFERENCE_RE",
            "_extract_step_numbers_for_relevance",
        ):
            assert fragment not in source


def test_no_exact_live_question_special_cases() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "graph.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "answer_adequacy.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "planner_node.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "verifier.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "retrieval" / "evidence.py").read_text(encoding="utf-8"),
    ]
    for source in sources:
        for fragment in (
            "тянет выпить",
            "тянеет выпить",
            "Поругались дома",
            "покупать акции",
            "покончить с собой",
            "чем помочь можешь",
            "одному не получается",
        ):
            assert fragment not in source
