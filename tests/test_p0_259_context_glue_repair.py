"""P0 kodmial/aa#268: model-driven turn understanding for glue and relevance.

The heuristic repair from #259 (greeting vocabularies, first-person
disclosure regexes, step-number extraction, token-overlap relevance) is
replaced by the model-driven architecture:

- the hidden planner returns mode + resolved_intent + queries, and only
  the schema decides glue vs substantive via ``is_conversational_plan``
  (conversational with zero queries is glue, retrieval is substantive);
- follow-up context arrives as planner-resolved ``resolved_intent`` read
  through ``effective_request`` (no regex or stem tables);
- whole-turn relevance comes from the unified verifier verdicts
  (per-unit ``addresses_intent`` plus turn-level ``answer_relevant``);
  a perfectly grounded citation that does not address the intent fails
  as irrelevant-citation;
- recovery uses the generic ``build_recovery_queries`` fallback with no
  domain branching;
- the live lane judges relevance with ``assess_reply_relevance_with_rubric``
  over telemetry (or an injected judge), failing closed otherwise.
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage


def _pack_entry(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Первый шаг: признание бессилия перед тягой помогает сегодня.",
) -> dict[str, Any]:
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
    }


def _grounding(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Первый шаг: признание бессилия перед тягой помогает сегодня.",
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
                "text": text,
                "addresses_intent": addresses_intent,
            }
        ],
    }


def test_planner_mode_decides_glue_vs_substantive() -> None:
    from aa.conversation.answer_adequacy import is_conversational_plan

    # Conversational mode with zero queries is glue.
    assert is_conversational_plan(mode="conversational", query_count=0) is True
    # Retrieval mode is substantive even with zero queries.
    assert is_conversational_plan(mode="retrieval", query_count=0) is False
    # Retrieval mode is substantive for any query count, so a short
    # disclosure planned as retrieval is never glue.
    assert is_conversational_plan(mode="retrieval", query_count=12) is False
    # Conversational mode carrying queries is not a clean glue plan.
    assert is_conversational_plan(mode="conversational", query_count=3) is False


def test_planner_reason_mapping_is_structural() -> None:
    from aa.conversation.answer_adequacy import planner_reason_for

    assert planner_reason_for(12, "ok") == "substantive-with-queries"
    assert planner_reason_for(0, "empty") == "legitimate-glue"


def test_query_plan_schema_normalizes_empty_to_conversational() -> None:
    import pytest

    from aa.conversation.planner_schema import (
        QueryPlan,
        QueryPlanValidationError,
        validate_query_plan,
    )

    # Model-driven (#268): an explicit retrieval plan with empty queries
    # or empty intent is invalid and fails closed; it is never
    # reinterpreted as conversational glue. Only an explicit
    # conversational plan carries zero queries.
    with pytest.raises(QueryPlanValidationError):
        validate_query_plan(QueryPlan(mode="retrieval", resolved_intent="", queries=[]))

    empty = validate_query_plan(QueryPlan(mode="conversational", resolved_intent="", queries=[]))
    assert empty.mode == "conversational"
    assert empty.queries == []

    retrieval = validate_query_plan(
        QueryPlan(
            mode="retrieval",
            resolved_intent="frozen-259-marker-alpha intent",
            queries=[f"frozen-259-marker-alpha query {i}" for i in range(12)],
        )
    )
    assert retrieval.mode == "retrieval"
    assert len(retrieval.queries) == 12


def test_effective_request_returns_resolved_intent() -> None:
    from aa.conversation.answer_adequacy import effective_request

    # Planner-resolved follow-ups travel as resolved_intent; no regex runs.
    assert (
        effective_request(
            resolved_intent="resolved follow-up intent",
            user_message="short follow-up turn",
        )
        == "resolved follow-up intent"
    )
    # Empty intent falls back to the raw turn text.
    assert effective_request(resolved_intent="", user_message="raw turn text") == "raw turn text"


def test_supported_relevant_citation_passes() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    text = "Первый шаг: признание бессилия перед тягой помогает сегодня."
    pack = [_pack_entry(text=text)]
    verdict = assess_turn_adequacy(
        user_message="frozen-259-marker-alpha turn",
        reply=text,
        evidence_pack=pack,
        grounding_result=_grounding(text=text, addresses_intent=True, answer_relevant=True),
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent="frozen-259-marker-alpha intent",
    )
    assert verdict.verdict == "pass"
    assert verdict.substantive_request is True
    assert verdict.answers_request is True


def test_grounded_but_irrelevant_citation_fails() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    text = "Первый шаг: признание бессилия перед тягой помогает сегодня."
    pack = [_pack_entry(text=text)]
    # Perfectly grounded (supported unit with evidence) yet the verifier
    # judges it off-intent at both levels: per-unit flag and turn flag.
    verdict = assess_turn_adequacy(
        user_message="frozen-259-marker-beta turn",
        reply=text,
        evidence_pack=pack,
        grounding_result=_grounding(text=text, addresses_intent=False, answer_relevant=False),
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent="frozen-259-marker-beta intent",
    )
    assert verdict.verdict == "fail"
    assert verdict.failure_category == "irrelevant-citation"
    assert verdict.technically_grounded is True
    assert verdict.answers_request is False


def test_top_level_irrelevance_overrides_support() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    text = "Первый шаг: признание бессилия перед тягой помогает сегодня."
    pack = [_pack_entry(text=text)]
    # Even with a per-unit positive flag, an explicit turn-level
    # answer_relevant False fails the turn as irrelevant-citation.
    grounding = _grounding(text=text, addresses_intent=True, answer_relevant=False)
    verdict = assess_turn_adequacy(
        user_message="frozen-259-marker-gamma turn",
        reply=text,
        evidence_pack=pack,
        grounding_result=grounding,
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent="frozen-259-marker-gamma intent",
    )
    assert verdict.verdict == "fail"
    assert verdict.failure_category == "irrelevant-citation"


def test_generic_fallback_queries_have_no_domain_branching() -> None:
    from aa.conversation.answer_adequacy import (
        build_generic_fallback_queries,
        build_recovery_queries,
    )

    # Any non-empty turn text yields a non-empty generic fallback.
    assert build_recovery_queries("frozen-259-marker-alpha turn text")
    assert build_recovery_queries("short note")
    assert build_generic_fallback_queries("frozen-259-marker-beta turn text")
    # Empty input yields no queries.
    assert build_recovery_queries("") == []
    assert build_recovery_queries("   ") == []
    assert build_generic_fallback_queries("") == []


async def test_pipeline_serves_grounded_relevant_answer() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    text = "Поддержка рядом помогает пережить тягу сегодня без выпивки."
    pack = [_pack_entry(passage_id="chapter-3#exp0002", text=text)]

    class _Grounded:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            return AIMessage(content=text)

    class _Verifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["chapter-3#exp0002"],
                "addresses_intent": True,
            }

    outcome = await run_v2_answer_turn(
        user_message="frozen-259-marker-alpha turn",
        summary="",
        recent=[HumanMessage(content="hello")],
        evidence_pack=pack,
        answer_model=_Grounded(),
        verifier_model=_Verifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        planner_mode="retrieval",
        resolved_intent="frozen-259-marker-alpha intent",
    )
    assert outcome["text"] == text
    assert outcome["telemetry"]["adequacy_verdict"] == "pass"


def test_live_lane_uses_rubric_relevance_without_domain_heuristics() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    live_source = (root / "src" / "aa" / "qualification" / "product_contract_live.py").read_text(
        encoding="utf-8"
    )
    assert "assess_reply_relevance_with_rubric" in live_source
    assert "_assess_live_relevance" in live_source
    assert "load_held_out_corpus" in live_source
    assert "held_out" in live_source
    assert "_extract_step_numbers_for_relevance" not in live_source
    assert "_content_tokens_for_relevance" not in live_source
    assert "_prompt_allows_context_rescue" not in live_source
    # Quoted stem literals must not drive recovery branching; the quoted
    # form keeps this check scoped to string literals in source.
    assert '"похмел"' not in live_source
    assert '"трезв"' not in live_source


def test_answer_adequacy_has_no_heuristic_tables() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    source = (root / "src" / "aa" / "conversation" / "answer_adequacy.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "_STEP_WORD_RE",
        "_STEP_DIGIT_RE",
        "_RECOVERY_DOMAIN_STEMS",
        "_GREETING_VOCABULARY",
        "_PERSONAL_DISCLOSURE_RE",
        "_FOLLOWUP_REFERENCE_RE",
        "_SECOND_PERSON_TOKENS",
        "_CAPABILITY_TOKENS",
    ):
        assert name not in source


def test_no_exact_question_whitelist_in_product() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "graph.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "answer_adequacy.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "planner_node.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "verifier.py").read_text(encoding="utf-8"),
    ]
    # Synthetic frozen markers (also used as fixed test data above) prove
    # no exact-question whitelist without embedding live prompts.
    frozen_fragments = (
        "frozen-259-marker-alpha",
        "frozen-259-marker-beta",
        "frozen-259-marker-gamma",
    )
    for source in sources:
        for fragment in frozen_fragments:
            assert fragment not in source
