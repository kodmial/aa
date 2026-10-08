"""P0 kodmial/aa#259: context-loss and false-glue on short disclosures.

Proven gap after #251: a follow-up about decisions in the current step was
answered from another step, and a short first-person daily-drinking
disclosure received a generic technical retry. Root causes were
architectural: conversational context was discarded in the glue decision,
short disclosures defaulted to glue, and whole-turn relevance relied on
single-prefix overlap without step fidelity.

This test locks the architecture-level repair (turn-independent, no exact
user-question whitelist, Product Contract unchanged):

- short personal statements are substantive even without a question mark;
- genuine greetings are glue only when positively established, with
  context participating;
- the resolved intent (live message plus summary plus recent turns) travels
  to the planner recovery, answer and independent adequacy gate;
- a cited passage from another numbered step fails adequacy and Gate C
  relevance even with generic vocabulary overlap;
- the live lane exercises raw-update multi-turn families with negative
  controls and requires no fallback in the final sent reply.
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
    }


def _grounding(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Первый шаг: признание бессилия перед тягой помогает сегодня.",
) -> dict[str, Any]:
    return {
        "verified": True,
        "all_required_supported": True,
        "units": [
            {
                "unit_id": "u1",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": [passage_id],
                "text": text,
            }
        ],
    }


def test_short_first_person_disclosure_is_substantive() -> None:
    from aa.conversation.answer_adequacy import is_proven_glue_message

    assert is_proven_glue_message("Привет") is True
    assert is_proven_glue_message("Здравствуйте!") is True
    assert is_proven_glue_message("Спасибо!") is True
    # Short disclosures are substantive, never proven glue.
    assert is_proven_glue_message("Пью каждый день") is False
    assert is_proven_glue_message("Я пью") is False
    assert is_proven_glue_message("Выпил вчера") is False
    # Context participates: the same short text after a recovery discussion
    # stays substantive rather than collapsing to glue.
    assert (
        is_proven_glue_message(
            "Пью каждый день",
            summary="разговор про тягу вечером",
            recent_count=2,
        )
        is False
    )
    assert (
        is_proven_glue_message(
            "Привет",
            summary="разговор про тягу вечером",
            recent_count=2,
        )
        is True
    )


def test_step_followup_resolves_through_context() -> None:
    from aa.conversation.answer_adequacy import (
        extract_step_numbers,
        resolve_effective_request,
    )

    summary = "Обсуждаем Первый шаг программы"
    resolved = resolve_effective_request(
        "А какие решения принимают в этом шаге?",
        summary=summary,
        recent=["Расскажи про Первый шаг"],
    )
    assert extract_step_numbers(resolved) == {1}
    assert extract_step_numbers("Третий шаг о решениях") == {3}


def test_short_explicit_topic_pivot_does_not_inherit_stale_context() -> None:
    from aa.conversation.answer_adequacy import resolve_effective_request

    resolved = resolve_effective_request(
        "А теперь про работу",
        summary="До этого обсуждали вечернюю тягу и выпивку",
        recent=["К вечеру очень тянет выпить"],
    )
    assert resolved == "А теперь про работу"


def test_short_explicit_topic_pivot_cannot_be_rescued_by_prior_recovery_context() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    text = "Поддержка рядом помогает пережить тягу сегодня без выпивки."
    pack = [_pack_entry(passage_id="chapter-3#exp0002", text=text)]
    verdict = assess_turn_adequacy(
        user_message="А теперь про работу",
        reply=text,
        evidence_pack=pack,
        grounding_result=_grounding(passage_id="chapter-3#exp0002", text=text),
        planner_reason="substantive-with-queries",
        summary="До этого обсуждали вечернюю тягу и выпивку",
        recent=["К вечеру очень тянет выпить"],
        prior_user_messages=["К вечеру очень тянет выпить"],
    )
    assert verdict.verdict == "fail"
    assert verdict.failure_category == "irrelevant-citation"


def test_step_number_extraction_requires_one_step_phrase() -> None:
    from aa.conversation.answer_adequacy import extract_step_numbers

    assert extract_step_numbers("В пятницу обсуждали шаг программы") == set()
    assert extract_step_numbers("Пятый шаг программы") == {5}
    assert extract_step_numbers("Шаг пятый") == {5}


def test_step_mismatch_fails_despite_generic_overlap() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    pack = [
        _pack_entry(
            passage_id="chapter-5#exp0001",
            text="Третий шаг говорит о решениях и воле.",
        )
    ]
    verdict = assess_turn_adequacy(
        user_message="А какие решения принимают в этом шаге?",
        reply="Третий шаг говорит о решениях и воле.",
        evidence_pack=pack,
        grounding_result=_grounding(
            passage_id="chapter-5#exp0001",
            text="Третий шаг говорит о решениях и воле.",
        ),
        planner_reason="substantive-with-queries",
        summary="Обсуждаем Первый шаг программы",
        recent=["Расскажи про Первый шаг"],
    )
    assert verdict.verdict == "fail"
    assert verdict.failure_category == "irrelevant-citation"


def test_same_step_passes_with_resolved_referent() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    pack = [_pack_entry()]
    verdict = assess_turn_adequacy(
        user_message="А какие решения принимают в этом шаге?",
        reply="Первый шаг: признание бессилия перед тягой помогает сегодня.",
        evidence_pack=pack,
        grounding_result=_grounding(),
        planner_reason="substantive-with-queries",
        summary="Обсуждаем Первый шаг программы",
        recent=["Расскажи про Первый шаг"],
    )
    assert verdict.verdict == "pass"
    assert verdict.answers_request is True


def test_short_disclosure_generic_retry_fails() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy
    from aa.conversation.turn_pipeline import NATURAL_RETRY_REPLY

    pack = [_pack_entry()]
    glue_grounding: dict[str, Any] = {
        "verified": True,
        "all_required_supported": True,
        "units": [
            {
                "unit_id": "u1",
                "scope": "conversation_glue",
                "supported": True,
                "evidence_passage_ids": [],
            }
        ],
    }
    verdict = assess_turn_adequacy(
        user_message="Пью каждый день",
        reply=NATURAL_RETRY_REPLY,
        evidence_pack=pack,
        grounding_result=glue_grounding,
        planner_reason="substantive-with-queries",
    )
    # A fallback template carries no practical book help for the disclosure.
    assert verdict.verdict == "fail"
    assert verdict.answers_request is False


def test_short_disclosure_relevant_answer_passes() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    text = "Поддержка рядом помогает пережить тягу сегодня без выпивки."
    pack = [_pack_entry(passage_id="chapter-3#exp0002", text=text)]
    verdict = assess_turn_adequacy(
        user_message="Пью каждый день",
        reply=text,
        evidence_pack=pack,
        grounding_result=_grounding(passage_id="chapter-3#exp0002", text=text),
        planner_reason="substantive-with-queries",
    )
    assert verdict.verdict == "pass"


def test_gate_c_relevance_requires_same_step() -> None:
    from aa.qualification.product_contract_live import _assess_prompt_reply_relevance

    assert (
        _assess_prompt_reply_relevance(
            "Расскажи про Первый шаг",
            "Третий шаг говорит о решениях и воле.",
        )
        is False
    )
    assert (
        _assess_prompt_reply_relevance(
            "А какие решения принимают в этом шаге?",
            "Первый шаг: признание бессилия помогает сегодня.",
            context="Обсуждаем Первый шаг",
        )
        is True
    )
    assert (
        _assess_prompt_reply_relevance(
            "Вечером тяжело пережить тягу",
            "Ведите финансовый бюджет спокойно.",
        )
        is False
    )


async def test_pipeline_short_disclosure_never_serves_glue_as_success() -> None:
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
            }

    outcome = await run_v2_answer_turn(
        user_message="Пью каждый день",
        summary="",
        recent=[HumanMessage(content="hello")],
        evidence_pack=pack,
        answer_model=_Grounded(),
        verifier_model=_Verifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
    )
    assert outcome["text"] == text
    assert outcome["telemetry"]["adequacy_verdict"] == "pass"


def test_live_lane_covers_multiturn_step_and_admission_families() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    live_source = (root / "src" / "aa" / "qualification" / "product_contract_live.py").read_text(
        encoding="utf-8"
    )
    assert "live-step-continuity-second-relevant" in live_source
    assert "live-short-admission-second-relevant" in live_source
    assert "live-typo-variant-grounded" in live_source
    assert "live-context-switch-helpful" in live_source
    assert "live-negative-control-wrong-step-fails" in live_source
    assert "_extract_step_numbers_for_relevance" in live_source
    assert "api.sent_texts[-1]" in live_source
    assert "last_telemetry_for_thread" in live_source
    assert "scenario_deliveries == len(scenarios)" in live_source
    assert "len(api.sent_texts) >= len(scenarios)" not in live_source


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
    frozen_fragments = (
        "Пью каждый день",
        "А какие решения принимают в этом шаге",
        "Пад вечер тянеет выпить",
    )
    for source in sources:
        for fragment in frozen_fragments:
            assert fragment not in source
