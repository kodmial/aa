"""P0 kodmial/aa#260: repair Gate C live-production-path failures.

Proven product failures on exact main c113c31920e5ed99f62d6bf400e03d2870c0b824
(run 37790650674):

- C:live-meta-direct-1 (a natural capability probe served an evasive or
  fallback reply instead of a direct capability answer);
- C:live-book-grounding-substantive-drinking-11 plus
  C:live-answer-relevance-long-conversation-17 (two book-grounded turns
  lost on relevance/grounding while siblings passed);
- C:live-answer-no-generic-collapse (one turn collapsed to a retry
  fallback) plus C:live-substantive-grounded-book-answer (6/8 grounded).

Diagnosis at the production-path boundary (turn-independent, no
exact-question special cases, Product Contract #110 unchanged):

- the meta classifier only recognized contiguous marker substrings, so
  ordinary capability paraphrases with split capability tokens were
  misrouted to the substantive book path, where adequacy demanded book
  evidence for a capability question and the turn collapsed to retry;
- whole-turn relevance compared only the immediate prompt, so a terse
  contextual follow-up in an ongoing conversation failed even when the
  evidence-backed answer continued the resolved topic.

Fix: token-level structural capability recognition (second person plus
capability vocabulary in a short single-segment probe, neutral filler
scaffolding ignored, recovery domain still substantive) and
same-conversation context resolution for relevance in both production
adequacy and the live qualification lane.
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
                "text": unit_text,
            }
        ],
    }


def test_natural_capability_probes_are_meta() -> None:
    from aa.conversation.answer_adequacy import is_meta_request

    assert is_meta_request("Ты кто?") is True
    assert is_meta_request("Что ты умеешь?") is True
    assert is_meta_request("Ты можешь быть полезен?") is True
    assert is_meta_request("А ты вообще что умеешь?") is True


def test_capability_prefix_with_recovery_request_stays_substantive() -> None:
    from aa.conversation.answer_adequacy import is_meta_request

    assert is_meta_request("Ты кто? Помоги с тягой вечером.") is False
    assert is_meta_request("Вечером тяжело, сильная тяга, как справиться?") is False
    assert is_meta_request("Подскажи, как справиться с желанием?") is False
    assert is_meta_request("Привет! Помоги разобраться с тягой вечером?") is False


def test_meta_without_book_passes_adequacy_as_glue() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    verdict = assess_turn_adequacy(
        user_message="Ты можешь быть полезен?",
        reply="Я помощник по материалам сообщества: поддерживаю разговор.",
        evidence_pack=[],
        grounding_result={
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
        },
        planner_reason="legitimate-glue",
    )
    assert verdict.verdict == "pass"
    assert verdict.substantive_request is False


def test_contextual_followup_resolves_against_prior_turns() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    prior = "Вечером тяжело пережить тягу, как обходиться?"
    generic_followup = "И что это значит для меня сейчас?"
    pack = [_pack_entry()]
    grounding = _book_grounding()
    reply = "Поддержка рядом помогает пережить тягу сегодня."

    without_context = assess_turn_adequacy(
        user_message=generic_followup,
        reply=reply,
        evidence_pack=pack,
        grounding_result=grounding,
        planner_reason="substantive-with-queries",
    )
    assert without_context.verdict == "fail"

    with_context = assess_turn_adequacy(
        user_message=generic_followup,
        reply=reply,
        evidence_pack=pack,
        grounding_result=grounding,
        planner_reason="substantive-with-queries",
        prior_user_messages=[prior],
    )
    assert with_context.verdict == "pass"
    assert with_context.answers_request is True


def test_unrelated_citation_still_fails_with_context() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    ev_text = "Финансовое планирование помогает вести бюджет спокойно."
    pack = [_pack_entry(passage_id="chapter-9#exp0001", text=ev_text)]
    grounding = {
        "verified": True,
        "all_required_supported": True,
        "units": [
            {
                "unit_id": "u1",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": ["chapter-9#exp0001"],
                "text": ev_text,
            }
        ],
    }
    verdict = assess_turn_adequacy(
        user_message="И что это значит для меня сейчас?",
        reply=ev_text,
        evidence_pack=pack,
        grounding_result=grounding,
        planner_reason="substantive-with-queries",
        prior_user_messages=["Вечером тяжело пережить тягу, как обходиться?"],
    )
    assert verdict.verdict == "fail"
    assert verdict.failure_category == "irrelevant-citation"


def test_qualification_relevance_resolves_followup_context() -> None:
    from aa.qualification.product_contract_live import _assess_prompt_reply_relevance

    prior = "Вечером тяжело пережить тягу, как обходиться?"
    generic_followup = "И что это значит для меня сейчас?"
    reply = "Поддержка рядом помогает пережить тягу сегодня."
    assert _assess_prompt_reply_relevance(generic_followup, reply) is False
    combined = f"{prior} {generic_followup}"
    assert _assess_prompt_reply_relevance(combined, reply) is True
    assert _assess_prompt_reply_relevance(prior, "Ведите финансовый бюджет.") is False


def test_no_exact_live_question_special_cases() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "graph.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "answer_adequacy.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "planner_node.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "verifier.py").read_text(encoding="utf-8"),
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
