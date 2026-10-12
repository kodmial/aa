"""P0 kodmial/aa#366: Gate C mixed-greeting-substantive must stay retrieval-grounded.

Proven product failure on exact main 4c86dacffe7d0cc5ea56908177b6ec8256a1bf38
(run 38253036929):

- C:live-book-grounding-mixed-greeting-substantive-9 on live-production-path
  (plus D:workflow-step-failed downstream and E:latency-budget-exceeded SLO).

Live diagnostics for the failing run show the functional defect is in the
production planning/answer stage, not only latency (Gate E owns the SLO):

- scenario 9 (mixed-greeting-substantive core) and scenario 18 (held-out
  variant) both delivered text with an entirely empty stage snapshot
  (planner_outcome "", production_outcome "", verified_book_units 0, no
  stage latencies) and the independent judge reported missing-turn-telemetry;
  scenario 11 (held-out substantive-drinking typo variant) failed the same
  way. The turns collapsed to a bookless fallback instead of a grounded
  answer.
- stage outcome counts show planner empty 2 plus skipped-initial 5 and
  retrieval preloaded 5 alongside 12 clarifications over 16 answer rounds:
  substantive turns served glue/clarification instead of book guidance.

Root cause (failure class, not one prompt): the planner and answer
instructions never state that a leading greeting, farewell, thanks, or
other conversational opener does not make a turn conversational when a
substantive request follows. A greeting-prefixed substantive turn is
therefore planned as conversational glue (zero queries, empty pack) and
answered with greeting-only filler, which fails the book-grounded contract
for the mixed-greeting-substantive family.

Repair (generic, no exact-question special cases; Product Contract #110
unchanged): state the same opener rule in every place the conversational
vs retrieval decision is issued (planner system prompt, planner text-JSON
fallback suffix) and in the answer contract (agent system prompt):
a greeting opener plus a substantive request is retrieval/substantive;
the resolved intent and queries focus on the substantive request while the
final reply may still acknowledge the opener briefly alongside grounded
guidance. Pure greeting alone stays conversational glue.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]

_GREETING_RULE = "greeting opener plus a substantive request is retrieval"
_OPENER_RULE = "opens with a greeting"
_ACK_RULE = "acknowledge"


def _read_prompt(name: str) -> str:
    return (ROOT / "prompts" / name).read_text(encoding="utf-8")


def test_planner_system_prompt_treats_greeting_plus_substantive_as_retrieval() -> None:
    text = _read_prompt("aa-planner-system-v2.md")
    lowered = text.casefold()
    assert "purely conversational" in lowered
    assert "greeting" in lowered
    assert "never purely conversational" in lowered
    assert "focused on the substantive request" in lowered


def test_planner_text_suffix_treats_greeting_plus_substantive_as_retrieval() -> None:
    from aa.conversation.planner_node import PLANNER_TEXT_JSON_SUFFIX

    lowered = PLANNER_TEXT_JSON_SUFFIX.casefold()
    assert "purely conversational" in lowered
    assert "greeting opener plus a substantive request is retrieval" in lowered
    assert "focused on the substantive request" in lowered
    assert _GREETING_RULE in PLANNER_TEXT_JSON_SUFFIX.casefold() or "greeting" in lowered


def test_agent_system_prompt_requires_grounded_answer_for_greeting_opener() -> None:
    text = _read_prompt("aa-agent-system-v2.md")
    lowered = text.casefold()
    assert "substantive turn that opens with a greeting" in lowered
    assert "still requires a book-supported answer" in lowered
    assert "brief natural acknowledgement" in lowered or "acknowledge" in lowered
    assert _OPENER_RULE in lowered
    assert _ACK_RULE in lowered


def _pack_entry(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Synthetic grounded guidance for the evening difficulty today.",
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


def test_mixed_greeting_glue_only_fails_adequacy() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    pack = [_pack_entry()]
    glue = "Synthetic greeting acknowledgement without book guidance today."
    verdict = assess_turn_adequacy(
        user_message="Synthetic greeting opener plus synthetic substantive request today",
        reply=glue,
        evidence_pack=pack,
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
    assert verdict.verdict == "fail"
    assert verdict.answers_request is False


def test_mixed_greeting_grounded_guidance_passes_adequacy() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    pack = [_pack_entry()]
    reply = "Synthetic grounded guidance for the evening difficulty today."
    verdict = assess_turn_adequacy(
        user_message="Synthetic greeting opener plus synthetic substantive request today",
        reply=reply,
        evidence_pack=pack,
        grounding_result={
            "verified": True,
            "all_required_supported": True,
            "answer_relevant": True,
            "units": [
                {
                    "unit_id": "u1",
                    "scope": "book",
                    "supported": True,
                    "evidence_passage_ids": ["chapter-3#exp0000"],
                    "addresses_intent": True,
                }
            ],
        },
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent="Synthetic substantive request without the opener",
        planner_query_count=3,
    )
    assert verdict.verdict == "pass"
    assert verdict.answers_request is True


def test_mixed_greeting_snapshot_needs_book_units() -> None:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    glue_snapshot: dict[str, Any] = {
        "answer_outcome": "conversational-generated",
        "verifier_outcome": "passed",
        "verifier_unavailable_units": 0,
        "turn_budget_exceeded": False,
        "planner_query_count": 0,
        "retrieval_passages": 0,
        "verified_book_units": 0,
        "response_units": 1,
        "adequacy_verdict": "pass",
        "failure_category": "",
        "planner_reason": "legitimate-glue",
        "answers_request": True,
        "technically_grounded": False,
        "qualified": True,
    }
    assert (
        _is_grounded_substantive_reply(
            glue_snapshot, "Synthetic greeting acknowledgement without book guidance today."
        )
        is False
    )


def test_no_exact_question_special_cases() -> None:
    planner = _read_prompt("aa-planner-system-v2.md")
    agent = _read_prompt("aa-agent-system-v2.md")
    import inspect

    from aa.conversation import planner_node as planner_mod

    suffix = planner_mod.PLANNER_TEXT_JSON_SUFFIX
    builder_source = inspect.getsource(planner_mod.build_planner_messages)
    for fragment in (
        "Чем ты вообще можешь быть полезен",
        "Слушай, а ты тут вообще чем помочь можешь",
        "К вечеру очень тянет выпить",
        "Под вечер опять тянеет выпить",
        "Дома снова ссора",
        "Поругались дома из-за бухла",
        "покупать акции",
        "Какой телефон сейчас выгоднее купить",
        "Привет! К вечеру очень тянет выпить",
        "Привет, подскажи че делать когда под вечер тянеет выпить",
    ):
        assert fragment not in planner
        assert fragment not in agent
        assert fragment not in suffix
        assert fragment not in builder_source
