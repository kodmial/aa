"""P0 kodmial/aa#364: Gate C substantive helpfulness needs verifier/judge alignment.

Proven product failure on exact main df04047 (run 38243944251):

- C:live-independent-helpfulness-substantive-drinking-2 on
  live-production-path (plus D:workflow-step-failed downstream and
  E:latency-budget-exceeded SLO).

Live diagnostics for the failing run show the functional defect is in
the grounding/relevance stage, not only latency (Gate E owns the SLO):

- scenario 2 (substantive-drinking) served ``narrowed-adequacy`` with
  verified_book_units=3, verifier_outcome=passed, grounded=True and
  telemetry-relevant=True, yet the procedurally independent whole-turn
  judge reported helpful=False (judge_overrode_telemetry). The narrowed
  subset passed the internal adequacy gate while the independent judge
  judged it unhelpful: background restatement served as a substitute
  for practical guidance.
- the verifier prompt asked only whether a unit "addresses the
  context-resolved intent", while the independent judge explicitly
  fails "generic background restatement, or difficulty description
  offered as a substitute for practical guidance". A grounded but
  background-only unit could therefore be marked addresses_intent=True
  by the verifier and pass adequacy, then fail the judge.

Root cause (failure class, not one prompt): the verifier relevance
instruction is weaker than the judge helpfulness instruction. Grounded
background without practical guidance counts as relevant upstream and
is served narrowed as success, then fails downstream as unhelpful.

Repair (generic, no exact-question special cases; Product Contract
#110 unchanged): state the same practical-guidance rule in the
verifier stage instruction everywhere it is issued (system prompt,
per-unit user prompt, and text-JSON fallback suffix), turn-independent.
Background-only restatement, generic repetition, off-topic drift, or
difficulty description as a substitute for practical guidance does not
address the intent, even when grounded. The deterministic aggregation
(verified addresses_intent=False fails relevance and adequacy as
irrelevant-citation, never served as grounded success) is unchanged
and locked below.
"""

from __future__ import annotations

import pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]

_PRACTICAL_CLAUSE = "as a substitute for practical guidance does not address the intent"


def _read_prompt(name: str) -> str:
    return (ROOT / "prompts" / name).read_text(encoding="utf-8")


def test_verifier_system_prompt_requires_practical_guidance() -> None:
    text = _read_prompt("aa-verifier-system-v2.md")
    lowered = text.casefold()
    assert "addresses_intent" in lowered
    assert "only restates background" in lowered
    assert "substitute for practical guidance" in lowered
    assert _PRACTICAL_CLAUSE in text


def test_verifier_text_suffix_requires_practical_guidance() -> None:
    from aa.conversation.verifier import VERIFIER_TEXT_JSON_SUFFIX

    assert _PRACTICAL_CLAUSE in VERIFIER_TEXT_JSON_SUFFIX
    assert "addresses_intent" in VERIFIER_TEXT_JSON_SUFFIX


def test_verifier_unit_prompt_requires_practical_guidance() -> None:
    from aa.conversation.response_units import ResponseUnitDraft
    from aa.conversation.verifier import build_single_unit_text

    unit = ResponseUnitDraft(unit_id="u1", text="Synthetic unit.", char_start=0, char_end=15)
    prompt = build_single_unit_text(
        unit=unit,
        passages=[],
        resolved_intent="synthetic intent",
        user_message="synthetic message",
    )
    assert _PRACTICAL_CLAUSE in prompt
    assert "addresses_intent" in prompt.casefold() or "addresses the" in prompt.casefold()


def _background_only_grounding() -> dict[str, Any]:
    return {
        "verified": True,
        "all_required_supported": True,
        "answer_relevant": False,
        "relevance_category": "irrelevant-citation",
        "units": [
            {
                "unit_id": "u1",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": ["chapter-3#exp0000"],
                "addresses_intent": False,
            },
            {
                "unit_id": "u2",
                "scope": "conversation_glue",
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": False,
            },
        ],
    }


def test_background_only_grounding_is_not_relevant() -> None:
    from aa.conversation.answer_adequacy import _turn_is_relevant

    assert _turn_is_relevant(_background_only_grounding()) is False


def test_background_only_narrowed_fails_adequacy() -> None:
    from aa.conversation.answer_adequacy import (
        FAILURE_IRRELEVANT_CITATION,
        assess_turn_adequacy,
    )

    assessment = assess_turn_adequacy(
        user_message="synthetic substantive request",
        reply="Synthetic background restatement without practical guidance.",
        evidence_pack=[
            {
                "passage_id": "chapter-3#exp0000",
                "text": "Synthetic passage.",
                "source_id": "ru-fourth-edition-txt",
                "section_id": "chapter-3",
                "char_start": 0,
                "char_end": 18,
                "text_sha256": "a" * 64,
                "source_sha256": "s" * 64,
                "corpus_version": "r" * 64,
            }
        ],
        grounding_result=_background_only_grounding(),
        planner_reason="substantive-with-queries",
        verifier_outcome="passed",
        unavailable_units=0,
        turn_budget_exceeded=False,
        planner_mode="retrieval",
        resolved_intent="synthetic resolved intent",
        planner_query_count=3,
    )
    assert assessment.verdict == "fail"
    assert assessment.failure_category == FAILURE_IRRELEVANT_CITATION
    assert assessment.answers_request is False


def test_background_only_snapshot_is_not_grounded_success() -> None:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    snapshot: dict[str, Any] = {
        "answer_outcome": "narrowed-adequacy",
        "verifier_outcome": "passed",
        "verifier_unavailable_units": 0,
        "turn_budget_exceeded": False,
        "planner_query_count": 3,
        "retrieval_passages": 3,
        "verified_book_units": 1,
        "response_units": 2,
        "adequacy_verdict": "fail",
        "failure_category": "irrelevant-citation",
        "planner_reason": "substantive-with-queries",
        "answers_request": False,
        "technically_grounded": True,
        "qualified": False,
    }
    assert (
        _is_grounded_substantive_reply(
            snapshot, "Synthetic background restatement without practical guidance."
        )
        is False
    )


def test_no_exact_question_special_cases() -> None:
    system = _read_prompt("aa-verifier-system-v2.md")
    import inspect

    from aa.conversation import verifier as verifier_mod

    builder_source = inspect.getsource(verifier_mod.build_single_unit_text)
    suffix = verifier_mod.VERIFIER_TEXT_JSON_SUFFIX
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
        assert fragment not in system
        assert fragment not in builder_source
        assert fragment not in suffix
