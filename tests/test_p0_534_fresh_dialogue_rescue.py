"""Qualification #7 / kodmial/aa#534: fresh no-evidence dialogue rescue.

Failure class: the Gate C canary ``meta-capability`` scenario can reach the
substantive pipeline when the planner stochastically emits retrieval, then
fail the verifier with ``verifier-incomplete-verdict`` because a claim-free
non-book unit was judged unsupported. Additional book retrieval cannot make
that wording supported, so the historical repair loop only burns provider
rounds before the delivered reply becomes a service error. When the verifier
reports no book scope and no unavailable unit, the pipeline generates one
fresh empty-evidence dialogue reply and serves it only after the independent
whole-turn judge certifies it helpful, intent-addressing and claim-free.

These are synthetic fixtures for the failure class, never the literal
qualification prompt.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage

_SYNTHETIC_USER_MESSAGE = "Синтетический вопрос о возможностях поддержки сегодня"
_SYNTHETIC_RESOLVED_INTENT = "синтетическое намерение о возможностях поддержки"
_SYNTHETIC_BAD_DRAFT = "Синтетическое обещание, которое не требует книги."
_SYNTHETIC_DIALOGUE = "Я рядом, готов спокойно поддержать разговор сегодня."


class _SequencedAnswer:
    """Answer double: first draft, then a fresh dialogue reply."""

    def __init__(self, first: str, second: str) -> None:
        self._texts = [first, second]
        self.calls = 0

    async def ainvoke(self, messages: Any) -> Any:
        _ = messages
        index = min(self.calls, len(self._texts) - 1)
        self.calls += 1
        return AIMessage(content=self._texts[index])


class _DialogueJudge:
    """Whole-turn judge double certifying pure dialogue."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
    ) -> dict[str, object]:
        self.calls += 1
        _ = (prompt, system, schema, retry_count)
        return {
            "helpful": True,
            "addresses_intent": True,
            "contains_substantive_claim": False,
        }


class _SubstantiveJudge:
    """Whole-turn judge double reporting a substantive claim."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
    ) -> dict[str, object]:
        self.calls += 1
        _ = (prompt, system, schema, retry_count)
        return {
            "helpful": True,
            "addresses_intent": True,
            "contains_substantive_claim": True,
        }


class _CountingPlanner:
    """Repair planner double recording any full repair round."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        return {
            "mode": "retrieval",
            "resolved_intent": _SYNTHETIC_RESOLVED_INTENT,
            "queries": ["синтетический запрос поддержки"],
        }


def _install_verifier(monkeypatch: pytest.MonkeyPatch, *, book_scope: bool) -> None:
    import aa.conversation.turn_pipeline as pipeline
    from aa.conversation.verifier_schema import (
        GroundingResult,
        UnitVerdict,
        validate_grounding_result,
    )

    async def _fake_run_verifier(units: Any, passages: Any, **kwargs: Any) -> GroundingResult:
        _ = (passages, kwargs)
        verdicts = [
            UnitVerdict(
                unit_id=unit.unit_id,
                scope="book" if book_scope else "conversation_glue",
                supported=False,
                evidence_passage_ids=[],
                addresses_intent=True,
                origin="book_claim" if book_scope else "conversation_glue",
            )
            for unit in units
        ]
        return validate_grounding_result(
            GroundingResult(
                verified=True,
                units=verdicts,
                all_required_supported=False,
                unavailable_unit_ids=[],
                answer_relevant=True,
            ),
            expected_unit_ids=[unit.unit_id for unit in units],
        )

    monkeypatch.setattr(pipeline, "run_verifier", _fake_run_verifier)


def _kwargs(answer: Any, judge: Any, planner: Any) -> dict[str, Any]:
    return {
        "user_message": _SYNTHETIC_USER_MESSAGE,
        "summary": "",
        "recent": [],
        "evidence_pack": [],
        "answer_model": answer,
        "verifier_model": object(),
        "planner_model": planner,
        "retrieval_index": None,
        "initial_query_count": 3,
        "planner_reason": "substantive-with-queries",
        "planner_mode": "retrieval",
        "resolved_intent": _SYNTHETIC_RESOLVED_INTENT,
        "whole_turn_judge_model": judge,
    }


async def test_unsupported_non_book_turn_serves_fresh_judged_dialogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.qualification.product_contract_live import _is_direct_meta_reply

    _install_verifier(monkeypatch, book_scope=False)
    answer = _SequencedAnswer(_SYNTHETIC_BAD_DRAFT, _SYNTHETIC_DIALOGUE)
    judge = _DialogueJudge()
    planner = _CountingPlanner()

    outcome = await run_v2_answer_turn(**_kwargs(answer, judge, planner))

    assert outcome["text"] == _SYNTHETIC_DIALOGUE
    assert answer.calls == 2
    assert judge.calls == 1
    assert planner.calls == 0
    telemetry = dict(outcome["telemetry"])
    assert telemetry["answer_outcome"] == "conversational-generated"
    assert telemetry["adequacy_verdict"] == "pass"
    assert telemetry["answers_request"] is True
    assert telemetry["technically_grounded"] is False
    assert telemetry["qualified"] is True
    assert _is_direct_meta_reply(outcome["text"], snapshot=telemetry) is True


async def test_unsupported_non_book_turn_not_rescued_by_substantive_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aa.conversation.failures import TurnFailed
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    _install_verifier(monkeypatch, book_scope=False)
    answer = _SequencedAnswer(_SYNTHETIC_BAD_DRAFT, _SYNTHETIC_DIALOGUE)
    judge = _SubstantiveJudge()
    planner = _CountingPlanner()

    with pytest.raises(TurnFailed):
        await run_v2_answer_turn(**_kwargs(answer, judge, planner))

    assert judge.calls >= 1


async def test_book_scoped_unsupported_turn_never_starts_fresh_dialogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aa.conversation.failures import TurnFailed
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    _install_verifier(monkeypatch, book_scope=True)
    answer = _SequencedAnswer(_SYNTHETIC_BAD_DRAFT, _SYNTHETIC_DIALOGUE)
    judge = _DialogueJudge()
    planner = _CountingPlanner()

    with pytest.raises(TurnFailed):
        await run_v2_answer_turn(**_kwargs(answer, judge, planner))

    # Only the initial draft generation ran; the fresh dialogue helper was
    # never allowed to turn a book-scoped gap into a conversational success.
    assert answer.calls == 1
    assert judge.calls == 0


def test_no_literal_qualification_branches_in_pipeline_or_prompts() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    seen = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "src/aa/conversation/turn_pipeline.py",
            "src/aa/conversation/planner_node.py",
            "prompts/aa-agent-system-v2.md",
            "prompts/aa-planner-system-v2.md",
        )
    )
    for fragment in (
        "Чем ты вообще можешь быть полезен",
        "Слушай, а ты тут вообще чем помочь можешь",
        "К вечеру очень тянет выпить",
        "Под вечер опять тянеет выпить",
    ):
        assert fragment not in seen
