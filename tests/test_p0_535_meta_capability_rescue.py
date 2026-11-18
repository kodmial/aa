"""Qualification #7 / kodmial/aa#535: complete negative non-book verdict rescue.

Failure class: the Gate C canary ``meta-capability`` scenario reaches the
substantive pipeline when the planner stochastically emits retrieval, then
fails the verifier with a *complete* set of negative non-book decisions.
Additional book retrieval cannot make a claim-free unit supported, so the
historical repair loop only burns provider rounds before the delivered reply
becomes a service error. When the verifier reports no book scope and no
unavailable unit, the pipeline generates one fresh empty-evidence dialogue
reply and serves it only after the independent whole-turn judge certifies it
helpful, intent-addressing and claim-free.

These are synthetic fixtures for the failure class, never the literal
qualification prompt.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

_SYNTHETIC_USER_MESSAGE = "Синтетический вопрос о возможностях поддержки сегодня"
_SYNTHETIC_RESOLVED_INTENT = "синтетическое намерение о возможностях поддержки"
_SYNTHETIC_BAD_DRAFT = "Синтетическое обещание, которое не требует книги."
_SYNTHETIC_DIALOGUE = "Я рядом, готов спокойно поддержать разговор сегодня."


class _SequencedAnswer:
    """Answer double: first draft, then one fresh dialogue reply."""

    def __init__(self, first: str, second: str) -> None:
        self._texts = [first, second]
        self.calls = 0

    async def ainvoke(self, messages: Any) -> Any:
        _ = messages
        index = min(self.calls, len(self._texts) - 1)
        self.calls += 1
        return AIMessage(content=self._texts[index])


class _FixedBatchedVerifier:
    """Batched verifier double returning one scripted decision per unit."""

    supports_batched_verifier = True

    def __init__(self, decision: dict[str, Any]) -> None:
        self._decision = dict(decision)
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        return {"verdicts": [dict(self._decision)]}


class _DialogueJudge:
    """Whole-turn judge double certifying pure dialogue."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
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
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
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


def _glue_negative_decision() -> dict[str, Any]:
    return {
        "requires_book_evidence": False,
        "supported": False,
        "evidence_passage_ids": [],
        "addresses_intent": True,
        "claim_origin": "assistant_capability",
    }


def _book_negative_decision() -> dict[str, Any]:
    return {
        "requires_book_evidence": True,
        "supported": False,
        "evidence_passage_ids": [],
        "addresses_intent": True,
        "claim_origin": "book_claim",
    }


def _kwargs(answer: Any, verifier: Any, judge: Any, planner: Any) -> dict[str, Any]:
    return {
        "user_message": _SYNTHETIC_USER_MESSAGE,
        "summary": "",
        "recent": [],
        "evidence_pack": [],
        "answer_model": answer,
        "verifier_model": verifier,
        "planner_model": planner,
        "retrieval_index": object(),
        "initial_query_count": 3,
        "planner_reason": "substantive-with-queries",
        "planner_mode": "retrieval",
        "resolved_intent": _SYNTHETIC_RESOLVED_INTENT,
        "whole_turn_judge_model": judge,
    }


@pytest.fixture(autouse=True)
def _enable_turn_time_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AA_TURN_TIME_LIMITS_ENABLED", "true")


async def test_complete_negative_non_book_verdict_serves_fresh_judged_dialogue() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.qualification.product_contract_live import _snapshot_product_failure_reason

    answer = _SequencedAnswer(_SYNTHETIC_BAD_DRAFT, _SYNTHETIC_DIALOGUE)
    verifier = _FixedBatchedVerifier(_glue_negative_decision())
    judge = _DialogueJudge()
    planner = _CountingPlanner()

    outcome = await run_v2_answer_turn(**_kwargs(answer, verifier, judge, planner))

    assert outcome["text"] == _SYNTHETIC_DIALOGUE
    # Initial draft + exactly one fresh dialogue generation; no repair round.
    assert answer.calls == 2
    assert verifier.calls == 1
    assert judge.calls == 1
    assert planner.calls == 0
    telemetry = dict(outcome["telemetry"])
    assert telemetry["answer_outcome"] == "conversational-generated"
    assert telemetry["adequacy_verdict"] == "pass"
    assert telemetry["answers_request"] is True
    assert telemetry["technically_grounded"] is False
    assert telemetry["qualified"] is True
    assert telemetry["repair_rounds"] == 0
    assert telemetry["dialogue_rescue_after_verification"] is True
    # The rejected draft's typed cause never describes the delivered reply.
    assert "answer_failure_cause" not in telemetry
    assert "answer_failure_cause_chain" not in telemetry
    assert _snapshot_product_failure_reason(telemetry) == ""


async def test_complete_negative_non_book_verdict_fails_closed_without_judge_cert() -> None:
    from aa.conversation.failures import TurnFailed
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    answer = _SequencedAnswer(_SYNTHETIC_BAD_DRAFT, _SYNTHETIC_DIALOGUE)
    verifier = _FixedBatchedVerifier(_glue_negative_decision())
    judge = _SubstantiveJudge()
    planner = _CountingPlanner()

    with pytest.raises(TurnFailed):
        await run_v2_answer_turn(**_kwargs(answer, verifier, judge, planner))

    # The judge rejected the fresh dialogue; no book repair was started.
    assert answer.calls == 2
    assert judge.calls >= 1
    assert planner.calls == 0


async def test_book_scoped_negative_verdict_keeps_repair_path() -> None:
    from aa.conversation.failures import TurnFailed
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    answer = _SequencedAnswer(_SYNTHETIC_BAD_DRAFT, _SYNTHETIC_DIALOGUE)
    verifier = _FixedBatchedVerifier(_book_negative_decision())
    judge = _DialogueJudge()
    planner = _CountingPlanner()

    with pytest.raises(TurnFailed):
        await run_v2_answer_turn(**_kwargs(answer, verifier, judge, planner))

    # A book-scoped gap is never converted into a conversational success:
    # only the initial draft ran and the dialogue rescue never started.
    assert answer.calls == 1
    assert judge.calls == 0
    assert planner.calls == 1


async def test_partially_unavailable_verdict_does_not_start_dialogue_rescue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aa.conversation.turn_pipeline as pipeline
    from aa.conversation.failures import TurnFailed
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.conversation.verifier_schema import (
        GroundingResult,
        unavailable_unit_verdict,
    )

    async def _partial(*args: Any, **kwargs: Any) -> GroundingResult:
        units = args[0]
        _ = kwargs
        verdicts = [unavailable_unit_verdict(unit.unit_id) for unit in units]
        return GroundingResult(
            units=verdicts,
            all_required_supported=False,
            unavailable_unit_ids=[unit.unit_id for unit in units],
        )

    monkeypatch.setattr(pipeline, "run_verifier", _partial)
    answer = _SequencedAnswer(_SYNTHETIC_BAD_DRAFT, _SYNTHETIC_DIALOGUE)
    judge = _DialogueJudge()
    planner = _CountingPlanner()

    with pytest.raises(TurnFailed):
        await run_v2_answer_turn(**_kwargs(answer, object(), judge, planner))

    assert answer.calls == 1
    assert judge.calls == 0


async def test_verifier_labels_complete_negative_verdict_as_unsupported_not_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fully adjudicated negative set is a support decision, not an outage."""
    import aa.conversation.turn_pipeline as pipeline
    from aa.conversation.answer_causality import (
        VERDICT_UNSUPPORTED,
        first_answer_failure_cause,
    )
    from aa.conversation.turn_pipeline import _verify_draft
    from aa.conversation.verifier_schema import GroundingResult, UnitVerdict

    async def _negative(*args: Any, **kwargs: Any) -> GroundingResult:
        units = args[0]
        _ = kwargs
        return GroundingResult(
            units=[
                UnitVerdict(
                    unit_id=unit.unit_id,
                    scope="conversation_glue",
                    supported=False,
                    evidence_passage_ids=[],
                    addresses_intent=True,
                    origin="conversation_glue",
                )
                for unit in units
            ],
            all_required_supported=False,
            unavailable_unit_ids=[],
        )

    monkeypatch.setattr(pipeline, "run_verifier", _negative)
    telemetry: dict[str, Any] = {}
    _units, result, passed = await _verify_draft(
        "Синтетическая фраза без опоры на книгу.",
        [],
        verifier_model=object(),
        telemetry=telemetry,
    )

    assert passed is False
    assert result is not None
    cause = first_answer_failure_cause(telemetry)
    assert cause is not None
    assert cause["reason"] == VERDICT_UNSUPPORTED
