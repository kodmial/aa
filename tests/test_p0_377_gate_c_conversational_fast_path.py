"""P0 SYSTEMIC kodmial/aa#377: conversational turns must bypass the substantive pipeline.

Systemic failure class (Gate C ``live-meta-direct-1`` on
``live-production-path``, exact main db14541 run 38297999328 with
repair_turns=6, repair_rounds=6, answer_rounds=32, budget_exceeded=5,
clarifications=13 and two turns with unavailable verifier units):
every planner-certified conversational turn previously paid the full
substantive cost first (one empty-pack answer draft plus one full
per-unit verifier round) and only then fell back to a model-composed
dialogue reply. That sequential draft+verify burns the shared
answer/verifier budget on turns needing no book evidence, exposes
them to verifier transport outages and turn-budget expiry, and
converts a direct natural meta reply into a clarification/retry that
Gate C counts as live-meta-direct failure while clean siblings pass.
Telemetry certification (#362) and unavailable narrowing (#290) could
not converge while the mechanism still routed glue through the
substantive pipeline.

Architecture repair (generic, no exact-question branches, no keyword
tables, no step parsing, no token-overlap relevance, no handcrafted
semantic predicates; Product Contract #110, Gate C semantics,
grounding, relevance, safety and Gate E SLO unchanged):
planner-certified conversational turns (conversational mode, zero
queries, legitimate-glue reason, empty pack) serve exactly one
model-composed dialogue reply directly with no draft, no verifier
round and no repair loop. A separately instantiated whole-turn judge
still overturns a substantive-claim reply so misclassified book
requests never serve as qualified glue.

All strings below are synthetic fixtures for this failure class,
never the literal qualification prompts.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage

_SYNTHETIC_USER_MESSAGE = "Синтетический вопрос о возможностях поддержки сегодня"
_SYNTHETIC_DIALOGUE = "Я рядом, готов спокойно поддержать разговор сегодня."


class _CountingAnswer:
    """Dialogue composer double that records every generation call."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.calls += 1
        _ = messages
        return AIMessage(content=self._text)


class _ExplodingVerifier:
    """Verifier double that fails the test when invoked."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        self.calls += 1
        _ = (prompt, system, schema, retry_count)
        raise AssertionError("conversational fast-path must not invoke the verifier")

    async def _ainvoke_text(self, text: str, *, system: str | None = None) -> str:
        self.calls += 1
        _ = (text, system)
        raise AssertionError("conversational fast-path must not invoke the verifier")


class _GlueJudge:
    """Whole-turn judge double reporting no substantive claim."""

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


class _SubstantiveClaimJudge:
    """Whole-turn judge double reporting a misclassified substantive claim."""

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


def _glue_kwargs(answer: Any, verifier: Any, judge: Any) -> dict[str, Any]:
    return {
        "user_message": _SYNTHETIC_USER_MESSAGE,
        "summary": "",
        "recent": [],
        "evidence_pack": [],
        "answer_model": answer,
        "verifier_model": verifier,
        "planner_model": None,
        "retrieval_index": None,
        "initial_query_count": 0,
        "planner_reason": "legitimate-glue",
        "planner_mode": "conversational",
        "resolved_intent": "",
        "whole_turn_judge_model": judge,
    }


async def test_conversational_fast_path_skips_draft_and_verifier() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.qualification.product_contract_live import _is_direct_meta_reply

    answer = _CountingAnswer(_SYNTHETIC_DIALOGUE)
    verifier = _ExplodingVerifier()
    judge = _GlueJudge()
    outcome = await run_v2_answer_turn(**_glue_kwargs(answer, verifier, judge))
    assert outcome["text"] == _SYNTHETIC_DIALOGUE
    assert answer.calls == 1
    assert verifier.calls == 0
    assert outcome["rounds"] == 0
    telemetry = dict(outcome["telemetry"])
    assert telemetry["answer_outcome"] == "conversational-generated"
    assert telemetry["adequacy_verdict"] == "pass"
    assert telemetry["answers_request"] is True
    assert telemetry["qualified"] is True
    assert telemetry["technically_grounded"] is False
    assert _is_direct_meta_reply(outcome["text"], snapshot=dict(telemetry)) is True


async def test_conversational_fast_path_withholds_substantive_claim() -> None:
    from aa.conversation.failures import TurnFailed
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    answer = _CountingAnswer(_SYNTHETIC_DIALOGUE)
    verifier = _ExplodingVerifier()
    judge = _SubstantiveClaimJudge()
    with pytest.raises(TurnFailed):
        await run_v2_answer_turn(**_glue_kwargs(answer, verifier, judge))
    assert verifier.calls == 0


async def test_substantive_turn_still_uses_draft_and_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    import aa.conversation.turn_pipeline as pipeline
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.conversation.verifier_schema import (
        GroundingResult,
        UnitVerdict,
        validate_grounding_result,
    )

    pack_text = "Синтетический отрывок о спокойном вечернем распорядке."
    pack_id = "chapter-3#exp0000"
    pack = [
        {
            "passage_id": pack_id,
            "text": pack_text,
            "source_id": "ru-fourth-edition-txt",
            "section_id": "chapter-3",
            "char_start": 0,
            "char_end": len(pack_text),
            "text_sha256": hashlib.sha256(pack_text.encode("utf-8")).hexdigest(),
            "source_sha256": "s" * 64,
            "corpus_version": "r" * 64,
        }
    ]
    draft = "Понимаю, вечернее напряжение переживается тяжело. Спокойный распорядок помогает."

    async def _fake_generate_draft(*args: Any, **kwargs: Any) -> str:
        _ = (args, kwargs)
        return draft

    verifier_calls = 0

    async def _fake_run_verifier(units: Any, passages: Any, **kwargs: Any) -> GroundingResult:
        nonlocal verifier_calls
        verifier_calls += 1
        _ = (passages, kwargs)
        verdicts = [
            UnitVerdict(
                unit_id=unit.unit_id,
                scope="book",
                supported=True,
                evidence_passage_ids=[pack_id],
                addresses_intent=True,
                origin="book_claim",
            )
            for unit in units
        ]
        return validate_grounding_result(
            GroundingResult(
                verified=True,
                units=verdicts,
                all_required_supported=True,
                unavailable_unit_ids=[],
            ),
            expected_unit_ids=[unit.unit_id for unit in units],
        )

    monkeypatch.setattr(pipeline, "generate_draft", _fake_generate_draft)
    monkeypatch.setattr(pipeline, "run_verifier", _fake_run_verifier)

    class _HelpfulJudge:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            return {
                "helpful": True,
                "addresses_intent": True,
                "contains_substantive_claim": True,
            }

    outcome = await run_v2_answer_turn(
        user_message="Синтетический вопрос о вечернем напряжении и распорядке",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=object(),
        verifier_model=object(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=3,
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent="синтетическое намерение разобраться с вечерним напряжением",
        whole_turn_judge_model=_HelpfulJudge(),
    )
    assert outcome["text"] == draft
    assert verifier_calls >= 1


def test_no_literal_qualification_branches() -> None:
    source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "aa"
        / "conversation"
        / "turn_pipeline.py"
    ).read_text(encoding="utf-8")
    for fragment in (
        "Чем ты вообще можешь быть полезен",
        "Слушай, а ты тут вообще чем помочь можешь",
        "К вечеру очень тянет выпить",
        "Под вечер опять тянеет выпить",
        "Дома снова ссора",
        "Поругались дома из-за бухла",
        "покупать акции",
        "Какой телефон сейчас выгоднее купить",
        "Изобретение спокойного вечернего распорядка",
    ):
        assert fragment not in source
