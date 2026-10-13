"""P0 SYSTEMIC kodmial/aa#385: judge-reconciled conversational delivery.

Failure class (Gate C ``live-meta-direct-1`` on ``live-production-path``,
exact main 7687436 run 38445287434 with repair_turns=3, repair_rounds=3,
answer_rounds=22, budget_exceeded=4, clarifications=13): every narrow patch
certified one planner-certified glue path (telemetry, narrowing,
fast-path) but delivery still gated on the planner's stochastic
conversational/retrieval bit plus an empty pack. A meta paraphrase whose
planner emits queries (or whose retrieval returns an irrelevant pack)
therefore collapses to clarification while clean siblings pass, moving
the same fingerprint across SHAs and paraphrases.

Architecture repair (generic, no exact-question branches, no keyword/stem
tables, no step parsing, no token-overlap relevance, no handcrafted
semantic predicates; Product Contract #110, Gate C semantics, grounding,
relevance, safety and Gate E SLO unchanged): the tail and
adequacy-repair-failed collapse paths rescue the existing candidate only
when two independent model judgments agree it carries no substantive
claim (verifier reports no book scope and the separately instantiated
whole-turn judge certifies helpful, addresses_intent and no substantive
claim) plus mechanical guards. Any book scope, any judge substantive
flag, or any unavailable unit fails closed. At most one bounded judge
call on the collapse path only, skipped when either budget is spent.

All strings below are synthetic fixtures for this failure class, never
the literal qualification prompts.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any

import pytest

_SYNTHETIC_USER_MESSAGE = "Синтетический вопрос о возможностях поддержки сегодня"
_SYNTHETIC_RESOLVED_INTENT = "синтетическое намерение о возможностях поддержки"
_SYNTHETIC_DIALOGUE = "Я рядом, готов спокойно поддержать разговор сегодня."
_SYNTHETIC_PACK_TEXT = "Синтетический отрывок о спокойном вечернем распорядке."
_SYNTHETIC_PACK_ID = "chapter-3#exp0000"


def _pack_entry() -> dict[str, Any]:
    text = _SYNTHETIC_PACK_TEXT
    return {
        "passage_id": _SYNTHETIC_PACK_ID,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
    }


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


def _install_doubles(monkeypatch: pytest.MonkeyPatch, *, draft: str, book_scope: bool) -> None:
    import aa.conversation.turn_pipeline as pipeline
    from aa.conversation.verifier_schema import (
        GroundingResult,
        UnitVerdict,
        validate_grounding_result,
    )

    async def _fake_generate_draft(*args: Any, **kwargs: Any) -> str:
        _ = (args, kwargs)
        return draft

    async def _fake_run_verifier(units: Any, passages: Any, **kwargs: Any) -> GroundingResult:
        _ = (passages, kwargs)
        verdicts = []
        for unit in units:
            if book_scope:
                verdicts.append(
                    UnitVerdict(
                        unit_id=unit.unit_id,
                        scope="book",
                        supported=True,
                        evidence_passage_ids=[_SYNTHETIC_PACK_ID],
                        addresses_intent=True,
                        origin="book_claim",
                    )
                )
            else:
                verdicts.append(
                    UnitVerdict(
                        unit_id=unit.unit_id,
                        scope="conversation_glue",
                        supported=True,
                        evidence_passage_ids=[],
                        addresses_intent=True,
                        origin="conversation_glue",
                    )
                )
        return validate_grounding_result(
            GroundingResult(
                verified=True,
                units=verdicts,
                all_required_supported=True,
                unavailable_unit_ids=[],
                answer_relevant=bool(book_scope),
            ),
            expected_unit_ids=[unit.unit_id for unit in units],
        )

    monkeypatch.setattr(pipeline, "generate_draft", _fake_generate_draft)
    monkeypatch.setattr(pipeline, "run_verifier", _fake_run_verifier)


def _misclassified_kwargs(judge: Any) -> dict[str, Any]:
    return {
        "user_message": _SYNTHETIC_USER_MESSAGE,
        "summary": "",
        "recent": [],
        "evidence_pack": [_pack_entry()],
        "answer_model": object(),
        "verifier_model": object(),
        "planner_model": None,
        "retrieval_index": None,
        "initial_query_count": 3,
        "planner_reason": "substantive-with-queries",
        "planner_mode": "retrieval",
        "resolved_intent": _SYNTHETIC_RESOLVED_INTENT,
        "whole_turn_judge_model": judge,
    }


async def test_misclassified_dialogue_rescued_as_direct_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.qualification.product_contract_live import _is_direct_meta_reply

    _install_doubles(monkeypatch, draft=_SYNTHETIC_DIALOGUE, book_scope=False)
    judge = _DialogueJudge()
    outcome = await run_v2_answer_turn(**_misclassified_kwargs(judge))
    assert outcome["text"] == _SYNTHETIC_DIALOGUE
    telemetry = dict(outcome["telemetry"])
    assert telemetry["answer_outcome"] == "conversational-generated"
    assert telemetry["adequacy_verdict"] == "pass"
    assert telemetry["answers_request"] is True
    assert telemetry["qualified"] is True
    assert telemetry["technically_grounded"] is False
    assert _is_direct_meta_reply(outcome["text"], snapshot=dict(telemetry)) is True
    assert judge.calls <= 1


async def test_book_scoped_turn_never_rescued_as_conversational(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    _install_doubles(monkeypatch, draft=_SYNTHETIC_DIALOGUE, book_scope=True)
    judge = _SubstantiveJudge()
    outcome = await run_v2_answer_turn(**_misclassified_kwargs(judge))
    telemetry = dict(outcome["telemetry"])
    assert telemetry["answer_outcome"] != "conversational-generated"


async def test_unhelpful_dialogue_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aa.conversation.failures import TurnFailed
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    _install_doubles(monkeypatch, draft=_SYNTHETIC_DIALOGUE, book_scope=False)

    class _UnhelpfulJudge:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            return {
                "helpful": False,
                "addresses_intent": False,
                "contains_substantive_claim": False,
            }

    with pytest.raises(TurnFailed):
        await run_v2_answer_turn(**_misclassified_kwargs(_UnhelpfulJudge()))


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
        "Привет! К вечеру очень тянет выпить",
        "Изобретение спокойного вечернего распорядка",
    ):
        assert fragment not in source
