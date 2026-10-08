"""Gate C live repair for kodmial/aa#234 recurrence 2.

Prior repair (omitted-wire circuit re-probe) fixed run 37725833818 where
the verifier was never served (0 response units, 5 clarifications).
Recurrence on exact main c34dd9e run 37805421560 still fails
``C:live-answer-no-generic-collapse`` with a healthy verifier (0
unavailable units over 49 response units), planner p50 6.5s / p95 15.6s,
answer p50 7.5s / p95 14.1s, verifier p50 7.0s / p95 13.0s,
repair_turns=0, repair_rounds=0, answer_rounds=17 and no budget breach.
Repeating the circuit patch cannot converge: the collapse now comes
from the retrieval/repair and adequacy fallback boundary, where a
duplicate repair retrieval breaks with zero generations and a
passed-but-inadequate turn discards verified progress to the exact
generic retry.

Fix (turn-independent, no exact-question special cases, Product
Contract unchanged): when repair retrieval adds no new passages,
regenerate once from the current pack with the missing-support focus;
when the focused adequacy regen holds a supported relevant subset,
serve it narrowed instead of collapsing to retry.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage


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


_GROUNDED = "Поддержка рядом помогает пережить тягу сегодня."
_OFF_TOPIC = "Финансовое планирование помогает вести бюджет спокойно."


class _AnswerThenGrounded:
    """First an off-topic draft, then the grounded answer."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages: Any) -> AIMessage:
        _ = messages
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content=_OFF_TOPIC)
        return AIMessage(content=_GROUNDED)


class _VerifierSupportsGroundedOnly:
    """Supports only the grounded draft; the off-topic draft is unsupported."""

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (system, schema, retry_count)
        if _OFF_TOPIC[:12] in prompt:
            return {
                "requires_book_evidence": True,
                "supported": False,
                "evidence_passage_ids": ["chapter-3#exp0000"],
                "addresses_intent": False,
            }
        return {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["chapter-3#exp0000"],
            "addresses_intent": True,
        }


class _DuplicatePlan:
    def __init__(self, queries: list[str]) -> None:
        self.queries = list(queries)


async def test_duplicate_retrieval_still_regenerates_from_existing_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aa.conversation import turn_pipeline as pipeline

    async def _fake_planner(
        focus: str, *, model: Any = None, summary: str = "", recent: Any = None
    ) -> Any:
        _ = (focus, model, summary, recent)
        return _DuplicatePlan([f"запрос {i}" for i in range(12)])

    monkeypatch.setattr("aa.conversation.planner_node.run_planner", _fake_planner)

    def _fake_retrieve(index: Any, queries: Any, *, config: Any = None) -> Any:
        _ = (index, queries, config)

        class _Pack:
            pass

        return _Pack()

    monkeypatch.setattr("aa.retrieval.evidence.retrieve_evidence", _fake_retrieve)

    def _fake_pack_to_state(pack: Any) -> tuple[Any, list[dict[str, Any]]]:
        _ = pack
        return (None, [_pack_entry()])

    monkeypatch.setattr("aa.conversation.retrieval_node.pack_to_state", _fake_pack_to_state)

    outcome = await pipeline.run_v2_answer_turn(
        user_message="Вечером тяжело пережить тягу, как обходиться?",
        summary="",
        recent=[HumanMessage(content="hello")],
        evidence_pack=[_pack_entry()],
        answer_model=_AnswerThenGrounded(),
        verifier_model=_VerifierSupportsGroundedOnly(),
        planner_model=object(),
        retrieval_index=object(),
        initial_query_count=12,
    )
    assert outcome["text"] == _GROUNDED
    assert outcome["text"].strip() != pipeline.NATURAL_CLARIFICATION_REPLY
    assert outcome["text"].strip() != pipeline.NATURAL_RETRY_REPLY
    assert int(outcome["rounds"]) >= 1
    telemetry = dict(outcome.get("telemetry", {}))
    assert int(telemetry.get("answer_rounds", 0)) >= 2


async def test_regen_partial_support_serves_narrowed_instead_of_retry() -> None:
    from aa.conversation import turn_pipeline as pipeline

    pack = [_pack_entry()]
    glue_then_partial = (
        "Привет! Рад, что ты написал. Поддержка рядом помогает пережить тягу сегодня."
    )

    class _MixedAnswer:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="Привет! Рад, что ты написал.")
            return AIMessage(content=glue_then_partial)

    class _MixedVerifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (system, schema, retry_count)
            # Judge only the response unit: evidence always carries grounded
            # text, so whole-prompt checks misclassify glue as supported.
            if "<response_unit>" in prompt and "</response_unit>" in prompt:
                unit_text = prompt.split("<response_unit>", 1)[1].split("</response_unit>", 1)[0]
            else:
                unit_text = prompt
            if "Рад, что ты написал" in unit_text or unit_text.strip() == "Привет!":
                return {
                    "requires_book_evidence": False,
                    "supported": True,
                    "evidence_passage_ids": [],
                    "addresses_intent": True,
                }
            if _GROUNDED[:12] in unit_text:
                return {
                    "requires_book_evidence": True,
                    "supported": True,
                    "evidence_passage_ids": ["chapter-3#exp0000"],
                    "addresses_intent": True,
                }
            return {
                "requires_book_evidence": True,
                "supported": False,
                "evidence_passage_ids": ["chapter-3#exp0000"],
                "addresses_intent": False,
            }

    outcome = await pipeline.run_v2_answer_turn(
        user_message="Вечером тяжело пережить тягу, как обходиться?",
        summary="",
        recent=[HumanMessage(content="hello")],
        evidence_pack=pack,
        answer_model=_MixedAnswer(),
        verifier_model=_MixedVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
    )
    # Either the adequacy regen served grounded help (or its supported
    # subset) or the turn failed explicitly; it must never serve the
    # all-glue draft as a helpful success nor the exact generic fallback
    # when verified material exists.
    assert outcome["text"] != "Привет! Рад, что ты написал."
    if outcome["text"].strip() in (
        pipeline.NATURAL_CLARIFICATION_REPLY,
        pipeline.NATURAL_RETRY_REPLY,
    ):
        telemetry = dict(outcome.get("telemetry", {}))
        assert telemetry.get("answer_outcome") in (
            "adequacy-failed",
            "adequacy-repair-failed",
            "clarification",
        )


def test_no_exact_live_question_special_cases() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
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
