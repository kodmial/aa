"""P0 kodmial/aa#217 regression: repair Gate C+E live failure in one cycle.

Live evidence on exact main 0202b0b (run 37664757721) showed the
combined product failure:

- C:live-answer-no-generic-collapse on live-production-path, and
- E:latency-budget-exceeded (p50 29.6s / p95 42.0s / max 45.5s over the
  30s budget, planner p50 5.7s / p95 12.1s, retrieval p50 0.7s).

The initial draft+verify chain already exceeds budget before any
repair (repair_turns=0, repair_rounds=0): the answer prompt carries
the full 16k-token Evidence Pack while the verifier display window is
already bounded, so every ordinary turn pays the largest provider
input on the answer call and then one verifier round-trip per draft
unit. Long-conversation tails additionally inflate the planner prompt
because the full history travels unbounded per message.

The repair bounds both generation inputs without changing grounding:
answer generation uses only the top-ranked window while verification,
checksum, quote and cite gates still use the full stored pack; planner
recent messages are per-message display-bounded with an explicit
marker while message count, order, the live user message and the
running summary travel untruncated. Turn-independent, never an
exact-question special case, Product Contract #110 unchanged.
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from aa.conversation.planner_node import (
    PLANNER_MAX_MESSAGE_CHARS,
    build_planner_messages,
)
from aa.conversation.response_units import split_response_units
from aa.conversation.turn_pipeline import (
    ANSWER_GENERATION_MAX_PASSAGES,
    run_v2_answer_turn,
)


def _pack_entry(index: int, text: str = "Фиктивная поддержка рядом помогает.") -> dict[str, Any]:
    body = f"{text} Отрывок {index}."
    return {
        "passage_id": f"chapter-3#exp{index:04d}",
        "text": body,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(body),
        "text_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }


class _AnswerModel:
    def __init__(self, drafts: list[str]) -> None:
        self._drafts = list(drafts)
        self.calls = 0
        self.seen_passage_counts: list[int] = []

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.calls += 1
        assert isinstance(messages, list) and messages
        final = str(messages[-1].content)
        self.seen_passage_counts.append(final.count("<passage "))
        if not self._drafts:
            raise AssertionError("answer model called more times than scripted")
        return AIMessage(content=self._drafts.pop(0))


class _VerifierModel:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results: list[dict[str, Any]] = []
        for entry in results:
            for unit in entry.get("units", []):
                self._results.append(
                    {
                        "requires_book_evidence": str(unit.get("scope", "book")) == "book",
                        "supported": bool(unit.get("supported", False)),
                        "evidence_passage_ids": list(unit.get("evidence_passage_ids", [])),
                    }
                )
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        if not self._results:
            raise AssertionError("verifier called more times than scripted")
        return dict(self._results.pop(0))


async def test_answer_generation_uses_bounded_window_but_verifier_sees_full_pack() -> None:
    """Generation input is bounded; grounding still validates full pack."""
    assert ANSWER_GENERATION_MAX_PASSAGES == 5
    pack = [_pack_entry(index) for index in range(10)]
    draft = "Поддержка рядом помогает пережить тягу спокойно."
    units = split_response_units(draft)
    answer = _AnswerModel([draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                    }
                    for unit in units
                ],
                "all_required_supported": True,
            }
        ]
    )
    outcome = await run_v2_answer_turn(
        user_message="что помогает при тяге?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
        planner_model=None,
        retrieval_index=None,
    )
    assert answer.calls == 1
    assert answer.seen_passage_counts == [ANSWER_GENERATION_MAX_PASSAGES]
    assert outcome["text"] == draft
    assert outcome["telemetry"]["answer_generation_window"] == ANSWER_GENERATION_MAX_PASSAGES


async def test_small_pack_passes_through_untruncated() -> None:
    """Packs at or below the window travel fully to generation."""
    pack = [_pack_entry(index) for index in range(2)]
    draft = "Поддержка рядом помогает пережить тягу спокойно."
    units = split_response_units(draft)
    answer = _AnswerModel([draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                    }
                    for unit in units
                ],
                "all_required_supported": True,
            }
        ]
    )
    outcome = await run_v2_answer_turn(
        user_message="что помогает при тяге?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
        planner_model=None,
        retrieval_index=None,
    )
    assert answer.seen_passage_counts == [2]
    assert outcome["text"] == draft


def test_planner_recent_bounded_but_count_and_live_turn_preserved() -> None:
    """Long recent messages truncate with a marker; count/order preserved."""
    assert PLANNER_MAX_MESSAGE_CHARS == 500
    long_text = "слово " + "x" * 2000
    live = "А почему это вообще важно?"
    summary = "пользователь обсуждает тягу"
    recent: list[BaseMessage] = [
        HumanMessage(content="сообщение 0 про тягу"),
        HumanMessage(content=long_text),
        HumanMessage(content="сообщение 29 про сон"),
    ]
    messages = build_planner_messages(user_message=live, summary=summary, recent=recent)
    assert messages[0].type == "system"
    body = str(messages[1].content)
    assert summary in body
    assert live in body
    assert "сообщение 0 про тягу" in body
    assert "сообщение 29 про сон" in body
    assert "truncated" in body
    assert long_text not in body
    assert body.count("user:") == 3
