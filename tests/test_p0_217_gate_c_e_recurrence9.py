"""P0 kodmial/aa#217 recurrence-9 regression: Gate C+E fail-fast repair.

Live evidence on exact main 922dd078acd06933318cb5e354e7cade9d08e273
(run 37720236046) shows the same stable failure set recurring after the
recurrence-8 answer-boundary repair:

- C:live-answer-no-generic-collapse on live-production-path, and
- E:latency-budget-exceeded p50 13.2s / p95 25.8s / max 31.6s.

Per-stage comparison with recurrence 8 (exact main 46cf046 run
37717319855: planner p50 5.5s / p95 8.2s, retrieval p50 0.4s, answer
p50 7.3s / p95 11.8s / max 25.3s, verifier p50 5.0s / p95 12.0s,
total p50 18.0s / p95 30.1s / max 43.7s) identifies the dominant
persistent cause in the ANSWER stage itself (repair_turns=0,
repair_rounds=0, budget_exceeded=0, unavailable_units_total=0 over 19
units, so no repair loop and no transport outage is the lever):

- the recurrence-8 bounds converged on siblings: planner is down to
  p50 4.8s / p95 6.9s / max 9.7s, verifier down to p50 3.7s / p95
  7.8s / max 8.7s, retrieval stays healthy at p50 0.4s;
- but answer is now the only stage whose p95 worsened (11.8s -> 15.1s,
  alone equal to the entire 15s P95 target) while its max 17.4s
  exceeds the single-call message-text max 12.7s (p50 5.0s / p95
  10.0s), proving the recurrence-8 in-turn minimal retry accumulates
  (10s timeout plus a second serve) instead of capping the tail, and
  its weak prompt can then verify unsupported and clarify into the 7
  generic clarifications.

Repeating timeout/trim tuning cannot converge. This repair changes
strategy at the answer-generation boundary (never an exact-question
special case, Product Contract #110 unchanged):

- the single answer draft attempt stays individually bounded
  (ANSWER_DRAFT_ATTEMPT_BUDGET_S); a true deadline expiry now fails
  fast with no second in-turn model call, capping answer at the budget
  and serving the natural retry reply (distinct from the generic
  clarification, preserving no-collapse/diversity) instead of grinding
  10s plus retry and risking a weak-prompt clarification;
- provider 429 always propagates from the answer path for runner
  retire/restart and never collapses to retry.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aa.conversation.turn_pipeline import (
    ANSWER_DRAFT_ATTEMPT_BUDGET_S,
    run_v2_answer_turn,
)
from aa.opencode.errors import OpenCodeRateLimitError


def _pack_dict(passage_id: str = "chapter-3#exp0000") -> dict[str, Any]:
    text = "Поддержка рядом помогает пережить тягу сегодня."
    import hashlib

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


def test_answer_budget_still_configured() -> None:
    """Recurrence-9 keeps the single-attempt budget (no retune)."""
    assert ANSWER_DRAFT_ATTEMPT_BUDGET_S == 35.0


async def test_answer_timeout_fails_fast_with_single_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung answer attempt serves retry with exactly one model call."""
    import aa.conversation.turn_pipeline as pipeline

    monkeypatch.setattr(pipeline, "ANSWER_DRAFT_ATTEMPT_BUDGET_S", 0.05)

    class _HangOnce:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages: Any) -> AIMessage:
            self.calls += 1
            _ = messages
            await asyncio.sleep(60.0)
            raise AssertionError("attempt must time out")

    from aa.conversation.failures import TurnFailed as _TF9

    model = _HangOnce()
    started = time.perf_counter()
    with __import__("pytest").raises(_TF9) as exc:
        await run_v2_answer_turn(
            user_message="К вечеру тянет выпить, как быть?",
            summary="",
            recent=[HumanMessage(content="hello")],
            evidence_pack=[_pack_dict()],
            answer_model=model,
            verifier_model=model,
            planner_model=None,
            retrieval_index=None,
            initial_query_count=12,
        )
    assert time.perf_counter() - started < 5.0
    assert exc.value.category == "answer-failed"
    assert model.calls == 1
    assert exc.value.telemetry["answer_rounds"] == 1


async def test_answer_timeout_never_serves_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-fast timeout preserves no-collapse (retry distinct from generic)."""
    import aa.conversation.turn_pipeline as pipeline

    monkeypatch.setattr(pipeline, "ANSWER_DRAFT_ATTEMPT_BUDGET_S", 0.05)

    class _AlwaysHang:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            await asyncio.sleep(60.0)
            raise AssertionError("must time out")

    from aa.conversation.failures import TurnFailed as _TF9b

    started = time.perf_counter()
    with __import__("pytest").raises(_TF9b) as exc:
        await run_v2_answer_turn(
            user_message="К вечеру тянет выпить, как быть?",
            summary="",
            recent=[HumanMessage(content="hello")],
            evidence_pack=[_pack_dict()],
            answer_model=_AlwaysHang(),
            verifier_model=_AlwaysHang(),
            planner_model=None,
            retrieval_index=None,
            initial_query_count=12,
        )
    assert time.perf_counter() - started < 5.0
    assert exc.value.category == "answer-failed"


async def test_answer_429_propagates_for_runner_retire() -> None:
    """Provider 429 from answer generation retires the runner, never retries."""

    class _RateLimited:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            raise OpenCodeRateLimitError("opencode request rate-limited: http=429")

    with pytest.raises(OpenCodeRateLimitError):
        await run_v2_answer_turn(
            user_message="К вечеру тянет выпить, как быть?",
            summary="",
            recent=[HumanMessage(content="hello")],
            evidence_pack=[_pack_dict()],
            answer_model=_RateLimited(),
            verifier_model=_RateLimited(),
            planner_model=None,
            retrieval_index=None,
            initial_query_count=12,
        )


def test_no_exact_live_question_special_cases() -> None:
    """The repair stays turn-independent: no frozen live prompt text in sources."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "prompt_builder.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "answer_node.py").read_text(encoding="utf-8"),
    ]
    frozen_fragments = (
        "тянет выпить",
        "тянеет выпить",
        "ссора из-за моей выпивки",
        "Поругались дома",
        "не могу успокоиться и уснуть",
        "мысли крутятся",
        "покупать акции",
        "выгоднее купить",
        "покончить с собой",
        "Не хочу жить",
        "одному не получается",
        "чем помочь можешь",
    )
    for source in sources:
        for fragment in frozen_fragments:
            assert fragment not in source
