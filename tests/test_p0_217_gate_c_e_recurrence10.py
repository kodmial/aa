"""P0 kodmial/aa#217 recurrence-10 regression: Gate C+E end-to-end repair.

Live evidence on exact main 94fd5b570e80ed5ae9ea8e517c3674452b888c15
(run 37722464604) shows the same stable failure set recurring after the
recurrence-9 answer-boundary repair:

- C:live-answer-no-generic-collapse on live-production-path (3 generic
  clarifications), and
- E:latency-budget-exceeded p50 10.0s / p95 21.2s / max 29.7s.

Per-stage comparison with recurrence 9 (exact main 922dd07 run
37720236046: planner p50 4.8s / p95 6.9s / max 9.7s, retrieval p50 0.4s,
answer p50 6.8s / p95 15.1s / max 17.4s, verifier p50 3.7s / p95 7.8s /
max 8.7s, repair_turns=0, repair_rounds=0, answer_rounds=16,
budget_exceeded=0, unavailable_units_total=0 over 19 units,
clarifications=7, text-path p50 5.0s / p95 10.0s / max 12.7s) identifies
the dominant persistent cause in the SEQUENTIAL SUM at the
turn-orchestration boundary (repair_turns=0, repair_rounds=0,
budget_exceeded=0, answer_rounds=9):

- the recurrence-9 answer fail-fast converged: answer is down to p50
  5.3s / p95 10.0s / max 10.0s (capped at the attempt budget);
- but every sibling stage stayed individually within its own bound
  (planner p50 5.4s / p95 8.8s <= 10s wall, retrieval p50 0.7s / p95
  3.5s, verifier p50 5ms / p95 12.0s pinned exactly at the 12s turn
  budget), so their sequential sum (8.8 + 3.5 + 10.0 + 12.0) still
  exceeds the 15s p95 target and the 30s max;
- the verifier budget expiry is simultaneously the C mechanism:
  unavailable_units_total regressed 0 -> 2 over 2 turns while those slow
  rounds grind the full 12s verifier budget and then clarify;
- one provider tail (message-text p50 5.3s / p95 12.0s / max 28.4s,
  past every stage budget) breaches whichever stage holds it.

Retuning any single stage timeout, or trimming input tokens a sixth
time, cannot converge because each stage already respects its bound.
This repair changes strategy at the turn-orchestration boundary (never
an exact-question special case, Product Contract #110 unchanged):

- the answer phase knows the already-spent upstream planner+retrieval
  cost (``upstream_latency_ms``) and owns one end-to-end budget
  (``TURN_END_TO_END_BUDGET_S``) for the whole graph turn; every further
  model call (answer attempt, verifier round, repair re-plan, envelope
  regen) is bounded by the REMAINING end-to-end budget instead of its
  full stage budget;
- a turn that has already exceeded the budget fails fast to
  deterministically narrowed supported material or the natural retry
  reply (distinct from the generic clarification, preserving
  no-collapse and diversity) instead of grinding another 12s verifier
  round that clarifies anyway; fast healthy turns behave exactly as
  before;
- provider 429 always propagates from the answer/verifier paths for
  runner retire/restart and never collapses to retry.
"""

from __future__ import annotations

import hashlib
import time
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aa.conversation.turn_pipeline import (
    ANSWER_DRAFT_ATTEMPT_BUDGET_S,
    TURN_ANSWER_MIN_SLICE_S,
    TURN_END_TO_END_BUDGET_S,
    TURN_VERIFIER_MIN_SLICE_S,
    run_v2_answer_turn,
)
from aa.conversation.verifier_schema import GroundingResult, UnitVerdict
from aa.opencode.errors import OpenCodeRateLimitError


def _pack_dict(passage_id: str = "chapter-3#exp0000") -> dict[str, Any]:
    text = "Поддержка рядом помогает пережить тягу сегодня."
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def test_end_to_end_budget_configured() -> None:
    """Recurrence-10 budget, E-aligned by kodmial/aa#240.

    The 14s guard converted ordinary answerable turns (sequential stage
    sum p50 ~10s / p95 ~21s) into bookless hash-selected filler. The
    budget now tracks the Gate E hard SLO (max < 120s, delivery margin
    kept) so ordinary slow turns complete as grounded answers; only a
    turn past the hard SLO still fails fast with explicit failure
    telemetry.
    """
    assert TURN_END_TO_END_BUDGET_S == 105.0
    assert TURN_END_TO_END_BUDGET_S < 120.0
    assert TURN_ANSWER_MIN_SLICE_S == 1.0
    assert TURN_VERIFIER_MIN_SLICE_S == 3.0
    # Stage budgets are unchanged (strategy change, not a retune).
    assert ANSWER_DRAFT_ATTEMPT_BUDGET_S == 35.0


async def test_slow_upstream_skips_answer_and_serves_retry() -> None:
    """A turn whose upstream already spent the budget makes no model call."""

    class _MustNotRun:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            raise AssertionError("no model call may start on a spent turn")

    from aa.conversation.failures import TurnFailed as _TF10a

    started = time.perf_counter()
    with __import__("pytest").raises(_TF10a) as exc:
        await run_v2_answer_turn(
            user_message="К вечеру тянет выпить, как быть?",
            summary="",
            recent=[HumanMessage(content="hello")],
            evidence_pack=[_pack_dict()],
            answer_model=_MustNotRun(),
            verifier_model=_MustNotRun(),
            planner_model=None,
            retrieval_index=None,
            initial_query_count=12,
            upstream_latency_ms=110000.0,
        )
    assert time.perf_counter() - started < 5.0
    assert exc.value.category == "answer-failed"
    assert exc.value.telemetry["turn_budget_exceeded"] is True
    assert exc.value.telemetry["answer_rounds"] == 0
    assert exc.value.telemetry["verifier_outcome"] == "skipped"


async def test_slow_upstream_plus_answer_skips_verifier_round() -> None:
    """Upstream + answer past budget skips the verifier grind, not clarify."""

    class _InstantAnswer:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            self.calls += 1
            return AIMessage(content="Поддержка рядом помогает пережить тягу сегодня.")

    class _MustNotVerify:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            raise AssertionError("doomed verifier round must not start")

    from aa.conversation.failures import TurnFailed as _TF10b

    answer = _InstantAnswer()
    with __import__("pytest").raises(_TF10b) as exc:
        await run_v2_answer_turn(
            user_message="К вечеру тянет выпить, как быть?",
            summary="",
            recent=[HumanMessage(content="hello")],
            evidence_pack=[_pack_dict()],
            answer_model=answer,
            verifier_model=_MustNotVerify(),
            planner_model=None,
            retrieval_index=None,
            initial_query_count=12,
            upstream_latency_ms=103000.0,
        )
    assert answer.calls == 1
    assert exc.value.category == "skipped-turn-budget"
    assert exc.value.telemetry["turn_budget_exceeded"] is True
    assert exc.value.telemetry["verifier_outcome"] == "skipped-turn-budget"


async def test_verifier_receives_reduced_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy-but-warm turn bounds (not skips) its verifier round."""
    import aa.conversation.turn_pipeline as pipeline

    captured: dict[str, Any] = {}

    async def _recorder(
        units: Any,
        passages: Any,
        *,
        model: Any,
        turn_budget_s: float | None = None,
        resolved_intent: str = "",
        user_message: str = "",
        conversation_context: str = "",
    ) -> GroundingResult:
        _ = (resolved_intent, user_message, conversation_context)
        captured["turn_budget_s"] = turn_budget_s
        captured["units"] = len(list(units))
        pack_ids = {
            str(item.get("passage_id", ""))
            for item in passages
            if isinstance(item, dict) and str(item.get("passage_id", ""))
        }
        cited = next(iter(sorted(pack_ids)))
        return GroundingResult(
            units=[
                UnitVerdict(
                    unit_id=unit.unit_id,
                    scope="book",
                    supported=True,
                    evidence_passage_ids=[cited],
                    addresses_intent=True,
                )
                for unit in units
            ],
            all_required_supported=True,
            unavailable_unit_ids=[],
            answer_relevant=True,
            relevance_category="",
        )

    monkeypatch.setattr(pipeline, "run_verifier", _recorder)

    class _InstantAnswer:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            return AIMessage(content="Поддержка рядом помогает пережить тягу сегодня.")

    outcome = await run_v2_answer_turn(
        user_message="К вечеру тянет выпить, как быть?",
        summary="",
        recent=[HumanMessage(content="hello")],
        evidence_pack=[_pack_dict()],
        answer_model=_InstantAnswer(),
        verifier_model=object(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        # Temporary quality-first SLO: 105s end-to-end budget with a 40s
        # verifier default. Upstream 97s leaves ~8s remaining: below the
        # 40s verifier default (reduced slice propagates) but above the
        # 3s skip floor (round still runs). Was 19s under the old 27s
        # budget with a 12s verifier default.
        upstream_latency_ms=97000.0,
    )
    # Remaining is ~8s: below the 40s verifier default (reduced slice
    # propagates) but above the 3s skip floor (round still runs).
    assert captured["units"] == 1
    assert captured["turn_budget_s"] is not None
    assert 3.0 < float(captured["turn_budget_s"]) < 40.0
    from aa.conversation.failures import is_service_error as _ise10

    assert not _ise10(outcome["text"])
    assert outcome["telemetry"]["turn_budget_exceeded"] is False


async def test_slow_turn_unsupported_serves_retry_not_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-budget unsupported turn retries; it must not clarify."""
    import aa.conversation.turn_pipeline as pipeline

    clock = {"now": 1000.0}
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(perf_counter=lambda: clock["now"]))

    class _InstantAnswer:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            return AIMessage(content="Поддержка рядом помогает спокойно разбирать тягу.")

    class _SlowUnsupportedVerifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            # Temporary quality-first SLO: the turn must spend past the
            # 105s end-to-end budget to serve retry (was 30s under the
            # old 27s budget).
            clock["now"] += 110.0
            return {
                "requires_book_evidence": True,
                "supported": False,
                "evidence_passage_ids": [],
                "addresses_intent": False,
            }

    from aa.conversation.failures import TurnFailed as _TF10c

    with __import__("pytest").raises(_TF10c) as exc:
        await run_v2_answer_turn(
            user_message="К вечеру тянет выпить, как быть?",
            summary="",
            recent=[HumanMessage(content="hello")],
            evidence_pack=[_pack_dict()],
            answer_model=_InstantAnswer(),
            verifier_model=_SlowUnsupportedVerifier(),
            planner_model=None,
            retrieval_index=None,
            initial_query_count=12,
        )
    assert exc.value.telemetry["verifier_outcome"] == "unsupported"
    assert exc.value.category == "turn-budget-exceeded"
    assert exc.value.telemetry["turn_budget_exceeded"] is True


async def test_fast_turn_unsupported_still_clarifies() -> None:
    """Fast-path grounding gaps keep the historical clarification reply."""

    class _InstantAnswer:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            return AIMessage(content="Поддержка рядом помогает спокойно разбирать тягу.")

    class _FastUnsupportedVerifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            return {
                "requires_book_evidence": True,
                "supported": False,
                "evidence_passage_ids": [],
                "addresses_intent": False,
            }

    from aa.conversation.failures import TurnFailed as _TF10d

    with __import__("pytest").raises(_TF10d) as exc:
        await run_v2_answer_turn(
            user_message="К вечеру тянет выпить, как быть?",
            summary="",
            recent=[HumanMessage(content="hello")],
            evidence_pack=[_pack_dict()],
            answer_model=_InstantAnswer(),
            verifier_model=_FastUnsupportedVerifier(),
            planner_model=None,
            retrieval_index=None,
            initial_query_count=12,
        )
    assert exc.value.telemetry["verifier_outcome"] == "unsupported"
    assert exc.value.category == "clarification-unavailable"
    assert exc.value.telemetry["turn_budget_exceeded"] is False


async def test_answer_429_still_propagates_with_upstream() -> None:
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
            upstream_latency_ms=5000.0,
        )
