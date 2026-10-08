"""P0 kodmial/aa#217 recurrence-8 regression: Gate C+E answer-boundary repair.

Live evidence on exact main 46cf046 (run 37717319855) shows the same
stable failure set recurring after the recurrence-7 provider-boundary
repair:

- C:live-answer-no-generic-collapse on live-production-path, and
- E:latency-budget-exceeded p50 18.0s / p95 30.1s / max 43.7s.

Per-stage comparison with recurrence 7 (exact main 58f943c run
37709271567: planner p50 8.4s / p95 9.9s, verifier p50 10.1s / p95
12.0s, retrieval p50 0.6s, 8 answer rounds) identifies the dominant
persistent cause in the ANSWER stage itself (repair_turns=0,
repair_rounds=0, budget_exceeded=0, so no repair loop is the lever):

- the recurrence-7 fixes worked: planner is no longer pinned (p50
  5.5s / p95 8.2s / max 8.8s, tailored plans via the deadline-aware
  capability cache: only 4 structured calls vs 74 text calls) and
  retrieval stays healthy (p50 0.4s / p95 0.6s / max 3.4s);
- but every substantive turn still pays the full slow-path sequence
  and the answer draft is now the slowest single model call (p50
  7.3s / p95 11.8s / max 25.3s, matching the message-text max 25.3s
  at p50 4.5s / p95 11.7s). The verifier (p50 5.0s / p95 12.0s,
  bounded by recurrence 7) plus planner plus one unbounded answer
  tail directly breaches the 30s max and pushes p95 to 30s, while the
  hung answer starves the downstream verifier into unavailable units
  (2 turns / 2 units over 36 units in 13 answer rounds) that narrow
  to nothing and clarify.
- verifier availability proves transport is healthy
  (unavailable_units_total=2 of 36, not a full outage); the collapse
  is budget starvation, not provider outage.

Repeating per-passage/per-message char or window trims cannot
converge (6x800 -> 5x600 already applied twice on both windows).
This repair changes strategy at the answer-generation boundary
(never an exact-question special case, Product Contract #110
unchanged):

- the single answer draft attempt is individually bounded
  (ANSWER_DRAFT_ATTEMPT_BUDGET_S); a true deadline expiry retries
  once with a minimal fast-path prompt (top-ranked passages only,
  last history messages only), still fully verified against the
  full stored pack. A second expiry serves the natural retry reply
  (distinct from the generic clarification, preserving
  no-collapse/diversity) instead of grinding a 25s tail.
- answer history message COUNT is bounded (ANSWER_MAX_HISTORY_MESSAGES);
  older continuity stays via the running summary and the live turn
  travels untruncated. This is a different dimension from the prior
  per-message char trims.
- provider 429 always propagates from the answer/verifier paths for
  runner retire/restart and never collapses to retry/unavailable.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aa.conversation.prompt_builder import (
    ANSWER_MAX_HISTORY_MESSAGES,
    build_answer_messages,
)
from aa.conversation.turn_pipeline import (
    ANSWER_DRAFT_ATTEMPT_BUDGET_S,
    ANSWER_FAST_RETRY_MAX_HISTORY,
    ANSWER_FAST_RETRY_MAX_PASSAGES,
    NATURAL_CLARIFICATION_REPLY,
    NATURAL_RETRY_VARIANTS,
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
    }


def test_answer_bounds_are_configured() -> None:
    """Recurrence-8 bounds: answer attempt budget plus history-count window."""
    assert ANSWER_DRAFT_ATTEMPT_BUDGET_S == 35.0
    assert ANSWER_FAST_RETRY_MAX_PASSAGES == 2
    assert ANSWER_FAST_RETRY_MAX_HISTORY == 2
    assert ANSWER_MAX_HISTORY_MESSAGES == 6
    assert ANSWER_FAST_RETRY_MAX_PASSAGES < 5
    assert ANSWER_FAST_RETRY_MAX_HISTORY < ANSWER_MAX_HISTORY_MESSAGES


def test_answer_history_count_bounded_live_turn_preserved() -> None:
    """Only the last N history messages travel; live turn is intact."""
    from langchain_core.messages import BaseMessage

    recent: list[BaseMessage] = [HumanMessage(content=f"history message {i}") for i in range(10)]
    live = "why does that matter tonight?"
    messages = build_answer_messages(
        recent=recent,
        summary="user discusses craving",
        passages=[],
        user_message=live,
    )
    assert messages[0].type == "system"
    body = "\n".join(str(item.content) for item in messages[1:])
    assert live in body
    # Last window preserved, oldest dropped.
    assert "history message 9" in body
    assert "history message 0" not in body
    # Count bound: system + 6 history + 1 turn payload.
    assert len(messages) == 1 + ANSWER_MAX_HISTORY_MESSAGES + 1


def test_answer_small_history_passes_through() -> None:
    """Short threads (live qualification size) are unaffected."""
    recent = [HumanMessage(content="short hello"), AIMessage(content="short reply")]
    messages = build_answer_messages(
        recent=recent,
        summary="",
        passages=[],
        user_message="why?",
    )
    body = "\n".join(str(item.content) for item in messages)
    assert "short hello" in body
    assert "short reply" in body
    assert "why?" in body


async def test_answer_attempt_timeout_uses_fast_minimal_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung answer attempt fails fast (recurrence 9 supersedes retry).

    Recurrence 9 on exact main 922dd07 (run 37720236046: answer p95
    15.1s, max 17.4s past the 12.7s single-call tail) proved the
    recurrence-8 minimal retry accumulates instead of capping. The
    single attempt now fails fast to the retry reply with one model
    call only.
    """
    import aa.conversation.turn_pipeline as pipeline

    monkeypatch.setattr(pipeline, "ANSWER_DRAFT_ATTEMPT_BUDGET_S", 0.05)
    seen: list[dict[str, Any]] = []

    class _HangOnceThenServe:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages: Any) -> AIMessage:
            self.calls += 1
            seen.append({"messages": len(messages)})
            if self.calls == 1:
                await asyncio.sleep(60.0)
                raise AssertionError("first attempt must time out")
            return AIMessage(content="Поддержка рядом помогает сегодня.")

    class _VerifyPass:
        async def ainvoke_structured(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("verifier must use text path in this test")

    # Verifier that accepts one unit via the shared test seam: use a fake
    # verifier model object with _ainvoke_text serving a supported verdict.
    # The structured channel raises a transient (not deterministic)
    # failure so the unit falls back to text WITHOUT poisoning the
    # process-wide verifier capability cache (a deterministic
    # "structured output missing" mark uses the shared "default" key
    # and would leak into later test files that serve structured-only
    # fakes). Turn-independent probe only.
    class _VerifierServes:
        primary_model = "recurrence8-probe-primary"
        fallback_model = "recurrence8-probe-fallback"
        agent = "recurrence8-probe-agent"

        async def ainvoke_structured(self, *args: Any, **kwargs: Any) -> Any:
            from aa.opencode.errors import OpenCodeTransientError

            raise OpenCodeTransientError("transient probe failure")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            return (
                '{"requires_book_evidence": true, "supported": true, '
                '"evidence_passage_ids": ["p1"], "addresses_intent": true}'
            )

    _ = _VerifyPass
    pack = [_pack_dict()]
    started = time.perf_counter()
    outcome = await run_v2_answer_turn(
        user_message="К вечеру тянет выпить, как быть?",
        summary="",
        recent=[HumanMessage(content="hello")],
        evidence_pack=pack,
        answer_model=_HangOnceThenServe(),
        verifier_model=_VerifierServes(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0
    assert outcome["text"] in NATURAL_RETRY_VARIANTS
    assert outcome["text"] != NATURAL_CLARIFICATION_REPLY
    assert outcome["telemetry"]["answer_rounds"] == 1


async def test_answer_double_timeout_serves_retry_not_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two hung attempts fail to retry (never fake-grounded, never collapse)."""
    import aa.conversation.turn_pipeline as pipeline

    monkeypatch.setattr(pipeline, "ANSWER_DRAFT_ATTEMPT_BUDGET_S", 0.05)

    class _AlwaysHang:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            await asyncio.sleep(60.0)
            raise AssertionError("must time out")

    started = time.perf_counter()
    outcome = await run_v2_answer_turn(
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
    assert outcome["text"] in NATURAL_RETRY_VARIANTS
    assert outcome["text"] != NATURAL_CLARIFICATION_REPLY


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
