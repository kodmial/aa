"""P0 kodmial/aa#217 convergence regression: Gate C+E in one cycle.

Live evidence on exact main a7d76f1 (run 37670332968) shows the same
stable failure set recurring after the prior token-count repair:

- C:live-answer-no-generic-collapse on live-production-path, and
- E:latency-budget-exceeded p50 25.2s / p95 33.6s / max 44.7s with
  planner p50 3.6s / p95 9.2s, retrieval p50 0.4s, repair_turns=0.

The prior repair bounded planner per-message display and answer passage
COUNT (6-window) and saved ~4s p50, but the persistent remainder
(total - planner - retrieval ~= 21s) is the initial answer draft plus
per-unit verifier round-trips: the answer prompt still carries full
per-passage text plus full per-message history. Repeating the same
count-window patch cannot converge.

This repair changes strategy at the responsible boundaries, neither an
exact-question special case, Product Contract #110 unchanged:

- answer-assembly display bounds: per-passage and per-message characters
  are display-bounded with explicit markers (matching the verifier 800
  window); count/order, the live user message and the summary travel
  untruncated; verification still uses the full stored pack;
- planner capability fallback: when the native json_schema channel is
  unavailable on the provider path, the same plan is retried once as
  bounded plain-text JSON and strictly validated, so paraphrased and
  typo-bearing turns still retrieve a usable pack instead of collapsing
  to weak-fallback empty queries and generic clarification. 429 always
  propagates.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from aa.conversation.planner_node import (
    PLANNER_TEXT_JSON_SUFFIX,
    parse_planner_text_json,
    run_planner,
)
from aa.conversation.planner_schema import QueryPlanValidationError
from aa.conversation.prompt_builder import (
    ANSWER_MAX_MESSAGE_CHARS,
    ANSWER_MAX_PASSAGE_CHARS,
    EvidencePassage,
    build_answer_messages,
    render_turn_context,
)
from aa.opencode.errors import (
    OpenCodeProviderAccessError,
    OpenCodeRateLimitError,
)


def _twelve_queries(base: str = "support sobriety") -> list[str]:
    return [f"{base} variant {index}" for index in range(12)]


def test_answer_passage_display_bounded_but_provenance_preserved() -> None:
    """Long passages truncate in display; ids and structure stay intact."""
    assert ANSWER_MAX_PASSAGE_CHARS == 800
    long_text = "x" * 3000
    context = render_turn_context(
        summary="",
        passages=[EvidencePassage(passage_id="p1", source="s", section="c", text=long_text)],
        user_message="what helps with craving?",
    )
    assert long_text not in context
    assert "truncated" in context
    assert "p1" in context
    assert "<book_evidence>" in context
    assert "<user_message>" in context


def test_answer_recent_bounded_but_count_and_live_turn_preserved() -> None:
    """Long history messages truncate; count/order and live turn preserved."""
    assert ANSWER_MAX_MESSAGE_CHARS == 500
    long_text = "word " + "y" * 2000
    live = "why does that matter?"
    recent: list[BaseMessage] = [
        HumanMessage(content="message 0 about craving"),
        HumanMessage(content=long_text),
        HumanMessage(content="message 29 about sleep"),
    ]
    messages = build_answer_messages(
        recent=recent,
        summary="user discusses craving",
        passages=[],
        user_message=live,
    )
    assert messages[0].type == "system"
    body = "\n".join(str(item.content) for item in messages[1:])
    assert live in body
    assert "message 0 about craving" in body
    assert "message 29 about sleep" in body
    assert "truncated" in body
    assert long_text not in body


def test_answer_small_inputs_pass_through_untruncated() -> None:
    """Short passages and history travel fully to generation."""
    short = "exact short passage text"
    messages = build_answer_messages(
        recent=[HumanMessage(content="short hello")],
        summary="",
        passages=[EvidencePassage(passage_id="p", source="s", section="c", text=short)],
        user_message="why?",
    )
    joined = "\n".join(str(item.content) for item in messages)
    assert short in joined
    assert "short hello" in joined
    assert "why?" in joined


def test_planner_text_suffix_demands_strict_json_only() -> None:
    """The text fallback prompt constrains output to one JSON object."""
    assert "Return ONLY a JSON object" in PLANNER_TEXT_JSON_SUFFIX
    assert "queries" in PLANNER_TEXT_JSON_SUFFIX


def test_parse_planner_text_json_accepts_fenced_payload() -> None:
    """Fenced text JSON with 12 queries parses for strict validation."""
    payload = '{"queries": ["' + '", "'.join(_twelve_queries()) + '"]}'
    parsed = parse_planner_text_json(f"```json\n{payload}\n```")
    assert len(parsed["queries"]) == 12


async def test_planner_structured_provider_error_falls_back_to_text_once() -> None:
    """A structured-channel failure serves via one bounded text call."""

    class _StructuredFailsOnce:
        def __init__(self) -> None:
            self.structured_calls = 0
            self.text_calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.structured_calls += 1
            raise OpenCodeProviderAccessError("opencode provider access rejected: http=403")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = system
            self.text_calls += 1
            assert "Return ONLY a JSON object" in prompt
            queries = _twelve_queries()
            return '{"queries": ["' + '", "'.join(queries) + '"]}'

    model = _StructuredFailsOnce()
    plan = await run_planner("evening craving", model=model)
    assert len(plan.queries) == 12
    assert model.structured_calls == 1
    assert model.text_calls == 1


async def test_planner_structured_429_never_falls_back() -> None:
    """Provider 429 propagates for runner retire/restart, never text fallback."""

    class _RateLimited:
        def __init__(self) -> None:
            self.text_calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            raise OpenCodeRateLimitError("opencode request rate-limited: http=429")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            self.text_calls += 1
            raise AssertionError("text fallback must not run on 429")

    model = _RateLimited()
    try:
        await run_planner("evening craving", model=model)
    except OpenCodeRateLimitError:
        pass
    else:
        raise AssertionError("429 must propagate")
    assert model.text_calls == 0


async def test_planner_plain_text_invalid_still_fails_closed_without_retry() -> None:
    """The legacy plain-text path still fails closed with exactly one call."""
    calls: list[list[BaseMessage]] = []

    class _PlainInvalid:
        async def ainvoke(self, messages: Any) -> AIMessage:
            assert isinstance(messages, list)
            calls.append(messages)
            return AIMessage(content="not json")

    try:
        await run_planner("evening craving", model=_PlainInvalid())
    except QueryPlanValidationError:
        pass
    else:
        raise AssertionError("invalid plain output must fail closed")
    assert len(calls) == 1
