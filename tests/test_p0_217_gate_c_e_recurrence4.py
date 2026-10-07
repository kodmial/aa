"""P0 kodmial/aa#217 recurrence-4 regression: Gate C+E planner strategy change.

Live evidence on exact main 7a9c907 (run 37697730282) shows the same
stable failure set recurring after three token-display repairs:

- C:live-answer-no-generic-collapse on live-production-path, and
- E:latency-budget-exceeded p50 20.4s / p95 33.3s / max 34.3s with
  planner p50 7.0s / p95 16.4s / max 28.7s, retrieval p50 0.5s,
  repair_turns=0, repair_rounds=0.

Prior repairs bounded per-message/per-passage display tokens at the
planner/answer/verifier boundaries. The persistent remainder is a
capability round-trip at the OpenCode-request boundary, not token
count: once the omitted structured channel is cached unavailable, every
later planner call still pays a weak-fallback structured round-trip
before the strong-model text fallback (two sequential slow calls per
turn), and each structured call pays up to 3 server-side invocations
(initial + 2 validation retries) before the AA-level fallback.

This repair changes strategy at the responsible boundary (never an
exact-question special case, Product Contract #110 unchanged):

- planner server retry budget 2 -> 1 (verifier precedent run
  37538518277), bounding the planner tail with unchanged strict
  Pydantic validation;
- omitted-wire planner goes directly to the single bounded text path
  once the omitted structured capability is cached unavailable,
  skipping the doomed weak structured call and serving strong-model
  queries (better pack, fewer verifier rejections into generic
  clarification). 429 always propagates.
"""

from __future__ import annotations

from typing import Any

import pytest

from aa.conversation.model_adapter import (
    clear_omitted_structured_cache,
    clear_primary_circuit,
    mark_omitted_structured_unavailable,
)
from aa.conversation.planner_node import run_planner
from aa.conversation.planner_schema import PLANNER_MAX_ATTEMPTS, QueryPlanValidationError
from aa.opencode.errors import OpenCodeRateLimitError


def _twelve_queries(base: str = "support sobriety") -> list[str]:
    return [f"{base} variant {index}" for index in range(12)]


def _twelve_payload() -> str:
    return '{"queries": ["' + '", "'.join(_twelve_queries()) + '"]}'


@pytest.fixture(autouse=True)
def _clear_shared_caches() -> Any:
    clear_primary_circuit()
    clear_omitted_structured_cache()
    yield
    clear_primary_circuit()
    clear_omitted_structured_cache()


def test_planner_retry_budget_is_single_server_retry() -> None:
    """The planner tail is bounded like the verifier precedent (no 2-retry tail)."""
    assert PLANNER_MAX_ATTEMPTS == 1


class _OmittedWireModel:
    """Minimal omitted-wire planner model with countable channels."""

    def __init__(self, *, text_reply: str) -> None:
        self.wire_agent = ""
        self.primary_model = "opencode/muse-spark-1.3-contributor-free"
        self.fallback_model = "opencode/space-bunny-free"
        self.agent = "aa-planner-v2"
        self.structured_calls = 0
        self.text_calls = 0
        self._text_reply = text_reply

    async def ainvoke_structured(
        self,
        prompt: str,
        *,
        system: str,
        schema: dict[str, object],
        retry_count: int = 1,
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.structured_calls += 1
        raise AssertionError("structured must be skipped when capability is cached")

    async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
        _ = system
        self.text_calls += 1
        assert "Return ONLY a JSON object" in prompt
        return self._text_reply


async def test_cached_omitted_planner_skips_weak_structured_call() -> None:
    """Cached capability serves strong text directly with one provider call."""
    model = _OmittedWireModel(text_reply=_twelve_payload())
    mark_omitted_structured_unavailable(model)
    plan = await run_planner("evening craving", model=model)
    assert len(plan.queries) == 12
    assert model.structured_calls == 0
    assert model.text_calls == 1


async def test_cached_omitted_planner_invalid_text_fails_closed() -> None:
    """Invalid cached-path text still fails closed without a structured call."""

    class _InvalidText(_OmittedWireModel):
        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            self.text_calls += 1
            return "not json"

    model = _InvalidText(text_reply="not json")
    mark_omitted_structured_unavailable(model)
    with pytest.raises(QueryPlanValidationError):
        await run_planner("evening craving", model=model)
    assert model.structured_calls == 0
    assert model.text_calls == 1


async def test_cached_omitted_planner_429_propagates_without_structured() -> None:
    """429 on the direct text path propagates for runner retire/restart."""

    class _RateLimited(_OmittedWireModel):
        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            self.text_calls += 1
            raise OpenCodeRateLimitError("opencode request rate-limited: http=429")

    model = _RateLimited(text_reply="")
    mark_omitted_structured_unavailable(model)
    with pytest.raises(OpenCodeRateLimitError):
        await run_planner("evening craving", model=model)
    assert model.structured_calls == 0
    assert model.text_calls == 1


async def test_custom_wire_planner_still_tries_structured_first() -> None:
    """The direct-text fast path never applies to custom-selector models."""

    class _CustomWire:
        def __init__(self) -> None:
            self.wire_agent = "aa-planner-v2"
            self.primary_model = "opencode/muse-spark-1.3-contributor-free"
            self.fallback_model = "opencode/space-bunny-free"
            self.agent = "aa-planner-v2"
            self.structured_calls = 0

        async def ainvoke_structured(
            self,
            prompt: str,
            *,
            system: str,
            schema: dict[str, object],
            retry_count: int = 1,
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.structured_calls += 1
            return {"queries": _twelve_queries()}

    model = _CustomWire()
    mark_omitted_structured_unavailable(model)
    plan = await run_planner("evening craving", model=model)
    assert len(plan.queries) == 12
    assert model.structured_calls == 1


async def test_uncached_omitted_planner_still_tries_structured_first() -> None:
    """Without a cached failure the native structured channel stays primary."""

    class _StructuredServes:
        def __init__(self) -> None:
            self.wire_agent = ""
            self.primary_model = "opencode/muse-spark-1.3-contributor-free"
            self.fallback_model = "opencode/space-bunny-free"
            self.agent = "aa-planner-v2"
            self.structured_calls = 0
            self.text_calls = 0

        async def ainvoke_structured(
            self,
            prompt: str,
            *,
            system: str,
            schema: dict[str, object],
            retry_count: int = 1,
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.structured_calls += 1
            assert retry_count == PLANNER_MAX_ATTEMPTS
            return {"queries": _twelve_queries()}

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            self.text_calls += 1
            raise AssertionError("text must not run when structured serves")

    model = _StructuredServes()
    plan = await run_planner("evening craving", model=model)
    assert len(plan.queries) == 12
    assert model.structured_calls == 1
    assert model.text_calls == 0
