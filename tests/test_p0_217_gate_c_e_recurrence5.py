"""P0 kodmial/aa#217 recurrence-5 regression: Gate C+E planner time budget.

Live evidence on exact main a9c5d3e (run 37701489048) shows the same
stable failure set recurring after the recurrence-4 capability repair:

- C:live-answer-no-generic-collapse on live-production-path (11
  clarifications), and
- E:latency-budget-exceeded p50 23.6s / p95 45.8s / max 67.6s with
  planner p50 11.1s / p95 42.8s / max 63.4s, retrieval p50 0.6s healthy,
  answer p50 3.8s / p95 6.4s, repair_turns=0, repair_rounds=0,
  structured p50 6.3s / p95 42.8s vs text p50 4.5s / p95 15.4s.

Comparison with recurrence 4 (exact main 7a9c907 run 37697730282:
planner p50 7.0s / p95 16.4s / max 28.7s, total p50 20.4s / p95 33.3s)
shows the tail worsened by +57% p50 / +161% p95 / +121% max despite the
retry-budget (2->1) and cached-direct-text repair. Retrieval stayed
healthy, so the dominant persistent cause is server-side provider tail
on the OpenCode-request boundary (structured p95 42.8s), not display
tokens or retry count. Repeating another token/retry patch cannot
converge.

This repair changes strategy at the responsible turn-orchestration
boundary (never an exact-question special case, Product Contract #110
unchanged): the whole planner provider sequence runs under one hard
deadline (PLANNER_TIME_BUDGET_S); on expiry a deterministic
broad-coverage plan (same 12 generic RU queries for any turn) is served
instead of waiting out the 40-60s tail and then serving weak-model
queries that retrieve a poor pack the verifier rejects into generic
clarification. Only timeout triggers the fallback: 429 propagates for
runner retire/restart and content validation failures still fail
closed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aa.conversation.model_adapter import (
    clear_omitted_structured_cache,
    clear_primary_circuit,
)
from aa.conversation.planner_node import (
    PLANNER_TIME_BUDGET_S,
    PLANNER_TIMEOUT_FALLBACK_QUERIES,
    deterministic_timeout_fallback_plan,
    run_planner,
)
from aa.conversation.planner_schema import QueryPlanValidationError, validate_query_plan
from aa.opencode.errors import (
    OpenCodeProviderAccessError,
    OpenCodeRateLimitError,
)


def _twelve_queries(base: str = "support sobriety") -> list[str]:
    return [f"{base} variant {index}" for index in range(12)]


@pytest.fixture(autouse=True)
def _clear_shared_caches() -> Any:
    clear_primary_circuit()
    clear_omitted_structured_cache()
    yield
    clear_primary_circuit()
    clear_omitted_structured_cache()


def test_time_budget_is_bounded() -> None:
    """The planner tail is hard-bounded well below the 30s turn budget."""
    assert PLANNER_TIME_BUDGET_S == 10.0


def test_timeout_fallback_plan_is_valid_broad_coverage() -> None:
    """The deterministic fallback satisfies the 10-16 structural contract."""
    assert len(PLANNER_TIMEOUT_FALLBACK_QUERIES) == 12
    plan = deterministic_timeout_fallback_plan()
    assert len(plan.queries) == 12
    validated = validate_query_plan(plan)
    assert len(validated.queries) == 12


async def test_slow_provider_serves_broad_fallback_within_budget() -> None:
    """A 40-60s provider tail degrades to a usable pack, not empty collapse."""

    class _Slow:
        def __init__(self) -> None:
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
            await asyncio.sleep(60.0)
            return {"queries": _twelve_queries()}

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            self.text_calls += 1
            await asyncio.sleep(60.0)
            return '{"queries": []}'

    model = _Slow()
    plan = await run_planner("evening craving", model=model, time_budget_s=0.05)
    assert list(plan.queries) == list(PLANNER_TIMEOUT_FALLBACK_QUERIES)
    assert len(plan.queries) == 12


async def test_timeout_fallback_is_turn_independent() -> None:
    """Different user turns receive the same broad fallback on timeout."""

    class _Slow:
        async def ainvoke_structured(
            self,
            prompt: str,
            *,
            system: str,
            schema: dict[str, object],
            retry_count: int = 1,
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            await asyncio.sleep(60.0)
            return {"queries": []}

    first = await run_planner("evening craving", model=_Slow(), time_budget_s=0.05)
    second = await run_planner("family quarrel tonight", model=_Slow(), time_budget_s=0.05)
    assert list(first.queries) == list(second.queries) == list(PLANNER_TIMEOUT_FALLBACK_QUERIES)


async def test_fast_structured_path_unchanged_under_budget() -> None:
    """Fast provider calls still serve model queries without fallback."""

    class _FastStructured:
        def __init__(self) -> None:
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
            return {"queries": _twelve_queries()}

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            self.text_calls += 1
            raise AssertionError("text must not run when structured serves fast")

    model = _FastStructured()
    plan = await run_planner("evening craving", model=model, time_budget_s=10.0)
    assert len(plan.queries) == 12
    assert model.structured_calls == 1
    assert model.text_calls == 0


async def test_429_propagates_without_timeout_fallback() -> None:
    """Provider 429 propagates for runner retire/restart, never fallback."""

    class _RateLimited:
        async def ainvoke_structured(
            self,
            prompt: str,
            *,
            system: str,
            schema: dict[str, object],
            retry_count: int = 1,
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            raise OpenCodeRateLimitError("opencode request rate-limited: http=429")

    with pytest.raises(OpenCodeRateLimitError):
        await run_planner("evening craving", model=_RateLimited(), time_budget_s=5.0)


async def test_provider_error_still_falls_back_to_text_within_budget() -> None:
    """A fast structured-channel failure still serves via one text call."""

    class _StructuredFailsFast:
        def __init__(self) -> None:
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
            raise OpenCodeProviderAccessError("opencode provider access rejected: http=403")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = system
            self.text_calls += 1
            assert "Return ONLY a JSON object" in prompt
            queries = _twelve_queries()
            return '{"queries": ["' + '", "'.join(queries) + '"]}'

    model = _StructuredFailsFast()
    plan = await run_planner("evening craving", model=model, time_budget_s=5.0)
    assert len(plan.queries) == 12
    assert model.structured_calls == 1
    assert model.text_calls == 1


async def test_invalid_text_content_fails_closed_without_fallback() -> None:
    """Invalid model content is never masked by the deterministic fallback."""

    class _InvalidText:
        async def ainvoke_structured(
            self,
            prompt: str,
            *,
            system: str,
            schema: dict[str, object],
            retry_count: int = 1,
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            raise OpenCodeProviderAccessError("opencode provider access rejected: http=403")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            return "not json"

    with pytest.raises(QueryPlanValidationError):
        await run_planner("evening craving", model=_InvalidText(), time_budget_s=5.0)


def test_non_positive_budget_rejected() -> None:
    """A non-positive budget is a programming error, never silent."""
    import asyncio as _asyncio

    class _Unused:
        pass

    with pytest.raises(QueryPlanValidationError):
        _asyncio.run(run_planner("evening craving", model=_Unused(), time_budget_s=0.0))
