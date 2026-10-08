"""P0 kodmial/aa#217 recurrence-6 regression: Gate C+E orchestration repair.

Live evidence on exact main 4a32481 (run 37705584457) shows the same
stable failure set recurring after the recurrence-5 wall-clock repair:

- C:live-answer-no-generic-collapse on live-production-path (8
  clarifications over 14 answer rounds), and
- E:latency-budget-exceeded p50 14.8s / p95 36.3s / max 64.6s.

Per-stage comparison with recurrence 5 (exact main a9c5d3e run
37701489048: planner p50 11.1s / p95 42.8s / max 63.4s, retrieval p50
0.6s, answer p50 3.8s / p95 6.4s, structured p50 6.3s / p95 42.8s vs
text p50 4.5s / p95 15.4s) identifies the dominant persistent cause:

- planner is now pinned at exactly the recurrence-5 10s wall on every
  turn (p50 10006ms / p95 10008ms / max 10024ms): the generic timeout
  fallback fires on ~100% of turns, so every substantive turn serves
  the same 12 identical generic queries, retrieves a generic pack the
  verifier rejects as unsupported (unavailable_units_total=0 proves
  transport is healthy), and collapses to the exact generic
  clarification, while the 10s floor anchors every turn (total p50
  14.8s). Retrieval stays healthy (p50 9ms / p95 486ms) and answer is
  stable (p50 3.5s / p95 8.1s); repair_turns=0 and repair_rounds=0, so
  the repair loop is not the lever.
- the remaining E tail concentrates in the unbounded verifier
  sequence (verifier p50 3ms / p95 25.4s / max 39.9s; text-path p95 15s
  / max 39.9s): tail turns grind 25-40s and then still clarify.

Repeating or retuning the generic-fallback wall cannot converge: the
fallback itself is the collapse mechanism. This repair changes strategy
at the turn-orchestration boundary (never an exact-question special
case, Product Contract #110 unchanged):

- planner: the generic fallback is removed. Only the single native
  structured attempt is individually bounded
  (PLANNER_STRUCTURED_ATTEMPT_BUDGET_S); a slow structured channel
  degrades to the single tailored text fallback (model-generated
  queries for this turn, strictly validated), never to identical
  generic queries. A wall-clock expiry of the tailored sequence fails
  closed as a provider timeout. 429 propagates; invalid content fails
  closed.
- verifier: the single concurrent per-unit round runs under one
  turn-level deadline (VERIFIER_TURN_BUDGET_S); on expiry the turn
  fails closed as verifier-unavailable (fast narrow/clarify
  downstream) instead of grinding 25-40s. Per-unit topology is
  unchanged (exactly one concurrent round, no batch round); 429
  propagates.

This file supersedes test_p0_217_gate_c_e_recurrence5.py, which locked
the disproven wall-clock generic fallback.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

import aa.conversation.planner_node as planner_module
from aa.conversation.model_adapter import (
    clear_omitted_structured_cache,
    clear_primary_circuit,
)
from aa.conversation.planner_node import (
    PLANNER_STRUCTURED_ATTEMPT_BUDGET_S,
    PLANNER_TIME_BUDGET_S,
    run_planner,
)
from aa.conversation.planner_schema import QueryPlanValidationError
from aa.conversation.response_units import split_response_units
from aa.conversation.verifier import VERIFIER_TURN_BUDGET_S, run_verifier
from aa.conversation.verifier_schema import VerifierValidationError
from aa.opencode.errors import (
    OpenCodeProviderAccessError,
    OpenCodeRateLimitError,
    OpenCodeTimeoutError,
)


def _twelve_queries(base: str = "support sobriety") -> list[str]:
    return [f"{base} variant {index}" for index in range(12)]


def _twelve_payload(base: str = "support sobriety") -> str:
    return '{"queries": ["' + '", "'.join(_twelve_queries(base)) + '"]}'


@pytest.fixture(autouse=True)
def _clear_shared_caches() -> Any:
    clear_primary_circuit()
    clear_omitted_structured_cache()
    yield
    clear_primary_circuit()
    clear_omitted_structured_cache()


def test_stage_budgets_are_bounded() -> None:
    """Planner wall, structured attempt and verifier turn each have a hard bound."""
    assert PLANNER_TIME_BUDGET_S == 10.0
    assert PLANNER_STRUCTURED_ATTEMPT_BUDGET_S == 3.0
    assert VERIFIER_TURN_BUDGET_S == 12.0
    assert PLANNER_STRUCTURED_ATTEMPT_BUDGET_S < PLANNER_TIME_BUDGET_S


async def test_slow_structured_serves_tailored_text_queries_not_generic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow structured channel degrades to this turn's text queries."""

    monkeypatch.setattr(planner_module, "PLANNER_STRUCTURED_ATTEMPT_BUDGET_S", 0.05)

    class _SlowStructured:
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
            return {"queries": _twelve_queries("unreachable structured")}

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = system
            self.text_calls += 1
            assert "Return ONLY a JSON object" in prompt
            return _twelve_payload("tailored turn query")

    model = _SlowStructured()
    plan = await run_planner("evening craving", model=model)
    assert [query.split(" variant ")[0] for query in plan.queries] == ["tailored turn query"] * 12
    assert model.structured_calls == 1
    assert model.text_calls == 1


async def test_overall_expiry_fails_closed_without_generic_plan() -> None:
    """A fully stalled provider sequence raises timeout; no generic plan is served."""

    class _Stalled:
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
            return {"queries": _twelve_queries()}

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            await asyncio.sleep(60.0)
            return '{"queries": []}'

    started = time.perf_counter()
    with pytest.raises(OpenCodeTimeoutError):
        await run_planner("evening craving", model=_Stalled(), time_budget_s=0.2)
    assert time.perf_counter() - started < 5.0


async def test_no_identical_generic_plan_for_different_slow_turns() -> None:
    """Different slow turns never receive the same deterministic generic plan."""

    class _Stalled:
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

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            await asyncio.sleep(60.0)
            return '{"queries": []}'

    for turn in ("evening craving", "family quarrel tonight"):
        with pytest.raises(OpenCodeTimeoutError):
            await run_planner(turn, model=_Stalled(), time_budget_s=0.1)


async def test_fast_structured_path_unchanged() -> None:
    """A healthy structured channel still serves with exactly one provider call."""

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
    plan = await run_planner("evening craving", model=model)
    assert len(plan.queries) == 12
    assert model.structured_calls == 1
    assert model.text_calls == 0


async def test_planner_429_propagates_without_text_fallback() -> None:
    """Provider 429 propagates for runner retire/restart, never tailored text."""

    class _RateLimited:
        def __init__(self) -> None:
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
            raise OpenCodeRateLimitError("opencode request rate-limited: http=429")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            self.text_calls += 1
            raise AssertionError("text fallback must not run on 429")

    model = _RateLimited()
    with pytest.raises(OpenCodeRateLimitError):
        await run_planner("evening craving", model=model)
    assert model.text_calls == 0


async def test_planner_provider_error_still_serves_tailored_text() -> None:
    """A fast structured-channel failure serves tailored queries via one text call."""

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
            return _twelve_payload("tailored fallback query")

    model = _StructuredFailsFast()
    plan = await run_planner("evening craving", model=model)
    assert [query.split(" variant ")[0] for query in plan.queries] == [
        "tailored fallback query"
    ] * 12
    assert model.structured_calls == 1
    assert model.text_calls == 1


async def test_planner_invalid_text_content_fails_closed() -> None:
    """Invalid model content is never masked by any fallback plan."""

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
        await run_planner("evening craving", model=_InvalidText())


def test_planner_non_positive_budget_rejected() -> None:
    """A non-positive budget is a programming error, never silent."""

    class _Unused:
        pass

    with pytest.raises(QueryPlanValidationError):
        asyncio.run(run_planner("evening craving", model=_Unused(), time_budget_s=0.0))


def _pack_entry(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Фиктивная поддержка рядом. Тяга проходит, если обратиться за помощью.",
) -> dict[str, Any]:
    import hashlib

    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": 120,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


async def test_slow_verifier_fails_closed_fast() -> None:
    """A grinding verifier round raises timeout quickly instead of burning 25-40s."""

    class _SlowVerifier:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke_structured(
            self,
            prompt: str,
            *,
            system: str,
            schema: dict[str, object],
            retry_count: int = 1,
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.calls += 1
            await asyncio.sleep(60.0)
            return {
                "requires_book_evidence": False,
                "supported": True,
                "evidence_passage_ids": [],
            }

    units = split_response_units("Понимаю. Поддержка рядом помогает.")
    assert len(units) == 2
    started = time.perf_counter()
    with pytest.raises(OpenCodeTimeoutError):
        await run_verifier(units, [_pack_entry()], model=_SlowVerifier(), turn_budget_s=0.2)
    assert time.perf_counter() - started < 5.0


async def test_slow_verifier_maps_to_unavailable_without_slow_grind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The answer pipeline maps a verifier timeout to fast unavailable handling."""
    import aa.conversation.verifier as verifier_module
    from aa.conversation.turn_pipeline import _verify_draft

    monkeypatch.setattr(verifier_module, "VERIFIER_TURN_BUDGET_S", 0.2)

    class _SlowVerifier:
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
            return {
                "requires_book_evidence": False,
                "supported": True,
                "evidence_passage_ids": [],
            }

    started = time.perf_counter()
    units, result, passed = await _verify_draft(
        "Поддержка рядом помогает.",
        [_pack_entry()],
        verifier_model=_SlowVerifier(),
    )
    assert passed is False
    assert result is None
    assert units != []
    assert time.perf_counter() - started < 5.0


async def test_verifier_429_propagates_despite_turn_budget() -> None:
    """Provider 429 propagates for runner retire/restart, never becomes a timeout."""

    class _Always429:
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

    units = split_response_units("Понимаю. Поддержка рядом помогает.")
    with pytest.raises(OpenCodeRateLimitError):
        await run_verifier(units, [_pack_entry()], model=_Always429(), turn_budget_s=5.0)


async def test_fast_verifier_path_unchanged_single_round() -> None:
    """A healthy verifier still completes one concurrent round with no batch."""

    class _ScriptedVerifier:
        def __init__(self, decisions: list[dict[str, Any]]) -> None:
            self._decisions = list(decisions)
            self.calls = 0

        async def ainvoke_structured(
            self,
            prompt: str,
            *,
            system: str,
            schema: dict[str, object],
            retry_count: int = 1,
        ) -> dict[str, object]:
            _ = (prompt, system, retry_count)
            self.calls += 1
            return dict(self._decisions.pop(0))

    pack = [_pack_entry()]
    units = split_response_units("Понимаю. Поддержка рядом помогает.")
    assert len(units) == 2
    model = _ScriptedVerifier(
        [
            {
                "requires_book_evidence": False,
                "supported": True,
                "evidence_passage_ids": [],
            },
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [pack[0]["passage_id"]],
            },
        ]
    )
    result = await run_verifier(units, pack, model=model)
    assert result.all_required_supported is True
    assert [verdict.unit_id for verdict in result.units] == ["u1", "u2"]
    assert model.calls == len(units)


def test_verifier_non_positive_budget_rejected() -> None:
    """A non-positive verifier turn budget is a programming error, never silent."""
    units = split_response_units("Понимаю.")

    class _Unused:
        pass

    with pytest.raises(VerifierValidationError):
        asyncio.run(run_verifier(units, [_pack_entry()], model=_Unused(), turn_budget_s=0.0))


def test_no_exact_live_question_special_cases() -> None:
    """The repair carries no question text and no question-specific branches.

    Run-id/SHA evidence citations in comments follow the established
    repair convention; what must stay out is live question wording and
    user-text-derived branching.
    """
    import pathlib

    for name in ("planner_node.py", "verifier.py"):
        source = (
            pathlib.Path(__file__).resolve().parents[1] / "src" / "aa" / "conversation" / name
        ).read_text(encoding="utf-8")
        for probe in (
            "тянет выпить",
            "покончить",
            "Ссора из-за моей выпивки",
            "ночью не могу успокоиться",
        ):
            assert probe not in source
