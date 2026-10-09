"""P0 kodmial/aa#217 recurrence-7 regression: Gate C+E base-chain repair.

Live evidence on exact main 58f943c (run 37709271567) shows the same
stable failure set recurring after the recurrence-6 orchestration repair:

- C:live-answer-no-generic-collapse on live-production-path (5
  clarifications over 8 answer rounds), and
- E:latency-budget-exceeded p50 10.0s / p95 23.2s / max 30.1s (max
  breaches the 30s hard guard; p95 target is 15s).

Per-stage comparison with recurrence 6 (exact main 4a32481 run
37705584457: planner pinned at the 10s wall serving generic queries,
verifier unbounded grind) identifies the dominant persistent cause in
the BASE chain itself (repair_turns=0, repair_rounds=0,
budget_exceeded=0, so no repair loop is the lever):

- the recurrence-6 fixes worked: planner is no longer pinned (p50
  8.4s / p95 9.9s, tailored plans) and the verifier serves with zero
  unavailable units (p50 10.1s / p95 12.0s, inside the 12s turn budget).
  Retrieval stays healthy (p50 0.6s).
- but every substantive turn still pays the full slow-path sequence:
  exactly one audited text request per planner turn (8/8) and per
  answer turn (8/8) while native structured calls serve fast when they
  serve at all (p50 0.5s) and the text path is slow (p50 6.0s / p95
  12.5s / max 26.0s). The verifier needs ~1.8 requests per response
  unit (structured miss plus text, plus validation-retry round-trips).
- planner stage cost (8.4s) is one caller-side structured-attempt
  budget plus one text call: the caller-side timeout never reaches the
  callee-observed capability cache, so EVERY turn re-burns the doomed
  structured attempt before the text fallback.
- verifier turn-budget expiry raises away the whole round, discarding
  completed sibling verdicts: expiry turns map to result None, skip
  repair, narrow to nothing, and clarify. The recurrence-6 budget is
  now the collapse mechanism on slow turns.

Repeating or retuning walls/budgets/loops cannot converge: the floor
(no-repair turns) already exceeds the SLO and the budget itself
collapses slow turns. This repair changes strategy at the
provider-interaction boundary (never an exact-question special case,
Product Contract #110 unchanged):

- deadline-aware capability inference: a caller-observed structured
  deadline expiry marks the capability cache (TTL-bounded, recovery
  re-probes), so later turns go text-direct instead of re-burning the
  attempt budget per turn. 429 never marks; content validation still
  fails closed; the custom-wire path is untouched.
- verifier: each unit's structured attempt is individually bounded and
  degrades to text on true deadline expiry (marking the capability for
  later rounds, which re-snapshot; the per-round snapshot keeps
  concurrent units deterministic).
- verifier text parsing keeps strict required-key/type validation but
  drops unknown envelope keys without trusting them (unit id stays
  bound by AA code, aggregate stays AA-computed), so a
  fully-determined verdict does not burn a second sequential text
  round-trip per unit for envelope prose.
- verifier turn-budget expiry preserves completed units as partial
  results (pending units become fail-closed unavailable-unit verdicts);
  only a round with no verified unit at all still fails fully closed.
  Provider 429 propagates promptly (first-exception) for runner
  retire/restart.
- answer/verifier generation display tightens 6x800 -> 5x600 (top
  RRF-ranked passages only); verification, checksum, quote and cite
  gates still use the full stored pack, so grounding strictness is
  unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

import pytest

import aa.conversation.planner_node as planner_module
import aa.conversation.verifier as verifier_module
from aa.conversation.model_adapter import (
    clear_omitted_structured_cache,
    clear_primary_circuit,
    omitted_structured_unavailable,
)
from aa.conversation.planner_node import (
    PLANNER_STRUCTURED_ATTEMPT_BUDGET_S,
    PLANNER_TIME_BUDGET_S,
    run_planner,
)
from aa.conversation.prompt_builder import (
    ANSWER_MAX_PASSAGE_CHARS,
)
from aa.conversation.response_units import split_response_units
from aa.conversation.turn_pipeline import ANSWER_GENERATION_MAX_PASSAGES
from aa.conversation.verifier import (
    VERIFIER_MAX_EVIDENCE_PASSAGES,
    VERIFIER_MAX_PASSAGE_CHARS,
    VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S,
    VERIFIER_TURN_BUDGET_S,
    parse_text_json_decision,
    run_verifier,
    structured_text_fallback_preferred,
)
from aa.conversation.verifier_schema import VerifierValidationError
from aa.opencode.errors import (
    OpenCodeRateLimitError,
    OpenCodeTimeoutError,
)


def _twelve_queries(base: str = "support sobriety") -> list[str]:
    return [f"{base} variant {index}" for index in range(12)]


def _twelve_payload(base: str = "support sobriety") -> str:
    return (
        '{"mode": "retrieval", "resolved_intent": "standalone intent for test turn", '
        '"queries": ["' + '", "'.join(_twelve_queries(base)) + '"]}'
    )


def _pack_entry(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Фиктивная поддержка рядом помогает.",
) -> dict[str, Any]:
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


@pytest.fixture(autouse=True)
def _clear_shared_caches() -> Any:
    clear_primary_circuit()
    clear_omitted_structured_cache()
    verifier_module.clear_verifier_capability_cache()
    yield
    clear_primary_circuit()
    clear_omitted_structured_cache()
    verifier_module.clear_verifier_capability_cache()


def test_stage_budgets_and_windows_are_bounded() -> None:
    """Recurrence-7 bounds: per-attempt deadlines; evidence is full-pack (#295)."""
    assert PLANNER_TIME_BUDGET_S == 25.0
    assert PLANNER_STRUCTURED_ATTEMPT_BUDGET_S == 6.0
    assert VERIFIER_TURN_BUDGET_S == 40.0
    assert VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S == 6.0
    assert VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S < VERIFIER_TURN_BUDGET_S
    assert ANSWER_GENERATION_MAX_PASSAGES == 0
    assert ANSWER_MAX_PASSAGE_CHARS == 0
    assert VERIFIER_MAX_EVIDENCE_PASSAGES == 0
    assert VERIFIER_MAX_PASSAGE_CHARS == 0


async def test_planner_structured_timeout_marks_capability_for_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-observed deadline falls back without pinning the fast path.

    Recurrence-2 update for kodmial/aa#244 on exact main 6f8d4e1 run
    37764195857 (structured p50 0.40s over 28 fast calls vs text p50
    5.1s over 73 slow calls; planner p50 5.7s pinned by 60s text
    preference): a deadline is latency, not capability evidence, so the
    slow turn serves tailored text once while the next turn re-probes
    fast structured instead of going text-direct for a full minute.
    """

    monkeypatch.setattr(planner_module, "PLANNER_STRUCTURED_ATTEMPT_BUDGET_S", 0.05)

    class _OmittedWireSlowStructured:
        wire_agent = ""

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
            return {
                "mode": "retrieval",
                "resolved_intent": "standalone intent for test turn",
                "queries": _twelve_queries("unreachable structured"),
            }

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = system
            self.text_calls += 1
            assert "Return ONLY a JSON object" in prompt
            return _twelve_payload("tailored turn query")

    model = _OmittedWireSlowStructured()
    first = await run_planner("evening craving", model=model)
    assert [query.split(" variant ")[0] for query in first.queries] == ["tailored turn query"] * 12
    assert model.structured_calls == 1
    assert model.text_calls == 1
    # The deadline expiry (latency, not capability) must not mark the
    # omitted-structured capability for this model path.
    assert omitted_structured_unavailable(model) is False

    second = await run_planner("family quarrel tonight", model=model)
    assert [query.split(" variant ")[0] for query in second.queries] == ["tailored turn query"] * 12
    # The next turn re-probes structured first instead of going text-direct.
    assert model.structured_calls == 2
    assert model.text_calls == 2


async def test_planner_structured_timeout_does_not_mark_custom_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deadline marking applies only to the omitted-wire capability path."""

    monkeypatch.setattr(planner_module, "PLANNER_STRUCTURED_ATTEMPT_BUDGET_S", 0.05)

    class _CustomWireSlowStructured:
        wire_agent = "aa-planner-v2"

        def __init__(self) -> None:
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
            await asyncio.sleep(60.0)
            return {
                "mode": "retrieval",
                "resolved_intent": "standalone intent for test turn",
                "queries": _twelve_queries("unreachable structured"),
            }

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            return _twelve_payload("tailored turn query")

    model = _CustomWireSlowStructured()
    plan = await run_planner("evening craving", model=model)
    assert len(plan.queries) == 12
    assert model.structured_calls == 1
    assert omitted_structured_unavailable(model) is False


async def test_verifier_slow_structured_degrades_within_unit_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One hung structured attempt degrades to text inside the unit, not the turn."""

    monkeypatch.setattr(verifier_module, "VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S", 0.05)

    class _SlowStructuredTextServes:
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
            raise AssertionError("structured must time out before serving")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            self.text_calls += 1
            return (
                '{"requires_book_evidence": true, "supported": true, '
                '"evidence_passage_ids": ["p1"], "addresses_intent": true}'
            )

    pack = [_pack_entry()]
    units = split_response_units("Поддержка рядом помогает пережить тягу.")
    assert len(units) == 1
    model = _SlowStructuredTextServes()
    started = time.perf_counter()
    result = await run_verifier(units, pack, model=model)
    elapsed = time.perf_counter() - started
    assert result.all_required_supported is True
    assert model.structured_calls == 1
    assert model.text_calls == 1
    assert elapsed < 5.0
    # The caller-observed deadline is latency, not capability evidence
    # (kodmial/aa#244 recurrence 2): later units/turns re-probe structured.
    assert structured_text_fallback_preferred(model) is False


async def test_verifier_turn_expiry_preserves_completed_units() -> None:
    """Budget expiry narrows to verified units instead of clarifying the turn."""

    class _MixedSpeed:
        async def ainvoke_structured(
            self,
            prompt: str,
            *,
            system: str,
            schema: dict[str, object],
            retry_count: int = 1,
        ) -> dict[str, object]:
            if "Тяга проходит" in prompt:
                return {
                    "requires_book_evidence": True,
                    "supported": True,
                    "evidence_passage_ids": ["chapter-3#exp0000"],
                    "addresses_intent": True,
                }
            await asyncio.sleep(60.0)
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["chapter-3#exp0000"],
                "addresses_intent": True,
            }

    pack = [_pack_entry()]
    units = split_response_units("Тяга проходит. Вторая долгая проверка тянется.")
    assert len(units) == 2
    started = time.perf_counter()
    result = await run_verifier(units, pack, model=_MixedSpeed(), turn_budget_s=0.3)
    assert time.perf_counter() - started < 5.0
    # The completed unit survives expiry; the pending unit fails closed.
    assert len(result.units) == 2
    assert len(result.unavailable_unit_ids) == 1
    assert result.all_required_supported is False
    by_id = {verdict.unit_id: verdict for verdict in result.units}
    verified = [verdict for verdict in result.units if verdict.supported]
    assert len(verified) == 1
    assert by_id[verified[0].unit_id].evidence_passage_ids == ["chapter-3#exp0000"]


async def test_verifier_turn_expiry_all_pending_still_raises_unavailable() -> None:
    """A round with no verified unit at all still fails fully closed."""

    class _AllSlowNoText:
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
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": True,
            }

    units = split_response_units("Понимаю. Поддержка рядом помогает.")
    assert len(units) == 2
    started = time.perf_counter()
    with pytest.raises(OpenCodeTimeoutError):
        await run_verifier(units, [_pack_entry()], model=_AllSlowNoText(), turn_budget_s=0.2)
    assert time.perf_counter() - started < 5.0


async def test_verifier_429_propagates_fast_despite_slow_sibling() -> None:
    """Provider 429 aborts the round promptly instead of waiting out the budget."""

    class _One429OneSlow:
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
            if "Тяга проходит" in prompt:
                raise OpenCodeRateLimitError("opencode request rate-limited: http=429")
            await asyncio.sleep(60.0)
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": True,
            }

    units = split_response_units("Тяга проходит. Вторая долгая проверка тянется.")
    assert len(units) == 2
    started = time.perf_counter()
    with pytest.raises(OpenCodeRateLimitError):
        await run_verifier(units, [_pack_entry()], model=_One429OneSlow(), turn_budget_s=5.0)
    assert time.perf_counter() - started < 2.0


def test_text_parser_drops_unknown_keys_without_trusting_them() -> None:
    """Envelope tolerance: verdict is a function of the four known keys only."""
    import json

    decision = parse_text_json_decision(
        json.dumps(
            {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["p1"],
                "addresses_intent": True,
                "unit_id": "u1",
                "all_required_supported": True,
                "reasoning": "model prose habit",
            }
        )
    )
    assert decision == {
        "requires_book_evidence": True,
        "supported": True,
        "evidence_passage_ids": ["p1"],
        "addresses_intent": True,
    }
    # Missing required keys and wrong value types still fail closed.
    with pytest.raises(VerifierValidationError):
        parse_text_json_decision(json.dumps({"supported": True}))
    with pytest.raises(VerifierValidationError):
        parse_text_json_decision(
            json.dumps(
                {
                    "requires_book_evidence": ["not-a-bool"],
                    "supported": True,
                    "evidence_passage_ids": [],
                    "addresses_intent": True,
                }
            )
        )


def test_no_exact_live_question_special_cases() -> None:
    """The repair stays turn-independent: no frozen live prompt text in sources."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "planner_node.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "verifier.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "prompt_builder.py").read_text(encoding="utf-8"),
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
