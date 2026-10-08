"""P0 kodmial/aa#244 recurrence 3: pin on persistent structured rejection.

Proven product failures on exact main 1d109a29724c15d90f07512dc81e7d6110b6312c
(run 37771443722):

- C:live-book-grounding-substantive-drinking-2 on live-production-path;
- E:latency-budget-exceeded on slo (p50 19459ms / p95 27020ms /
  max 27038ms with planner p50 6418ms / p95 8303ms, retrieval p50
  583ms / p95 627ms, answer p50 6954ms / p95 8883ms, verifier p50
  5802ms / p95 12007ms still pinned at its 12s turn wall,
  message-structured p50 450ms / p95 697ms / max 1542ms over 33 fast
  calls vs message-text p50 4898ms / p95 10131ms / max 26907ms over
  79 slow calls, 2 turns with unavailable units (2 total) over 41
  response units with max 4, answer_rounds=14, budget_exceeded=1).

Compared with the recurrence-2 base (exact main 6f8d4e1 run
37764195857: p50 18306ms / p95 26644ms / max 27034ms, planner p50
5730ms / p95 10007ms pinned at its 10s wall, answer p50 6002ms / p95
10007ms pinned, verifier p50 4712ms / p95 12007ms at its
wall, message-structured p50 404ms / p95 678ms over 28 calls vs
message-text p50 5098ms / p95 9994ms over 73 calls, same 2
unavailable turns), the recurrence-2 re-probe policy did not
converge: structured attempts rose (+5) while text attempts rose in
lockstep (+6), so re-probed structured attempts converted zero turns
back to the fast path and only added a second paid provider
interaction per invocation (planner p50 even rose 5.7s->6.4s).
No structured timeout fired in either run (max 1.5s is well under
the 2s attempt budget), so the remaining waste is call COUNT, not
attempt duration: the provider persistently rejects native
json_schema on both the primary and the fallback routes (no fallback
model is ever recorded as serving; ~40 of 112 HTTP interactions
record no tokens at all), and every TTL-window re-probe round burns
a full concurrent fan-out (structured-reject plus text-fallback per
unit) before re-marking.

Strategy change at the capability-cache boundary (not another
duration retune, and not a return to blip-pinning): pin on
PERSISTENT rejection evidence. Two consecutive capability-suggestive
structured rejections (deterministic, or generic provider errors)
mark the path for the TTL; one isolated rejection still re-probes.
Caller-observed deadline expiries, transient/timeout failures,
content-validation failures and provider 429 never count (latency
or content, not capability evidence); any structured success resets
the streak; TTL expiry still re-probes a recovered provider; 429
always propagates and never marks. Later turns on a persistently
rejecting path then go directly to the proven bounded text path
instead of burning the doomed structured round-trip per unit, which
cuts provider contention for Gate E and wall-expiry pressure
(unavailable units -> ungrounded retry) for the drinking-2 Gate C
mechanism. Turn-independent, Product Contract #110 unchanged, no
exact-question special cases, no SLO weakening; both paths stay
fully Pydantic-validated.
"""

from __future__ import annotations

import pytest

import aa.conversation.verifier as verifier_module
from aa.conversation.model_adapter import (
    OMITTED_STRUCTURED_CAPABILITY_TTL_S,
    OMITTED_STRUCTURED_PERSISTENT_THRESHOLD,
    clear_omitted_structured_cache,
    omitted_structured_unavailable,
)
from aa.conversation.planner_node import (
    PLANNER_STRUCTURED_ATTEMPT_BUDGET_S,
    PLANNER_TIME_BUDGET_S,
    run_planner,
)
from aa.conversation.response_units import split_response_units
from aa.conversation.turn_pipeline import (
    ANSWER_DRAFT_ATTEMPT_BUDGET_S,
    TURN_END_TO_END_BUDGET_S,
)
from aa.conversation.verifier import (
    VERIFIER_CAPABILITY_TTL_S,
    VERIFIER_PERSISTENT_REJECTION_THRESHOLD,
    VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S,
    VERIFIER_TURN_BUDGET_S,
    clear_verifier_capability_cache,
    run_verifier,
    structured_text_fallback_preferred,
)
from aa.opencode.errors import (
    OpenCodeDeterministicError,
    OpenCodeNotReadyError,
    OpenCodeRateLimitError,
    OpenCodeTimeoutError,
    OpenCodeTransientError,
)
from aa.qualification.self_proving import ORDINARY_TURN_BUDGET_MS, P95_TARGET_MS

PRIMARY = "opencode/muse-spark-1.3-contributor-free"
FALLBACK = "opencode/space-bunny-free"


@pytest.fixture(autouse=True)
def _clear_state() -> object:
    clear_verifier_capability_cache()
    clear_omitted_structured_cache()
    yield
    clear_verifier_capability_cache()
    clear_omitted_structured_cache()
    return None


def _pack_entry(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Support nearby helps to get through craving today.",
) -> dict[str, object]:
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


def _decision_json() -> str:
    import json

    return json.dumps(
        {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["p1"],
        }
    )


def _twelve_query_json() -> str:
    import json

    return json.dumps({"queries": [f"tailored turn query variant {i}" for i in range(12)]})


def test_budgets_thresholds_and_slo_unchanged() -> None:
    """Recurrence 3 adds streak evidence only; walls, windows and SLO stay strict."""
    assert PLANNER_STRUCTURED_ATTEMPT_BUDGET_S == 2.0
    assert PLANNER_TIME_BUDGET_S == 10.0
    assert VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S == 2.0
    assert VERIFIER_TURN_BUDGET_S == 12.0
    assert ANSWER_DRAFT_ATTEMPT_BUDGET_S == 10.0
    assert TURN_END_TO_END_BUDGET_S == 27.0
    assert P95_TARGET_MS == 15_000
    assert ORDINARY_TURN_BUDGET_MS == 30_000
    assert VERIFIER_CAPABILITY_TTL_S == 60.0
    assert OMITTED_STRUCTURED_CAPABILITY_TTL_S == 60.0
    assert VERIFIER_PERSISTENT_REJECTION_THRESHOLD == 2
    assert OMITTED_STRUCTURED_PERSISTENT_THRESHOLD == 2


async def test_verifier_single_generic_rejection_still_reprobes() -> None:
    """One isolated generic rejection is not capability evidence (no pin)."""
    pack = [_pack_entry()]
    units = split_response_units("Support nearby helps to get through craving today.")
    assert len(units) == 1

    class _GenericOnce:
        agent = "aa-verifier-v2"
        primary_model = PRIMARY
        fallback_model = ""

        def __init__(self) -> None:
            self.structured_calls = 0
            self.text_calls = 0
            self.failures = 1

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.structured_calls += 1
            if self.failures > 0:
                self.failures -= 1
                raise OpenCodeNotReadyError("opencode runtime has not been started")
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["p1"],
            }

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            self.text_calls += 1
            return _decision_json()

    model = _GenericOnce()
    first = await run_verifier(units, pack, model=model)
    assert first.all_required_supported is True
    assert structured_text_fallback_preferred(model) is False
    second = await run_verifier(units, pack, model=model)
    assert second.all_required_supported is True
    assert model.structured_calls == 2
    assert model.text_calls == 1


async def test_verifier_consecutive_generic_rejections_pin() -> None:
    """Two consecutive generic rejections pin the path until the TTL expires."""
    pack = [_pack_entry()]
    units = split_response_units("Support nearby helps to get through craving today.")

    class _AlwaysGeneric:
        agent = "aa-verifier-v2"
        primary_model = PRIMARY
        fallback_model = ""

        def __init__(self) -> None:
            self.structured_calls = 0
            self.text_calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.structured_calls += 1
            raise OpenCodeNotReadyError("opencode runtime has not been started")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            self.text_calls += 1
            return _decision_json()

    model = _AlwaysGeneric()
    assert await run_verifier(units, pack, model=model)
    assert structured_text_fallback_preferred(model) is False
    assert await run_verifier(units, pack, model=model)
    assert structured_text_fallback_preferred(model) is True
    assert await run_verifier(units, pack, model=model)
    assert model.structured_calls == 2
    assert model.text_calls == 3


async def test_verifier_success_resets_rejection_streak() -> None:
    """A structured success between generic rejections keeps re-probing."""
    pack = [_pack_entry()]
    units = split_response_units("Support nearby helps to get through craving today.")

    class _FlakyStructured:
        agent = "aa-verifier-v2"
        primary_model = PRIMARY
        fallback_model = ""

        def __init__(self) -> None:
            self.structured_calls = 0
            self.text_calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.structured_calls += 1
            if self.structured_calls == 2:
                return {
                    "requires_book_evidence": True,
                    "supported": True,
                    "evidence_passage_ids": ["p1"],
                }
            raise OpenCodeNotReadyError("opencode runtime has not been started")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            self.text_calls += 1
            return _decision_json()

    model = _FlakyStructured()
    assert await run_verifier(units, pack, model=model)
    assert await run_verifier(units, pack, model=model)
    assert structured_text_fallback_preferred(model) is False
    assert await run_verifier(units, pack, model=model)
    assert model.structured_calls == 3
    assert structured_text_fallback_preferred(model) is False


async def test_verifier_transient_and_timeout_stay_streak_neutral() -> None:
    """Blips never count: two transients pin nothing; later generics still need two."""

    pack = [_pack_entry()]
    units = split_response_units("Support nearby helps to get through craving today.")

    class _BlipsThenGeneric:
        agent = "aa-verifier-v2"
        primary_model = PRIMARY
        fallback_model = ""

        def __init__(self) -> None:
            self.structured_calls = 0
            self.text_calls = 0
            self.mode = "transient"

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.structured_calls += 1
            if self.mode == "transient":
                raise OpenCodeTransientError("transient")
            if self.mode == "timeout":
                raise OpenCodeTimeoutError("opencode request timed out")
            raise OpenCodeNotReadyError("opencode runtime has not been started")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            self.text_calls += 1
            return _decision_json()

    model = _BlipsThenGeneric()
    assert await run_verifier(units, pack, model=model)
    model.mode = "timeout"
    assert await run_verifier(units, pack, model=model)
    assert structured_text_fallback_preferred(model) is False
    model.mode = "generic"
    assert await run_verifier(units, pack, model=model)
    assert structured_text_fallback_preferred(model) is False
    assert await run_verifier(units, pack, model=model)
    assert structured_text_fallback_preferred(model) is True


async def test_verifier_deterministic_still_marks_immediately() -> None:
    """Deterministic capability failures keep the immediate mark (no streak needed)."""
    pack = [_pack_entry()]
    units = split_response_units("Support nearby helps to get through craving today.")

    class _DeterministicMissing:
        agent = "aa-verifier-v2"
        primary_model = PRIMARY
        fallback_model = ""

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            raise OpenCodeDeterministicError("opencode structured output missing")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            return _decision_json()

    model = _DeterministicMissing()
    result = await run_verifier(units, pack, model=model)
    assert result.all_required_supported is True
    assert structured_text_fallback_preferred(model) is True


async def test_verifier_429_never_records_or_marks() -> None:
    """Provider 429 propagates for runner retire/restart without streak or mark."""
    pack = [_pack_entry()]
    units = split_response_units("Support nearby helps to get through craving today.")

    class _RateLimited:
        agent = "aa-verifier-v2"
        primary_model = PRIMARY
        fallback_model = ""

        def __init__(self) -> None:
            self.structured_calls = 0
            self.text_calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.structured_calls += 1
            raise OpenCodeRateLimitError("opencode request rate-limited: http=429")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            self.text_calls += 1
            raise AssertionError("text fallback must not run on 429")

    model = _RateLimited()
    try:
        await run_verifier(units, pack, model=model)
    except OpenCodeRateLimitError:
        pass
    else:
        raise AssertionError("429 must propagate")
    assert model.text_calls == 0
    assert structured_text_fallback_preferred(model) is False


async def test_verifier_streak_expires_with_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired streak re-probes: a recovered provider is retried after the TTL."""
    pack = [_pack_entry()]
    units = split_response_units("Support nearby helps to get through craving today.")

    class _AlwaysGeneric:
        agent = "aa-verifier-v2"
        primary_model = PRIMARY
        fallback_model = ""

        def __init__(self) -> None:
            self.structured_calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.structured_calls += 1
            raise OpenCodeNotReadyError("opencode runtime has not been started")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            return _decision_json()

    model = _AlwaysGeneric()
    assert await run_verifier(units, pack, model=model)
    assert await run_verifier(units, pack, model=model)
    assert structured_text_fallback_preferred(model) is True
    monkeypatch.setattr(verifier_module, "VERIFIER_CAPABILITY_TTL_S", 0.0)
    assert structured_text_fallback_preferred(model) is False
    assert await run_verifier(units, pack, model=model)
    assert model.structured_calls == 3


async def test_planner_consecutive_generic_rejections_pin_omitted_path() -> None:
    """Two consecutive generic planner rejections switch later turns to direct text."""

    class _GenericStructuredPlanner:
        agent = "aa-planner-v2"
        primary_model = PRIMARY
        fallback_model = FALLBACK
        wire_agent = ""

        def __init__(self) -> None:
            self.structured_calls = 0
            self.text_calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.structured_calls += 1
            raise OpenCodeNotReadyError("opencode runtime has not been started")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            self.text_calls += 1
            return _twelve_query_json()

    model = _GenericStructuredPlanner()
    first = await run_planner("evening craving", model=model)
    assert len(first.queries) == 12
    assert omitted_structured_unavailable(model) is False
    second = await run_planner("family quarrel tonight", model=model)
    assert len(second.queries) == 12
    assert omitted_structured_unavailable(model) is True
    third = await run_planner("sleepless night", model=model)
    assert len(third.queries) == 12
    assert model.structured_calls == 2
    assert model.text_calls == 3


async def test_planner_success_resets_omitted_streak() -> None:
    """A structured planner success between generic rejections keeps re-probing."""

    class _FlakyPlanner:
        agent = "aa-planner-v2"
        primary_model = PRIMARY
        fallback_model = FALLBACK
        wire_agent = ""

        def __init__(self) -> None:
            self.structured_calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.structured_calls += 1
            if self.structured_calls == 2:
                return {"queries": [f"tailored turn query variant {i}" for i in range(12)]}
            raise OpenCodeNotReadyError("opencode runtime has not been started")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            return _twelve_query_json()

    model = _FlakyPlanner()
    assert len((await run_planner("evening craving", model=model)).queries) == 12
    assert len((await run_planner("family quarrel tonight", model=model)).queries) == 12
    assert omitted_structured_unavailable(model) is False
    assert len((await run_planner("sleepless night", model=model)).queries) == 12
    assert model.structured_calls == 3


def test_hardened_gate_c_and_slo_stay_required() -> None:
    """The #244 repair must never weaken Gate C grounding or Gate E SLO."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    live_source = (root / "src" / "aa" / "qualification" / "product_contract_live.py").read_text(
        encoding="utf-8"
    )
    assert "live-substantive-grounded-book-answer" in live_source
    assert "verified_book_units" in live_source
    assert "book_grounded_families" in live_source
    assert "held_out_scenarios" in live_source
    runner_source = (root / "scripts" / "run_self_proving_qualification.py").read_text(
        encoding="utf-8"
    )
    assert "live-substantive-grounded-book-answer" in runner_source
    dag_source = (root / "src" / "aa" / "qualification" / "self_proving.py").read_text(
        encoding="utf-8"
    )
    assert "P95_TARGET_MS = 15_000" in dag_source or "P95_TARGET_MS" in dag_source
    assert "ORDINARY_TURN_BUDGET_MS = 30_000" in dag_source


def test_no_exact_live_question_special_cases() -> None:
    """The repair stays turn-independent: no live prompt text in product sources."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "planner_node.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "verifier.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "model_adapter.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "prompt_builder.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
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
