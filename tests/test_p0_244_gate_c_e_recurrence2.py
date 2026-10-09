"""P0 kodmial/aa#244 recurrence 2: stop timeout-pinning the slow text path.

Proven product failures on exact main 6f8d4e1 (run 37764195857):

- C:live-book-grounding-substantive-drinking-2 on live-production-path;
- E:latency-budget-exceeded on slo (p50 18306ms / p95 26644ms /
  max 27034ms with planner p50 5730ms / p95 10007ms pinned at its 10s
  wall, answer p50 6002ms / p95 10007ms pinned at its 10s wall, verifier
  p50 4712ms / p95 12007ms at its 12s turn wall, message-structured p50
  404ms / p95 678ms over 28 fast calls vs message-text p50 5098ms / p95
  9994ms over 73 slow calls, 2 turns with unavailable units,
  answer_rounds=14, budget_exceeded=1).

Compared with run 37757356193 (structured p50 0.47s over only 5 calls,
planner 5.3s, verifier 6.1s, total p50 20.0s / p95 27.0s), the prior
3s->2s attempt cut plus 300s->60s TTL did not converge (p50 -1.7s, p95
-0.4s; planner p50 even rose 5.3s->5.7s with p95 pinned at the wall).
The dominant persistent cause is the timeout-marking itself: one slow
structured tail marks the lane to the slow text path for 60s, so healthy
turns that would serve in ~0.4s structured instead pay ~5s text and the
sequential planner+answer+verifier sum stays at ~18s p50.

Strategy change at the capability-cache boundary (not another duration
retune): a caller-observed deadline, a transient/timeout, a generic
provider error, or a structured content-validation failure is latency or
content, not capability evidence, so it falls back once without marking.
Only deterministic capability failures (schema rejected, provider access
rejected, missing structured payload) mark. Every turn re-probes the
fast structured path first. Turn-independent, Product Contract #110
unchanged, no exact-question special cases, no SLO weakening.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import aa.conversation.planner_node as planner_module
import aa.conversation.verifier as verifier_module
from aa.conversation.model_adapter import clear_omitted_structured_cache
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
    VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S,
    VERIFIER_TURN_BUDGET_S,
    clear_verifier_capability_cache,
    run_verifier,
    structured_text_fallback_preferred,
)
from aa.opencode.errors import (
    OpenCodeDeterministicError,
    OpenCodeTransientError,
)
from aa.qualification.self_proving import ORDINARY_TURN_BUDGET_MS, P95_TARGET_MS


@pytest.fixture(autouse=True)
def _clear_state() -> Any:
    clear_verifier_capability_cache()
    clear_omitted_structured_cache()
    yield
    clear_verifier_capability_cache()
    clear_omitted_structured_cache()


def _pack_entry(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Support nearby helps to get through craving today.",
) -> dict[str, Any]:
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


def test_budgets_and_slo_unchanged() -> None:
    """Recurrence 2 changes marking policy only, never walls or SLO."""
    assert PLANNER_STRUCTURED_ATTEMPT_BUDGET_S == 6.0
    assert PLANNER_TIME_BUDGET_S == 25.0
    assert VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S == 6.0
    assert VERIFIER_TURN_BUDGET_S == 40.0
    assert ANSWER_DRAFT_ATTEMPT_BUDGET_S == 35.0
    assert TURN_END_TO_END_BUDGET_S == 105.0
    assert P95_TARGET_MS == 60_000
    assert ORDINARY_TURN_BUDGET_MS == 120_000


async def test_planner_timeout_reprobes_structured_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow planner turn does not pin the next turn to slow text."""
    from aa.conversation.model_adapter import omitted_structured_unavailable

    monkeypatch.setattr(planner_module, "PLANNER_STRUCTURED_ATTEMPT_BUDGET_S", 0.05)

    class _SlowStructured:
        wire_agent = ""

        def __init__(self) -> None:
            self.structured_calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.structured_calls += 1
            await asyncio.sleep(60.0)
            raise AssertionError("must time out")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            queries = ", ".join(f'"tailored query variant {i}"' for i in range(12))
            return (
                '{"mode": "retrieval", "resolved_intent": "standalone intent for test turn", '
                '"queries": [' + queries + "]}"
            )

    model = _SlowStructured()
    first = await run_planner("evening craving", model=model)
    assert len(first.queries) == 12
    assert omitted_structured_unavailable(model) is False
    second = await run_planner("family quarrel tonight", model=model)
    assert len(second.queries) == 12
    assert model.structured_calls == 2


async def test_verifier_timeout_and_transient_do_not_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifier deadline and transient fall back once without marking."""
    monkeypatch.setattr(verifier_module, "VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S", 0.05)
    pack = [_pack_entry()]
    units = split_response_units("Support nearby helps to get through craving today.")
    assert len(units) == 1

    class _HangingStructured:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            await asyncio.sleep(60.0)
            raise AssertionError("must time out")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            return (
                '{"requires_book_evidence": true, "supported": true, '
                '"evidence_passage_ids": ["p1"], "addresses_intent": true}'
            )

    hanging = _HangingStructured()
    result = await run_verifier(units, pack, model=hanging)
    assert result.all_required_supported is True
    assert structured_text_fallback_preferred(hanging) is False

    class _TransientStructured:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            self.calls += 1
            raise OpenCodeTransientError("transient")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            return (
                '{"requires_book_evidence": true, "supported": true, '
                '"evidence_passage_ids": ["p1"], "addresses_intent": true}'
            )

    transient = _TransientStructured()
    result = await run_verifier(units, pack, model=transient)
    assert result.all_required_supported is True
    assert structured_text_fallback_preferred(transient) is False
    again = await run_verifier(units, pack, model=transient)
    assert again.all_required_supported is True
    assert transient.calls == 2


async def test_verifier_deterministic_still_marks() -> None:
    """Only true capability failures keep pinning the text path."""
    pack = [_pack_entry()]
    units = split_response_units("Support nearby helps to get through craving today.")

    class _DeterministicMissing:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            raise OpenCodeDeterministicError("opencode structured output missing")

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            _ = (prompt, system)
            return (
                '{"requires_book_evidence": true, "supported": true, '
                '"evidence_passage_ids": ["p1"], "addresses_intent": true}'
            )

    model = _DeterministicMissing()
    result = await run_verifier(units, pack, model=model)
    assert result.all_required_supported is True
    assert structured_text_fallback_preferred(model) is True
