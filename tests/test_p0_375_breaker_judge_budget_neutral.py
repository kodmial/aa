"""P0 BREAKER kodmial/aa#375: substantive judge must be budget-neutral.

Convergence scope:
``fingerprint:C:live-book-grounding-substantive-drinking-2:live-production-path|D:workflow-step-failed:workflow-step``
(blocking gate C, component live-production-path, main 47e1783 run
38288847872, local recurrence 3, systemic recurrence 3).

Reconstruction across recent exact-main history: the same drinking-2
scope failed with different surface categories on successive mains
(grounding, relevance, helpfulness, empty-telemetry, budget-exceeded)
while stage timings show retrieval p95 61s / max 75s and total p95
pinned at the 105s turn budget with repair_turns=3 and
budget_exceeded=2. Verifier availability was clean (0 unavailable
units). Each prior repair added another sequential provider round
(repair loop, then the #372 judge with up to 3 x 20s rounds checked
only against a 3s slice and ignoring the 50s repair budget), so tail
turns necessarily exceeded the shared deadline and collapsed to
budget-exceeded grounding failures. Prompt-wording alignment cannot
converge while the mechanism adds the latency that creates the next
failure.

Smallest architecture invariant: sequential quality gates must be
jointly budget-bounded. No gate may launch new provider rounds once
the repair budget or the end-to-end turn budget is spent, and no
single judge round may consume the downstream verifier slice.

Final one-shot repair (generic, no prompt branches, no keyword/stem
tables, no step parsing, no token-overlap relevance, no handcrafted
semantic predicates; Product Contract #110, Gate semantics,
grounding, relevance, safety and Gate E SLO unchanged): the internal
substantive judge returns to the adequacy path without a model call
once either budget is spent, and its call cap is carved out of the
remaining budget after reserving the verifier slice. Fast turns keep
the exact #372 gating behavior.

If this scope fails again after this repair, the Product Contract
stays UNQUALIFIED and the breaker stays blocked: no further automatic
code mutations for the same scope (asserted below via the existing
#290 breaker routing).

All strings below are synthetic fixtures for this failure class, never
the literal qualification prompts.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

_SYNTHETIC_USER_MESSAGE = "Синтетический вопрос о вечернем напряжении и спокойном распорядке"
_SYNTHETIC_RESOLVED_INTENT = "синтетическое намерение разобраться с вечерним напряжением"
_SYNTHETIC_DRAFT = (
    "Понимаю, вечернее напряжение переживается тяжело. "
    "Спокойный распорядок и поддержка рядом помогают пройти этот час."
)
_SYNTHETIC_PACK_TEXT = "Синтетический отрывок о спокойном вечернем распорядке."
_SYNTHETIC_PACK_ID = "chapter-3#exp0000"


def _pack_entry() -> dict[str, Any]:
    text = _SYNTHETIC_PACK_TEXT
    return {
        "passage_id": _SYNTHETIC_PACK_ID,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_sha256": "s" * 64,
        "corpus_version": "r" * 64,
    }


class _StubJudge:
    """Separately-identified whole-turn judge double with a fixed verdict."""

    def __init__(self, *, helpful: bool) -> None:
        self._helpful = helpful
        self.calls = 0
        self.agent = "stub-judge-375"

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
        return {
            "helpful": self._helpful,
            "addresses_intent": self._helpful,
            "contains_substantive_claim": True,
        }


def _install_pipeline_doubles(monkeypatch: pytest.MonkeyPatch) -> None:
    import aa.conversation.turn_pipeline as pipeline
    from aa.conversation.verifier_schema import (
        GroundingResult,
        UnitVerdict,
        validate_grounding_result,
    )

    async def _fake_generate_draft(*args: Any, **kwargs: Any) -> str:
        _ = (args, kwargs)
        return _SYNTHETIC_DRAFT

    async def _fake_run_verifier(units: Any, passages: Any, **kwargs: Any) -> GroundingResult:
        _ = (passages, kwargs)
        verdicts = [
            UnitVerdict(
                unit_id=unit.unit_id,
                scope="book",
                supported=True,
                evidence_passage_ids=[_SYNTHETIC_PACK_ID],
                addresses_intent=True,
                origin="book_claim",
            )
            for unit in units
        ]
        return validate_grounding_result(
            GroundingResult(
                verified=True,
                units=verdicts,
                all_required_supported=True,
                unavailable_unit_ids=[],
            ),
            expected_unit_ids=[unit.unit_id for unit in units],
        )

    monkeypatch.setattr(pipeline, "generate_draft", _fake_generate_draft)
    monkeypatch.setattr(pipeline, "run_verifier", _fake_run_verifier)


def _substantive_kwargs(judge: Any, **extra: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "user_message": _SYNTHETIC_USER_MESSAGE,
        "summary": "",
        "recent": [],
        "evidence_pack": [_pack_entry()],
        "answer_model": object(),
        "verifier_model": object(),
        "planner_model": None,
        "retrieval_index": None,
        "initial_query_count": 3,
        "planner_reason": "substantive-with-queries",
        "planner_mode": "retrieval",
        "resolved_intent": _SYNTHETIC_RESOLVED_INTENT,
        "whole_turn_judge_model": judge,
    }
    kwargs.update(extra)
    return kwargs


async def test_judge_skipped_when_repair_budget_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spent repair budget preserves adequacy instead of adding judge rounds."""
    import aa.conversation.turn_pipeline as pipeline
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    _install_pipeline_doubles(monkeypatch)
    monkeypatch.setattr(pipeline, "TURN_REPAIR_TIME_BUDGET_S", -1.0)
    assert pipeline.TURN_REPAIR_TIME_BUDGET_S < 0
    judge = _StubJudge(helpful=False)
    outcome = await run_v2_answer_turn(**_substantive_kwargs(judge))
    assert outcome["text"] == _SYNTHETIC_DRAFT
    assert judge.calls == 0
    telemetry = dict(outcome["telemetry"])
    assert telemetry["adequacy_verdict"] == "pass"
    assert telemetry["qualified"] is True


async def test_judge_skipped_when_turn_budget_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spent end-to-end budget launches no extra judge model call."""
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    _install_pipeline_doubles(monkeypatch)
    judge = _StubJudge(helpful=False)
    try:
        await run_v2_answer_turn(**_substantive_kwargs(judge, upstream_latency_ms=200_000.0))
    except Exception as exc:
        telemetry = dict(getattr(exc, "telemetry", {}) or {})
        assert judge.calls == 0
        assert telemetry.get("turn_budget_exceeded") is True
        return
    assert judge.calls == 0


async def test_judge_still_gates_fast_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fast turns keep the #372 gating behavior byte-identically."""
    from aa.conversation.failures import TurnFailed
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    _install_pipeline_doubles(monkeypatch)
    judge = _StubJudge(helpful=False)
    with pytest.raises(TurnFailed):
        await run_v2_answer_turn(**_substantive_kwargs(judge))
    assert judge.calls == 1


def test_breaker_scope_stays_blocked() -> None:
    """After the one-shot breaker, the same scope dispatches no more mutations."""
    from aa.qualification import mandatory_lifecycle as lifecycle

    fingerprint_c = "C:live-book-grounding-substantive-drinking-2:live-production-path"
    combined = fingerprint_c + "|D:workflow-step-failed:workflow-step"
    assert lifecycle.is_repair_allowed(fingerprint_c) is False
    assert lifecycle.is_repair_allowed(fingerprint_c, scope="gate:C") is False
    assert lifecycle.is_repair_allowed(combined, scope="gate:C") is False
    assert (
        lifecycle.needs_scheduled_recovery(
            failure_kind="rate_limited_429",
            attempt=0,
            fingerprint=fingerprint_c,
            scope="gate:C",
        )
        is False
    )
    assert (
        lifecycle.needs_scheduled_recovery(
            failure_kind="rate_limited_429",
            attempt=0,
            fingerprint=combined,
            scope="gate:C",
        )
        is False
    )
    assert lifecycle.is_repair_allowed("C:some-new-component:live-production-path") is True


def test_no_literal_qualification_branches() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    source = (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8")
    for fragment in (
        "Чем ты вообще можешь быть полезен",
        "Слушай, а ты тут вообще чем помочь можешь",
        "К вечеру очень тянет выпить",
        "Под вечер опять тянеет выпить",
        "Дома снова ссора",
        "Поругались дома из-за бухла",
        "покупать акции",
        "Какой телефон сейчас выгоднее купить",
        "Привет! К вечеру очень тянет выпить",
        "Привет, подскажи че делать когда под вечер тянеет выпить",
    ):
        assert fragment not in source
    for stem in ("тяне", "тянет выпить", "бухл", "похмел"):
        assert stem not in source
