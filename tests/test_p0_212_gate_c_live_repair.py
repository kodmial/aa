"""P0 kodmial/aa#212 regression: repair Gate C live-path-failed.

Live evidence on exact main e2e42de (run 37638264853) showed the
live-telegram-evidence lane failing with the single concrete component
signature:

- live-text-max-over-budget (max 50.6s over the 30s hard budget, p50
  31.1s, p95 44.5s over 14 ordinary turns),

with no generic collapse (clarification 0), diversity passing, stage
telemetry present (14 snapshots), and healthy served-model identities
(planner via weak Space Bunny fallback, answer/summarizer/verifier via
strong Muse Spark). The tail is long overflowing drafts: a passed draft
that misses the #83 envelope pays a full extra answer+verifier provider
round via compact regeneration, pushing an already-slow turn further
over budget.

The repair bounds that extra round with the existing live-SLO repair
budget: when a passed turn already exceeds the budget before the
envelope check, regeneration is skipped and the turn compacts
deterministically to leading supported units instead. Grounding stays
strict (only Pydantic-validated supported units are served, otherwise
clarification); fast turns still use the single compact regeneration.
Turn-independent, never an exact-question special case, Product
Contract #110 unchanged.
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.messages import AIMessage

from aa.conversation import turn_pipeline as turn_pipeline_module
from aa.conversation.output_limits import envelope_passes
from aa.conversation.response_units import split_response_units
from aa.conversation.turn_pipeline import TURN_REPAIR_TIME_BUDGET_S, run_v2_answer_turn


def _pack_entry(text: str = "Фиктивная поддержка рядом помогает пережить тягу.") -> dict[str, Any]:
    return {
        "passage_id": "chapter-3#exp0000",
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


class _AnswerModel:
    def __init__(self, drafts: list[str]) -> None:
        self._drafts = list(drafts)
        self.calls = 0

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.calls += 1
        if not self._drafts:
            raise AssertionError("answer model called more times than scripted")
        return AIMessage(content=self._drafts.pop(0))


class _VerifierModel:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results: list[dict[str, Any]] = []
        for entry in results:
            for unit in entry.get("units", []):
                supported = bool(unit.get("supported", False))
                self._results.append(
                    {
                        "requires_book_evidence": str(unit.get("scope", "book")) == "book",
                        "supported": supported,
                        "evidence_passage_ids": list(unit.get("evidence_passage_ids", [])),
                        "addresses_intent": bool(unit.get("addresses_intent", supported)),
                    }
                )
        self.calls = 0

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        self.calls += 1
        if not self._results:
            raise AssertionError("verifier called more times than scripted")
        return dict(self._results.pop(0))


async def test_over_budget_envelope_overflow_skips_regeneration(
    monkeypatch: Any,
) -> None:
    """Issue #295: a slow verified overflowing turn splits (no 2nd round, no cut)."""
    from aa.conversation.output_limits import MAX_TRANSPORT_SEGMENTS

    pack = [_pack_entry()]
    sentence = "Поддержка рядом помогает пережить тягу спокойно"
    long_draft = " ".join(f"{sentence}." for _ in range(60))
    assert not envelope_passes(long_draft)
    long_units = split_response_units(long_draft)
    assert len(long_units) > 1
    answer = _AnswerModel([long_draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                    }
                    for unit in long_units
                ],
                "all_required_supported": True,
            },
        ]
    )
    monkeypatch.setattr(turn_pipeline_module, "TURN_REPAIR_TIME_BUDGET_S", -1.0)
    assert TURN_REPAIR_TIME_BUDGET_S > 0

    outcome = await run_v2_answer_turn(
        user_message="что помогает при тяге?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
        planner_model=None,
        retrieval_index=None,
    )
    assert answer.calls == 1
    # No extra provider round burns the SLO budget: the complete verified
    # answer is preserved via bounded transport split, not truncated.
    assert outcome["telemetry"].get("transport_split") is True
    segments = outcome.get("segments") or []
    assert 1 < len(segments) <= MAX_TRANSPORT_SEGMENTS
    assert all(envelope_passes(seg) for seg in segments)
    assert outcome["text"] == long_draft


async def test_fast_envelope_overflow_still_regenerates() -> None:
    """Issue #295: a fast verified overflowing turn splits (no cut, no regen)."""
    from aa.conversation.output_limits import MAX_TRANSPORT_SEGMENTS

    pack = [_pack_entry()]
    sentence = "Поддержка рядом помогает пережить тягу спокойно"
    long_draft = " ".join(f"{sentence}." for _ in range(60))
    short_draft = f"{sentence}."
    assert not envelope_passes(long_draft)
    assert envelope_passes(short_draft)
    long_units = split_response_units(long_draft)
    answer = _AnswerModel([long_draft])
    verifier = _VerifierModel(
        [
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "scope": "book",
                        "supported": True,
                        "evidence_passage_ids": [pack[0]["passage_id"]],
                    }
                    for unit in long_units
                ],
                "all_required_supported": True,
            },
        ]
    )
    assert TURN_REPAIR_TIME_BUDGET_S > 0

    outcome = await run_v2_answer_turn(
        user_message="что помогает при тяге?",
        summary="",
        recent=[],
        evidence_pack=pack,
        answer_model=answer,
        verifier_model=verifier,
        planner_model=None,
        retrieval_index=None,
    )
    assert answer.calls == 1
    assert outcome["telemetry"]["repair_budget_exceeded"] is False
    assert outcome["telemetry"].get("transport_split") is True
    segments = outcome.get("segments") or []
    assert 1 < len(segments) <= MAX_TRANSPORT_SEGMENTS
    assert all(envelope_passes(seg) for seg in segments)
    assert outcome["text"] == long_draft
