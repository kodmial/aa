"""P0 kodmial/aa#240 regression: bookless hash-selected filler behind Gate C PASS.

Proven product regression (manual Telegram evidence 2026-10-08): a real
user asked how to stop drinking and then asked for recommendations; the
bot returned polite repetitive filler without one actionable
book-grounded claim, and a later meta turn returned a generic
non-answer. Root-cause mechanics:

- the 14s end-to-end budget / 10s draft bound skipped answer/verifier
  on time exhaustion and served ``select_retry_reply()`` with zero
  verified book units (other paths served ``NATURAL_CLARIFICATION_REPLY``);
- the 10 SHA-256-selected ``NATURAL_RETRY_VARIANTS`` beat the
  8-distinct-string Gate C diversity floor instead of providing book
  information;
- old live Gate C checked one exact clarification string plus reply
  diversity, so it reported PASS on an all-fallback run.

Repair (turn-independent, Product Contract #110 unchanged):

- the end-to-end budget tracks the Gate E hard SLO (27s < 30s max) so
  ordinary answerable turns (sequential stage sum p50 ~10s / p95 ~21s)
  complete as verified grounded answers; only a turn past the hard SLO
  still fails fast with explicit failure telemetry that hardened Gate C
  counts as failure, never completion;
- SHA-256 hash selection is removed: every degraded turn serves the one
  stable retry, so fallback runs collapse visibly instead of mimicking
  helpful variety;
- planner provider/timeout failures are classified in stage telemetry
  instead of crashing the graph into a telemetry-less fallback.

The hardened Gate C checks (``live-substantive-grounded-book-answer``,
book-unit provenance, all-template-retries-count-as-failed) are locked
here and must never be weakened to get green.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from aa.conversation.graph import build_turn_graph, make_planner_node, turn_input
from aa.conversation.turn_pipeline import (
    NATURAL_CLARIFICATION_REPLY,
    NATURAL_RETRY_REPLY,
    NATURAL_RETRY_VARIANTS,
    TURN_END_TO_END_BUDGET_S,
    run_v2_answer_turn,
    select_retry_reply,
)
from aa.opencode.errors import OpenCodeRateLimitError, OpenCodeTimeoutError


def _pack_dict(passage_id: str = "chapter-3#exp0000") -> dict[str, Any]:
    text = "Поддержка рядом помогает пережить тягу сегодня."
    return {
        "passage_id": passage_id,
        "text": text,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _snapshot_for_gate_c(outcome: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the Gate C stage snapshot shape from an answer-turn outcome."""
    telemetry = dict(outcome.get("telemetry", {}))
    verification = outcome.get("verification", {})
    units = verification.get("units", []) if isinstance(verification, dict) else []
    verified_book_units = sum(
        1
        for unit in units
        if isinstance(unit, dict)
        and unit.get("scope") == "book"
        and unit.get("supported") is True
        and bool(unit.get("evidence_passage_ids"))
    )
    snapshot = dict(telemetry)
    snapshot["verified_book_units"] = verified_book_units
    return snapshot


def test_retry_selection_is_single_stable_reply() -> None:
    """No hash-selected variation: every input maps to the one retry."""
    assert select_retry_reply("любой текст") == NATURAL_RETRY_REPLY
    assert select_retry_reply("") == NATURAL_RETRY_REPLY
    assert select_retry_reply("   ") == NATURAL_RETRY_REPLY
    variants = {select_retry_reply(f"вариант {index}") for index in range(16)}
    assert variants == {NATURAL_RETRY_REPLY}


def test_no_hash_selection_in_turn_pipeline() -> None:
    """The answer pipeline must not hash user text into reply variety."""
    source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "aa"
        / "conversation"
        / "turn_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "sha256" not in source
    assert "hashlib" not in source


def test_end_to_end_budget_tracks_hard_slo_not_p95_target() -> None:
    """Ordinary 14-27s turns must complete; only hard-SLO breaches fail fast."""
    assert TURN_END_TO_END_BUDGET_S == 105.0
    assert TURN_END_TO_END_BUDGET_S < 120.0


async def test_slow_but_answerable_turn_serves_grounded_answer() -> None:
    """Upstream 16s (past the old 14s guard) still serves verified help."""

    class _InstantAnswer:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            return AIMessage(content="Поддержка рядом помогает пережить тягу сегодня.")

    class _SupportingVerifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (prompt, system, schema, retry_count)
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["p1"],
                "addresses_intent": True,
            }

    outcome = await run_v2_answer_turn(
        user_message="вечером тяжело без выпивки",
        summary="",
        recent=[HumanMessage(content="hello")],
        evidence_pack=[_pack_dict()],
        answer_model=_InstantAnswer(),
        verifier_model=_SupportingVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        upstream_latency_ms=16000.0,
    )
    assert outcome["telemetry"]["turn_budget_exceeded"] is False
    assert outcome["telemetry"]["answer_outcome"] == "served"
    assert outcome["text"] not in (*NATURAL_RETRY_VARIANTS, NATURAL_CLARIFICATION_REPLY)

    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    # Production enriches planner/retrieval counts in answer_pipeline_node;
    # mirror that enrichment for the direct turn-pipeline call.
    snapshot = _snapshot_for_gate_c(outcome)
    snapshot["planner_query_count"] = 12
    assert snapshot["retrieval_passages"] == 1
    assert _is_grounded_substantive_reply(snapshot, outcome["text"]) is True


async def test_budget_breach_retry_is_explicit_failure_for_gate_c() -> None:
    """A turn past the hard SLO fails with zero book units; Gate C rejects it."""

    class _MustNotRun:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            raise AssertionError("no model call may start on a spent turn")

    outcome = await run_v2_answer_turn(
        user_message="вечером тяжело без выпивки",
        summary="",
        recent=[HumanMessage(content="hello")],
        evidence_pack=[_pack_dict()],
        answer_model=_MustNotRun(),
        verifier_model=_MustNotRun(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        # Temporary quality-first SLO: end-to-end budget is 105s (< 120s
        # hard SLO), so a breach needs upstream past 105s (was 35s under
        # the old 27s budget).
        upstream_latency_ms=110000.0,
    )
    assert outcome["text"] == NATURAL_RETRY_REPLY
    assert outcome["telemetry"]["turn_budget_exceeded"] is True

    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    snapshot = _snapshot_for_gate_c(outcome)
    assert snapshot["verified_book_units"] == 0
    assert _is_grounded_substantive_reply(snapshot, outcome["text"]) is False
    assert outcome["text"].strip() in {*NATURAL_RETRY_VARIANTS, NATURAL_CLARIFICATION_REPLY}


async def test_planner_timeout_is_classified_not_graph_crash() -> None:
    """Planner provider timeouts land in stage telemetry with fallback queries."""

    def _timeout(messages: Any) -> Any:
        _ = messages
        raise OpenCodeTimeoutError("planner time budget exceeded")

    node = make_planner_node(planner_model=RunnableLambda(_timeout))
    result = await node(turn_input("вечером тяжело без выпивки"))
    assert result["planner_invoked"] is True
    # Generic fallback supplies retrieval queries instead of empty.
    assert len(result["search_queries"]) >= 1
    assert result["planner_mode"] == "retrieval"
    retry_state = result["retry_state"]
    assert retry_state["planner_query_count"] == len(result["search_queries"])
    assert retry_state["planner_outcome"] == "timeout"
    assert retry_state["planner_reason"] == "timeout"


async def test_planner_timeout_in_graph_keeps_telemetry() -> None:
    """The compiled graph records a planner timeout instead of raising."""

    def _timeout(messages: Any) -> Any:
        _ = messages
        raise OpenCodeTimeoutError("planner time budget exceeded")

    graph = build_turn_graph(planner_model=RunnableLambda(_timeout))
    result = await graph.ainvoke(turn_input("вечером тяжело без выпивки"))
    assert result["planner_invoked"] is True
    assert len(result["search_queries"]) >= 1
    assert result["planner_mode"] == "retrieval"
    assert result["retry_state"]["planner_outcome"] == "timeout"


async def test_planner_429_still_propagates_for_runner_retire() -> None:
    """Provider 429 from the planner retires the runner, never classifies."""

    def _limited(messages: Any) -> Any:
        _ = messages
        raise OpenCodeRateLimitError("opencode request rate-limited: http=429")

    node = make_planner_node(planner_model=RunnableLambda(_limited))
    with pytest.raises(OpenCodeRateLimitError):
        await node(turn_input("вечером тяжело без выпивки"))


def test_hardened_gate_c_checks_stay_required() -> None:
    """The #240 guard must never be weakened to get green."""
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


def test_no_exact_live_question_special_cases() -> None:
    """The repair stays turn-independent: no live prompt text in product sources."""
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "graph.py").read_text(encoding="utf-8"),
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
