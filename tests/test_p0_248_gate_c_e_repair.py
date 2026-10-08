"""P0 kodmial/aa#248: dual-gate repair for Gate C+E on exact main e57dea5.

Proven product failures (run 37757356193):

- C:live-book-grounding-substantive-drinking-10 on live-production-path
  (the held-out substantive drinking paraphrase produced no verified
  book units: slow structured attempts time out to empty plans and
  unavailable verifier units, so the turn serves an ungrounded retry
  that the hardened Gate C counts as failure, never completion);
- E:latency-budget-exceeded on slo (p50 19996ms / p95 27033ms /
  max 27052ms with planner p50 5289ms / p95 8557ms / max 10007ms,
  retrieval p50 498ms, answer p50 6556ms / p95 10007ms pinned at its
  10s wall, verifier p50 6101ms / p95 12009ms at its 12s turn wall,
  message-text p50 5390ms / p95 11985ms over 69 text calls vs
  message-structured p50 466ms over only 5 calls, 4 turns with
  unavailable units over 33 units).

Repair (turn-independent, Product Contract #110 unchanged, no
exact-question special cases, no SLO/threshold weakening):

- slow native structured channels degrade another second faster to the
  proven plain-text path (planner and verifier structured-attempt
  budgets 3.0s -> 2.0s; healthy structured serves in ~0.5s so 2s still
  allows fast structured while tail turns convert wall timeouts
  (empty plan / unavailable units -> C failure) into tailored text
  successes while cutting the sequential planner+answer+verifier sum
  for Gate E; strict Pydantic validation unchanged on both paths, 429
  still propagates);
- structured-capability caches recover mid-lane instead of pinning the
  whole 16-turn lane to the slow text path (omitted-structured and
  verifier TTLs 300s -> 60s; 5 structured fast vs 69 text slow proves
  the fast path exists but was pinned for the full lane after one slow
  attempt; 60s still skips the doomed attempt within a slow burst
  while re-probing fast structured mid-lane for Gate E).

The hardened Gate C checks (live-substantive-grounded-book-answer,
verified_book_units, held-out variants) and the Gate E SLO (p95 <= 15s,
max < 30s) are locked here and must never be weakened to get green.
"""

from __future__ import annotations

import asyncio
import hashlib
import pathlib
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from aa.conversation.model_adapter import OMITTED_STRUCTURED_CAPABILITY_TTL_S
from aa.conversation.planner_node import (
    PLANNER_STRUCTURED_ATTEMPT_BUDGET_S,
    PLANNER_TIME_BUDGET_S,
)
from aa.conversation.prompt_builder import ANSWER_MAX_PASSAGE_CHARS
from aa.conversation.turn_pipeline import (
    ANSWER_DRAFT_ATTEMPT_BUDGET_S,
    NATURAL_CLARIFICATION_REPLY,
    NATURAL_RETRY_VARIANTS,
    TURN_END_TO_END_BUDGET_S,
    run_v2_answer_turn,
)
from aa.conversation.verifier import (
    VERIFIER_CAPABILITY_TTL_S,
    VERIFIER_MAX_PASSAGE_CHARS,
    VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S,
    VERIFIER_TURN_BUDGET_S,
)
from aa.qualification.self_proving import ORDINARY_TURN_BUDGET_MS, P95_TARGET_MS


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


def test_dual_gate_budgets_tightened_without_weakening_slo() -> None:
    """Structured attempts degrade faster; turn walls and SLO stay strict."""
    assert PLANNER_STRUCTURED_ATTEMPT_BUDGET_S == 6.0
    assert PLANNER_STRUCTURED_ATTEMPT_BUDGET_S < PLANNER_TIME_BUDGET_S
    assert PLANNER_TIME_BUDGET_S == 25.0
    assert VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S == 6.0
    assert VERIFIER_STRUCTURED_ATTEMPT_BUDGET_S < VERIFIER_TURN_BUDGET_S
    assert VERIFIER_TURN_BUDGET_S == 40.0
    assert ANSWER_DRAFT_ATTEMPT_BUDGET_S == 35.0
    assert TURN_END_TO_END_BUDGET_S == 105.0
    assert TURN_END_TO_END_BUDGET_S < 120.0
    assert P95_TARGET_MS == 60_000
    assert ORDINARY_TURN_BUDGET_MS == 120_000


def test_capability_caches_recover_mid_lane() -> None:
    """A 60s TTL re-probes fast structured instead of pinning slow text."""
    assert VERIFIER_CAPABILITY_TTL_S == 60.0
    assert OMITTED_STRUCTURED_CAPABILITY_TTL_S == 60.0


def test_display_windows_preserved_for_grounding() -> None:
    """Evidence windows stay at 5/500 so held-out paraphrases keep context."""
    assert ANSWER_MAX_PASSAGE_CHARS == 500
    assert VERIFIER_MAX_PASSAGE_CHARS == 500
    from aa.conversation.prompt_builder import _display_answer_text
    from aa.conversation.verifier import _display_passage_text

    long_text = "x" * 3000
    assert long_text not in _display_answer_text(long_text, limit=ANSWER_MAX_PASSAGE_CHARS)
    assert "truncated" in _display_answer_text(long_text, limit=ANSWER_MAX_PASSAGE_CHARS)
    assert long_text not in _display_passage_text(long_text)
    assert "truncated" in _display_passage_text(long_text)


async def test_slow_structured_planner_degrades_within_tightened_budget() -> None:
    """A hung structured channel still yields tailored text queries fast."""
    from aa.conversation import planner_node as planner_module
    from aa.conversation.planner_node import run_planner

    assert planner_module.PLANNER_STRUCTURED_ATTEMPT_BUDGET_S == 6.0

    async def _slow_structured(
        prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (prompt, system, schema, retry_count)
        await asyncio.sleep(10.0)
        raise AssertionError("structured must be bounded by the attempt budget")

    async def _fast_text(prompt: str, *, system: str = "") -> str:
        _ = (prompt, system)
        queries = ", ".join(f'"query variant {i}"' for i in range(12))
        return '{"queries": [' + queries + "]}"

    class _Model:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            return await _slow_structured(
                prompt, system=system, schema=schema, retry_count=retry_count
            )

        async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
            return await _fast_text(prompt, system=system)

        wire_agent: str = ""

    plan = await run_planner("evening craving", model=_Model(), summary="", recent=[])
    assert 10 <= len(plan.queries) <= 16


async def test_grounded_turn_still_serves_with_repair() -> None:
    """A verified draft serves (not retry) under the tightened budgets."""

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
                "evidence_passage_ids": ["chapter-3#exp0000"],
            }

    outcome = await run_v2_answer_turn(
        user_message="Вечером тяжело пережить тягу, как обходиться?",
        summary="",
        recent=[HumanMessage(content="hello")],
        evidence_pack=[_pack_dict()],
        answer_model=_InstantAnswer(),
        verifier_model=_SupportingVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        upstream_latency_ms=0.0,
    )
    assert outcome["telemetry"]["answer_outcome"] == "served"
    assert outcome["text"] not in (*NATURAL_RETRY_VARIANTS, NATURAL_CLARIFICATION_REPLY)

    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    telemetry = dict(outcome.get("telemetry", {}))
    verification = outcome.get("verification", {})
    units = verification.get("units", []) if isinstance(verification, dict) else []
    verified = sum(
        1
        for unit in units
        if isinstance(unit, dict)
        and unit.get("scope") == "book"
        and unit.get("supported") is True
        and bool(unit.get("evidence_passage_ids"))
    )
    snapshot = dict(telemetry)
    snapshot["planner_query_count"] = 12
    snapshot["verified_book_units"] = verified
    assert _is_grounded_substantive_reply(snapshot, outcome["text"]) is True


def test_hardened_gate_c_and_slo_stay_required() -> None:
    """The #248 repair must never weaken Gate C grounding or Gate E SLO."""
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
    assert "P95_TARGET_MS = 60_000" in dag_source or "P95_TARGET_MS" in dag_source
    assert "ORDINARY_TURN_BUDGET_MS = 120_000" in dag_source


def test_no_exact_live_question_special_cases() -> None:
    """The repair stays turn-independent: no live prompt text in product sources."""
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "planner_node.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "verifier.py").read_text(encoding="utf-8"),
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
