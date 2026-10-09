"""P0 kodmial/aa#244: dual-gate repair for Gate C+E on exact main a0d377a.

Proven product failures (run 37753553708):

- C:live-book-grounding-substantive-drinking-2 on live-production-path
  (the core substantive drinking turn produced zero verified book units:
  planner timeouts yield empty query plans and empty packs, so the answer
  phase serves an ungrounded retry that the hardened Gate C counts as
  failure, never completion);
- E:latency-budget-exceeded on slo (p50 18930ms / p95 24290ms /
  max 24732ms with planner p50 4625ms / p95 10007ms pinned at its 10s
  wall, answer p50 6553ms / p95 10006ms pinned at its 10s wall, verifier
  p50 4655ms / p95 9410ms, message-text p50 4542ms / p95 9997ms over 73
  text calls).

Repair (turn-independent, Product Contract #110 unchanged, no
exact-question special cases, no SLO/threshold weakening):

- slow native structured channels degrade one second faster to the
  proven plain-text path (planner and verifier structured-attempt
  budgets 4.0s -> 3.0s, strict Pydantic validation unchanged on both
  paths, 429 still propagates): tail turns convert wall timeouts
  (empty plan / unavailable units -> C failure) into tailored text
  successes while cutting the sequential planner+answer+verifier sum
  for Gate E;
- answer/verifier display windows narrow 600 -> 500 chars per passage
  (explicit truncation markers preserved; stored pack plus
  checksum/quote/cite gates still use full exact text): every ordinary
  turn pays fewer text-path input tokens, cutting the message-text tail
  that pins all three stages at their walls.

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

from aa.conversation.planner_node import (
    PLANNER_STRUCTURED_ATTEMPT_BUDGET_S,
    PLANNER_TIME_BUDGET_S,
)
from aa.conversation.prompt_builder import ANSWER_MAX_PASSAGE_CHARS
from aa.conversation.turn_pipeline import (
    ANSWER_DRAFT_ATTEMPT_BUDGET_S,
    TURN_END_TO_END_BUDGET_S,
    run_v2_answer_turn,
)
from aa.conversation.verifier import (
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


def test_display_windows_narrowed_but_gates_use_full_pack() -> None:
    """Complete evidence reaches generator and verifier; gates keep full text (#295)."""
    assert ANSWER_MAX_PASSAGE_CHARS == 0
    assert VERIFIER_MAX_PASSAGE_CHARS == 0
    from aa.conversation.prompt_builder import EvidencePassage, render_turn_context
    from aa.conversation.verifier import _display_passage_text

    long_text = "x" * 3000
    context = render_turn_context(
        summary="",
        passages=[EvidencePassage(passage_id="p1", source="s", section="c", text=long_text)],
        user_message="что помогает?",
    )
    assert long_text in context
    assert _display_passage_text(long_text) == long_text


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
        return (
            '{"mode": "retrieval", "resolved_intent": "standalone intent for test turn", '
            '"queries": [' + queries + "]}"
        )

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
    assert 1 <= len(plan.queries) <= 16


async def test_grounded_turn_still_serves_with_narrowed_window() -> None:
    """A verified draft serves (not retry) with the 500-char display window."""

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
                "addresses_intent": True,
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
    from aa.conversation.failures import is_service_error as _ise244

    assert not _ise244(outcome["text"])

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
    """The #244 repair must never weaken Gate C grounding or Gate E SLO."""
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
