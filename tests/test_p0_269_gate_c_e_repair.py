"""P0 kodmial/aa#269: dual-gate repair for Gate C+E on exact main 90ff8b8.

Proven product failures (run 37813016105):

- C:live-answer-relevance-substantive-drinking-2 on live-production-path
  (the core substantive evening-craving turn served a reply with no
  topical relevance to the live request);
- E:latency-budget-exceeded on slo (p50 25027ms / p95 71451ms /
  max 81435ms with planner p50 5926ms / p95 17059ms, answer p50 9162ms /
  p95 20936ms, verifier p50 9632ms / p95 23197ms, repair_turns=2,
  answer_rounds=24, message-text p50 5310ms / p95 14271ms over 150
  text calls, message-structured p50 399ms / p95 687ms over 45 calls).

Repair (turn-independent, Product Contract #110 unchanged, no
exact-question special cases, no SLO/threshold weakening):

- Gate C: repair and adequacy-regeneration prompts keep the live request
  last (supplementary model-generated unsupported context first). The
  previous order buried the live request under trailing supplementary
  prose, so regeneration drifted off-topic and served verified-but-
  irrelevant replies. Ordering is generic for every turn; no prompt,
  reply, or corpus text is matched.
- Gate E: the repair time budget becomes binding (90s -> 50s). The 90s
  budget never fired while the p95 tail breached the 60s Gate E target,
  letting slow tail turns burn a second full
  planner+retrieval+answer+verifier sequence after an already-slow
  initial chain. At 50s ordinary turns still complete one full repair
  round while slow tail turns skip further re-planning and narrow to
  verified supported material instead of grinding another sequence.

The Gate E SLO (p95 <= 60s, max < 120s) and the hardened Gate C
relevance/grounding checks are locked here and must never be weakened
to get green.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage


def _pack_entry(
    passage_id: str = "chapter-3#exp0000",
    text: str = "Поддержка рядом помогает пережить тягу сегодня.",
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


def test_repair_budgets_binding_without_weakening_slo() -> None:
    """The repair budget binds tail turns; turn walls and SLO stay strict."""
    from aa.conversation.turn_pipeline import (
        TURN_END_TO_END_BUDGET_S,
        TURN_REPAIR_TIME_BUDGET_S,
    )
    from aa.qualification.self_proving import ORDINARY_TURN_BUDGET_MS, P95_TARGET_MS

    assert TURN_REPAIR_TIME_BUDGET_S == 50.0
    assert TURN_REPAIR_TIME_BUDGET_S < TURN_END_TO_END_BUDGET_S
    assert TURN_END_TO_END_BUDGET_S == 105.0
    assert TURN_END_TO_END_BUDGET_S < 120.0
    assert P95_TARGET_MS == 60_000
    assert ORDINARY_TURN_BUDGET_MS == 120_000


def test_repair_focus_keeps_live_request_last() -> None:
    """Supplementary context first, live request last (generic ordering)."""
    from aa.conversation.turn_pipeline import anchored_repair_focus

    request = "Вечером тяжело пережить тягу, как обходиться?"
    missing = ["Неподдержанное утверждение про распорядок."]
    focus = anchored_repair_focus(request, missing)
    assert focus.endswith(request)
    assert focus.index("Недостающая поддержка") < focus.rindex(request)


def test_repair_focus_empty_missing_returns_request() -> None:
    """No supplementary context means no prompt change at all."""
    from aa.conversation.turn_pipeline import anchored_repair_focus

    request = "Вечером тяжело пережить тягу, как обходиться?"
    assert anchored_repair_focus(request, []) == request
    assert anchored_repair_focus(request, ["   "]) == request


def test_adequacy_regen_prompt_keeps_request_last() -> None:
    """The adequacy instruction leads; the resolved request anchors last."""
    from aa.conversation.turn_pipeline import anchored_adequacy_regen_prompt

    request = "Вечером тяжело пережить тягу, как обходиться?"
    prompt = anchored_adequacy_regen_prompt(request)
    assert prompt.endswith(request)
    assert "практичный ответ" in prompt


async def test_repair_replan_focus_stays_anchored_to_live_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The planner repair focus sent on the wire ends with the live request."""
    from aa.conversation import turn_pipeline as pipeline

    live_request = "Вечером тяжело пережить тягу, как обходиться?"
    grounded = "Поддержка рядом помогает пережить тягу сегодня."
    off_topic = "Финансовое планирование помогает вести бюджет спокойно."
    seen: list[str] = []

    class _AnswerOffTopicThenGrounded:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content=off_topic)
            return AIMessage(content=grounded)

    class _VerifierRejectsOffTopicOnly:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (system, schema, retry_count)
            if off_topic[:12] in prompt:
                return {
                    "requires_book_evidence": True,
                    "supported": False,
                    "evidence_passage_ids": ["chapter-3#exp0000"],
                    "addresses_intent": False,
                }
            return {
                "requires_book_evidence": True,
                "supported": True,
                "evidence_passage_ids": ["chapter-3#exp0000"],
                "addresses_intent": True,
            }

    class _CapturePlan:
        def __init__(self, queries: list[str]) -> None:
            self.queries = list(queries)

    async def _fake_planner(
        focus: str, *, model: Any = None, summary: str = "", recent: Any = None
    ) -> Any:
        _ = (model, summary, recent)
        seen.append(focus)
        return _CapturePlan([f"запрос {i}" for i in range(12)])

    monkeypatch.setattr("aa.conversation.planner_node.run_planner", _fake_planner)

    def _fake_retrieve(index: Any, queries: Any, *, config: Any = None, **kwargs: Any) -> Any:
        _ = (index, queries, config)

        class _Pack:
            pass

        return _Pack()

    monkeypatch.setattr("aa.retrieval.evidence.retrieve_evidence", _fake_retrieve)

    def _fake_pack_to_state(pack: Any) -> tuple[Any, list[dict[str, Any]]]:
        _ = pack
        return (None, [_pack_entry(), _pack_entry(passage_id="chapter-3#exp0001")])

    monkeypatch.setattr("aa.conversation.retrieval_node.pack_to_state", _fake_pack_to_state)

    outcome = await pipeline.run_v2_answer_turn(
        user_message=live_request,
        summary="",
        recent=[HumanMessage(content="hello")],
        evidence_pack=[_pack_entry()],
        answer_model=_AnswerOffTopicThenGrounded(),
        verifier_model=_VerifierRejectsOffTopicOnly(),
        planner_model=object(),
        retrieval_index=object(),
        initial_query_count=12,
    )
    assert outcome["text"] == grounded
    assert seen, "repair must re-plan at least once"
    for focus in seen:
        assert focus.endswith(live_request)


async def test_binding_repair_budget_skips_replan_on_slow_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already-over-budget turn skips re-planning instead of grinding."""
    from aa.conversation import turn_pipeline as pipeline

    async def _noisy_planner(
        focus: str, *, model: Any = None, summary: str = "", recent: Any = None
    ) -> Any:
        _ = (focus, model, summary, recent)
        raise AssertionError("repair re-plan must be skipped over budget")

    monkeypatch.setattr("aa.conversation.planner_node.run_planner", _noisy_planner)
    monkeypatch.setattr(pipeline, "TURN_REPAIR_TIME_BUDGET_S", 0.0)

    class _AlwaysUnsupportedAnswer:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            return AIMessage(content="Финансовое планирование помогает вести бюджет.")

    class _AlwaysUnsupportedVerifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (system, schema, retry_count)
            return {
                "requires_book_evidence": True,
                "supported": False,
                "evidence_passage_ids": ["chapter-3#exp0000"],
                "addresses_intent": False,
            }

    import pytest as _pt269

    from aa.conversation.failures import TurnFailed as _TF269

    with _pt269.raises(_TF269) as _exc269:
        await pipeline.run_v2_answer_turn(
            user_message="Вечером тяжело пережить тягу, как обходиться?",
            summary="",
            recent=[HumanMessage(content="hello")],
            evidence_pack=[_pack_entry()],
            answer_model=_AlwaysUnsupportedAnswer(),
            verifier_model=_AlwaysUnsupportedVerifier(),
            planner_model=object(),
            retrieval_index=object(),
            initial_query_count=12,
        )
    telemetry = dict(_exc269.value.telemetry)
    assert telemetry["repair_budget_exceeded"] is True
    assert telemetry["repair_rounds"] == 0
