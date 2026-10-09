"""P0 kodmial/aa#251: stop bookless all-glue replies; whole-turn adequacy.

Proven incident: a normal greeting plus a direct recovery question was
answered with bookless conversational filler and zero practical
book-based help. Root causes were systemic: planner glue/error
conflation, an agent prompt permitting pure glue without evidence, and
per-unit verification without whole-turn answer adequacy.

This test locks the production repair (turn-independent, no
exact-question whitelist, Product Contract unchanged):

- planner errors are never legitimate glue and trigger one bounded
  recovery from the actual turn context against the canonical book;
- a substantive turn requires a relevant verifier-supported book unit;
  all-glue, irrelevant citations, offers plus questions, unrelated facts
  and unsupported paraphrases fail adequacy;
- pure greetings and honest identity keep natural glue;
- telemetry is causal and privacy-safe with trace, reason, adequacy and
  runtime SHA correlated to the sent message;
- Gate C judges the sent message plus this turn's snapshot with an
  independent semantic relevance signal, never sentence shape alone.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any

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
    }


def _grounded_snapshot(**overrides: object) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "answer_outcome": "served",
        "verifier_outcome": "passed",
        "verifier_unavailable_units": 0,
        "turn_budget_exceeded": False,
        "planner_query_count": 12,
        "planner_reason": "substantive-with-queries",
        "retrieval_passages": 3,
        "verified_book_units": 1,
        "response_units": 1,
        "adequacy_verdict": "pass",
        "failure_category": "",
        "answers_request": True,
        "technically_grounded": True,
        "qualified": True,
    }
    snapshot.update(overrides)
    return snapshot


def test_planner_reason_never_conflates_error_with_glue() -> None:
    from aa.conversation.answer_adequacy import planner_reason_for

    assert planner_reason_for(0, "timeout") == "timeout"
    assert planner_reason_for(0, "failed") == "provider-error"
    assert planner_reason_for(0, "invalid") == "invalid"
    assert planner_reason_for(0, "ok") == "legitimate-glue"
    assert planner_reason_for(0, "empty") == "legitimate-glue"
    assert planner_reason_for(12, "ok") == "substantive-with-queries"


def test_mixed_greeting_plus_request_is_not_proven_glue() -> None:
    # Model-driven turn understanding (#268): schema-only planner mode
    # decides glue, and the planner-resolved intent carries the request.
    from aa.conversation.answer_adequacy import effective_request, is_conversational_plan

    assert is_conversational_plan(mode="conversational", query_count=0) is True
    assert is_conversational_plan(mode="retrieval", query_count=12) is False
    assert is_conversational_plan(mode="conversational", query_count=12) is False
    assert is_conversational_plan(mode="retrieval", query_count=0) is False
    resolved = "how to handle evening craving"
    assert (
        effective_request(resolved_intent=resolved, user_message="Привет! Что делать?") == resolved
    )
    assert (
        effective_request(resolved_intent="", user_message="А что мне делать-то с этим?")
        == "А что мне делать-то с этим?"
    )


def test_recovery_queries_use_actual_turn_not_canned_generic() -> None:
    from aa.conversation.answer_adequacy import build_recovery_queries

    queries = build_recovery_queries("Вечером тяжело, тяга сильная", summary="разговор о вечере")
    assert queries
    assert any("Вечером тяжело" in item for item in queries)
    assert len(queries) <= 6
    assert build_recovery_queries("") == []


def test_all_glue_with_available_pack_fails_adequacy() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    pack = [_pack_entry()]
    glue_reply = "Привет! Рад, что ты написал. Расскажешь чуть больше о своей ситуации?"
    verdict = assess_turn_adequacy(
        user_message="Привет! Как обходиться с тягой вечером?",
        reply=glue_reply,
        evidence_pack=pack,
        grounding_result={
            "verified": True,
            "all_required_supported": True,
            "units": [
                {
                    "unit_id": "u1",
                    "scope": "conversation_glue",
                    "supported": True,
                    "evidence_passage_ids": [],
                }
            ],
        },
        planner_reason="legitimate-glue",
    )
    assert verdict.verdict == "fail"
    assert verdict.failure_category == "all-glue-for-substantive"
    assert verdict.answers_request is False


def test_irrelevant_citation_fails_adequacy_despite_verified_unit() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    ev_text = "Финансовое планирование помогает вести бюджет спокойно."
    pack = [_pack_entry(passage_id="chapter-9#exp0001", text=ev_text)]
    reply = "Финансовое планирование помогает вести бюджет спокойно."
    verdict = assess_turn_adequacy(
        user_message="Вечером тяжело пережить тягу, как обходиться?",
        reply=reply,
        evidence_pack=pack,
        grounding_result={
            "verified": True,
            "all_required_supported": True,
            "answer_relevant": False,
            "units": [
                {
                    "unit_id": "u1",
                    "scope": "book",
                    "supported": True,
                    "evidence_passage_ids": ["chapter-9#exp0001"],
                    "addresses_intent": False,
                }
            ],
        },
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent="Вечером тяжело пережить тягу, как обходиться?",
    )
    assert verdict.verdict == "fail"
    assert verdict.failure_category == "irrelevant-citation"


def test_relevant_grounded_answer_passes_adequacy() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    ev_text = "Поддержка рядом помогает пережить тягу сегодня."
    pack = [_pack_entry(text=ev_text)]
    reply = "Поддержка рядом помогает пережить тягу сегодня."
    verdict = assess_turn_adequacy(
        user_message="Вечером тяжело пережить тягу, как обходиться?",
        reply=reply,
        evidence_pack=pack,
        grounding_result={
            "verified": True,
            "all_required_supported": True,
            "answer_relevant": True,
            "units": [
                {
                    "unit_id": "u1",
                    "scope": "book",
                    "supported": True,
                    "evidence_passage_ids": ["chapter-3#exp0000"],
                    "addresses_intent": True,
                }
            ],
        },
        planner_reason="substantive-with-queries",
        planner_mode="retrieval",
        resolved_intent="Вечером тяжело пережить тягу, как обходиться?",
    )
    assert verdict.verdict == "pass"
    assert verdict.answers_request is True


def test_pure_greeting_keeps_glue_without_book() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    verdict = assess_turn_adequacy(
        user_message="Привет",
        reply="Привет! Как ты сегодня?",
        evidence_pack=[],
        grounding_result={
            "verified": True,
            "all_required_supported": True,
            "units": [
                {
                    "unit_id": "u1",
                    "scope": "conversation_glue",
                    "supported": True,
                    "evidence_passage_ids": [],
                }
            ],
        },
        planner_reason="legitimate-glue",
        planner_mode="conversational",
        resolved_intent="",
    )
    assert verdict.verdict == "pass"


def test_meta_identity_keeps_glue_without_book() -> None:
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    verdict = assess_turn_adequacy(
        user_message="Ты кто?",
        reply="Я ИИ-помощник. Помогаю разбирать тягу и ближайшие шаги.",
        evidence_pack=[],
        grounding_result={
            "verified": True,
            "all_required_supported": True,
            "units": [
                {
                    "unit_id": "u1",
                    "scope": "conversation_glue",
                    "supported": True,
                    "evidence_passage_ids": [],
                }
            ],
        },
        planner_reason="legitimate-glue",
        planner_mode="conversational",
        resolved_intent="",
    )
    assert verdict.verdict == "pass"


def test_substantive_delivery_invariant_splits_statuses() -> None:
    from aa.conversation.answer_adequacy import (
        check_substantive_delivery_invariant,
        snapshot_statuses,
    )

    ok_snapshot = _grounded_snapshot()
    ok, _ = check_substantive_delivery_invariant(ok_snapshot)
    assert ok is True
    statuses = snapshot_statuses(ok_snapshot, delivered=True)
    assert statuses == {
        "technically_grounded": True,
        "answers_request": True,
        "delivered": True,
        "qualified": True,
    }
    bad = _grounded_snapshot(verified_book_units=0, adequacy_verdict="fail")
    ok_bad, _ = check_substantive_delivery_invariant(bad)
    assert ok_bad is False
    statuses_bad = snapshot_statuses(bad, delivered=True)
    assert statuses_bad["qualified"] is False


async def test_planner_timeout_empty_pack_serves_honest_unavailability() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    class _MustNotRun:
        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            raise AssertionError("no model call may start on a spent turn")

    import pytest as _pt251

    from aa.conversation.failures import TurnFailed as _TF251

    with _pt251.raises(_TF251) as _exc251:
        await run_v2_answer_turn(
            user_message="Привет! Как обходиться с тягой вечером?",
            summary="",
            recent=[HumanMessage(content="hello")],
            evidence_pack=[],
            answer_model=_MustNotRun(),
            verifier_model=_MustNotRun(),
            planner_model=None,
            retrieval_index=None,
            initial_query_count=0,
            upstream_latency_ms=120000.0,
        )
    telemetry = dict(_exc251.value.telemetry)
    assert telemetry["turn_budget_exceeded"] is True

    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    snapshot = dict(telemetry)
    snapshot["verified_book_units"] = 0
    assert _is_grounded_substantive_reply(snapshot, "") is False


async def test_all_glue_draft_with_pack_regenerates_or_fails_explicitly() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    pack = [_pack_entry()]
    glue = "Привет! Рад, что ты написал. Расскажешь чуть больше о своей ситуации?"

    class _GlueThenGrounded:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages: Any) -> AIMessage:
            _ = messages
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content=glue)
            return AIMessage(content="Поддержка рядом помогает пережить тягу сегодня.")

    class _Verifier:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            _ = (system, schema, retry_count)
            # Per-unit routing on the judged unit only: the passage text
            # itself lives in every prompt, so match <response_unit>.
            import re as _re

            match = _re.search(r"<response_unit>(.*?)</response_unit>", prompt, _re.S)
            unit = match.group(1).strip() if match else prompt
            if "Поддержка рядом" in unit:
                return {
                    "requires_book_evidence": True,
                    "supported": True,
                    "evidence_passage_ids": ["p1"],
                    "addresses_intent": True,
                }
            return {
                "requires_book_evidence": False,
                "supported": True,
                "evidence_passage_ids": [],
                "addresses_intent": False,
            }

    outcome = await run_v2_answer_turn(
        user_message="Привет! Как обходиться с тягой вечером?",
        summary="",
        recent=[HumanMessage(content="hello")],
        evidence_pack=pack,
        answer_model=_GlueThenGrounded(),
        verifier_model=_Verifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
    )
    # Either the bounded adequacy regeneration served grounded help or the
    # turn failed explicitly to honest unavailability; it must never serve
    # the all-glue draft as a helpful success.
    assert outcome["text"] != glue


def test_gate_c_negative_controls_fail_while_grounded_passes() -> None:
    from aa.conversation import turn_pipeline as _tp  # noqa: F401
    from aa.qualification.product_contract_live import (
        _is_grounded_substantive_reply,
        assess_reply_relevance_with_rubric,
    )

    prompt = "Вечером тяжело пережить тягу, как обходиться?"
    grounded_reply = "Поддержка рядом помогает пережить тягу сегодня."
    assert _is_grounded_substantive_reply(_grounded_snapshot(), grounded_reply) is True
    passing_telemetry = {
        "adequacy_verdict": "pass",
        "answers_request": True,
        "technically_grounded": True,
        "answer_relevant": True,
    }
    assert (
        assess_reply_relevance_with_rubric(prompt, grounded_reply, telemetry=passing_telemetry)
        is True
    )

    # Greeting plus a generic non-answer is not helpful even with counts:
    # production adequacy fails it, so Gate C fails it despite identifiers.
    generic = "Привет! Рад, что ты написал. Расскажешь чуть больше о своей ситуации?"
    generic_snapshot = _grounded_snapshot(
        adequacy_verdict="fail",
        failure_category="all-glue-for-substantive",
        answers_request=False,
        technically_grounded=False,
        qualified=False,
    )
    assert _is_grounded_substantive_reply(generic_snapshot, generic) is False
    failing_telemetry = {
        "adequacy_verdict": "fail",
        "answers_request": False,
        "technically_grounded": False,
        "answer_relevant": False,
    }
    assert assess_reply_relevance_with_rubric(prompt, generic, telemetry=failing_telemetry) is False

    # Out-of-context supported book quote is irrelevant.
    assert (
        assess_reply_relevance_with_rubric(
            prompt,
            "Ведите финансовый бюджет спокойно.",
            telemetry=dict(failing_telemetry),
        )
        is False
    )

    # Citation present but no useful step fails relevance.
    assert (
        assess_reply_relevance_with_rubric(
            prompt, "См. источник PC-S-1.", telemetry=dict(failing_telemetry)
        )
        is False
    )

    # Planner timeout with empty queries never counts as grounded help.
    timeout_snapshot = _grounded_snapshot(
        planner_query_count=0,
        planner_reason="timeout",
        retrieval_passages=0,
        verified_book_units=0,
        adequacy_verdict="fail",
        failure_category="no-evidence-substantive",
        answers_request=False,
        technically_grounded=False,
        qualified=False,
    )
    assert _is_grounded_substantive_reply(timeout_snapshot, grounded_reply) is False

    # Successful zero-query pure greeting is glue, not a substantive answer.
    from aa.conversation.failures import SERVICE_ERROR_REPLY as _SVC251

    assert _is_grounded_substantive_reply(_grounded_snapshot(), _SVC251) is False
    assert _is_grounded_substantive_reply(_grounded_snapshot(), "") is False

    # Successful zero-query pure greeting passes adequacy as glue (separate path).
    from aa.conversation.answer_adequacy import assess_turn_adequacy

    glue_ok = assess_turn_adequacy(
        user_message="Привет",
        reply="Привет! Как ты сегодня?",
        evidence_pack=[],
        grounding_result={
            "verified": True,
            "all_required_supported": True,
            "units": [
                {
                    "unit_id": "u1",
                    "scope": "conversation_glue",
                    "supported": True,
                    "evidence_passage_ids": [],
                }
            ],
        },
        planner_reason="legitimate-glue",
        planner_mode="conversational",
        resolved_intent="",
    )
    assert glue_ok.verdict == "pass"


def test_telemetry_is_causal_and_privacy_safe() -> None:
    import asyncio

    from langchain_core.runnables import RunnableLambda

    from aa.conversation.graph import make_planner_node, turn_input
    from aa.conversation.graph_runtime import GraphTurnRuntime

    async def _run() -> None:
        def _timeout(messages: Any) -> Any:
            _ = messages
            from aa.opencode.errors import OpenCodeTimeoutError

            raise OpenCodeTimeoutError("planner time budget exceeded")

        node = make_planner_node(planner_model=RunnableLambda(_timeout))
        result = await node(turn_input("Привет! Как обходиться с тягой вечером?"))
        assert result["retry_state"]["planner_outcome"] == "timeout"
        assert result["retry_state"]["planner_reason"] == "timeout"
        # Model-driven fallback: a planner error never counts as glue, so
        # the node emits a bounded generic retrieval fallback from the
        # actual turn context instead of an empty query list.
        queries = result["search_queries"]
        assert queries
        assert any("Привет! Как обходиться с тягой вечером?" in item for item in queries)
        assert result["retry_state"]["planner_mode"] == "retrieval"

        runtime = GraphTurnRuntime()
        graph_result = {
            "retry_state": {
                "turn_telemetry": {
                    "planner_outcome": "timeout",
                    "planner_reason": "timeout",
                    "planner_latency_ms": 5.0,
                    "retrieval_outcome": "empty-after-recovery",
                    "retrieval_latency_ms": 1.0,
                    "answer_outcome": "unavailable-substantive-no-evidence",
                    "answer_latency_ms": 1.0,
                    "answer_rounds": 0,
                    "verifier_outcome": "skipped",
                    "verifier_latency_ms": 0.0,
                    "verifier_unavailable_units": 0,
                    "repair_rounds": 0,
                    "repair_budget_exceeded": False,
                    "turn_budget_exceeded": False,
                    "adequacy_verdict": "fail",
                    "failure_category": "no-evidence-substantive",
                    "answers_request": False,
                    "technically_grounded": False,
                    "qualified": False,
                    "turn_trace_id": "abc123def4567890",
                    "runtime_sha": "6f8d4e1e13a2bd1684ad7f91314b1934dc4b99c8",
                }
            },
            "search_queries": [],
            "evidence_pack": [],
            "grounding_result": {"all_required_supported": False, "units": []},
        }
        runtime._record_stage_telemetry("thread", graph_result, 10.0, 42)
        snapshot = runtime.last_telemetry_for_thread("thread")
        assert snapshot["planner_reason"] == "timeout"
        assert snapshot["adequacy_verdict"] == "fail"
        assert snapshot["failure_category"] == "no-evidence-substantive"
        assert snapshot["turn_trace_id"] == "abc123def4567890"
        assert snapshot["runtime_sha"] == "6f8d4e1e13a2bd1684ad7f91314b1934dc4b99c8"
        assert snapshot["verified_book_units"] == 0
        dumped = repr(snapshot)
        assert "тягой" not in dumped

    asyncio.run(_run())


def test_live_lane_proves_usefulness_on_sent_message() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    live_source = (root / "src" / "aa" / "qualification" / "product_contract_live.py").read_text(
        encoding="utf-8"
    )
    assert "mixed-greeting-substantive" in live_source
    assert "held_out_scenarios" in live_source
    assert "book_grounded_families" in live_source
    assert "live-answer-relevance-" in live_source
    assert "assess_reply_relevance_with_rubric" in live_source
    assert "_assess_live_relevance" in live_source
    assert "load_held_out_corpus" in live_source
    assert "api.sent_texts[-1]" in live_source
    assert "last_telemetry_for_thread" in live_source
    assert "adequacy_verdict" in live_source
    assert "_assess_prompt_reply_relevance" not in live_source
    assert "_extract_step_numbers_for_relevance" not in live_source
    assert "_prompt_allows_context_rescue" not in live_source


def test_no_exact_question_whitelist_in_product() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [
        (root / "src" / "aa" / "conversation" / "turn_pipeline.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "graph.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "answer_adequacy.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "planner_node.py").read_text(encoding="utf-8"),
        (root / "src" / "aa" / "conversation" / "verifier.py").read_text(encoding="utf-8"),
    ]
    frozen_fragments = (
        "тянет выпить",
        "тянеет выпить",
        "что мне делать-то",
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
        "Поддержка рядом помогает",
    )
    for source in sources:
        for fragment in frozen_fragments:
            assert fragment not in source
