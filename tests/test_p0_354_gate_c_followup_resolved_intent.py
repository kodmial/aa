"""P0 kodmial/aa#354: Gate C follow-up judged against raw ellipsis text.

Proven product failure on exact main 80439a6 (run 38214364493):

- C:live-continuation-helpful-followup-ellipsis-4 (plus D:workflow-step-failed
  downstream and E:latency-budget-exceeded SLO).

Root cause: ``GraphTurnRuntime._record_stage_telemetry`` stored only
counts/latencies/outcomes. It dropped the planner-resolved substantive
intent (``ResolvedTurn.resolved_intent``) and the verifier whole-turn
relevance verdict (``grounding_result.answer_relevant``). The live lane
judges every continuation with
``assess_live_helpfulness_with_judge_metrics``, which reads
``snapshot.get("resolved_intent", "") or cleaned_prompt``. For a terse
follow-up the raw surface prompt carries almost no topical content, so a
correctly resolved and grounded continuation was judged against the
generic ellipsis instead of the substantive resolved intent, while first
turns (whose raw prompts are already substantive) kept passing. That is
exactly the observed split.

Repair (generic, no exact-question special cases; Product Contract #110
unchanged): carry the canonical resolved intent, the explicit
whole-turn relevance verdict and the context digest in the in-memory
stage snapshot (never logged, never persisted as gate evidence) so the
independent judge and the telemetry relevance check see the same
context-resolved intent the planner, selector, generator and verifier
used.
"""

from __future__ import annotations

from typing import Any

from aa.conversation.graph_runtime import GraphTurnRuntime


def _graph_result(
    *,
    resolved_intent: str = "",
    resolved_turn_intent: str | None = None,
    answer_relevant: bool | None = None,
    context_digest: str = "",
) -> dict[str, Any]:
    grounding: dict[str, Any] = {
        "all_required_supported": True,
        "units": [
            {
                "unit_id": "u1",
                "scope": "book",
                "supported": True,
                "evidence_passage_ids": ["chapter-3#exp0000"],
                "addresses_intent": True,
            }
        ],
    }
    if answer_relevant is not None:
        grounding["answer_relevant"] = bool(answer_relevant)
    resolved_turn: dict[str, Any] = {}
    if resolved_turn_intent is not None:
        resolved_turn = {
            "resolved_intent": resolved_turn_intent,
            "context_digest": context_digest or "digest-turn",
        }
    return {
        "retry_state": {
            "turn_telemetry": {
                "planner_outcome": "ok",
                "planner_latency_ms": 10.0,
                "retrieval_outcome": "evidence-ready",
                "answer_outcome": "served",
                "verifier_outcome": "passed",
                "adequacy_verdict": "pass",
                "answers_request": True,
                "technically_grounded": True,
                "qualified": True,
            },
            "resolved_intent": resolved_intent,
        },
        "resolved_turn": resolved_turn,
        "resolved_intent": resolved_intent,
        "context_digest": context_digest,
        "search_queries": ["q1", "q2"],
        "evidence_pack": [{"passage_id": "chapter-3#exp0000"}],
        "grounding_result": grounding,
    }


def test_telemetry_carries_canonical_resolved_intent() -> None:
    runtime = GraphTurnRuntime()
    substantive = "resolved substantive intent for the evening craving follow-up"
    result = _graph_result(
        resolved_intent=substantive,
        resolved_turn_intent=substantive,
        answer_relevant=True,
        context_digest="digest-354",
    )
    runtime._record_stage_telemetry("thread-354", result, 1000.0, 42)
    snapshot = runtime.last_telemetry_for_thread("thread-354")
    assert snapshot["resolved_intent"] == substantive
    assert snapshot["answer_relevant"] is True
    assert snapshot["context_digest"] == "digest-354"


def test_telemetry_prefers_resolved_turn_over_raw_state() -> None:
    runtime = GraphTurnRuntime()
    canonical = "canonical resolved intent from the shared resolved turn"
    result = _graph_result(
        resolved_intent="stale state intent",
        resolved_turn_intent=canonical,
        answer_relevant=True,
    )
    runtime._record_stage_telemetry("thread-354-prefer", result, 1000.0, 42)
    snapshot = runtime.last_telemetry_for_thread("thread-354-prefer")
    assert snapshot["resolved_intent"] == canonical


def test_telemetry_explicit_irrelevant_survives_as_false() -> None:
    runtime = GraphTurnRuntime()
    result = _graph_result(
        resolved_intent="some intent",
        resolved_turn_intent="some intent",
        answer_relevant=False,
    )
    runtime._record_stage_telemetry("thread-354-irrelevant", result, 1000.0, 42)
    snapshot = runtime.last_telemetry_for_thread("thread-354-irrelevant")
    assert snapshot["answer_relevant"] is False


def test_telemetry_missing_relevance_stays_none() -> None:
    runtime = GraphTurnRuntime()
    result = _graph_result(
        resolved_intent="some intent",
        resolved_turn_intent="some intent",
        answer_relevant=None,
    )
    result["grounding_result"] = {
        "all_required_supported": True,
        "units": [],
    }
    runtime._record_stage_telemetry("thread-354-missing", result, 1000.0, 42)
    snapshot = runtime.last_telemetry_for_thread("thread-354-missing")
    assert snapshot["answer_relevant"] is None


def test_telemetry_drives_rubric_relevance_without_keyword_heuristics() -> None:
    from aa.qualification.product_contract_live import assess_reply_relevance_with_rubric

    passing = {
        "adequacy_verdict": "pass",
        "answers_request": True,
        "technically_grounded": True,
        "answer_relevant": True,
    }
    failing = {
        "adequacy_verdict": "pass",
        "answers_request": True,
        "technically_grounded": True,
        "answer_relevant": False,
    }
    assert assess_reply_relevance_with_rubric("prompt", "reply", telemetry=passing) is True
    assert assess_reply_relevance_with_rubric("prompt", "reply", telemetry=failing) is False


async def test_live_helpfulness_uses_snapshot_resolved_intent() -> None:
    from aa.conversation.whole_turn_judge import WholeTurnJudgement
    from aa.qualification.product_contract_live import (
        assess_live_helpfulness_with_judge_metrics,
    )

    seen_intents: list[str] = []

    class _CapturingJudge:
        async def ainvoke_structured(
            self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
        ) -> dict[str, object]:
            del prompt, system, schema, retry_count
            return {
                "helpful": True,
                "addresses_intent": True,
                "contains_substantive_claim": True,
            }

    async def _fake_judge_whole_turn(
        *,
        resolved_intent: str,
        reply: str,
        context: str = "",
        model: Any = None,
    ) -> WholeTurnJudgement:
        del reply, context, model
        seen_intents.append(str(resolved_intent))
        return WholeTurnJudgement(
            helpful=True,
            addresses_intent=True,
            contains_substantive_claim=True,
            model_identity="test-judge",
        )

    import aa.conversation.whole_turn_judge as judge_mod

    raw_generic_prompt = "generic terse follow-up surface text"
    resolved_substantive = "context-resolved substantive intent for the follow-up"
    snapshot = {
        "adequacy_verdict": "pass",
        "answers_request": True,
        "technically_grounded": True,
        "answer_relevant": True,
        "resolved_intent": resolved_substantive,
    }
    original = judge_mod.judge_whole_turn
    judge_mod.judge_whole_turn = _fake_judge_whole_turn
    try:
        helpful, _metrics = await assess_live_helpfulness_with_judge_metrics(
            prompt=raw_generic_prompt,
            snapshot=dict(snapshot),
            reply="substantive grounded reply text",
            context="prior user context",
            judge_model=_CapturingJudge(),
        )
    finally:
        judge_mod.judge_whole_turn = original
    assert helpful is True
    assert seen_intents == [resolved_substantive]


def test_no_exact_question_special_cases() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    source = (root / "src" / "aa" / "conversation" / "graph_runtime.py").read_text(encoding="utf-8")
    for fragment in (
        "почему это вообще важно",
        "что мне делать-то",
        "ночью не могу уснуть",
        "покупать акции",
    ):
        assert fragment not in source
