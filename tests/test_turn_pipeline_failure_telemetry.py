"""Regression tests for failure-path graph telemetry enrichment."""

from __future__ import annotations

from typing import Any

import pytest

from aa.conversation import turn_pipeline
from aa.conversation.failures import TurnFailed


@pytest.mark.asyncio
async def test_answer_pipeline_node_preserves_upstream_stages_when_answer_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure_telemetry: dict[str, Any] = {
        "planner_outcome": "skipped-initial",
        "planner_latency_ms": 0.0,
        "planner_query_count": 0,
        "retrieval_outcome": "preloaded",
        "retrieval_latency_ms": 0.0,
        "answer_outcome": "clarification-unavailable",
    }

    async def _fail_answer(**_kwargs: Any) -> dict[str, Any]:
        raise TurnFailed(
            "clarification-unavailable",
            "no verified answer",
            telemetry=failure_telemetry,
        )

    monkeypatch.setattr(turn_pipeline, "run_v2_answer_turn", _fail_answer)
    state = {
        "route": "normal",
        "current_user_message": "Почему это важно?",
        "messages": [],
        "search_queries": ["контекстно разрешённый запрос"],
        "retry_state": {
            "planner_outcome": "ok",
            "planner_latency_ms": 1250.0,
            "planner_query_count": 1,
            "planner_reason": "substantive",
        },
        "planner_invoked": True,
        "retrieval_latency_ms": 450.0,
        "retrieval_over_budget": False,
        "evidence_pack": [{"source_id": "fixture-source"}],
        "retrieval_metadata": {},
    }

    with pytest.raises(TurnFailed) as caught:
        await turn_pipeline.answer_pipeline_node(
            state,
            answer_model=object(),
            verifier_model=object(),
            planner_model=object(),
        )

    assert caught.value.category == "clarification-unavailable"
    assert caught.value.detail == "no verified answer"
    assert caught.value.telemetry["planner_outcome"] == "ok"
    assert caught.value.telemetry["planner_latency_ms"] == 1250.0
    assert caught.value.telemetry["planner_query_count"] == 1
    assert caught.value.telemetry["planner_reason"] == "substantive"
    assert caught.value.telemetry["retrieval_outcome"] == "evidence-ready"
    assert caught.value.telemetry["retrieval_latency_ms"] == 450.0
    assert caught.value.telemetry["answer_outcome"] == "clarification-unavailable"
