"""Regression for a real, runnable whole-turn independent AA judge.

Gate C run 37865118363 recorded 17 independent judge calls with zero
helpful verdicts and 14 production-verdict overrides. The judge was
requested under logical/wire name 'aa-judge-v2', absent from opencode.json.
An unknown agent plus swallowed transport/validation errors looked like
seventeen genuine negative semantic judgments. The product must never
claim PASS on an unavailable judge; the next run must report unavailable
counts and sanitized failure categories separately from semantic FAIL.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aa.conversation.whole_turn_judge import (
    WHOLE_TURN_JUDGE_AGENT_V2,
    judge_whole_turn,
)


def test_real_whole_turn_judge_registered_for_opencode() -> None:
    from aa.qualification.product_contract_live import build_live_whole_turn_judge

    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "opencode.json").read_text(encoding="utf-8"))
    judge = data["agent"][WHOLE_TURN_JUDGE_AGENT_V2]
    assert judge["permission"]["StructuredOutput"] == "allow"
    assert judge["permission"]["*"] == "deny"
    assert judge["temperature"] == 0
    assert judge["model"]
    settings = SimpleNamespace(
        opencode_model="opencode/muse-spark-1.3-contributor-free",
        opencode_fallback_model="opencode/space-bunny-free",
    )
    model = build_live_whole_turn_judge(object(), settings)
    assert model.agent == WHOLE_TURN_JUDGE_AGENT_V2
    assert model.wire_agent == WHOLE_TURN_JUDGE_AGENT_V2


class _TransportUnavailable:
    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
    ) -> dict[str, object]:
        from aa.opencode.errors import OpenCodeDeterministicError

        raise OpenCodeDeterministicError("missing agent")

    async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
        from aa.opencode.errors import OpenCodeDeterministicError

        raise OpenCodeDeterministicError("missing agent")


class _NegativeSemanticVerdict:
    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
    ) -> dict[str, object]:
        return {
            "helpful": False,
            "addresses_intent": False,
            "contains_substantive_claim": True,
        }


class _429Failure:
    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 1
    ) -> dict[str, object]:
        from aa.opencode.errors import OpenCodeRateLimitError

        raise OpenCodeRateLimitError("429")

    async def _ainvoke_text(self, prompt: str, *, system: str = "") -> str:
        raise AssertionError("429 must never run a text fallback")


async def test_transport_unavailable_cannot_be_misreported_as_semantic_negative() -> None:
    failed = await judge_whole_turn(
        resolved_intent="synthetic question",
        reply="Синтетический ответ.",
        model=_TransportUnavailable(),
    )
    assert failed.helpful is False
    assert failed.available is False
    assert failed.failure_category == "OpenCodeDeterministicError"

    negative = await judge_whole_turn(
        resolved_intent="synthetic question",
        reply="Неподходящий ответ.",
        model=_NegativeSemanticVerdict(),
    )
    assert negative.helpful is False
    assert negative.available is True
    assert negative.failure_category == ""


async def test_independent_judge_429_propagates_to_hosted_runner() -> None:
    from aa.opencode.errors import OpenCodeRateLimitError

    with pytest.raises(OpenCodeRateLimitError):
        await judge_whole_turn(
            resolved_intent="synthetic question",
            reply="Тестовый ответ.",
            model=_429Failure(),
        )


async def test_live_judge_429_is_not_counted_as_unhelpful() -> None:
    from aa.opencode.errors import OpenCodeRateLimitError
    from aa.qualification.product_contract_live import (
        assess_live_helpfulness_with_judge_metrics,
    )

    snapshot = {
        "adequacy_verdict": "pass",
        "answers_request": True,
        "technically_grounded": True,
        "answer_relevant": True,
    }
    with pytest.raises(OpenCodeRateLimitError):
        await assess_live_helpfulness_with_judge_metrics(
            prompt="synthetic question",
            snapshot=snapshot,
            reply="Тестовый ответ.",
            judge_model=_429Failure(),
        )
