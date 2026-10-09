"""P0 kodmial/aa#257 recurrence 4: safety-recovered delivery contract.

Proven failure on exact main 922c521e8ccef44d4b45839749fcff7e7fea5902
(run 37830100780): ``C:live-book-grounding-substantive-drinking-2`` on
``live-production-path`` with repair_turns=1, repair_rounds=1,
answer_rounds=21, budget_exceeded=0, zero verifier-unavailable units
over 47 response units (max 4).

Per-stage comparison with the recurrence-3 base (exact main 3665f17
run 37824621714: repair_turns=0, repair_rounds=0, budget_exceeded=0,
zero unavailable over 50 units, answer_rounds=21, every stage within
budget): transport is healthy in both runs and the narrowing-adequacy
allowlist (recurrence 3), the predicate-count relaxation
(recurrence 1) and the outbound-safety clause rescoping (recurrence 2)
all left this path untouched, so repeating any of them cannot
converge. The planner/retrieval/answer/verifier p50s are all within
budget in the new run, and the single repair round proves the
targeted/adequacy repair fires yet the drinking turn still fails while
relevance passes.

Dominant persistent cause at the production/qualification delivery
boundary: the outbound safety gate blocks a verifier-passing draft
that advises drinking, then serves a diversified safe book-supported
regeneration as ``served`` with ``outbound_safety=repaired`` but
without ever recording whole-turn adequacy. The returned telemetry
keeps ``adequacy_verdict=unknown`` and
``answers_request/technically_grounded/qualified=False``, so Gate C
counts a certified safe book delivery as ungrounded. The same turn
may also carry the one targeted/adequacy repair round, which explains
the new repair_turns=1 signature on the same stable category.

Fix (turn-independent, no exact-question special cases, Product
Contract #110 unchanged): certify the safety-recovered text exactly
like a normal served candidate -- refresh the evidence window,
refresh verifier telemetry, reset the stale history mark, record
whole-turn adequacy on the recovered text, and serve it only when
adequate. An inadequate safe recovery falls through to the
transparent safety-blocked reply instead of a false success.
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

_HARMFUL_DRAFT = (
    "Многие пытались доказать себе, что они исключение, с помощью разных уловок. "
    "Честно разобраться помогает такой опыт: попробовать начать пить "
    "и резко прекратить, несколько раз."
)

_SAFE_RECOVERY = (
    "Поддержка рядом помогает пережить тягу сегодня, "
    "а честный разговор проясняет ближайшие трезвые шаги."
)


def _pack_entry(passage_id: str, text: str) -> dict[str, Any]:
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


class _HarmfulThenSafeAnswer:
    """Serve an unsafe verifier-passing draft, then a safe recovery draft."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages: object) -> AIMessage:
        _ = messages
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content=_HARMFUL_DRAFT)
        return AIMessage(content=_SAFE_RECOVERY)


class _SupportingVerifier:
    """Support every unit with the pack's display passage."""

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


def _fresh_pack() -> Any:
    from aa.retrieval.evidence import EvidencePack, EvidencePassageData

    text = "Честный разговор проясняет ближайшие трезвые шаги."
    passage = EvidencePassageData(
        passage_id="chapter-3#exp0001",
        exact_text=text,
        source_id="ru-fourth-edition-txt",
        section_id="chapter-3",
        child_chunk_ids=("chapter-3#exp0001",),
        char_start=0,
        char_end=len(text),
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_sha256=hashlib.sha256(b"src").hexdigest(),
    )
    return EvidencePack(
        passages=(passage,),
        total_tokens=10,
        corpus_version="test",
        retrieval_metadata={},
    )


async def test_safety_recovery_serves_certified_delivery() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    monkey_calls = {"retrieval": 0}

    def _fake_retrieve(  # noqa: E501
        index: object, queries: object, *, config: object = None, **kwargs: Any
    ) -> Any:
        _ = (index, queries, config)
        monkey_calls["retrieval"] += 1
        return _fresh_pack()

    import aa.retrieval.evidence as _evidence

    _original = _evidence.retrieve_evidence
    _evidence.retrieve_evidence = _fake_retrieve
    try:
        outcome = await run_v2_answer_turn(
            user_message="Как мне бросить пить?",
            summary="",
            recent=[HumanMessage(content="Здравствуйте")],
            evidence_pack=[
                _pack_entry(
                    "chapter-3#exp0000",
                    "Поддержка рядом помогает пережить тягу сегодня.",
                )
            ],
            answer_model=_HarmfulThenSafeAnswer(),
            verifier_model=_SupportingVerifier(),
            planner_model=object(),
            retrieval_index=object(),
            initial_query_count=12,
            upstream_latency_ms=0.0,
        )
    finally:
        _evidence.retrieve_evidence = _original

    telemetry = dict(outcome.get("telemetry", {}))
    assert outcome["text"] == _SAFE_RECOVERY
    assert telemetry.get("answer_outcome") == "served"
    assert telemetry.get("outbound_safety") == "repaired"
    assert telemetry.get("adequacy_verdict") == "pass"
    assert telemetry.get("answers_request") is True
    assert telemetry.get("technically_grounded") is True
    assert telemetry.get("qualified") is True
    assert telemetry.get("failure_category") == ""
    assert telemetry.get("verifier_unavailable_units") == 0


async def test_safety_recovered_snapshot_passes_grounding_and_relevance() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    def _fake_retrieve(  # noqa: E501
        index: object, queries: object, *, config: object = None, **kwargs: Any
    ) -> Any:
        _ = (index, queries, config)
        return _fresh_pack()

    import aa.retrieval.evidence as _evidence

    _original = _evidence.retrieve_evidence
    _evidence.retrieve_evidence = _fake_retrieve
    try:
        outcome = await run_v2_answer_turn(
            user_message="Как мне бросить пить?",
            summary="",
            recent=[HumanMessage(content="Здравствуйте")],
            evidence_pack=[
                _pack_entry(
                    "chapter-3#exp0000",
                    "Поддержка рядом помогает пережить тягу сегодня.",
                )
            ],
            answer_model=_HarmfulThenSafeAnswer(),
            verifier_model=_SupportingVerifier(),
            planner_model=object(),
            retrieval_index=object(),
            initial_query_count=12,
            upstream_latency_ms=0.0,
        )
    finally:
        _evidence.retrieve_evidence = _original

    telemetry = dict(outcome.get("telemetry", {}))
    verification = dict(outcome.get("verification", {}))
    units = verification.get("units", [])
    verified = sum(
        1
        for unit in units
        if isinstance(unit, dict)
        and unit.get("scope") == "book"
        and unit.get("supported") is True
        and bool(unit.get("evidence_passage_ids"))
    )
    snapshot = {
        "answer_outcome": str(telemetry.get("answer_outcome", "")),
        "verifier_outcome": str(telemetry.get("verifier_outcome", "")),
        "verifier_unavailable_units": int(telemetry.get("verifier_unavailable_units", 0)),
        "turn_budget_exceeded": bool(telemetry.get("turn_budget_exceeded", False)),
        "planner_query_count": int(telemetry.get("planner_query_count", 12) or 12),
        "retrieval_passages": int(telemetry.get("retrieval_passages", 1) or 1),
        "verified_book_units": int(verified),
        "response_units": int(len(units) if isinstance(units, list) else 0),
        "adequacy_verdict": str(telemetry.get("adequacy_verdict", "")),
        "failure_category": str(telemetry.get("failure_category", "")),
        "planner_reason": str(telemetry.get("planner_reason", "substantive-with-queries")),
        "answers_request": bool(telemetry.get("answers_request", False)),
        "technically_grounded": bool(telemetry.get("technically_grounded", False)),
        "qualified": bool(telemetry.get("qualified", False)),
    }
    reply = str(outcome["text"])
    # Task #268: relevance comes from the single structured semantic
    # verifier invocation (groundedness + answer relevance together),
    # never from lexical prefix/token-overlap or domain-stem heuristics.
    assert verification.get("answer_relevant") is True
    assert any(
        isinstance(unit, dict)
        and unit.get("scope") == "book"
        and unit.get("supported") is True
        and unit.get("addresses_intent") is True
        for unit in units
    )
    assert _is_grounded_substantive_reply(snapshot, reply) is True


def test_no_exact_live_question_special_cases() -> None:
    import inspect

    from aa.conversation import turn_pipeline as _pipeline

    source = inspect.getsource(_pipeline.run_v2_answer_turn)
    for fragment in (
        "тянет выпить",
        "тянеет выпить",
        "Поругались дома",
        "одному не получается",
        "покупать акции",
        "покончить с собой",
        "чем помочь можешь",
    ):
        assert fragment not in source
