"""P0 kodmial/aa#257 recurrence 3: production/qualification delivery contract.

Proven failure on exact main 3665f171ae03588d8be95405990e7613d7b53513
(run 37824621714): ``C:live-book-grounding-substantive-drinking-2`` on
``live-production-path``.

Per-stage evidence comparison with the recurrence-2 base (exact main
5b3fd77 run 37808560496: repair_turns=0, repair_rounds=0,
budget_exceeded=0, zero verifier-unavailable units over 46 response
units, answer_rounds=20): the new run shows the identical signature
(repair_turns=0, repair_rounds=0, budget_exceeded=0, zero unavailable
units over 50 response units, answer_rounds=21) with every stage within
budget (planner p50 5.1s, retrieval p50 0.6s, answer p50 7.6s, verifier
p50 8.1s). Neither the recurrence-1 predicate-count relaxation nor the
recurrence-2 outbound-safety clause rescoping converged, and transport
is healthy (unavailable 0/0), so repeating either symptom patch cannot
converge.

Dominant persistent cause at the production/qualification delivery
boundary (touched by neither prior repair): the pipeline serves a
verifier-passing draft whose whole-turn adequacy initially fails, then
delivers the verified adequate subset as ``narrowed-adequacy`` or
``narrowed-adequacy-regen`` with ``repair_rounds == 0`` (the adequacy
regen path does not count as a repair round) while keeping the
repair-history mark ``adequacy-repair-failed`` on the turn. The Gate C
grounding predicate rejected both the outcome tokens and the history
mark, so a certified book-supported delivery counted as ungrounded and
the single drinking turn failed while relevance passed.

Fix (turn-independent, no exact-question special cases, Product
Contract #110 unchanged): the grounding allowlist recognizes the two
pipeline-certified narrowing outcomes, the existing-pack repair success
token on served answers, and exactly the subset-delivery history mark
on exactly the two narrowing outcomes. Every other outcome, mark, or
unavailable/budget failure still fails closed.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

_REQUEST = "Вечером тяжело пережить тягу, как обходиться?"
_PASSAGE = "Спокойный вечер с книгой помогает отдохнуть."
_INITIAL_DRAFT = "Вечером можно спокойно почитать книгу и отдохнуть."
_REGEN_DRAFT = "Вечерняя тяга проходит, поддержка спонсора помогает сегодня. Купите акции сегодня."
_NARROWED_HELP = "Вечерняя тяга проходит, поддержка спонсора помогает сегодня."


def _pack_entry() -> dict[str, Any]:
    return {
        "passage_id": "chapter-3#exp0000",
        "text": _PASSAGE,
        "source_id": "ru-fourth-edition-txt",
        "section_id": "chapter-3",
        "char_start": 0,
        "char_end": len(_PASSAGE),
        "text_sha256": hashlib.sha256(_PASSAGE.encode("utf-8")).hexdigest(),
    }


class _TwoDraftAnswer:
    """Serve an inadequate-then-partially-adequate draft sequence."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages: object) -> AIMessage:
        _ = messages
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content=_INITIAL_DRAFT)
        return AIMessage(content=_REGEN_DRAFT)


class _MarkerVerifier:
    """Support every unit except the explicit off-topic filler."""

    async def ainvoke_structured(
        self, prompt: str, *, system: str, schema: dict[str, object], retry_count: int = 2
    ) -> dict[str, object]:
        _ = (system, schema, retry_count)
        if "Купите акции" in prompt:
            return {
                "requires_book_evidence": True,
                "supported": False,
                "evidence_passage_ids": ["chapter-3#exp0000"],
            }
        return {
            "requires_book_evidence": True,
            "supported": True,
            "evidence_passage_ids": ["chapter-3#exp0000"],
        }


def _snapshot(**overrides: object) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "answer_outcome": "narrowed-adequacy-regen",
        "verifier_outcome": "passed",
        "verifier_unavailable_units": 0,
        "turn_budget_exceeded": False,
        "planner_query_count": 12,
        "retrieval_passages": 1,
        "verified_book_units": 1,
        "response_units": 1,
        "adequacy_verdict": "pass",
        "failure_category": "adequacy-repair-failed",
        "planner_reason": "substantive-with-queries",
        "answers_request": True,
        "technically_grounded": True,
        "qualified": True,
    }
    snapshot.update(overrides)
    return snapshot


async def test_pipeline_serves_narrowed_subset_with_zero_repair_rounds() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn

    outcome = await run_v2_answer_turn(
        user_message=_REQUEST,
        summary="",
        recent=[HumanMessage(content="Здравствуйте")],
        evidence_pack=[_pack_entry()],
        answer_model=_TwoDraftAnswer(),
        verifier_model=_MarkerVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        upstream_latency_ms=0.0,
    )
    telemetry = dict(outcome.get("telemetry", {}))
    assert outcome["text"] == _NARROWED_HELP
    assert telemetry.get("answer_outcome") == "narrowed-adequacy-regen"
    assert int(telemetry.get("repair_rounds", -1)) == 0
    assert telemetry.get("adequacy_verdict") == "pass"
    assert telemetry.get("failure_category") == "adequacy-repair-failed"


async def test_grounding_accepts_pipeline_narrowed_delivery() -> None:
    from aa.conversation.turn_pipeline import run_v2_answer_turn
    from aa.qualification.product_contract_live import (
        _assess_prompt_reply_relevance,
        _is_grounded_substantive_reply,
    )

    outcome = await run_v2_answer_turn(
        user_message=_REQUEST,
        summary="",
        recent=[HumanMessage(content="Здравствуйте")],
        evidence_pack=[_pack_entry()],
        answer_model=_TwoDraftAnswer(),
        verifier_model=_MarkerVerifier(),
        planner_model=None,
        retrieval_index=None,
        initial_query_count=12,
        upstream_latency_ms=0.0,
    )
    telemetry = dict(outcome.get("telemetry", {}))
    # Shape the snapshot exactly as GraphTurnRuntime maps pipeline
    # telemetry into the qualification snapshot.
    snapshot = _snapshot(
        answer_outcome=str(telemetry.get("answer_outcome", "unknown")),
        verifier_outcome=str(telemetry.get("verifier_outcome", "unknown")),
        adequacy_verdict=str(telemetry.get("adequacy_verdict", "unknown")),
        failure_category=str(telemetry.get("failure_category", "")),
        answers_request=bool(telemetry.get("answers_request", False)),
        technically_grounded=bool(telemetry.get("technically_grounded", False)),
        qualified=bool(telemetry.get("qualified", False)),
    )
    reply = str(outcome["text"])
    assert _assess_prompt_reply_relevance(_REQUEST, reply) is True
    assert _is_grounded_substantive_reply(snapshot, reply) is True


def test_grounding_accepts_narrowed_adequacy_outcome() -> None:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    snapshot = _snapshot(answer_outcome="narrowed-adequacy")
    assert _is_grounded_substantive_reply(snapshot, _NARROWED_HELP) is True


def test_grounding_accepts_served_existing_pack_repair_token() -> None:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    snapshot = _snapshot(
        answer_outcome="served",
        verifier_outcome="passed-after-repair-existing-pack",
        failure_category="",
    )
    assert _is_grounded_substantive_reply(snapshot, _NARROWED_HELP) is True


def test_grounding_still_fails_closed_on_other_marks_and_outcomes() -> None:
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    # A different failure mark on a narrowing outcome still fails.
    assert (
        _is_grounded_substantive_reply(
            _snapshot(failure_category="irrelevant-citation"), _NARROWED_HELP
        )
        is False
    )
    # The subset-delivery mark on a non-narrowing outcome still fails.
    assert (
        _is_grounded_substantive_reply(
            _snapshot(answer_outcome="narrowed-supported"), _NARROWED_HELP
        )
        is False
    )
    assert (
        _is_grounded_substantive_reply(
            _snapshot(answer_outcome="adequacy-repair-failed"), _NARROWED_HELP
        )
        is False
    )
    # A served turn with an unsupported verifier verdict still fails.
    assert (
        _is_grounded_substantive_reply(
            _snapshot(
                answer_outcome="served",
                verifier_outcome="unsupported",
                failure_category="",
            ),
            _NARROWED_HELP,
        )
        is False
    )
    # A served existing-pack token with a failed adequacy verdict still fails.
    assert (
        _is_grounded_substantive_reply(
            _snapshot(
                answer_outcome="served",
                verifier_outcome="passed-after-repair-existing-pack",
                failure_category="",
                adequacy_verdict="fail",
            ),
            _NARROWED_HELP,
        )
        is False
    )


def test_no_exact_live_question_special_cases() -> None:
    from aa.qualification import product_contract_live as live

    source = inspect.getsource(live._is_grounded_substantive_reply)
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
